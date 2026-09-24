from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import threading
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from PIL import Image, PngImagePlugin
from starlette.applications import Starlette

import anpr_engine.inference_service.runtime as inference_runtime
import anpr_engine.integration.anpr_v1 as anpr_v1
from anpr_engine.inference_service.app import create_app
from anpr_engine.inference_service.config import ServiceConfig
from anpr_engine.inference_service.logging import SafeJsonLogger
from anpr_engine.integration.anpr_v1 import DETECTOR_SHA, RECOGNIZER_SHA, FrozenBundle
from anpr_engine.web_contracts.image_validation import (
    ImageContractError,
    validate_encoded_image,
    validate_image_dimensions,
)

REQUEST_ID = "01K5DP5YEX8ZDB7MVT1J0B4G6Q"
ADMISSION_ID = "admission_00000000000000000000001"
SECRET = "test-internal-secret-32-characters-minimum"


class FakePipeline:
    load_count = 1

    def __init__(
        self,
        *,
        pipeline_status: str = "ACCEPTED",
        candidate_count: int = 1,
        fail: bool = False,
        gate: threading.Event | None = None,
        structural_valid: bool = True,
        decoded_text: str | None = "34ABC123",
    ) -> None:
        self.pipeline_status = pipeline_status
        self.candidate_count = candidate_count
        self.fail = fail
        self.gate = gate
        self.structural_valid = structural_valid
        self.decoded_text = decoded_text
        self.calls = 0
        self.observed_shape: tuple[int, ...] | None = None

    def health(self) -> dict[str, object]:
        return {"status": "HEALTHY"}

    def infer(self, image: np.ndarray[Any, Any], **kwargs: object) -> dict[str, Any]:
        assert kwargs["output_directory"] is None
        assert kwargs["filename"] == "not-retained"
        assert kwargs["input_sha256"] == "not-retained"
        self.calls += 1
        self.observed_shape = image.shape
        if self.gate is not None:
            self.gate.wait(timeout=2)
        if self.fail:
            raise RuntimeError("private/path/sensitive-plate-value")
        candidates = [
            {"bbox_xyxy": [10.0 + index, 5.0, 30.0, 15.0], "confidence": 0.9 - index * 0.1}
            for index in range(self.candidate_count)
        ]
        recognition = None
        if candidates and self.pipeline_status != "INVALID_CROP":
            recognition = {
                "tr_text": self.decoded_text,
                "decoder_confidence": 0.88 if self.decoded_text is not None else None,
                "decoding_changed_raw": False,
                "structural_validation": (
                    {"valid": self.structural_valid} if self.decoded_text is not None else None
                ),
            }
        return {
            "pipeline_status": self.pipeline_status,
            "detection": {"candidates": candidates},
            "recognition": recognition,
            "timings_ms": {
                "detection_inference_ms": 1.0,
                "crop_preprocessing_ms": 2.0,
                "recognition_inference_ms": 3.0,
            },
        }


def _config(tmp_path: Path, **updates: object) -> ServiceConfig:
    values: dict[str, object] = {
        "repository": tmp_path,
        "model_bundle": tmp_path / "bundle",
        "environment": "production",
        "internal_credential": SECRET,
        "device": "cpu",
        "inference_deadline_seconds": 5.0,
    }
    values.update(updates)
    return ServiceConfig(**values)  # type: ignore[arg-type]


def _image(image_format: str = "PNG", *, animated: bool = False) -> bytes:
    output = io.BytesIO()
    first = Image.new("RGB", (40, 20), (20, 80, 160))
    if animated:
        second = Image.new("RGB", (40, 20), (160, 80, 20))
        first.save(output, format=image_format, save_all=True, append_images=[second])
    else:
        first.save(output, format=image_format)
    return output.getvalue()


def _headers(media_type: str, payload: bytes, **changes: str | None) -> dict[str, str]:
    headers = {
        "content-type": media_type,
        "content-length": str(len(payload)),
        "x-anpr-request-id": REQUEST_ID,
        "x-anpr-admission-id": ADMISSION_ID,
        "x-anpr-locale": "en",
        "x-anpr-privacy-acknowledged": "true",
        "authorization": f"Bearer {SECRET}",
    }
    for key, value in changes.items():
        if value is None:
            headers.pop(key, None)
        else:
            headers[key] = value
    return headers


async def _request(
    app: Starlette,
    method: str,
    path: str,
    *,
    body: bytes = b"",
    headers: Mapping[str, str] | None = None,
    chunks: tuple[bytes, ...] | None = None,
    block_after_first_chunk: asyncio.Event | None = None,
) -> tuple[int, dict[str, Any] | None]:
    sent: list[dict[str, Any]] = []
    pending = list(chunks if chunks is not None else (body,))
    received = 0

    async def receive() -> dict[str, Any]:
        nonlocal received
        if received and block_after_first_chunk is not None:
            await block_after_first_chunk.wait()
        chunk = pending.pop(0) if pending else b""
        received += 1
        return {"type": "http.request", "body": chunk, "more_body": bool(pending)}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(dict(message))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
        ],
        "client": ("127.0.0.1", 10000),
        "server": ("127.0.0.1", 8080),
    }
    await app(scope, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    raw = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return status, json.loads(raw) if raw else None


async def _ready_app(
    tmp_path: Path,
    pipeline: FakePipeline | None = None,
    *,
    logger: SafeJsonLogger | None = None,
) -> Starlette:
    selected = pipeline or FakePipeline()
    app = create_app(
        _config(tmp_path),
        pipeline_factory=lambda _config: selected,
        logger=logger,
        initialize_on_startup=False,
    )
    await app.state.runtime.initialize()
    return app


def test_health_is_liveness_only_and_readiness_transitions(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = create_app(
            _config(tmp_path),
            pipeline_factory=lambda _config: FakePipeline(),
            initialize_on_startup=False,
        )
        status, payload = await _request(app, "GET", "/healthz")
        assert (status, payload) == (200, {"status": "alive"})
        status, payload = await _request(app, "GET", "/readyz")
        assert status == 503 and payload == {"status": "not_ready", "reason": "not_initialized"}
        await app.state.runtime.initialize()
        status, payload = await _request(app, "GET", "/readyz")
        assert (status, payload) == (200, {"status": "ready"})
        await app.state.runtime.close()
        assert not app.state.runtime.ready
        assert app.state.runtime.pipeline is None

    asyncio.run(scenario())


def test_production_without_credential_fails_closed(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = create_app(
            _config(tmp_path, internal_credential=None),
            pipeline_factory=lambda _config: FakePipeline(),
            initialize_on_startup=False,
        )
        await app.state.runtime.initialize()
        assert not app.state.runtime.ready
        status, payload = await _request(app, "GET", "/readyz")
        assert status == 503 and "credential" not in json.dumps(payload)
        await app.state.runtime.close()

    asyncio.run(scenario())


def test_missing_bundle_and_each_hash_mismatch_fail_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    bundle = FrozenBundle.open(bundle_root)
    with pytest.raises(FileNotFoundError):
        bundle.verify()

    async def assert_unready() -> None:
        def verify_then_build(_config: ServiceConfig) -> FakePipeline:
            bundle.verify()
            return FakePipeline()

        app = create_app(
            _config(tmp_path),
            pipeline_factory=verify_then_build,
            initialize_on_startup=False,
        )
        await app.state.runtime.initialize()
        assert not app.state.runtime.ready
        await app.state.runtime.close()

    asyncio.run(assert_unready())
    bundle.manifest.write_text(
        json.dumps(
            {
                "schema_version": "anpr-engine-private-model-bundle-v1",
                "classification": "PRIVATE_OWNER_ONLY",
                "models": [
                    {"role": "detection", "sha256": DETECTOR_SHA},
                    {"role": "recognition", "sha256": RECOGNIZER_SHA},
                ],
            }
        )
    )
    bundle.detector.write_bytes(b"detection")
    bundle.recognizer.write_bytes(b"recognition")
    monkeypatch.setattr(
        anpr_v1,
        "sha256_file",
        lambda path: "wrong" if path == bundle.detector else RECOGNIZER_SHA,
    )
    with pytest.raises(ValueError, match="detector bundle hash mismatch"):
        bundle.verify()
    asyncio.run(assert_unready())
    monkeypatch.setattr(
        anpr_v1,
        "sha256_file",
        lambda path: DETECTOR_SHA if path == bundle.detector else "wrong",
    )
    with pytest.raises(ValueError, match="recognizer bundle hash mismatch"):
        bundle.verify()
    asyncio.run(assert_unready())


@pytest.mark.parametrize(
    ("image_format", "media_type"), [("JPEG", "image/jpeg"), ("PNG", "image/png")]
)
def test_valid_raw_images_return_versioned_provenance(
    tmp_path: Path, image_format: str, media_type: str
) -> None:
    async def scenario() -> None:
        app = await _ready_app(tmp_path)
        payload = _image(image_format)
        status, result = await _request(
            app, "POST", "/api/v1/inference", body=payload, headers=_headers(media_type, payload)
        )
        assert status == 200 and result is not None
        assert result["schema_version"] == "anpr-private-inference-response-v1"
        assert result["prediction"] == "34ABC123"
        assert result["models"]["detection"]["sha256"] == DETECTOR_SHA
        assert result["models"]["recognition"]["sha256"] == RECOGNIZER_SHA
        assert result["crop_policy"] == "BLUE_BAND_GLYPH_QUAD_V1"
        assert not {"filename", "path", "image_hash"} & set(result)
        schema = json.loads(
            (
                Path(__file__).parents[1]
                / "schemas/internal/v1/anpr-private-inference-response-v1.schema.json"
            ).read_text()
        )
        Draft202012Validator(schema).validate(result)
        await app.state.runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"authorization": None}, 403),
        ({"authorization": "Bearer wrong"}, 403),
        ({"x-anpr-privacy-acknowledged": None}, 400),
        ({"x-anpr-admission-id": "short"}, 400),
    ],
)
def test_internal_authentication_and_metadata_fail_closed(
    tmp_path: Path, change: dict[str, str | None], expected: int
) -> None:
    async def scenario() -> None:
        app = await _ready_app(tmp_path)
        payload = _image()
        status, _result = await _request(
            app,
            "POST",
            "/api/v1/inference",
            body=payload,
            headers=_headers("image/png", payload, **change),
        )
        assert status == expected
        await app.state.runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("payload", "media_type", "expected"),
    [
        (_image("PNG"), "application/octet-stream", "UNSUPPORTED_MEDIA_TYPE"),
        (_image("PNG"), "multipart/form-data", "UNSUPPORTED_MEDIA_TYPE"),
        (_image("PNG"), "image/jpeg", "UNSUPPORTED_MEDIA_TYPE"),
        (b"not-an-image", "image/png", "UNSUPPORTED_MEDIA_TYPE"),
        (_image("PNG")[:-8], "image/png", "IMAGE_TRUNCATED"),
        (_image("PNG", animated=True), "image/png", "IMAGE_MULTIFRAME_REJECTED"),
        (_image("PNG") + _image("PNG"), "image/png", "IMAGE_MULTIFRAME_REJECTED"),
    ],
)
def test_invalid_image_inputs_are_safe(
    tmp_path: Path, payload: bytes, media_type: str, expected: str
) -> None:
    async def scenario() -> None:
        app = await _ready_app(tmp_path)
        status, result = await _request(
            app, "POST", "/api/v1/inference", body=payload, headers=_headers(media_type, payload)
        )
        assert status in {413, 415, 422}
        assert result is not None and result["error"]["code"] == expected
        assert "not-an-image" not in json.dumps(result)
        await app.state.runtime.close()

    asyncio.run(scenario())


def test_stream_limit_applies_without_or_with_misleading_content_length(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = await _ready_app(tmp_path)
        oversized = b"x" * (10 * 1024 * 1024 + 1)
        headers = _headers("image/png", oversized, **{"content-length": "1"})
        status, result = await _request(
            app,
            "POST",
            "/api/v1/inference",
            headers=headers,
            chunks=(oversized[:5_000_000], oversized[5_000_000:]),
        )
        assert status == 413 and result is not None
        assert result["error"]["code"] == "IMAGE_TOO_LARGE"
        await app.state.runtime.close()

    asyncio.run(scenario())


def test_dimension_and_decompression_bomb_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ImageContractError, match="IMAGE_DIMENSIONS_EXCEEDED"):
        validate_image_dimensions(8193, 1)
    with pytest.raises(ImageContractError, match="IMAGE_DIMENSIONS_EXCEEDED"):
        validate_image_dimensions(5000, 5000)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ImageContractError, match="IMAGE_DIMENSIONS_EXCEEDED"):
        validate_encoded_image(_image(), "image/png")


def test_busy_request_fails_without_queue_or_duplicate_execution(tmp_path: Path) -> None:
    async def scenario() -> None:
        gate = threading.Event()
        pipeline = FakePipeline(gate=gate)
        app = await _ready_app(tmp_path, pipeline)
        payload = _image()
        first = asyncio.create_task(
            _request(
                app,
                "POST",
                "/api/v1/inference",
                body=payload,
                headers=_headers("image/png", payload),
            )
        )
        while not app.state.runtime.active:
            await asyncio.sleep(0)
        second_headers = _headers(
            "image/png",
            payload,
            **{
                "x-anpr-admission-id": "admission_00000000000000000000002",
                "x-anpr-request-id": "01K5DP5YEX8ZDB7MVT1J0B4G6R",
            },
        )
        status, result = await _request(
            app, "POST", "/api/v1/inference", body=payload, headers=second_headers
        )
        assert status == 503 and result is not None
        assert result["error"]["code"] == "CAPACITY_UNAVAILABLE"
        assert pipeline.calls <= 1
        gate.set()
        assert (await first)[0] == 200
        assert pipeline.calls == 1
        await app.state.runtime.close()

    asyncio.run(scenario())


def test_timeout_and_client_cancellation_keep_slot_until_execution_stops(tmp_path: Path) -> None:
    async def scenario() -> None:
        gate = threading.Event()
        pipeline = FakePipeline(gate=gate)
        app = create_app(
            _config(tmp_path, inference_deadline_seconds=1.0),
            pipeline_factory=lambda _config: pipeline,
            initialize_on_startup=False,
        )
        await app.state.runtime.initialize()
        payload = _image()
        status, result = await _request(
            app,
            "POST",
            "/api/v1/inference",
            body=payload,
            headers=_headers("image/png", payload),
        )
        assert status == 504 and result is not None
        assert result["error"]["code"] == "INFERENCE_DEADLINE_EXCEEDED"
        assert app.state.runtime.active
        gate.set()
        while app.state.runtime.active:
            await asyncio.sleep(0)

        gate.clear()
        next_headers = _headers(
            "image/png", payload, **{"x-anpr-request-id": "01K5DP5YEX8ZDB7MVT1J0B4G6R"}
        )
        task = asyncio.create_task(
            _request(
                app,
                "POST",
                "/api/v1/inference",
                body=payload,
                headers=next_headers,
            )
        )
        while not app.state.runtime.active:
            await asyncio.sleep(0)
        # Admission becomes active before image decoding and worker execution.
        # Cancel only after infer has entered the gated worker; otherwise an
        # immediate release is correct and this test races under CPU load.
        while pipeline.calls < 2:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert app.state.runtime.active
        gate.set()
        while app.state.runtime.active:
            await asyncio.sleep(0)
        await app.state.runtime.close()

    asyncio.run(scenario())


def test_stream_timeout_and_cancellation_release_unstarted_admission(tmp_path: Path) -> None:
    async def scenario() -> None:
        pipeline = FakePipeline()
        app = create_app(
            _config(tmp_path, inference_deadline_seconds=1.0),
            pipeline_factory=lambda _config: pipeline,
            initialize_on_startup=False,
        )
        await app.state.runtime.initialize()
        payload = _image()
        blocked = asyncio.Event()
        chunks = (payload[:8], payload[8:])
        status, response = await _request(
            app,
            "POST",
            "/api/v1/inference",
            body=payload,
            headers=_headers("image/png", payload),
            chunks=chunks,
            block_after_first_chunk=blocked,
        )
        assert status == 504 and response is not None
        assert response["error"]["code"] == "INFERENCE_DEADLINE_EXCEEDED"
        assert not app.state.runtime.active
        assert pipeline.calls == 0

        task = asyncio.create_task(
            _request(
                app,
                "POST",
                "/api/v1/inference",
                body=payload,
                headers=_headers(
                    "image/png", payload, **{"x-anpr-request-id": "01K5DP5YEX8ZDB7MVT1J0B4G6R"}
                ),
                chunks=chunks,
                block_after_first_chunk=blocked,
            )
        )
        while not app.state.runtime.active:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not app.state.runtime.active
        status, _response = await _request(
            app,
            "POST",
            "/api/v1/inference",
            body=payload,
            headers=_headers(
                "image/png", payload, **{"x-anpr-request-id": "01K5DP5YEX8ZDB7MVT1J0B4G6S"}
            ),
        )
        assert status == 200 and pipeline.calls == 1
        await app.state.runtime.close()

    asyncio.run(scenario())


def test_decode_timeout_keeps_slot_until_worker_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    gate = threading.Event()
    original = inference_runtime._validate_decode

    def slow_decode(payload: bytes, media_type: str) -> np.ndarray[Any, Any]:
        entered.set()
        gate.wait(timeout=3)
        return original(payload, media_type)

    monkeypatch.setattr(inference_runtime, "_validate_decode", slow_decode)

    async def scenario() -> None:
        pipeline = FakePipeline()
        app = create_app(
            _config(tmp_path, inference_deadline_seconds=1.0),
            pipeline_factory=lambda _config: pipeline,
            initialize_on_startup=False,
        )
        await app.state.runtime.initialize()
        payload = _image()
        first = asyncio.create_task(
            _request(
                app,
                "POST",
                "/api/v1/inference",
                body=payload,
                headers=_headers("image/png", payload),
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0)
        assert (await _request(app, "GET", "/healthz"))[0] == 200
        status, response = await first
        assert status == 504 and response is not None
        assert response["error"]["code"] == "INFERENCE_DEADLINE_EXCEEDED"
        assert app.state.runtime.active
        status, response = await _request(
            app,
            "POST",
            "/api/v1/inference",
            body=payload,
            headers=_headers(
                "image/png", payload, **{"x-anpr-request-id": "01K5DP5YEX8ZDB7MVT1J0B4G6S"}
            ),
        )
        assert status == 503 and response is not None
        assert response["error"]["code"] == "CAPACITY_UNAVAILABLE"
        assert pipeline.calls == 0
        gate.set()
        while app.state.runtime.active:
            await asyncio.sleep(0)
        gate.clear()
        entered.clear()
        cancelled = asyncio.create_task(
            _request(
                app,
                "POST",
                "/api/v1/inference",
                body=payload,
                headers=_headers(
                    "image/png", payload, **{"x-anpr-request-id": "01K5DP5YEX8ZDB7MVT1J0B4G6T"}
                ),
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert app.state.runtime.active
        gate.set()
        while app.state.runtime.active:
            await asyncio.sleep(0)
        assert pipeline.calls == 0
        await app.state.runtime.close()

    asyncio.run(scenario())


def test_orientation_geometry_uses_normalized_pixel_dimensions(tmp_path: Path) -> None:
    async def scenario() -> None:
        pipeline = FakePipeline()
        app = await _ready_app(tmp_path, pipeline)
        output = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (40, 20)).save(output, format="JPEG", exif=exif)
        payload = output.getvalue()
        status, response = await _request(
            app,
            "POST",
            "/api/v1/inference",
            body=payload,
            headers=_headers("image/jpeg", payload),
        )
        assert status == 200 and response is not None
        assert pipeline.observed_shape == (40, 20, 3)
        assert response["detection"]["box_xyxy_normalized"][0] == pytest.approx(0.5)
        await app.state.runtime.close()

    asyncio.run(scenario())


def test_embedded_png_signature_in_metadata_is_one_valid_image() -> None:
    metadata = PngImagePlugin.PngInfo()
    metadata.add(b"tEXt", b"note\x00\x89PNG\r\n\x1a\n")
    output = io.BytesIO()
    Image.new("RGB", (4, 3)).save(output, format="PNG", pnginfo=metadata)
    info = validate_encoded_image(output.getvalue(), "image/png")
    assert (info.width, info.height) == (4, 3)


def test_embedded_jpeg_signature_in_comment_and_progressive_jpeg_are_valid() -> None:
    output = io.BytesIO()
    Image.new("RGB", (4, 3)).save(output, format="JPEG", progressive=True)
    payload = output.getvalue()
    comment = b"note\xff\xd8\xffembedded-marker"
    segment = b"\xff\xfe" + (len(comment) + 2).to_bytes(2, "big") + comment
    with_comment = payload[:2] + segment + payload[2:]
    info = validate_encoded_image(with_comment, "image/jpeg")
    assert (info.width, info.height) == (4, 3)
    with pytest.raises(ImageContractError, match="IMAGE_MULTIFRAME_REJECTED"):
        validate_encoded_image(payload + payload, "image/jpeg")


def test_repeated_generated_request_id_is_rejected_without_second_inference(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        pipeline = FakePipeline()
        app = await _ready_app(tmp_path, pipeline)
        payload = _image()
        headers = _headers("image/png", payload)
        assert (await _request(app, "POST", "/api/v1/inference", body=payload, headers=headers))[
            0
        ] == 200
        status, result = await _request(
            app, "POST", "/api/v1/inference", body=payload, headers=headers
        )
        assert status == 409 and result is not None
        assert result["error"]["code"] == "INVALID_REQUEST"
        assert pipeline.calls == 1
        await app.state.runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("pipeline_status", "candidate_count", "status", "error_code"),
    [
        ("NO_PLATE_DETECTED", 0, "no_plate_detected", "NO_PLATE_DETECTED"),
        (
            "MULTIPLE_PLATE_CANDIDATES",
            2,
            "multiple_detections",
            "MULTIPLE_DETECTIONS_TOP1_SELECTED",
        ),
        ("INVALID_CROP", 1, "crop_refinement_failed", "CROP_REFINEMENT_FAILED"),
    ],
)
def test_deterministic_pipeline_outcomes(
    tmp_path: Path,
    pipeline_status: str,
    candidate_count: int,
    status: str,
    error_code: str,
) -> None:
    async def scenario() -> None:
        app = await _ready_app(
            tmp_path,
            FakePipeline(pipeline_status=pipeline_status, candidate_count=candidate_count),
        )
        payload = _image()
        response_status, result = await _request(
            app, "POST", "/api/v1/inference", body=payload, headers=_headers("image/png", payload)
        )
        assert response_status == 200 and result is not None
        assert (result["status"], result["error_code"]) == (status, error_code)
        if candidate_count > 1:
            assert result["detection"]["candidate_count"] == 2
            assert result["detection"]["box_xyxy_normalized"][0] == pytest.approx(0.25)
        await app.state.runtime.close()

    asyncio.run(scenario())


def test_no_file_creation_and_no_sensitive_error_or_log_leakage(tmp_path: Path) -> None:
    async def scenario() -> None:
        stream = io.StringIO()
        app = await _ready_app(
            tmp_path,
            FakePipeline(fail=True),
            logger=SafeJsonLogger(stream),
        )
        before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
        payload = _image()
        headers = _headers("image/png", payload)
        headers["authorization"] = "Bearer credential-canary"
        status, _result = await _request(
            app, "POST", "/api/v1/inference", body=payload, headers=headers
        )
        assert status == 403
        status, result = await _request(
            app, "POST", "/api/v1/inference", body=payload, headers=_headers("image/png", payload)
        )
        assert status == 500 and result is not None
        serialized = json.dumps(result) + stream.getvalue()
        for canary in ("credential-canary", "34ABC123", "private/path", "sensitive-plate-value"):
            assert canary not in serialized
        after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
        assert after == before
        await app.state.runtime.close()

    asyncio.run(scenario())


def test_recognition_failure_and_structural_validation_outcomes(tmp_path: Path) -> None:
    async def scenario() -> None:
        payload = _image()
        failing = await _ready_app(tmp_path, FakePipeline(fail=True))
        status, result = await _request(
            failing,
            "POST",
            "/api/v1/inference",
            body=payload,
            headers=_headers("image/png", payload),
        )
        assert status == 500 and result is not None
        assert result["error"]["code"] == "INFERENCE_FAILED"
        await failing.state.runtime.close()

        invalid = await _ready_app(tmp_path, FakePipeline(structural_valid=False))
        status, result = await _request(
            invalid,
            "POST",
            "/api/v1/inference",
            body=payload,
            headers=_headers("image/png", payload),
        )
        assert status == 200 and result is not None
        assert result["structural_validity"] == "invalid"
        assert result["error_code"] == "STRUCTURAL_VALIDATION_FAILED"
        await invalid.state.runtime.close()

        unresolved = await _ready_app(tmp_path, FakePipeline(decoded_text=None))
        status, result = await _request(
            unresolved,
            "POST",
            "/api/v1/inference",
            body=payload,
            headers=_headers("image/png", payload),
        )
        assert status == 200 and result is not None
        assert result["status"] == "recognition_unresolved"
        assert result["prediction"] is None
        assert result["error_code"] == "RECOGNITION_UNRESOLVED"
        schema = json.loads(
            (
                Path(__file__).parents[1]
                / "schemas/internal/v1/anpr-private-inference-response-v1.schema.json"
            ).read_text()
        )
        Draft202012Validator(schema).validate(result)
        await unresolved.state.runtime.close()

    asyncio.run(scenario())


def test_production_service_source_isolated_from_legacy_and_external_models() -> None:
    root = Path(__file__).parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "src/anpr_engine/inference_service").glob("*.py")
    ).lower()
    assert "integration.browser" not in source
    assert "review_store" not in source
    forbidden = ("parseq", "easyocr", "vitstr", "abinet", "strhub", "external-models")
    assert not any(value in source for value in forbidden)
    assert not (root.parent / "external-models").exists()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import anpr_engine.inference_service.app; "
                "print('\\n'.join(sorted(name for name in sys.modules "
                "if name.startswith('anpr_engine.integration.'))))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    imported = set(result.stdout.splitlines())
    assert "anpr_engine.integration.review_store" not in imported
    assert "anpr_engine.integration.browser" not in imported
    assert "anpr_engine.integration.project_models_browser_ui" not in imported
