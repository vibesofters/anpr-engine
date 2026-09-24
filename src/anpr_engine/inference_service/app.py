"""Minimal private ASGI service for one no-retention ANPR request at a time."""

from __future__ import annotations

import asyncio
import hmac
import re
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from anpr_engine.inference_service.config import ServiceConfig
from anpr_engine.inference_service.logging import SafeJsonLogger
from anpr_engine.inference_service.runtime import (
    CROP_POLICY,
    PipelineFactory,
    ServiceRuntime,
    _default_pipeline_factory,
    model_identities,
)
from anpr_engine.web_contracts.image_validation import (
    ALLOWED_MEDIA_TYPES,
    MAX_COMPRESSED_BYTES,
    ImageContractError,
)

REQUEST_ID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
ADMISSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,64}$")
RESPONSE_SCHEMA = "anpr-private-inference-response-v1"
ERROR_SCHEMA = "anpr-api-error-v1"


def _bounded_ms(value: float) -> float:
    return round(max(0.0, min(120_000.0, value)), 3)


def _new_request_id() -> str:
    return secrets.token_hex(13).upper()


def _error(
    request_id: str,
    code: str,
    status_code: int,
    *,
    retryable: bool = False,
) -> JSONResponse:
    return JSONResponse(
        {
            "schema_version": ERROR_SCHEMA,
            "request_id": request_id,
            "error": {
                "code": code,
                "message_key": f"error.{code.lower()}",
                "retryable": retryable,
            },
        },
        status_code=status_code,
    )


async def _read_bounded_body(request: Request) -> bytes:
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            declared_length = int(raw_length)
        except ValueError as exc:
            raise ImageContractError("INVALID_REQUEST") from exc
        if declared_length < 1:
            raise ImageContractError("INVALID_REQUEST")
        if declared_length > MAX_COMPRESSED_BYTES:
            raise ImageContractError("IMAGE_TOO_LARGE")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_COMPRESSED_BYTES:
            raise ImageContractError("IMAGE_TOO_LARGE")
        body.extend(chunk)
    if not body:
        raise ImageContractError("INVALID_REQUEST")
    return bytes(body)


def _pipeline_response(
    result: dict[str, Any],
    *,
    request_id: str,
    admission_id: str,
    validation_ms: float,
    execution_ms: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    candidates = result.get("detection", {}).get("candidates", [])
    selected = candidates[0] if candidates else None
    normalized_box: list[float] | None = None
    confidence: float | None = None
    if selected:
        box = selected["bbox_xyxy"]
        normalized_box = [
            max(0.0, min(1.0, float(box[0]) / width)),
            max(0.0, min(1.0, float(box[1]) / height)),
            max(0.0, min(1.0, float(box[2]) / width)),
            max(0.0, min(1.0, float(box[3]) / height)),
        ]
        confidence = float(selected["confidence"])
    recognition = result.get("recognition") or {}
    prediction = recognition.get("tr_text")
    structural = recognition.get("structural_validation")
    pipeline_status = str(result.get("pipeline_status", ""))
    status = "prediction_available"
    error_code: str | None = None
    if pipeline_status == "NO_PLATE_DETECTED":
        status, error_code = "no_plate_detected", "NO_PLATE_DETECTED"
    elif pipeline_status == "MULTIPLE_PLATE_CANDIDATES":
        status, error_code = "multiple_detections", "MULTIPLE_DETECTIONS_TOP1_SELECTED"
    elif pipeline_status == "INVALID_CROP":
        status, error_code = "crop_refinement_failed", "CROP_REFINEMENT_FAILED"
    elif prediction is None:
        status, error_code = "recognition_unresolved", "RECOGNITION_UNRESOLVED"
    elif structural is not None and structural.get("valid") is False:
        error_code = "STRUCTURAL_VALIDATION_FAILED"
    timings = result.get("timings_ms", {})
    return {
        "schema_version": RESPONSE_SCHEMA,
        "request_id": request_id,
        "admission_id": admission_id,
        "status": status,
        "prediction": prediction,
        "prediction_source": "existing_project_decoder",
        "decoding_changed_raw": bool(recognition.get("decoding_changed_raw", False)),
        "structural_validity": (
            "valid"
            if structural and structural.get("valid") is True
            else "invalid"
            if structural and structural.get("valid") is False
            else "not_run"
        ),
        "informational_confidence": recognition.get("decoder_confidence"),
        "detection": {
            "candidate_count": len(candidates),
            "selection_rule": "highest_confidence_then_box_coordinates",
            "box_xyxy_normalized": normalized_box,
            "informational_confidence": confidence,
        },
        "models": model_identities(),
        "crop_policy": CROP_POLICY,
        "timings_ms": {
            "total": _bounded_ms(validation_ms + execution_ms),
            "validation": _bounded_ms(validation_ms),
            "detection": _bounded_ms(float(timings.get("detection_inference_ms", 0.0))),
            "crop_refinement": _bounded_ms(float(timings.get("crop_preprocessing_ms", 0.0))),
            "recognition": _bounded_ms(float(timings.get("recognition_inference_ms", 0.0))),
        },
        "error_code": error_code,
    }


def create_app(
    config: ServiceConfig | None = None,
    *,
    pipeline_factory: PipelineFactory = _default_pipeline_factory,
    logger: SafeJsonLogger | None = None,
    initialize_on_startup: bool = True,
) -> Starlette:
    service_config = config or ServiceConfig.from_environment()
    runtime = ServiceRuntime(service_config, pipeline_factory=pipeline_factory)
    safe_logger = logger or SafeJsonLogger()

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        if initialize_on_startup:
            await runtime.initialize()
        safe_logger.emit(
            severity="INFO",
            route_category="lifecycle",
            application_version=service_config.application_version,
            api_version="v1",
            schema_version=RESPONSE_SCHEMA,
            readiness_state="ready" if runtime.ready else "not_ready",
        )
        try:
            yield
        finally:
            await runtime.close()

    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "alive"})

    async def readyz(_request: Request) -> JSONResponse:
        if runtime.ready and not runtime.active:
            return JSONResponse({"status": "ready"})
        reason = "busy" if runtime.ready and runtime.active else "not_initialized"
        return JSONResponse({"status": "not_ready", "reason": reason}, status_code=503)

    async def inference(request: Request) -> Response:
        request_id = request.headers.get("x-anpr-request-id", "")
        if not REQUEST_ID_PATTERN.fullmatch(request_id):
            request_id = _new_request_id()
            return _error(request_id, "INVALID_REQUEST", 400)
        if not runtime.ready:
            return _error(request_id, "MODEL_NOT_READY", 503, retryable=True)
        expected = service_config.internal_credential
        supplied = request.headers.get("authorization", "")
        if expected is not None:
            valid_auth = supplied.startswith("Bearer ") and hmac.compare_digest(
                supplied[7:].encode(), expected.encode()
            )
            if not valid_auth:
                return _error(request_id, "ORIGIN_REJECTED", 403)
        admission_id = request.headers.get("x-anpr-admission-id", "")
        locale = request.headers.get("x-anpr-locale", "")
        privacy_acknowledged = request.headers.get("x-anpr-privacy-acknowledged", "")
        if (
            not ADMISSION_ID_PATTERN.fullmatch(admission_id)
            or locale not in {"en", "tr"}
            or privacy_acknowledged != "true"
        ):
            code = (
                "PRIVACY_ACKNOWLEDGEMENT_REQUIRED"
                if privacy_acknowledged != "true"
                else "INVALID_REQUEST"
            )
            return _error(request_id, code, 400)
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type not in ALLOWED_MEDIA_TYPES:
            return _error(request_id, "UNSUPPORTED_MEDIA_TYPE", 415)
        admission, rejection = runtime.try_admit(request_id, admission_id)
        if admission is None:
            if rejection == "duplicate":
                return _error(request_id, "INVALID_REQUEST", 409)
            return _error(request_id, "CAPACITY_UNAVAILABLE", 503, retryable=True)
        validation_started = time.perf_counter()
        try:
            async with asyncio.timeout(service_config.inference_deadline_seconds):
                payload = await _read_bounded_body(request)
                image = await runtime.prepare_image(payload, media_type)
                validation_ms = (time.perf_counter() - validation_started) * 1000
                del payload
                result, execution_ms = await runtime.execute(
                    admission, image, request_id=request_id
                )
            response = _pipeline_response(
                result,
                request_id=request_id,
                admission_id=admission_id,
                validation_ms=validation_ms,
                execution_ms=execution_ms,
                width=image.shape[1],
                height=image.shape[0],
            )
            safe_logger.emit(
                severity="INFO",
                request_id=request_id,
                route_category="inference",
                status_code=200,
                error_code=response["error_code"],
                stage_timings_ms=response["timings_ms"],
                capacity_state="available",
                application_version=service_config.application_version,
                api_version="v1",
                schema_version=RESPONSE_SCHEMA,
                model_versions={
                    key: value["model_id"] for key, value in model_identities().items()
                },
            )
            return JSONResponse(response)
        except ImageContractError as exc:
            runtime.abandon(admission)
            status_code = (
                413 if exc.code in {"IMAGE_TOO_LARGE", "IMAGE_DIMENSIONS_EXCEEDED"} else 422
            )
            return _error(request_id, exc.code, status_code)
        except ClientDisconnect:
            runtime.abandon(admission)
            safe_logger.emit(
                severity="WARNING",
                request_id=request_id,
                route_category="inference",
                status_code=499,
                error_code="CLIENT_DISCONNECTED",
                capacity_state="available",
                application_version=service_config.application_version,
                api_version="v1",
                schema_version=RESPONSE_SCHEMA,
            )
            return Response(status_code=499)
        except TimeoutError:
            runtime.abandon(admission)
            return _error(request_id, "INFERENCE_DEADLINE_EXCEEDED", 504, retryable=True)
        except asyncio.CancelledError:
            runtime.abandon(admission)
            raise
        except Exception:
            runtime.abandon(admission)
            safe_logger.emit(
                severity="ERROR",
                request_id=request_id,
                route_category="inference",
                status_code=500,
                error_code="INFERENCE_FAILED",
                capacity_state="available",
                application_version=service_config.application_version,
                api_version="v1",
                schema_version=RESPONSE_SCHEMA,
            )
            return _error(request_id, "INFERENCE_FAILED", 500)
        finally:
            if "image" in locals():
                del image

    app = Starlette(
        debug=False,
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/readyz", readyz, methods=["GET"]),
            Route("/api/v1/inference", inference, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    return app


app = create_app()
