from __future__ import annotations

import copy
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from openapi_spec_validator import validate as validate_openapi_document
from openapi_spec_validator.readers import read_from_filename
from PIL import Image

from anpr_engine.web_contracts import (
    ImageContractError,
    SafeLogContractError,
    validate_encoded_image,
    validate_image_dimensions,
    validate_safe_log_record,
)

ROOT = Path(__file__).parents[1]
CONTRACTS = ROOT / "packages/api-contract"
SCHEMAS = {
    "metadata": ROOT / "schemas/api/v1/anpr-image-inference-metadata-v1.schema.json",
    "response": ROOT / "schemas/api/v1/anpr-image-inference-response-v1.schema.json",
    "error": ROOT / "schemas/api/v1/anpr-api-error-v1.schema.json",
    "export": ROOT / "schemas/exports/v1/anpr-session-result-v1.schema.json",
    "review": ROOT / "schemas/review/v1/anpr-human-review-v1.schema.json",
    "notice": ROOT / "schemas/privacy/v1/anpr-processing-notice-v1.schema.json",
    "lifecycle": ROOT / "schemas/privacy/v1/anpr-data-lifecycle-v1.schema.json",
}
OPENAPI = ROOT / "schemas/api/v1/anpr-inference-v1.openapi.yaml"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _fixture(group: str, name: str) -> dict[str, Any]:
    return _load(CONTRACTS / "fixtures" / group / name)


def _validator(name: str) -> Draft202012Validator:
    schema = _load(SCHEMAS[name])
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _encoded_image(image_format: str, *, animated: bool = False) -> bytes:
    output = BytesIO()
    first = Image.new("RGB", (4, 3), color=(20, 80, 160))
    if animated:
        second = Image.new("RGB", (4, 3), color=(160, 80, 20))
        first.save(output, format=image_format, save_all=True, append_images=[second])
    else:
        first.save(output, format=image_format)
    return output.getvalue()


@pytest.mark.parametrize(
    ("schema_name", "fixture_name"),
    [
        ("metadata", "image-metadata-jpeg.json"),
        ("metadata", "image-metadata-png.json"),
        ("response", "image-inference-response.json"),
        ("response", "image-inference-no-detection.json"),
        ("response", "image-inference-unresolved.json"),
        ("error", "api-error.json"),
        ("export", "session-result-confirmed.json"),
        ("export", "session-result-corrected.json"),
        ("export", "session-result-unresolved.json"),
        ("review", "human-review-confirmed.json"),
        ("review", "human-review-corrected.json"),
        ("review", "human-review-unresolved.json"),
        ("notice", "processing-notice.json"),
        ("lifecycle", "data-lifecycle.json"),
    ],
)
def test_valid_contract_fixtures(schema_name: str, fixture_name: str) -> None:
    _validator(schema_name).validate(_fixture("valid", fixture_name))


@pytest.mark.parametrize(
    ("schema_name", "fixture_name"),
    [
        ("metadata", "metadata-octet-stream.json"),
        ("metadata", "metadata-missing-acknowledgement.json"),
        ("metadata", "metadata-oversized.json"),
        ("metadata", "metadata-mime-mismatch.json"),
        ("metadata", "metadata-filename.json"),
        ("metadata", "metadata-url.json"),
        ("metadata", "metadata-batch.json"),
        ("metadata", "metadata-model-selection.json"),
        ("response", "response-private-path.json"),
    ],
)
def test_invalid_contract_fixtures_are_rejected(schema_name: str, fixture_name: str) -> None:
    assert list(_validator(schema_name).iter_errors(_fixture("invalid", fixture_name)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image", "encoded"),
        ("image_bytes", "encoded"),
        ("base64", "encoded"),
        ("remote_url", "https://example.invalid/image.png"),
        ("path", "/tmp/image.png"),
        ("filename", "image.png"),
        ("batch", []),
        ("model_id", "user-model"),
        ("session_id", "user-session"),
        ("unknown", True),
    ],
)
def test_metadata_rejects_browser_controlled_or_unknown_fields(field: str, value: object) -> None:
    instance = _fixture("valid", "image-metadata-jpeg.json")
    instance[field] = value
    assert list(_validator("metadata").iter_errors(instance))


def test_all_json_schema_objects_are_closed() -> None:
    def inspect(value: object, path: str) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False, path
            for key, child in value.items():
                inspect(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}/{index}")

    for name, schema_path in SCHEMAS.items():
        inspect(_load(schema_path), name)


def test_openapi_is_valid_and_exposes_only_raw_jpeg_png() -> None:
    specification, base_uri = read_from_filename(str(OPENAPI))
    validate_openapi_document(specification, base_uri=base_uri)
    operation = specification["paths"]["/api/inference"]["post"]
    content = operation["requestBody"]["content"]
    assert set(content) == {"image/jpeg", "image/png"}
    assert all(value["schema"]["format"] == "binary" for value in content.values())
    assert all(value["schema"]["maxLength"] == 10_485_760 for value in content.values())
    assert operation["security"] == [{"sessionCookie": []}]
    parameter_names = {parameter["name"].lower() for parameter in operation["parameters"]}
    assert parameter_names == {"x-privacy-acknowledged"}
    assert not parameter_names & {"x-request-id", "x-session-id", "x-model-id"}


def test_transport_negative_fixture_covers_reviewed_rejections() -> None:
    fixture = _load(CONTRACTS / "fixtures/transport/invalid-public-requests.json")
    case_ids = {case["case_id"] for case in fixture["cases"]}
    assert case_ids == {
        "multipart",
        "octet_stream",
        "base64_json",
        "remote_url",
        "path",
        "filename",
        "batch",
        "gif",
        "webp",
        "svg",
        "tiff",
        "pdf",
        "video",
        "archive",
        "empty_body",
        "compressed_size_exceeded",
        "width_exceeded",
        "height_exceeded",
        "pixel_count_exceeded",
        "multiframe",
        "truncated",
        "mime_signature_mismatch",
        "user_model_id",
        "browser_session_id",
        "browser_request_id",
        "missing_privacy_acknowledgement",
        "unknown_property",
    }


@pytest.mark.parametrize(
    ("image_format", "media_type"), [("JPEG", "image/jpeg"), ("PNG", "image/png")]
)
def test_bounded_jpeg_and_png_are_accepted(image_format: str, media_type: str) -> None:
    payload = _encoded_image(image_format)
    info = validate_encoded_image(payload, media_type)
    assert info.media_type == media_type
    assert (info.width, info.height) == (4, 3)
    assert info.compressed_size_bytes == len(payload)


@pytest.mark.parametrize(
    ("payload", "media_type"),
    [
        (b"GIF89a" + b"x" * 20, "image/gif"),
        (b"RIFF" + b"x" * 20 + b"WEBP", "image/webp"),
        (b"<svg></svg>", "image/svg+xml"),
        (b"II*\x00" + b"x" * 20, "image/tiff"),
        (b"%PDF-1.7", "application/pdf"),
        (b"\x00\x00\x00\x18ftypmp42", "video/mp4"),
        (b"PK\x03\x04" + b"x" * 20, "application/zip"),
        (_encoded_image("JPEG"), "application/octet-stream"),
        (_encoded_image("JPEG"), "multipart/form-data"),
    ],
)
def test_non_jpeg_png_and_wrapped_media_types_are_rejected(payload: bytes, media_type: str) -> None:
    with pytest.raises(ImageContractError) as error:
        validate_encoded_image(payload, media_type)
    assert error.value.code == "UNSUPPORTED_MEDIA_TYPE"


def test_empty_and_oversized_compressed_bodies_are_rejected() -> None:
    with pytest.raises(ImageContractError, match="INVALID_REQUEST"):
        validate_encoded_image(b"", "image/jpeg")
    with pytest.raises(ImageContractError, match="IMAGE_TOO_LARGE"):
        validate_encoded_image(b"\xff\xd8\xff" + bytes(10_485_758), "image/jpeg")


def test_mime_signature_mismatch_is_rejected() -> None:
    with pytest.raises(ImageContractError, match="UNSUPPORTED_MEDIA_TYPE"):
        validate_encoded_image(_encoded_image("PNG"), "image/jpeg")


def test_multiframe_png_is_rejected() -> None:
    with pytest.raises(ImageContractError, match="IMAGE_MULTIFRAME_REJECTED"):
        validate_encoded_image(_encoded_image("PNG", animated=True), "image/png")


@pytest.mark.parametrize("image_format,media_type", [("JPEG", "image/jpeg"), ("PNG", "image/png")])
def test_truncated_images_are_rejected(image_format: str, media_type: str) -> None:
    payload = _encoded_image(image_format)
    with pytest.raises(ImageContractError) as error:
        validate_encoded_image(payload[: max(12, len(payload) // 2)], media_type)
    assert error.value.code in {"IMAGE_TRUNCATED", "IMAGE_DECODE_FAILED"}


@pytest.mark.parametrize(
    ("width", "height"),
    [(8_193, 1), (1, 8_193), (5_000, 5_000)],
)
def test_decoded_dimension_and_pixel_limits(width: int, height: int) -> None:
    with pytest.raises(ImageContractError, match="IMAGE_DIMENSIONS_EXCEEDED"):
        validate_image_dimensions(width, height)


def test_response_fixes_models_and_requires_human_review() -> None:
    valid = _fixture("valid", "image-inference-response.json")
    changed_model = copy.deepcopy(valid)
    changed_model["models"]["detection"]["model_id"] = "user-model"
    assert list(_validator("response").iter_errors(changed_model))
    no_review = copy.deepcopy(valid)
    no_review["human_review_required"] = False
    assert list(_validator("response").iter_errors(no_review))


def test_browser_only_review_and_export_outcomes() -> None:
    review = _load(SCHEMAS["review"])
    assert review["properties"]["storage_scope"]["const"] == "active_page_memory_only"
    export = _load(SCHEMAS["export"])
    assert set(export["properties"]["review_outcome"]["enum"]) == {
        "OWNER_CONFIRMED_PROJECT_MODEL",
        "OWNER_CORRECTED",
        "UNRESOLVED",
    }
    routes = _load(CONTRACTS / "policy/public-route-allowlist.json")
    assert all("correction" not in route["path"] for route in routes["routes"])


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "validation_failure",
        "no_detection",
        "recognition_failure",
        "timeout",
        "cancellation",
        "client_disconnect",
        "internal_exception",
        "service_shutdown",
    ],
)
def test_cleanup_contract_covers_every_outcome(outcome: str) -> None:
    lifecycle = _fixture("valid", "data-lifecycle.json")
    assert outcome in lifecycle["cleanup_outcomes"]
    assert lifecycle["lock_release"] == "finally_on_every_outcome"
    assert set(lifecycle["released_buffers"]) == {
        "source_image",
        "decoded_image",
        "crop",
        "tensor",
        "annotated_result",
        "request_buffer",
    }
    assert lifecycle["retained_server_side"] == []
    assert set(lifecycle["prohibited_uses"]) == {
        "training",
        "evaluation",
        "analytics",
        "dataset_construction",
        "debug_archive",
        "unrelated_reuse",
        "future_processing",
        "request_replay",
        "result_cache",
    }


def test_error_envelope_cannot_contain_sensitive_details() -> None:
    instance = _fixture("valid", "api-error.json")
    for field in ("plate_text", "filename", "path", "exception", "image_details"):
        invalid = copy.deepcopy(instance)
        invalid["error"][field] = "sentinel"
        assert list(_validator("error").iter_errors(invalid))


def test_notice_and_glossary_have_english_turkish_parity() -> None:
    notice = _fixture("valid", "processing-notice.json")
    assert set(notice["notices"]["en"]) == set(notice["notices"]["tr"])
    glossary = _load(CONTRACTS / "glossary.seed.json")
    assert glossary["authoritative_locale"] == "en"
    assert glossary["turkish_review_status"] == "native_review_required"
    assert all(term["en"] and term["tr"] for term in glossary["terms"])


def test_route_allowlist_and_resource_invariants_are_closed() -> None:
    routes = _load(CONTRACTS / "policy/public-route-allowlist.json")
    assert routes["implemented"] is True
    pairs = {(route["method"], route["path"]) for route in routes["routes"]}
    assert ("POST", "/api/inference") in pairs
    prohibited_fragments = {
        "batch",
        "remote",
        "camera",
        "video",
        "websocket",
        "history",
        "training",
        "model-management",
        "dataset",
        "correction",
        "analytics",
        "debug",
        "files",
    }
    assert not any(fragment in path for _, path in pairs for fragment in prohibited_fragments)
    resources = _load(CONTRACTS / "policy/resource-invariants.json")
    assert resources["global_inference_concurrency"] == 1
    assert resources["per_session_active_inference"] == 1
    assert resources["queue_depth"] == 0
    assert resources["model_bearing_processes"] == 1
    assert resources["batch_processing"] is False
    assert resources["public_python_inference_endpoint"] is False


def test_legacy_browser_is_local_only_and_excluded_from_public_deployment() -> None:
    exclusions = _load(CONTRACTS / "policy/public-deployment-exclusions.json")
    excluded = set(exclusions["excluded_entry_points"]) | set(exclusions["excluded_modules"])
    assert excluded == {
        "scripts/launch_anpr_v1_browser.py",
        "src/anpr_engine/integration/browser.py",
        "src/anpr_engine/integration/project_models_browser_ui.py",
        "src/anpr_engine/integration/review_store.py",
    }
    assert exclusions["requirements"]["future_build_must_fail_if_included"] is True
    launcher = (ROOT / "scripts/launch_anpr_v1_browser.py").read_text(encoding="utf-8")
    transport = (ROOT / "src/anpr_engine/integration/browser.py").read_text(encoding="utf-8")
    assert "LOCAL_ONLY_NOT_FOR_PUBLIC_DEPLOYMENT = True" in launcher
    assert 'ThreadingHTTPServer(("127.0.0.1", port)' in transport
    assert "0.0.0.0" not in launcher + transport
    routes = _load(CONTRACTS / "policy/public-route-allowlist.json")
    public_paths = {route["path"] for route in routes["routes"]}
    assert not public_paths & {"/api/batches", "/api/artifact", "/api/batches/export"}


def test_error_codes_match_policy_and_define_cleanup() -> None:
    schema = _load(SCHEMAS["error"])
    policy = _load(CONTRACTS / "policy/error-codes.json")
    records = policy["errors"]
    assert set(schema["$defs"]["ErrorCode"]["enum"]) == {record["code"] for record in records}
    assert all(200 <= record["http_status"] <= 599 for record in records)
    assert all(record["message_semantics"] and record["cleanup"] for record in records)


def test_safe_logging_rejects_fields_and_sensitive_canaries() -> None:
    contract = _load(CONTRACTS / "policy/logging-contract.json")
    allowed = set(contract["allowed_fields"])
    prohibited = set(contract["prohibited_fields"])
    assert allowed.isdisjoint(prohibited)
    for field in prohibited:
        with pytest.raises(SafeLogContractError):
            validate_safe_log_record({field: "value"})
    for canary in contract["canary_values"]:
        with pytest.raises(SafeLogContractError, match="sensitive_log_canary"):
            validate_safe_log_record(
                {"error_code": canary}, canary_values=tuple(contract["canary_values"])
            )


def test_generated_contracts_have_no_drift() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_api_contracts.py", "--check"], cwd=ROOT, check=True
    )


def test_public_boundary_validator_passes() -> None:
    for check in ("openapi", "scans", "contracts", "ci"):
        subprocess.run(
            [sys.executable, "scripts/validate_public_candidate.py", check], cwd=ROOT, check=True
        )


def test_openapi_contains_no_unexpected_public_route() -> None:
    specification = yaml.safe_load(OPENAPI.read_text(encoding="utf-8"))
    assert set(specification["paths"]) == {"/api/inference"}
