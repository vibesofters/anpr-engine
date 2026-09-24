#!/usr/bin/env python3
"""Validate the retained public candidate and Phase 2A real-image boundary."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from openapi_spec_validator import validate as validate_openapi_document
from openapi_spec_validator.readers import read_from_filename

ROOT = Path(__file__).parents[1]
CONTRACT_ROOT = ROOT / "packages/api-contract"
SCHEMA_MAP = {
    "image-metadata": ROOT / "schemas/api/v1/anpr-image-inference-metadata-v1.schema.json",
    "image-inference": ROOT / "schemas/api/v1/anpr-image-inference-response-v1.schema.json",
    "api-error": ROOT / "schemas/api/v1/anpr-api-error-v1.schema.json",
    "session-result": ROOT / "schemas/exports/v1/anpr-session-result-v1.schema.json",
    "human-review": ROOT / "schemas/review/v1/anpr-human-review-v1.schema.json",
    "processing-notice": ROOT / "schemas/privacy/v1/anpr-processing-notice-v1.schema.json",
    "data-lifecycle": ROOT / "schemas/privacy/v1/anpr-data-lifecycle-v1.schema.json",
    "private-inference": ROOT
    / "schemas/internal/v1/anpr-private-inference-response-v1.schema.json",
}
OPENAPI_PATHS = (
    ROOT / "schemas/api/v1/anpr-inference-v1.openapi.yaml",
    ROOT / "schemas/internal/v1/anpr-private-inference-v1.openapi.yaml",
)
FORBIDDEN_REQUEST_FIELDS = {
    "file",
    "files",
    "image_bytes",
    "encoded_image",
    "bytes",
    "base64",
    "data_url",
    "url",
    "remote_url",
    "path",
    "filepath",
    "filename",
    "model",
    "model_id",
    "model_path",
    "batch",
    "items",
    "session_id",
    "cookie",
}
EXTERNAL_OCR_PATTERN = re.compile(
    r"parseq|easyocr|vitstr|abinet|strhub|english_g2|pretrained[-_ ]ocr|external-models",
    re.IGNORECASE,
)


def public_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(
        path for line in result.stdout.splitlines() if line and (path := ROOT / line).is_file()
    )


def validate_json() -> None:
    paths = [path for path in public_files() if path.suffix in {".json", ".jsonl"}]
    for path in paths:
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
    print(f"validated {len(paths)} JSON/JSONL files")


def validate_yaml() -> None:
    paths = [path for path in public_files() if path.suffix in {".yaml", ".yml"}]
    for path in paths:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    print(f"validated {len(paths)} YAML files")


def schema_for_fixture(path: Path) -> Path:
    stem = path.stem
    for key, schema in SCHEMA_MAP.items():
        if key in stem:
            return schema
    if stem.startswith("metadata-"):
        return SCHEMA_MAP["image-metadata"]
    if stem.startswith("response-"):
        return SCHEMA_MAP["image-inference"]
    if stem.startswith("private-"):
        return SCHEMA_MAP["private-inference"]
    raise ValueError(f"fixture has no schema mapping: {path.relative_to(ROOT)}")


def validate_schemas() -> None:
    validators: dict[Path, Draft202012Validator] = {}
    for path in SCHEMA_MAP.values():
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        validators[path] = Draft202012Validator(schema)
    valid_paths = sorted((CONTRACT_ROOT / "fixtures/valid").glob("*.json"))
    invalid_paths = sorted((CONTRACT_ROOT / "fixtures/invalid").glob("*.json"))
    for path in valid_paths:
        validators[schema_for_fixture(path)].validate(json.loads(path.read_text(encoding="utf-8")))
    for path in invalid_paths:
        errors = list(
            validators[schema_for_fixture(path)].iter_errors(
                json.loads(path.read_text(encoding="utf-8"))
            )
        )
        if not errors:
            raise ValueError(f"invalid fixture unexpectedly passed: {path.relative_to(ROOT)}")
    print(
        f"compiled {len(validators)} Draft 2020-12 schemas; "
        f"validated {len(valid_paths)} valid and {len(invalid_paths)} invalid fixtures"
    )


def validate_openapi() -> None:
    documents = []
    for path in OPENAPI_PATHS:
        specification, base_uri = read_from_filename(str(path))
        validate_openapi_document(specification, base_uri=base_uri)
        documents.append(specification)
    specification = documents[0]
    operation = specification["paths"]["/api/inference"]["post"]
    content = operation["requestBody"]["content"]
    if set(content) != {"image/jpeg", "image/png"}:
        raise ValueError("OpenAPI raw-image content types drifted")
    for media_type in content:
        schema = content[media_type]["schema"]
        if schema != {"type": "string", "format": "binary", "maxLength": 10_485_760}:
            raise ValueError(f"OpenAPI body contract drifted: {media_type}")
    if "multipart/form-data" in content or "application/octet-stream" in content:
        raise ValueError("OpenAPI exposes a prohibited body media type")
    internal = documents[1]
    internal_operation = internal["paths"]["/api/v1/inference"]["post"]
    internal_content = internal_operation["requestBody"]["content"]
    if set(internal_content) != {"image/jpeg", "image/png"}:
        raise ValueError("internal OpenAPI raw-image content types drifted")
    if internal_operation["security"] != [{"internalBearer": []}]:
        raise ValueError("internal OpenAPI authentication drifted")
    print("validated 2 OpenAPI 3.1 raw JPEG/PNG contracts")


def validate_links() -> None:
    checked = 0
    errors: list[str] = []
    for path in public_files():
        if path.suffix != ".md":
            continue
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.split()[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            checked += 1
            destination = (path.parent / target.split("#", 1)[0]).resolve()
            if not destination.exists():
                errors.append(f"{path.relative_to(ROOT)} -> {target}")
    if errors:
        raise ValueError("broken documentation links:\n" + "\n".join(errors))
    print(f"validated {checked} local documentation links")


def text_files() -> list[Path]:
    allowed = {".md", ".py", ".toml", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".ts"}
    return [path for path in public_files() if path.suffix in allowed]


def validate_scans() -> None:
    files = public_files()
    substantive = [path for path in files if path.stat().st_size > 0]
    external_matches: list[str] = []
    absolute_matches: list[str] = []
    secret_matches: list[str] = []
    for path in text_files():
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        if relative != "scripts/validate_public_candidate.py" and not relative.startswith(
            ("tests/", "packages/api-contract/fixtures/invalid/")
        ):
            if EXTERNAL_OCR_PATTERN.search(text):
                external_matches.append(relative)
            if re.search(r"/Users/[A-Za-z0-9._-]+/|[A-Z]:\\\\Users\\\\", text):
                absolute_matches.append(relative)
        if relative not in {".env.example", "scripts/validate_public_candidate.py"} and re.search(
            r"(?i)(api[_-]?key|password|client[_-]?secret|private[_-]?key)\s*[:=]\s*['\"][^'\"]{8,}",
            text,
        ):
            secret_matches.append(relative)
    if external_matches:
        raise ValueError(f"external OCR/import/path references: {external_matches}")
    if absolute_matches:
        raise ValueError(f"absolute workstation paths: {absolute_matches}")
    if secret_matches:
        raise ValueError(f"credential-like assignments: {secret_matches}")
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors"}
    weights = [
        str(path.relative_to(ROOT)) for path in files if path.suffix.lower() in forbidden_suffixes
    ]
    dataset_payloads = [
        str(path.relative_to(ROOT))
        for path in files
        if path.relative_to(ROOT).parts[0] == "datasets"
    ]
    allowed_large_prefixes = ("assets/fonts/",)
    large = [
        str(path.relative_to(ROOT))
        for path in substantive
        if path.stat().st_size >= 1_000_000
        and not path.relative_to(ROOT).as_posix().startswith(allowed_large_prefixes)
    ]
    if weights or dataset_payloads or large:
        raise ValueError(
            "public binary/data boundary failed: "
            f"weights={weights}, datasets={dataset_payloads}, large={large}"
        )
    direct_dependencies = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    prohibited_dependencies = (
        "python-multipart",
        "easyocr",
        "strhub",
        "requests==",
        "httpx==",
        "aiohttp==",
    )
    if any(name in direct_dependencies for name in prohibited_dependencies):
        raise ValueError("prohibited multipart, remote-image, or external OCR dependency")
    if (ROOT.parent / "external-models").exists():
        raise ValueError("external-models directory was recreated")
    print(
        f"scanned {len(files)} public files; no prohibited paths, secrets, "
        "weights, datasets, or large files"
    )


def property_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        names.update(value.get("properties", {}).keys())
        for child in value.values():
            names.update(property_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(property_names(child))
    return names


def validate_contracts() -> None:
    metadata_schema = json.loads(SCHEMA_MAP["image-metadata"].read_text(encoding="utf-8"))
    forbidden = property_names(metadata_schema) & FORBIDDEN_REQUEST_FIELDS
    if forbidden:
        raise ValueError(f"prohibited request properties: {sorted(forbidden)}")
    if set(metadata_schema["properties"]) != {
        "schema_version",
        "request_id",
        "admission_id",
        "locale",
        "declared_media_type",
        "detected_media_type",
        "compressed_size_bytes",
        "privacy_acknowledged",
        "source_metadata_handling",
    }:
        raise ValueError("request root is not the reviewed internal contract")
    invariants = json.loads((CONTRACT_ROOT / "policy/resource-invariants.json").read_text())
    expected = {
        "one_image_per_request": True,
        "batch_processing": False,
        "global_inference_concurrency": 1,
        "per_session_active_inference": 1,
        "queue_depth": 0,
        "model_bearing_processes": 1,
        "unsafe_concurrency_override_allowed": False,
        "automatic_retry": False,
        "background_inference": False,
        "database": False,
        "redis": False,
        "object_storage": False,
        "persistent_history": False,
        "analytics": False,
        "external_error_monitoring": False,
        "public_python_inference_endpoint": False,
    }
    for key, value in expected.items():
        if invariants.get(key) != value:
            raise ValueError(f"resource invariant drift: {key}")
    routes = json.loads((CONTRACT_ROOT / "policy/public-route-allowlist.json").read_text())
    pairs = [(entry["method"], entry["path"]) for entry in routes["routes"]]
    expected_routes = {
        ("GET", "/"),
        ("GET", "/en"),
        ("GET", "/tr"),
        ("GET", "/en/privacy"),
        ("GET", "/tr/privacy"),
        ("GET", "/en/responsible-use"),
        ("GET", "/tr/responsible-use"),
        ("GET", "/en/limitations"),
        ("GET", "/tr/limitations"),
        ("POST", "/api/inference"),
    }
    if set(pairs) != expected_routes or len(pairs) != len(expected_routes):
        raise ValueError("route allowlist differs from the Phase 2A reviewed set")
    if routes["implemented"] is not True:
        raise ValueError("Phase 2C route allowlist implementation status drift")
    error_schema = json.loads(SCHEMA_MAP["api-error"].read_text())
    error_policy = json.loads((CONTRACT_ROOT / "policy/error-codes.json").read_text())
    schema_codes = set(error_schema["$defs"]["ErrorCode"]["enum"])
    policy_codes = {item["code"] for item in error_policy["errors"]}
    if schema_codes != policy_codes:
        raise ValueError("error-code drift")
    if any(
        not item.get("cleanup") or not item.get("message_semantics")
        for item in error_policy["errors"]
    ):
        raise ValueError("error cleanup/message semantics missing")
    glossary = json.loads((CONTRACT_ROOT / "glossary.seed.json").read_text())
    documented = (ROOT / "docs/reference/terminology.en-tr.md").read_text()
    missing = [
        term["term_id"] for term in glossary["terms"] if f"`{term['term_id']}`" not in documented
    ]
    if missing:
        raise ValueError(f"glossary/documentation drift: {missing}")
    logging = json.loads((CONTRACT_ROOT / "policy/logging-contract.json").read_text())
    sensitive = {
        "image",
        "image_bytes",
        "encoded_image",
        "filename",
        "plate_text",
        "prediction",
        "correction",
        "crop",
        "payload",
        "session_id",
        "admission_id",
        "raw_ip",
        "secret",
    }
    if sensitive & set(logging["allowed_fields"]):
        raise ValueError("sensitive logging field allowed")
    serialized_canaries = " ".join(logging["canary_values"])
    if any(canary not in serialized_canaries for canary in logging["canary_values"]):
        raise ValueError("logging canary contract drift")
    image_policy = json.loads((CONTRACT_ROOT / "policy/image-validation-contract.json").read_text())
    expected_limits = {
        "maximum_compressed_bytes": 10_485_760,
        "maximum_decoded_pixels": 20_000_000,
        "maximum_width": 8_192,
        "maximum_height": 8_192,
        "required_frame_count": 1,
    }
    for key, value in expected_limits.items():
        if image_policy.get(key) != value:
            raise ValueError(f"image-validation limit drift: {key}")
    if image_policy["allowed_media_types"] != ["image/jpeg", "image/png"]:
        raise ValueError("image-validation media type drift")
    lifecycle = json.loads((CONTRACT_ROOT / "fixtures/valid/data-lifecycle.json").read_text())
    required_outcomes = {
        "success",
        "validation_failure",
        "no_detection",
        "recognition_failure",
        "timeout",
        "cancellation",
        "client_disconnect",
        "internal_exception",
        "service_shutdown",
    }
    if set(lifecycle["cleanup_outcomes"]) != required_outcomes:
        raise ValueError("data-lifecycle cleanup outcomes drift")
    exclusions = json.loads(
        (CONTRACT_ROOT / "policy/public-deployment-exclusions.json").read_text()
    )
    expected_excluded_paths = {
        "scripts/launch_anpr_v1_browser.py",
        "src/anpr_engine/integration/browser.py",
        "src/anpr_engine/integration/project_models_browser_ui.py",
        "src/anpr_engine/integration/review_store.py",
    }
    actual_excluded_paths = set(exclusions["excluded_entry_points"]) | set(
        exclusions["excluded_modules"]
    )
    if actual_excluded_paths != expected_excluded_paths:
        raise ValueError("legacy local-browser deployment exclusion drift")
    requirements = exclusions["requirements"]
    if requirements != {
        "loopback_binding_only": True,
        "public_container_entry_point": False,
        "public_service_import": False,
        "public_route_allowlist_overlap": False,
        "future_build_must_fail_if_included": True,
    }:
        raise ValueError("legacy local-browser deployment requirements drift")
    legacy_launcher = (ROOT / "scripts/launch_anpr_v1_browser.py").read_text()
    legacy_transport = (ROOT / "src/anpr_engine/integration/browser.py").read_text()
    if "LOCAL_ONLY_NOT_FOR_PUBLIC_DEPLOYMENT = True" not in legacy_launcher:
        raise ValueError("legacy browser lacks local-only marker")
    if 'ThreadingHTTPServer(("127.0.0.1", port)' not in legacy_transport:
        raise ValueError("legacy browser is not fixed to loopback")
    if "0.0.0.0" in legacy_launcher or "0.0.0.0" in legacy_transport:
        raise ValueError("legacy browser exposes a non-loopback binding")
    public_docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "README.md", ROOT / "docs/policies/privacy-and-responsible-use.md")
    ).lower()
    obsolete = (
        "public processing of real vehicle or plate images is not authorized",
        "public demonstration is restricted to synthetic",
        "public real-image processing remains blocked",
    )
    if any(statement in public_docs for statement in obsolete):
        raise ValueError("obsolete synthetic-only public restriction remains")
    print(
        "validated raw-image boundary, lifecycle, resource invariants, routes, "
        "errors, glossary, and logging"
    )


def validate_ci() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    paths = set(re.findall(r"(?:python|check)\s+([A-Za-z0-9_./-]+\.py)", workflow))
    missing = [path for path in sorted(paths) if not (ROOT / path).is_file()]
    if missing:
        raise ValueError(f"CI references missing scripts: {missing}")
    prohibited = ["private-governance", "external-models", "ANPR_PRIVATE_MODEL_BUNDLE"]
    found = [term for term in prohibited if term in workflow]
    if found:
        raise ValueError(f"ordinary CI references private inputs: {found}")
    print(f"validated {len(paths)} CI script paths")


VALIDATORS = {
    "json": validate_json,
    "yaml": validate_yaml,
    "schemas": validate_schemas,
    "openapi": validate_openapi,
    "links": validate_links,
    "scans": validate_scans,
    "contracts": validate_contracts,
    "ci": validate_ci,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=(*VALIDATORS, "all"))
    arguments = parser.parse_args()
    selected = VALIDATORS.values() if arguments.check == "all" else (VALIDATORS[arguments.check],)
    for validator in selected:
        validator()


if __name__ == "__main__":
    main()
