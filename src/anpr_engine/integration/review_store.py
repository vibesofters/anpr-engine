"""Persistent, append-audited local batch storage for the ANPR V1 browser."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from anpr_engine.integration.anpr_v1 import DETECTOR_SHA, RECOGNIZER_SHA, content_sha256
from anpr_engine.integration.project_models_release import (
    CROP_REFINEMENT,
    RELEASE_VERSION,
    frozen_identities,
    resolve_owner_review,
    unresolved_decision,
)
from anpr_engine.recognition.normalization import PlateNormalizationError, normalize_plate_text

BATCH_SCHEMA = "anpr-v1-browser-batch-v1"
REVIEW_SCHEMA = "real-recognition-intake-event-v3"
DETECTION_STATES = {"UNREVIEWED", "CORRECT", "INCORRECT"}
RECOGNITION_STATES = {"UNREVIEWED", "CORRECT", "INCORRECT", "NOT_APPLICABLE"}
RECOGNITION_ASSESSMENT_FIELDS = ("official_raw_assessment",)
DETECTION_REASONS = {
    "",
    "NO_PLATE_DETECTED",
    "WRONG_OBJECT",
    "WRONG_TOP1",
    "BOX_TOO_TIGHT",
    "BOX_TOO_LOOSE",
    "PARTIAL_PLATE",
    "MULTIPLE_CANDIDATES",
    "OTHER",
}
DATASET_STATES = {
    "INCOMING_QUARANTINED",
    "OWNER_LABELED",
    "AUDITED",
    "TRAINING_ELIGIBLE",
    "DEVELOPMENT_REGRESSION",
    "LOCKED_INDEPENDENT_HOLDOUT",
}
AUDIT_STATES = {"PENDING", "OWNER_REVIEWED", "AUDITED", "CONTRADICTORY"}
CAPTURE_CONDITIONS = {
    "DAYLIGHT",
    "NIGHT",
    "LOW_LIGHT",
    "GLARE",
    "RAIN",
    "MOTION_BLUR",
    "DEFOCUS_BLUR",
    "PERSPECTIVE",
    "ROTATED",
    "SMALL_PLATE",
    "OCCLUDED",
    "DIRTY",
    "TR_BAND_VISIBLE",
    "PLATE_FRAME",
    "SURROUNDING_CONTEXT",
}
NEAR_DUPLICATE_HAMMING_THRESHOLD = 6


def _project_decision(result: dict[str, Any]) -> dict[str, Any]:
    """Read or derive the active project-models-only unresolved decision."""
    stored = result.get("project_release_decision")
    if isinstance(stored, dict) and "owner_review_required" in stored:
        return stored
    official = str((result.get("recognition") or {}).get("raw_text", ""))
    return unresolved_decision(str(result.get("pipeline_status")), official or None)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_name(value: str) -> str:
    leaf = Path(value).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf).strip("._") or "image"
    return stem[:160]


def perceptual_hash_bytes(content: bytes) -> str:
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("cannot decode image for perceptual fingerprint")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(resized)[:8, :8].reshape(-1)
    median = float(np.median(cast(Any, coefficients[1:])))
    bits = "".join("1" if value > median else "0" for value in coefficients)
    return f"{int(bits, 2):016x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def validate_group_safe_assignments(records: list[dict[str, Any]]) -> list[str]:
    """Return leakage errors for any jointly governed identity crossing a split."""
    errors: list[str] = []
    for field in ("physical_plate_id", "vehicle_group_id", "capture_session_id"):
        locations: dict[str, set[str]] = {}
        for row in records:
            value, split = str(row.get(field, "")).strip(), str(row.get("split", "")).strip()
            if value and split:
                locations.setdefault(value, set()).add(split)
        for value, splits in sorted(locations.items()):
            if len(splits) > 1:
                errors.append(f"{field}={value} crosses splits: {sorted(splits)}")
    return errors


def validate_source_group_consistency(records: list[dict[str, Any]]) -> list[str]:
    """Reject one immutable source digest being assigned contradictory identities."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        digest = str(row.get("input_sha256", "")).strip()
        if digest:
            grouped.setdefault(digest, []).append(row)
    errors: list[str] = []
    for digest, members in sorted(grouped.items()):
        for field in ("physical_plate_id", "vehicle_group_id", "capture_session_id"):
            values = {str(row.get(field, "")).strip() for row in members if row.get(field)}
            if len(values) > 1:
                errors.append(f"source {digest} has contradictory {field}: {sorted(values)}")
    return errors


class BatchReviewStore:
    def __init__(
        self, root: Path, *, max_file_bytes: int = 20 * 1024 * 1024, max_batch: int = 50
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_bytes = max_file_bytes
        self.max_batch = max_batch

    def _batch(self, batch_id: str) -> Path:
        if not re.fullmatch(r"batch-[0-9TZ-]+-[0-9a-f]{8}", batch_id):
            raise ValueError("invalid batch ID")
        path = (self.root / batch_id).resolve()
        if self.root not in path.parents:
            raise ValueError("unsafe batch path")
        return path

    def create_batch(self, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        if not files or len(files) > self.max_batch:
            raise ValueError(f"batch must contain 1-{self.max_batch} images")
        for name, content in files:
            if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"unsupported filename extension: {_safe_name(name)}")
            if not content or len(content) > self.max_file_bytes:
                raise ValueError(f"invalid file size: {_safe_name(name)}")
        ordered = sorted(
            files, key=lambda item: (_safe_name(item[0]).casefold(), content_sha256(item[1]))
        )
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        batch_id = f"batch-{timestamp}-{secrets.token_hex(4)}"
        directory = self._batch(batch_id)
        (directory / "inputs").mkdir(parents=True)
        (directory / "records").mkdir()
        prior: list[tuple[str, str, str]] = []
        for manifest_path in sorted(self.root.glob("batch-*/batch-manifest.json")):
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for prior_row in prior_manifest.get("records", []):
                prior_path = manifest_path.parent / str(prior_row.get("input_reference", ""))
                if not prior_path.is_file():
                    continue
                prior_hash = str(prior_row.get("perceptual_hash", ""))
                if not prior_hash:
                    prior_hash = perceptual_hash_bytes(prior_path.read_bytes())
                prior.append(
                    (
                        f"{prior_manifest['batch_id']}/{prior_row['request_id']}",
                        str(prior_row["input_sha256"]),
                        prior_hash,
                    )
                )
        seen: dict[str, str] = {}
        seen_perceptual: list[tuple[str, str]] = []
        records: list[dict[str, Any]] = []
        duplicates: list[dict[str, str]] = []
        near_duplicates: list[dict[str, str]] = []
        for index, (original, content) in enumerate(ordered, start=1):
            digest = content_sha256(content)
            if digest in seen:
                duplicates.append(
                    {
                        "filename": _safe_name(original),
                        "sha256": digest,
                        "duplicate_of": seen[digest],
                    }
                )
                continue
            request_id = f"request-{index:04d}-{digest[:12]}"
            phash = perceptual_hash_bytes(content)
            seen[digest] = request_id
            extension = Path(original).suffix.lower()
            safe_input = f"{request_id}{extension}"
            (directory / "inputs" / safe_input).write_bytes(content)
            records.append(
                {
                    "request_id": request_id,
                    "queue_index": len(records) + 1,
                    "safe_original_filename": _safe_name(original),
                    "input_sha256": digest,
                    "perceptual_hash": phash,
                    "input_reference": f"inputs/{safe_input}",
                    "state": "PENDING",
                    "review": {
                        "detection_assessment": "UNREVIEWED",
                        "recognition_assessment": "UNREVIEWED",
                    },
                }
            )
            for other_id, other_hash in seen_perceptual:
                distance = hamming_distance(phash, other_hash)
                if distance <= NEAR_DUPLICATE_HAMMING_THRESHOLD:
                    near_duplicates.append(
                        {
                            "filename": _safe_name(original),
                            "sha256": digest,
                            "near_duplicate_of": other_id,
                            "perceptual_hamming_distance": str(distance),
                        }
                    )
            seen_perceptual.append((request_id, phash))
            for other_id, other_digest, other_hash in prior:
                if digest == other_digest:
                    duplicates.append(
                        {
                            "filename": _safe_name(original),
                            "sha256": digest,
                            "duplicate_of": other_id,
                        }
                    )
                    continue
                distance = hamming_distance(phash, other_hash)
                if distance <= NEAR_DUPLICATE_HAMMING_THRESHOLD:
                    near_duplicates.append(
                        {
                            "filename": _safe_name(original),
                            "sha256": digest,
                            "near_duplicate_of": other_id,
                            "perceptual_hamming_distance": str(distance),
                        }
                    )
        manifest = {
            "schema_version": BATCH_SCHEMA,
            "batch_id": batch_id,
            "created_at": _now(),
            "queue_order": "safe filename casefold, then content SHA-256",
            "records": records,
            "duplicate_uploads": duplicates,
            "near_duplicate_candidates": near_duplicates,
            "limits": {"max_images": self.max_batch, "max_file_bytes": self.max_file_bytes},
        }
        _write_json(directory / "batch-manifest.json", manifest)
        self._materialize(batch_id)
        return cast(dict[str, Any], manifest)

    def load(self, batch_id: str) -> dict[str, Any]:
        directory = self._batch(batch_id)
        manifest = json.loads((directory / "batch-manifest.json").read_text(encoding="utf-8"))
        for row in manifest["records"]:
            result_path = directory / "records" / f"{row['request_id']}.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result.setdefault("project_release_decision", _project_decision(result))
                row["result"] = result
        manifest["summary"] = self.summary(batch_id)
        return cast(dict[str, Any], manifest)

    def list_batches(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob("batch-*/batch-manifest.json"), reverse=True):
            value = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "batch_id": value["batch_id"],
                    "created_at": value["created_at"],
                    "summary": self.summary(value["batch_id"]),
                }
            )
        return rows

    def pending(self, batch_id: str) -> dict[str, Any] | None:
        manifest = self.load(batch_id)
        return next((row for row in manifest["records"] if row["state"] == "PENDING"), None)

    def save_result(self, batch_id: str, request_id: str, result: dict[str, Any]) -> None:
        directory = self._batch(batch_id)
        manifest_path = directory / "batch-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row = next((item for item in manifest["records"] if item["request_id"] == request_id), None)
        if row is None:
            raise ValueError("unknown request ID")
        _write_json(directory / "records" / f"{request_id}.json", result)
        row["state"] = (
            "FAILED"
            if result["pipeline_status"]
            in {"IMAGE_DECODE_ERROR", "MODEL_BUNDLE_ERROR", "INFERENCE_ERROR"}
            else "COMPLETED"
        )
        row["review"] = result["manual_review"]
        _write_json(manifest_path, manifest)
        self._materialize(batch_id)

    def save_review(
        self, batch_id: str, request_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        detection = str(payload.get("detection_assessment", "UNREVIEWED"))
        legacy_recognition = str(payload.get("recognition_assessment", "UNREVIEWED"))
        explicit_assessments = {
            field: str(payload.get(field, "UNREVIEWED")) for field in RECOGNITION_ASSESSMENT_FIELDS
        }
        reason = str(payload.get("detection_error_reason", ""))
        if detection not in DETECTION_STATES or legacy_recognition not in RECOGNITION_STATES:
            raise ValueError("invalid explicit assessment state")
        if any(value not in RECOGNITION_STATES for value in explicit_assessments.values()):
            raise ValueError("invalid output-specific assessment state")
        if reason not in DETECTION_REASONS:
            raise ValueError("invalid Detection error reason")
        corrected = str(payload.get("corrected_plate_text", "")).strip()
        corrected_validation: dict[str, Any] | None = None
        if corrected:
            try:
                normalized = normalize_plate_text(corrected).normalized_text
            except PlateNormalizationError as error:
                raise ValueError(str(error)) from error
            corrected = normalized
            from anpr_engine.recognition.profiles import TR_PROFILE

            corrected_validation = TR_PROFILE.validate(corrected).model_dump(mode="json")
        dataset_state = str(payload.get("dataset_state", "INCOMING_QUARANTINED"))
        audit_status = str(payload.get("audit_status", "PENDING"))
        if dataset_state not in DATASET_STATES or audit_status not in AUDIT_STATES:
            raise ValueError("invalid dataset or audit state")
        group_fields = {
            name: str(payload.get(name, "")).strip()[:200] or None
            for name in ("physical_plate_id", "vehicle_group_id", "capture_session_id")
        }
        conditions = sorted({str(value) for value in payload.get("capture_conditions", [])})
        if any(value not in CAPTURE_CONDITIONS for value in conditions):
            raise ValueError("unsupported capture condition")
        directory = self._batch(batch_id)
        manifest_path = directory / "batch-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row = next((item for item in manifest["records"] if item["request_id"] == request_id), None)
        if row is None or row["state"] not in {"COMPLETED", "FAILED"}:
            raise ValueError("request is not reviewable")
        result_path = directory / "records" / f"{request_id}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if dataset_state == "TRAINING_ELIGIBLE":
            if any(value is None for value in group_fields.values()):
                raise ValueError("training eligibility requires plate, vehicle, and session IDs")
            if not corrected or audit_status != "AUDITED":
                raise ValueError("training eligibility requires audited owner ground truth")
            if corrected_validation is None or not corrected_validation.get("valid"):
                raise ValueError("training eligibility requires a supported TR plate label")
        recognition_result = result.get("recognition") or {}
        has_explicit_assessment = any(field in payload for field in RECOGNITION_ASSESSMENT_FIELDS)
        if has_explicit_assessment:
            recognition = explicit_assessments["official_raw_assessment"]
            if legacy_recognition not in {"UNREVIEWED", recognition}:
                raise ValueError("legacy Recognition assessment must match Official RAW")
            legacy_recognition = recognition
            legacy_target = "OFFICIAL_RAW"
        else:
            recognition = legacy_recognition
            legacy_target = "LEGACY_UNSCOPED"
        output_values = {value for value in (recognition_result.get("raw_text"),) if value}
        any_incorrect = any(value == "INCORRECT" for value in explicit_assessments.values())
        if has_explicit_assessment and not corrected and (len(output_values) > 1 or any_incorrect):
            raise ValueError(
                "correct plate text is required when Recognition outputs disagree "
                "or any is incorrect"
            )
        if (
            recognition == "CORRECT"
            and corrected
            and corrected
            not in {recognition_result.get("raw_text"), recognition_result.get("tr_text")}
        ):
            raise ValueError(
                "contradictory review: correct Recognition does not match ground truth"
            )
        # Historical intake events used ``corrected_plate_text`` as the audited GT even
        # when it exactly matched a CORRECT project prediction. Keep those events
        # readable/write-compatible; only a differing value is an owner correction.
        if recognition == "NOT_APPLICABLE" and corrected:
            raise ValueError("owner correction requires an incorrect project-model assessment")
        resolution = resolve_owner_review(
            str(recognition_result.get("raw_text")) if recognition_result.get("raw_text") else None,
            recognition,
            corrected,
        )
        adjusted_bbox = payload.get("owner_adjusted_bbox_xyxy")
        adjusted_crop: dict[str, Any] | None = None
        if adjusted_bbox not in (None, "", []):
            if not isinstance(adjusted_bbox, list) or len(adjusted_bbox) != 4:
                raise ValueError("owner-adjusted bounding box must contain x1,y1,x2,y2")
            values = [int(value) for value in adjusted_bbox]
            width = int(result.get("input", {}).get("width", 0))
            height = int(result.get("input", {}).get("height", 0))
            x1, y1, x2, y2 = values
            if width < 1 or height < 1 or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("owner-adjusted bounding box is outside the source image")
            raw_path = directory / row["input_reference"]
            image = cv2.imread(raw_path.as_posix(), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cannot decode immutable source image")
            crop = image[y1:y2, x1:x2]
            crop_path = directory / "artifacts" / request_id / "owner-adjusted-crop.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(crop_path.as_posix(), crop):
                raise ValueError("cannot write owner-adjusted crop")
            adjusted_crop = {
                "artifact": "owner-adjusted-crop.jpg",
                "bbox_xyxy": values,
                "crop_policy": "OWNER_ADJUSTED_TIGHT_XYXY_V1",
                "software_version": "real-recognition-intake-v1",
                "source_image_sha256": row["input_sha256"],
                "crop_sha256": content_sha256(crop_path.read_bytes()),
                "width": x2 - x1,
                "height": y2 - y1,
            }
        events_path = directory / "review-events.jsonl"
        previous = "0" * 64
        if events_path.exists() and events_path.stat().st_size:
            previous = json.loads(events_path.read_text(encoding="utf-8").splitlines()[-1])[
                "current_event_hash"
            ]
        event = {
            "schema_version": REVIEW_SCHEMA
            if has_explicit_assessment
            else "real-recognition-intake-event-v1",
            "batch_id": batch_id,
            "request_id": request_id,
            "input_sha256": row["input_sha256"],
            "safe_original_filename": row["safe_original_filename"],
            "detection_model_sha256": DETECTOR_SHA,
            "recognition_model_sha256": RECOGNIZER_SHA,
            "detection_result_identity": hashlib.sha256(
                json.dumps(result.get("detection"), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "recognition_result_identity": hashlib.sha256(
                json.dumps(
                    result.get("recognition"), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "official_raw_prediction": recognition_result.get("raw_text"),
            "project_recognition_raw_prediction": recognition_result.get("raw_text"),
            "project_recognition_tr_prediction": recognition_result.get("tr_text"),
            "project_recognition_confidence": recognition_result.get("raw_confidence"),
            "structural_validation": recognition_result.get("structural_validation"),
            "release_version": RELEASE_VERSION,
            "release_primary_status": resolution["primary_status"],
            "resolved_plate_text": resolution["resolved_plate_text"],
            "resolution_source": resolution["resolution_source"],
            "owner_review_required": resolution["owner_review_required"],
            "detection_assessment": detection,
            "project_detection_assessment": detection,
            "recognition_assessment": legacy_recognition,
            "project_recognition_assessment": recognition,
            "recognition_assessment_target": legacy_target,
            **explicit_assessments,
            "assessment_targets": {
                "detection_assessment": {
                    "output": "DETECTION_TOP1_BOUNDING_BOX",
                    "result_identity": hashlib.sha256(
                        json.dumps(
                            result.get("detection"), sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                },
                "official_raw_assessment": {
                    "output": "OFFICIAL_RAW",
                    "model_sha256": RECOGNIZER_SHA,
                    "prediction": recognition_result.get("raw_text"),
                },
            },
            "detection_error_reason": reason or None,
            "corrected_plate_text": corrected or None,
            "corrected_plate_validation": corrected_validation,
            "owner_comment": str(payload.get("owner_comment", ""))[:4000] or None,
            "ground_truth_plate_text": resolution["resolved_plate_text"],
            "dataset_state": dataset_state,
            "audit_status": audit_status,
            **group_fields,
            "capture_conditions": conditions,
            "owner_adjusted_crop": adjusted_crop,
            "training_eligible": dataset_state == "TRAINING_ELIGIBLE",
            "reviewed_timestamp": _now(),
            "previous_event_hash": previous,
        }
        canonical = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
        event["current_event_hash"] = hashlib.sha256(previous.encode() + canonical).hexdigest()
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        row["review"] = event
        result["manual_review"] = event
        _write_json(result_path, result)
        _write_json(manifest_path, manifest)
        self._materialize(batch_id)
        return dict(event)

    def verify_audit_chain(self, batch_id: str) -> bool:
        path = self._batch(batch_id) / "review-events.jsonl"
        previous = "0" * 64
        if not path.exists():
            return True
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            current = event.pop("current_event_hash")
            if event["previous_event_hash"] != previous:
                return False
            canonical = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
            if current != hashlib.sha256(previous.encode() + canonical).hexdigest():
                return False
            previous = current
        return True

    def summary(self, batch_id: str) -> dict[str, Any]:
        manifest = json.loads(
            (self._batch(batch_id) / "batch-manifest.json").read_text(encoding="utf-8")
        )
        records = manifest["records"]
        results = []
        for row in records:
            path = self._batch(batch_id) / "records" / f"{row['request_id']}.json"
            if path.exists():
                results.append(json.loads(path.read_text(encoding="utf-8")))
        reviews = [row.get("review", {}) for row in records]
        legacy_fully = [
            r
            for r in reviews
            if r.get("detection_assessment") != "UNREVIEWED"
            and r.get("recognition_assessment") != "UNREVIEWED"
        ]
        fully = [
            r
            for r in reviews
            if r.get("detection_assessment") != "UNREVIEWED"
            and all(
                r.get(field) not in {None, "UNREVIEWED"} for field in RECOGNITION_ASSESSMENT_FIELDS
            )
        ]
        det_reviewed = [
            r for r in reviews if r.get("detection_assessment") in {"CORRECT", "INCORRECT"}
        ]
        rec_conditional = [
            r
            for r in reviews
            if r.get("detection_assessment") == "CORRECT"
            and r.get("official_raw_assessment") in {"CORRECT", "INCORRECT"}
        ]
        both = [
            r
            for r in reviews
            if r.get("detection_assessment") == "CORRECT"
            and r.get("official_raw_assessment") == "CORRECT"
        ]

        def rate(numerator: int, denominator: int) -> dict[str, Any]:
            return {
                "numerator": numerator,
                "denominator": denominator,
                "percent": None if not denominator else 100 * numerator / denominator,
            }

        return {
            "uploaded_images": len(records),
            "successfully_decoded": sum(
                x["pipeline_status"] != "IMAGE_DECODE_ERROR" for x in results
            ),
            "detection_completed": sum(
                x["pipeline_status"]
                not in {"IMAGE_DECODE_ERROR", "MODEL_BUNDLE_ERROR", "INFERENCE_ERROR"}
                for x in results
            ),
            "no_plate_detected": sum(x["pipeline_status"] == "NO_PLATE_DETECTED" for x in results),
            "multiple_candidate_cases": sum(
                x["pipeline_status"] == "MULTIPLE_PLATE_CANDIDATES" for x in results
            ),
            "recognition_completed": sum(x.get("recognition") is not None for x in results),
            "inference_failures": sum(
                x["pipeline_status"]
                in {"IMAGE_DECODE_ERROR", "MODEL_BUNDLE_ERROR", "INFERENCE_ERROR"}
                for x in results
            ),
            "pending": sum(r["state"] == "PENDING" for r in records),
            "reviewed": len(fully),
            "unreviewed": len(records) - len(fully),
            "legacy_reviewed": len(legacy_fully),
            "detection_correct": sum(r.get("detection_assessment") == "CORRECT" for r in reviews),
            "detection_incorrect": sum(
                r.get("detection_assessment") == "INCORRECT" for r in reviews
            ),
            "recognition_correct": sum(
                r.get("official_raw_assessment") == "CORRECT" for r in reviews
            ),
            "recognition_incorrect": sum(
                r.get("official_raw_assessment") == "INCORRECT" for r in reviews
            ),
            "recognition_not_applicable": sum(
                r.get("official_raw_assessment") == "NOT_APPLICABLE" for r in reviews
            ),
            "legacy_recognition_assessment_ambiguous": sum(
                r.get("recognition_assessment") in {"CORRECT", "INCORRECT"}
                and r.get("recognition_assessment_target") != "OFFICIAL_RAW"
                for r in reviews
            ),
            "official_raw_correct": sum(
                r.get("official_raw_assessment") == "CORRECT" for r in reviews
            ),
            "manual_detection_accuracy": rate(
                sum(r.get("detection_assessment") == "CORRECT" for r in det_reviewed),
                len(det_reviewed),
            ),
            "manual_recognition_accuracy_conditional_on_accepted_detection": rate(
                sum(r.get("official_raw_assessment") == "CORRECT" for r in rec_conditional),
                len(rec_conditional),
            ),
            "manual_end_to_end_exact_success": rate(len(both), len(det_reviewed)),
            "audit_chain_valid": self.verify_audit_chain(batch_id),
        }

    def _materialize(self, batch_id: str) -> None:
        _write_json(self._batch(batch_id) / "batch-summary.json", self.summary(batch_id))

    def export(self, batch_id: str, kind: str) -> tuple[bytes, str]:
        manifest = self.load(batch_id)
        rows = []
        for item in manifest["records"]:
            result = item.get("result", {})
            detection, recognition, review = (
                result.get("detection", {}),
                result.get("recognition") or {},
                item.get("review", {}),
            )
            project = _project_decision(result)
            if recognition.get("raw_text"):
                resolution = resolve_owner_review(
                    str(recognition["raw_text"]),
                    review.get("official_raw_assessment") or review.get("recognition_assessment"),
                    review.get("corrected_plate_text"),
                )
            else:
                resolution = {
                    "primary_status": project.get("primary_status"),
                    "resolved_plate_text": None,
                    "resolution_source": "UNRESOLVED",
                    "owner_review_required": project.get("owner_review_required"),
                }
            selected = detection.get("selected") or {}
            rows.append(
                {
                    "batch_id": batch_id,
                    "request_id": item["request_id"],
                    "filename": item["safe_original_filename"],
                    "image_sha256": item["input_sha256"],
                    "immutable_source_reference": item["input_reference"],
                    "pipeline_status": result.get("pipeline_status"),
                    "detection_candidate_count": detection.get("candidate_count"),
                    "detection_prediction": selected,
                    "project_detection_prediction": selected,
                    "selected_bounding_box": selected.get("bbox_xyxy"),
                    "detection_crop_provenance": result.get("crop_reference"),
                    "recognition_crop_provenance": result.get("recognition_crop_reference"),
                    "detection_confidence": selected.get("confidence"),
                    "raw_prediction": recognition.get("raw_text"),
                    "official_raw_prediction": recognition.get("raw_text"),
                    "project_recognition_raw_prediction": recognition.get("raw_text"),
                    "project_recognition_tr_prediction": recognition.get("tr_text"),
                    "project_recognition_confidence": recognition.get("raw_confidence"),
                    "structural_validation": recognition.get("structural_validation"),
                    "release_version": RELEASE_VERSION,
                    "release_decision_policy": "OWNER_CONFIRM_OR_CORRECT_PROJECT_MODEL",
                    "release_primary_status": resolution.get("primary_status"),
                    "manual_review_required": resolution.get("owner_review_required"),
                    "owner_review_required": resolution.get("owner_review_required"),
                    "resolved_plate_text": resolution.get("resolved_plate_text"),
                    "final_resolved_plate_text": resolution.get("resolved_plate_text"),
                    "resolution_source": resolution.get("resolution_source"),
                    "tr_prediction": recognition.get("tr_text"),
                    "recognition_confidence": recognition.get("raw_confidence"),
                    "detection_assessment": review.get("detection_assessment", "UNREVIEWED"),
                    "project_detection_assessment": review.get(
                        "detection_assessment", "UNREVIEWED"
                    ),
                    "recognition_assessment": review.get("recognition_assessment", "UNREVIEWED"),
                    "recognition_assessment_target": review.get(
                        "recognition_assessment_target", "LEGACY_UNSCOPED"
                    ),
                    "official_raw_assessment": review.get("official_raw_assessment", "UNREVIEWED"),
                    "project_recognition_assessment": review.get(
                        "official_raw_assessment",
                        review.get("recognition_assessment", "UNREVIEWED"),
                    ),
                    "assessment_targets": review.get("assessment_targets"),
                    "detection_error_reason": review.get("detection_error_reason"),
                    "corrected_plate_text": review.get("corrected_plate_text"),
                    "owner_corrected_plate_text": review.get("corrected_plate_text"),
                    "owner_comment": review.get("owner_comment"),
                    "dataset_state": review.get("dataset_state", "INCOMING_QUARANTINED"),
                    "audit_status": review.get("audit_status", "PENDING"),
                    "physical_plate_id": review.get("physical_plate_id"),
                    "vehicle_group_id": review.get("vehicle_group_id"),
                    "capture_session_id": review.get("capture_session_id"),
                    "capture_conditions": review.get("capture_conditions", []),
                    "perceptual_hash": item.get("perceptual_hash"),
                    "owner_adjusted_crop": review.get("owner_adjusted_crop"),
                    "timings_ms": result.get("timings_ms"),
                    "detection_model_sha256": DETECTOR_SHA,
                    "detection_checkpoint_sha256": DETECTOR_SHA,
                    "recognition_model_sha256": RECOGNIZER_SHA,
                    "recognition_checkpoint_sha256": RECOGNIZER_SHA,
                    "crop_refinement_identity": CROP_REFINEMENT,
                    "crop_refinement_provenance": result.get("recognition_crop_reference"),
                    "frozen_identities": frozen_identities(),
                    "processing_errors": result.get("errors", []),
                    "error": result.get("errors", []),
                    "append_only_review_event_integrity": self.verify_audit_chain(batch_id),
                    "audit_chain_valid": self.verify_audit_chain(batch_id),
                    "reviewed_timestamp": review.get("reviewed_timestamp"),
                }
            )
        if kind == "json":
            return (
                json.dumps(
                    {"manifest": manifest, "records": rows}, indent=2, sort_keys=True
                ).encode(),
                "application/json",
            )
        if kind == "jsonl":
            return (
                ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode(),
                "application/x-ndjson",
            )
        if kind == "csv":
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else ["batch_id"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()
                    }
                )
            return output.getvalue().encode(), "text/csv"
        raise ValueError("unsupported export format")
