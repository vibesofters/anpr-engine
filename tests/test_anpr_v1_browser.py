from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from anpr_engine.integration.anpr_v1 import (
    DETECTOR_SHA,
    RECOGNIZER_SHA,
    FrozenBundle,
    decode_upload,
)
from anpr_engine.integration.browser import HTML, parse_multipart
from anpr_engine.integration.review_store import BatchReviewStore


def _jpeg(color: int = 100) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((20, 40, 3), color, dtype=np.uint8)).save(buffer, "JPEG")
    return buffer.getvalue()


def _complete(
    store: BatchReviewStore, batch_id: str, request_id: str, *, status: str = "ACCEPTED"
) -> None:
    store.save_result(
        batch_id,
        request_id,
        {
            "schema_version": "anpr-v1-browser-result-v1",
            "pipeline_status": status,
            "detection": {
                "candidate_count": 1,
                "selected": {"bbox_xyxy": [1, 2, 3, 4], "confidence": 0.9},
            },
            "recognition": {"raw_text": "34ABC12", "tr_text": "34ABC12", "raw_confidence": 0.8},
            "manual_review": {
                "detection_assessment": "UNREVIEWED",
                "recognition_assessment": "UNREVIEWED",
            },
            "timings_ms": {"total_request_ms": 2.0},
        },
    )


def test_decode_validates_real_content_and_orientation() -> None:
    image, kind = decode_upload(_jpeg())
    assert image.shape == (20, 40, 3)
    assert image.dtype == np.uint8
    assert kind == "JPEG"
    with pytest.raises(ValueError, match="corrupt"):
        decode_upload(b"not an image")


def test_multipart_multi_file_parsing() -> None:
    boundary = "abc123"
    payload = b""
    for name in ("b.jpg", "a.jpg"):
        payload += (
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="images"; filename="{name}"\r\n'
                "Content-Type: image/jpeg\r\n\r\n"
            ).encode()
            + _jpeg()
            + b"\r\n"
        )
    payload += f"--{boundary}--\r\n".encode()
    files = parse_multipart(f"multipart/form-data; boundary={boundary}", payload)
    assert [name for name, _ in files] == ["b.jpg", "a.jpg"]


def test_batch_order_duplicates_limits_and_safe_paths(tmp_path: Path) -> None:
    store = BatchReviewStore(tmp_path, max_file_bytes=10_000, max_batch=3)
    first, second = _jpeg(10), _jpeg(20)
    batch = store.create_batch([("../B.jpg", second), ("a.jpg", first), ("copy.jpg", first)])
    assert [row["safe_original_filename"] for row in batch["records"]] == ["a.jpg", "B.jpg"]
    assert len(batch["duplicate_uploads"]) == 1
    assert not any(".." in row["input_reference"] for row in batch["records"])
    with pytest.raises(ValueError, match="1-3"):
        store.create_batch([("x.jpg", first)] * 4)
    with pytest.raises(ValueError, match="unsupported"):
        store.create_batch([("x.gif", first)])


def test_review_revision_chain_resume_summary_and_exports(tmp_path: Path) -> None:
    store = BatchReviewStore(tmp_path)
    batch = store.create_batch([("one.jpg", _jpeg())])
    batch_id, request_id = batch["batch_id"], batch["records"][0]["request_id"]
    _complete(store, batch_id, request_id)
    first = store.save_review(
        batch_id,
        request_id,
        {
            "detection_assessment": "CORRECT",
            "recognition_assessment": "INCORRECT",
            "corrected_plate_text": "34 abc 12",
            "owner_comment": "visible",
        },
    )
    second = store.save_review(
        batch_id,
        request_id,
        {
            "detection_assessment": "CORRECT",
            "recognition_assessment": "CORRECT",
            "official_raw_assessment": "CORRECT",
        },
    )
    assert second["previous_event_hash"] == first["current_event_hash"]
    assert first["corrected_plate_text"] == "34ABC12"
    assert store.verify_audit_chain(batch_id)
    resumed = BatchReviewStore(tmp_path).load(batch_id)
    assert resumed["records"][0]["review"]["recognition_assessment"] == "CORRECT"
    summary = resumed["summary"]
    assert summary["manual_detection_accuracy"] == {
        "numerator": 1,
        "denominator": 1,
        "percent": 100.0,
    }
    assert (
        summary["manual_recognition_accuracy_conditional_on_accepted_detection"]["denominator"] == 1
    )
    for kind in ("json", "jsonl", "csv"):
        content, mime = store.export(batch_id, kind)
        assert batch_id.encode() in content
        assert mime


def test_tri_state_validation_and_no_ambiguous_checkbox(tmp_path: Path) -> None:
    store = BatchReviewStore(tmp_path)
    batch = store.create_batch([("one.jpg", _jpeg())])
    batch_id, request_id = batch["batch_id"], batch["records"][0]["request_id"]
    _complete(store, batch_id, request_id)
    with pytest.raises(ValueError, match="explicit assessment"):
        store.save_review(
            batch_id, request_id, {"detection_assessment": "", "recognition_assessment": "CORRECT"}
        )
    assert 'type="radio"' in HTML
    assert "UNREVIEWED','CORRECT','INCORRECT" in HTML


def test_browser_contains_required_columns_and_proportional_images() -> None:
    ordered = [
        "<th>Raw Image</th>",
        "<th>Detection</th>",
        "<th>Project Model Prediction</th>",
        "<th>Resolved Result</th>",
        "<th>Owner Assessments</th>",
    ]
    offsets = [HTML.index(label) for label in ordered]
    assert offsets == sorted(offsets)
    assert "object-fit:contain" in HTML
    assert "Save & Continue" in HTML
    assert "MULTIPLE_CANDIDATES" in HTML
    assert "Detection-Annotated Image" in HTML
    assert "INFORMATIONAL ONLY" in HTML
    assert "MODEL PREDICTION" in HTML
    assert "RESOLVED RESULT" in HTML
    assert "project models only" in HTML


def test_result_storage_failure_isolation(tmp_path: Path) -> None:
    store = BatchReviewStore(tmp_path)
    batch = store.create_batch([("a.jpg", _jpeg(1)), ("b.jpg", _jpeg(2))])
    first, second = batch["records"]
    _complete(store, batch["batch_id"], first["request_id"], status="INFERENCE_ERROR")
    pending = store.pending(batch["batch_id"])
    assert pending is not None
    assert pending["request_id"] == second["request_id"]


def test_safe_batch_and_audit_tamper_detection(tmp_path: Path) -> None:
    store = BatchReviewStore(tmp_path)
    with pytest.raises(ValueError, match="invalid batch"):
        store.load("../../etc")
    batch = store.create_batch([("a.jpg", _jpeg())])
    bid, rid = batch["batch_id"], batch["records"][0]["request_id"]
    _complete(store, bid, rid)
    store.save_review(
        bid, rid, {"detection_assessment": "CORRECT", "recognition_assessment": "CORRECT"}
    )
    path = store._batch(bid) / "review-events.jsonl"
    event = json.loads(path.read_text())
    event["owner_comment"] = "tampered"
    path.write_text(json.dumps(event) + "\n")
    assert not store.verify_audit_chain(bid)


def test_cv2_available_for_real_artifact_path() -> None:
    encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1]
    assert encoded.size > 0


def test_bundle_wrong_checkpoint_fails_closed(tmp_path: Path) -> None:
    detector = tmp_path / "detector.pt"
    recognizer = tmp_path / "recognizer.pt"
    manifest = tmp_path / "bundle-manifest.json"
    detector.write_bytes(b"wrong")
    recognizer.write_bytes(b"wrong")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "anpr-engine-private-model-bundle-v1",
                "classification": "PRIVATE_OWNER_ONLY",
                "models": [
                    {
                        "role": "detection",
                        "sha256": DETECTOR_SHA,
                    },
                    {
                        "role": "recognition",
                        "sha256": RECOGNIZER_SHA,
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="detector bundle hash"):
        FrozenBundle(tmp_path, detector, recognizer, manifest).verify()


def test_bundle_missing_checkpoint_fails_closed(tmp_path: Path) -> None:
    bundle = FrozenBundle(
        tmp_path,
        tmp_path / "missing-detector.pt",
        tmp_path / "missing-recognizer.pt",
        tmp_path / "missing-manifest.json",
    )
    with pytest.raises(FileNotFoundError):
        bundle.verify()
