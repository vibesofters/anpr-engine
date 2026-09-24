from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from anpr_engine.integration.project_models_release import (
    RELEASE_VERSION,
    resolve_owner_review,
    unresolved_decision,
)
from anpr_engine.integration.review_store import BatchReviewStore

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/launch_anpr_v1_browser.py"
BROWSER = ROOT / "src/anpr_engine/integration/browser.py"
PIPELINE = ROOT / "src/anpr_engine/integration/anpr_v1.py"


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((20, 40, 3), 120, dtype=np.uint8)).save(buffer, "JPEG")
    return buffer.getvalue()


def _result() -> dict[str, object]:
    return {
        "pipeline_status": "ACCEPTED",
        "detection": {
            "candidate_count": 1,
            "selected": {"bbox_xyxy": [1, 2, 30, 15], "confidence": 0.91},
        },
        "crop_reference": {"artifact": "crop.jpg", "crop_policy": "TOP1_ZERO_PADDING"},
        "recognition_crop_reference": {
            "artifact": "refined-crop.jpg",
            "refinement_policy": "BLUE_BAND_GLYPH_QUAD_V1",
        },
        "recognition": {
            "raw_text": "34ABC12",
            "tr_text": "34ABC12",
            "raw_confidence": 0.99,
            "structural_validation": {"valid": True},
        },
        "project_release_decision": unresolved_decision("ACCEPTED", "34ABC12"),
        "manual_review": {
            "detection_assessment": "UNREVIEWED",
            "recognition_assessment": "UNREVIEWED",
        },
        "errors": [],
        "timings_ms": {},
    }


def _review_payload(assessment: str, corrected: str = "") -> dict[str, str]:
    return {
        "detection_assessment": "CORRECT",
        "recognition_assessment": assessment,
        "official_raw_assessment": assessment,
        "corrected_plate_text": corrected,
    }


def test_project_only_resolution_requires_owner_action() -> None:
    assert unresolved_decision("ACCEPTED", "34ABC12")["resolved_plate_text"] is None
    assert resolve_owner_review("34ABC12", "CORRECT", None) == {
        "primary_status": "OWNER CONFIRMED",
        "resolved_plate_text": "34ABC12",
        "resolution_source": "OWNER_CONFIRMED_PROJECT_MODEL",
        "owner_review_required": False,
    }
    assert (
        resolve_owner_review("34ABC12", "INCORRECT", "34ABC13")["resolution_source"]
        == "OWNER_CORRECTED"
    )


def test_new_exports_contain_only_project_results(tmp_path: Path) -> None:
    store = BatchReviewStore(tmp_path)
    for index, (assessment, corrected, expected, source) in enumerate(
        (
            ("CORRECT", "", "34ABC12", "OWNER_CONFIRMED_PROJECT_MODEL"),
            ("INCORRECT", "34ABC13", "34ABC13", "OWNER_CORRECTED"),
        )
    ):
        batch = store.create_batch([(f"{index}.jpg", _jpeg())])
        batch_id, row = batch["batch_id"], batch["records"][0]
        store.save_result(batch_id, row["request_id"], _result())
        event = store.save_review(
            batch_id, row["request_id"], _review_payload(assessment, corrected)
        )
        assert event["resolved_plate_text"] == expected
        assert event["resolution_source"] == source
        exported = json.loads(store.export(batch_id, "json")[0])["records"][0]
        assert exported["release_version"] == RELEASE_VERSION
        assert exported["resolved_plate_text"] == expected
        assert exported["resolution_source"] == source
        assert exported["project_recognition_raw_prediction"] == "34ABC12"
        assert exported["project_detection_assessment"] == "CORRECT"
        assert "secondary_raw_prediction" not in exported
        assert "fused_candidate" not in exported
        assert "final_automatic_plate_text" not in exported
        assert exported["append_only_review_event_integrity"] is True
        assert exported["audit_chain_valid"] is True
        assert exported["detection_checkpoint_sha256"]
        assert exported["recognition_checkpoint_sha256"]
        assert exported["crop_refinement_identity"] == "BLUE_BAND_GLYPH_QUAD_V1"


def test_active_runtime_accepts_only_project_model_options() -> None:
    help_result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for removed in (
        "--secondary-python",
        "--secondary-source-root",
        "--secondary-weight",
        "--secondary-timeout-ms",
        "--development-multi-view-consensus",
    ):
        assert removed not in help_result.stdout
    active_browser = BROWSER.read_text(encoding="utf-8")
    assert "subprocess" not in active_browser
    assert "worker" not in active_browser
    assert "weight" not in help_result.stdout
    assert "_fusion_evidence" not in PIPELINE.read_text(encoding="utf-8")
