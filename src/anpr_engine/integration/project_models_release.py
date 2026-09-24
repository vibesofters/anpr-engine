"""Owner-reviewed decision contract for the project-models-only ANPR V1 release."""

from __future__ import annotations

from typing import Any

from anpr_engine.integration.anpr_v1 import DETECTOR_SHA, RECOGNIZER_SHA

RELEASE_VERSION = "anpr-v1-project-models-only-20260916"
OWNER_DECISION = "OWNER_REQUIRES_PROJECT_OWNED_MODELS_ONLY"
CROP_REFINEMENT = "BLUE_BAND_GLYPH_QUAD_V1"
UNREVIEWED_STATUS = "MODEL PREDICTION — OWNER REVIEW REQUIRED"


def unresolved_decision(pipeline_status: str, raw_prediction: str | None) -> dict[str, Any]:
    if pipeline_status == "NO_PLATE_DETECTED":
        status = "NO PLATE DETECTED"
        review_required = False
    elif raw_prediction:
        status = UNREVIEWED_STATUS
        review_required = True
    else:
        status = "INFERENCE FAILURE"
        review_required = True
    return {
        "release_version": RELEASE_VERSION,
        "owner_decision": OWNER_DECISION,
        "primary_status": status,
        "model_prediction": raw_prediction,
        "resolved_plate_text": None,
        "resolution_source": "UNRESOLVED",
        "owner_review_required": review_required,
        "confidence_is_informational_only": True,
    }


def resolve_owner_review(
    raw_prediction: str | None,
    recognition_assessment: str | None,
    owner_corrected_plate_text: str | None,
) -> dict[str, Any]:
    assessment = (recognition_assessment or "UNREVIEWED").strip()
    corrected = (owner_corrected_plate_text or "").strip() or None
    if assessment == "CORRECT" and raw_prediction:
        return {
            "primary_status": "OWNER CONFIRMED",
            "resolved_plate_text": raw_prediction,
            "resolution_source": "OWNER_CONFIRMED_PROJECT_MODEL",
            "owner_review_required": False,
        }
    if assessment == "INCORRECT" and corrected:
        return {
            "primary_status": "OWNER CORRECTED",
            "resolved_plate_text": corrected,
            "resolution_source": "OWNER_CORRECTED",
            "owner_review_required": False,
        }
    return {
        "primary_status": UNREVIEWED_STATUS,
        "resolved_plate_text": None,
        "resolution_source": "UNRESOLVED",
        "owner_review_required": True,
    }


def frozen_identities() -> dict[str, str]:
    return {
        "detection_checkpoint_sha256": DETECTOR_SHA,
        "official_recognition_checkpoint_sha256": RECOGNIZER_SHA,
        "crop_refinement": CROP_REFINEMENT,
    }
