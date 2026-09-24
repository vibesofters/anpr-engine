"""Deterministic sample-level comparison of two validation analyses."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def sample_outcome(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    baseline_matched = bool(baseline["matched_at_iou_0_50"])
    candidate_matched = bool(candidate["matched_at_iou_0_50"])
    if baseline_matched != candidate_matched:
        return "candidate_win" if candidate_matched else "candidate_regression"
    baseline_false_positives = int(baseline["false_positive_count"])
    candidate_false_positives = int(candidate["false_positive_count"])
    if baseline_false_positives != candidate_false_positives:
        return (
            "candidate_win"
            if candidate_false_positives < baseline_false_positives
            else "candidate_regression"
        )
    baseline_iou = float(baseline["best_iou"] or 0.0)
    candidate_iou = float(candidate["best_iou"] or 0.0)
    if candidate_iou >= baseline_iou + 0.05:
        return "candidate_win"
    if baseline_iou >= candidate_iou + 0.05:
        return "candidate_regression"
    return "tie"


def compare_samples(
    baseline_samples: list[dict[str, Any]], candidate_samples: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_by_id = {sample["sample_id"]: sample for sample in baseline_samples}
    candidate_by_id = {sample["sample_id"]: sample for sample in candidate_samples}
    if len(baseline_by_id) != len(baseline_samples) or len(candidate_by_id) != len(
        candidate_samples
    ):
        raise ValueError("validation analyses contain duplicate sample IDs")
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("validation analyses do not contain the same sample IDs")
    records: list[dict[str, Any]] = []
    for sample_id in sorted(baseline_by_id):
        baseline = baseline_by_id[sample_id]
        candidate = candidate_by_id[sample_id]
        if baseline["plate_size_slice"] != candidate["plate_size_slice"]:
            raise ValueError(f"validation size slice differs for sample: {sample_id}")
        records.append(
            {
                "sample_id": sample_id,
                "plate_size_slice": candidate["plate_size_slice"],
                "outcome": sample_outcome(baseline, candidate),
                "baseline": baseline,
                "candidate": candidate,
            }
        )
    outcome_counts = {
        outcome: sum(record["outcome"] == outcome for record in records)
        for outcome in ("candidate_win", "tie", "candidate_regression")
    }
    residuals = {
        "candidate_miss_count": sum(
            not record["candidate"]["matched_at_iou_0_50"] for record in records
        ),
        "candidate_multiple_detection_count": sum(
            record["candidate"]["detection_count"] > 1 for record in records
        ),
        "candidate_false_positive_image_count": sum(
            record["candidate"]["false_positive_count"] > 0 for record in records
        ),
        "candidate_iou_below_0_75_count": sum(
            float(record["candidate"]["best_iou"] or 0.0) < 0.75 for record in records
        ),
    }
    return {
        "population": len(records),
        "outcome_rule": (
            "match at IoU 0.50, then fewer false positives, then at least 0.05 IoU gain; "
            "otherwise tie"
        ),
        "outcome_counts": outcome_counts,
        "candidate_residuals": residuals,
    }, records


def select_review_cases(records: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    selectors: dict[str, Callable[[dict[str, Any]], bool]] = {
        "ordinary": lambda record: (
            record["plate_size_slice"] == "medium"
            and record["baseline"]["matched_at_iou_0_50"]
            and record["baseline"]["false_positive_count"] == 0
        ),
        "small_plate": lambda record: record["plate_size_slice"] == "small",
        "baseline_1_fp_heavy": lambda record: record["baseline"]["false_positive_count"] > 0,
        "localization_failure": lambda record: (
            record["baseline"]["matched_at_iou_0_50"]
            and float(record["baseline"]["best_iou"] or 0.0) < 0.75
        ),
        "difficult_or_miss": lambda record: not record["baseline"]["matched_at_iou_0_50"],
        "regression": lambda record: record["outcome"] == "candidate_regression",
    }
    sort_keys: dict[str, Callable[[dict[str, Any]], Any]] = {
        "ordinary": lambda record: (
            -float(record["baseline"]["best_iou"] or 0.0),
            record["sample_id"],
        ),
        "small_plate": lambda record: (
            float(record["baseline"].get("plate_area_ratio", 1.0)),
            record["sample_id"],
        ),
        "baseline_1_fp_heavy": lambda record: (
            -int(record["baseline"]["background_false_positive_count"]),
            -int(record["baseline"]["duplicate_false_positive_count"]),
            record["sample_id"],
        ),
        "localization_failure": lambda record: (
            float(record["baseline"]["best_iou"] or 0.0),
            -float(record["baseline"]["best_normalized_center_offset"] or 0.0),
            record["sample_id"],
        ),
        "difficult_or_miss": lambda record: (
            int(record["baseline"]["detection_count"] > 0),
            record["sample_id"],
        ),
        "regression": lambda record: (
            int(record["candidate"]["matched_at_iou_0_50"]),
            -(
                int(record["candidate"]["false_positive_count"])
                - int(record["baseline"]["false_positive_count"])
            ),
            float(record["candidate"]["best_iou"] or 0.0)
            - float(record["baseline"]["best_iou"] or 0.0),
            record["sample_id"],
        ),
    }
    selected: dict[str, dict[str, Any] | None] = {}
    used: set[str] = set()
    for category, selector in selectors.items():
        candidates = sorted(
            (record for record in records if selector(record)), key=sort_keys[category]
        )
        unused = [record for record in candidates if record["sample_id"] not in used]
        chosen = unused[0] if unused else candidates[0] if candidates else None
        selected[category] = chosen
        if chosen is not None:
            used.add(str(chosen["sample_id"]))
    return selected
