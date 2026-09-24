"""Validation-only, deterministic ANPR-oriented detector analysis helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from anpr_engine.detection.metrics import SamplePredictions, classify_sample_predictions
from anpr_engine.detection.model import box_iou


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None}
    return {"mean": mean(values), "median": median(values)}


def top1_analysis(
    predictions: tuple[SamplePredictions, ...], confidence_threshold: float
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Classify every image under the one-plate, one-crop ANPR contract."""
    records: dict[str, dict[str, Any]] = {}
    correct_ious: list[float] = []
    counts: Counter[str] = Counter()
    for prediction in predictions:
        boxes, scores, _ious, _match = classify_sample_predictions(prediction, confidence_threshold)
        if not len(scores):
            status = "TOP1_MISS"
            iou: float | None = None
            score: float | None = None
        else:
            winner = int(scores.argmax())
            iou = float(box_iou(prediction.target, boxes[winner].unsqueeze(0))[0])
            score = float(scores[winner])
            status = "TOP1_CORRECT" if iou >= 0.50 else "TOP1_WRONG_WINNER"
            if status == "TOP1_CORRECT":
                correct_ious.append(iou)
        counts[status] += 1
        records[prediction.sample_id] = {
            "top1_status": status,
            "top1_iou": iou,
            "top1_score": score,
            "top1_surviving_detection_count": len(scores),
        }
    population = len(predictions)
    return {
        "population": population,
        "top1_correct_count": counts["TOP1_CORRECT"],
        "top1_correct_rate": counts["TOP1_CORRECT"] / max(1, population),
        "top1_wrong_winner_count": counts["TOP1_WRONG_WINNER"],
        "top1_wrong_winner_rate": counts["TOP1_WRONG_WINNER"] / max(1, population),
        "top1_miss_count": counts["TOP1_MISS"],
        "top1_miss_rate": counts["TOP1_MISS"] / max(1, population),
        "top1_matched_iou": _summary(correct_ious),
    }, records


def slice_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-image metric records; AP is deliberately not sliced."""
    population = len(records)
    matched = [record for record in records if record["matched_at_iou_0_50"]]
    false_positives = sum(int(record["false_positive_count"]) for record in records)
    ious = [float(record["best_iou"]) for record in matched]
    top1 = Counter(str(record["top1_status"]) for record in records)
    return {
        "population": population,
        "true_positive_count": len(matched),
        "false_positive_count": false_positives,
        "detection_miss_count": population - len(matched),
        "precision": len(matched) / max(1, len(matched) + false_positives),
        "recall": len(matched) / max(1, population),
        "matched_iou": _summary(ious),
        "top1_correct_count": top1["TOP1_CORRECT"],
        "top1_correct_rate": top1["TOP1_CORRECT"] / max(1, population),
        "top1_wrong_winner_count": top1["TOP1_WRONG_WINNER"],
        "top1_wrong_winner_rate": top1["TOP1_WRONG_WINNER"] / max(1, population),
        "top1_miss_count": top1["TOP1_MISS"],
        "top1_miss_rate": top1["TOP1_MISS"] / max(1, population),
    }


def grouped_slice_metrics(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    return {name: slice_metrics(group) for name, group in sorted(grouped.items())}


def atomic_group_balanced_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Give each validation atomic group total weight one, not each image."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["atomic_group_id"])].append(record)
    group_values: list[dict[str, float]] = []
    for members in grouped.values():
        population = len(members)
        matched = [member for member in members if member["matched_at_iou_0_50"]]
        top1_correct = [member for member in members if member["top1_status"] == "TOP1_CORRECT"]
        ious = [float(member["best_iou"]) for member in matched]
        group_values.append(
            {
                "detection_success": len(matched) / population,
                "top1_correct": len(top1_correct) / population,
                "matched_iou": mean(ious) if ious else 0.0,
            }
        )
    return {
        "validation_group_count": len(group_values),
        "group_balanced_detection_success": mean(
            value["detection_success"] for value in group_values
        ),
        "group_balanced_miss_rate": mean(
            1.0 - value["detection_success"] for value in group_values
        ),
        "group_balanced_top1_correct_rate": mean(value["top1_correct"] for value in group_values),
        "group_balanced_matched_iou_mean": mean(value["matched_iou"] for value in group_values),
    }


def residual_error_records(
    records: list[dict[str, Any]],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Create objective residual labels without inferring visual conditions."""
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        classifications: list[str] = []
        if not record["matched_at_iou_0_50"]:
            classifications.append("MISS")
        if record["top1_status"] == "TOP1_WRONG_WINNER":
            classifications.append("WRONG_TOP1_WINNER")
        if int(record["background_false_positive_count"]) > 0:
            classifications.append("BACKGROUND_FALSE_POSITIVE")
        if int(record["duplicate_false_positive_count"]) > 0:
            classifications.append("DUPLICATE_FALSE_POSITIVE")
        if record["matched_at_iou_0_50"] and float(record["best_iou"]) < 0.75:
            classifications.append("LOCALIZATION_IOU_BELOW_0_75")
        if record["too_tight_for_crop"]:
            classifications.append("CROP_TOO_TIGHT")
        if record["too_loose_for_crop"]:
            classifications.append("CROP_TOO_LOOSE")
        for classification in classifications:
            counts[classification] += 1
        if classifications:
            errors.append(
                {
                    "sample_id": record["sample_id"],
                    "source": record["source"],
                    "annotation_origin": record["annotation_origin"],
                    "plate_size_slice": record["plate_size_slice"],
                    "classifications": classifications,
                    "best_iou": record["best_iou"],
                    "top1_iou": record["top1_iou"],
                }
            )
    return dict(sorted(counts.items())), errors
