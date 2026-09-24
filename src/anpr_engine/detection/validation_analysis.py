"""Deterministic validation-only detector calibration and localization analysis."""

from __future__ import annotations

from statistics import median
from typing import Any

from anpr_engine.detection.metrics import (
    SamplePredictions,
    classify_sample_predictions,
    detection_metrics,
)


def select_operating_points(grid: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not grid:
        raise ValueError("calibration grid cannot be empty")
    maximum_f1 = max(
        grid,
        key=lambda point: (
            float(point["f1"]),
            float(point["precision"]),
            float(point["recall"]),
            -float(point["confidence_threshold"]),
            -float(point["nms_iou_threshold"]),
        ),
    )
    maximum_recall = max(float(point["recall"]) for point in grid)
    high_recall_candidates = [
        point for point in grid if float(point["recall"]) >= maximum_recall - 0.02
    ]
    high_recall = max(
        high_recall_candidates,
        key=lambda point: (
            float(point["precision"]),
            float(point["f1"]),
            -float(point["confidence_threshold"]),
            -float(point["nms_iou_threshold"]),
        ),
    )
    balanced_candidates = [point for point in grid if float(point["recall"]) >= 0.60]
    balanced = max(
        balanced_candidates or grid,
        key=lambda point: (
            float(point["f1"]),
            float(point["precision"]),
            -float(point["confidence_threshold"]),
            -float(point["nms_iou_threshold"]),
        ),
    )
    return {
        "maximum_f1": maximum_f1,
        "high_recall": high_recall,
        "recommended_balanced": balanced,
    }


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "median": median(ordered),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
    }


def localization_analysis(
    predictions: tuple[SamplePredictions, ...],
    confidence_threshold: float,
    image_size: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered_by_area = sorted(
        predictions,
        key=lambda prediction: (
            float(
                (prediction.target[2] - prediction.target[0])
                * (prediction.target[3] - prediction.target[1])
            ),
            prediction.sample_id,
        ),
    )
    slice_by_id = {
        prediction.sample_id: (
            "small"
            if index < len(predictions) / 3
            else "medium"
            if index < 2 * len(predictions) / 3
            else "large"
        )
        for index, prediction in enumerate(ordered_by_area)
    }
    matched_ious: list[float] = []
    area_ratios: list[float] = []
    center_offsets: list[float] = []
    width_errors: list[float] = []
    height_errors: list[float] = []
    sample_records: list[dict[str, Any]] = []
    for prediction in predictions:
        boxes, scores, ious, matched_index = classify_sample_predictions(
            prediction, confidence_threshold
        )
        target_width = float(prediction.target[2] - prediction.target[0])
        target_height = float(prediction.target[3] - prediction.target[1])
        target_area = float(
            (prediction.target[2] - prediction.target[0])
            * (prediction.target[3] - prediction.target[1])
        )
        best_iou: float | None = None
        best_area_ratio: float | None = None
        best_center_offset: float | None = None
        best_width_error: float | None = None
        best_height_error: float | None = None
        too_tight: bool | None = None
        too_loose: bool | None = None
        if len(boxes):
            best_index = int(ious.argmax())
            best_iou = float(ious[best_index])
            best_box = boxes[best_index]
            predicted_area = float((best_box[2] - best_box[0]) * (best_box[3] - best_box[1]))
            best_area_ratio = predicted_area / max(target_area, 1e-8)
            predicted_width = float(best_box[2] - best_box[0])
            predicted_height = float(best_box[3] - best_box[1])
            best_width_error = (predicted_width - target_width) / max(target_width, 1e-8)
            best_height_error = (predicted_height - target_height) / max(target_height, 1e-8)
            target_center = (prediction.target[:2] + prediction.target[2:]) / 2.0
            predicted_center = (best_box[:2] + best_box[2:]) / 2.0
            normalized_delta = (predicted_center - target_center) / prediction.target.new_tensor(
                (target_width, target_height)
            ).clamp(min=1e-8)
            best_center_offset = float(normalized_delta.square().sum().sqrt())
            too_tight = best_width_error < -0.10 or best_height_error < -0.10
            too_loose = best_width_error > 0.25 or best_height_error > 0.25
        matched = matched_index is not None
        if matched and best_iou is not None and best_area_ratio is not None:
            matched_ious.append(best_iou)
            area_ratios.append(best_area_ratio)
            if best_center_offset is not None:
                center_offsets.append(best_center_offset)
            if best_width_error is not None:
                width_errors.append(best_width_error)
            if best_height_error is not None:
                height_errors.append(best_height_error)
        false_positive_indices = [index for index in range(len(scores)) if index != matched_index]
        duplicate_indices = [
            index
            for index in false_positive_indices
            if matched_index is not None and float(ious[index]) >= 0.50
        ]
        background_indices = [
            index for index in false_positive_indices if index not in duplicate_indices
        ]
        sample_records.append(
            {
                "sample_id": prediction.sample_id,
                "plate_area_ratio": target_area,
                "plate_size_slice": slice_by_id[prediction.sample_id],
                "target": prediction.target.tolist(),
                "boxes": boxes.tolist(),
                "scores": scores.tolist(),
                "detection_count": len(scores),
                "false_positive_count": len(false_positive_indices),
                "false_positive_scores": [float(scores[index]) for index in false_positive_indices],
                "duplicate_false_positive_count": len(duplicate_indices),
                "duplicate_false_positive_scores": [
                    float(scores[index]) for index in duplicate_indices
                ],
                "background_false_positive_count": len(background_indices),
                "background_false_positive_scores": [
                    float(scores[index]) for index in background_indices
                ],
                "best_iou": best_iou,
                "matched_at_iou_0_50": matched,
                "best_predicted_to_target_area_ratio": best_area_ratio,
                "best_normalized_center_offset": best_center_offset,
                "best_relative_width_error": best_width_error,
                "best_relative_height_error": best_height_error,
                "too_tight_for_crop": too_tight,
                "too_loose_for_crop": too_loose,
                "model_plate_width_px": target_width * image_size if image_size else None,
                "model_plate_height_px": target_height * image_size if image_size else None,
            }
        )
    slices: dict[str, dict[str, Any]] = {}
    for name in ("small", "medium", "large"):
        records = [record for record in sample_records if record["plate_size_slice"] == name]
        matched_records = [record for record in records if record["matched_at_iou_0_50"]]
        slice_ious = [float(record["best_iou"]) for record in matched_records]
        slices[name] = {
            "population": len(records),
            "matched_count": len(matched_records),
            "match_rate": len(matched_records) / max(1, len(records)),
            "recall": len(matched_records) / max(1, len(records)),
            "miss_count": len(records) - len(matched_records),
            "matched_iou": _quantiles(slice_ious),
        }
    localization = {
        "population": len(predictions),
        "matched_count": len(matched_ious),
        "matched_iou": _quantiles(matched_ious),
        "iou_ge_0_50_rate": len(matched_ious) / max(1, len(predictions)),
        "iou_ge_0_75_count": sum(value >= 0.75 for value in matched_ious),
        "iou_ge_0_75_rate": sum(value >= 0.75 for value in matched_ious) / max(1, len(predictions)),
        "iou_ge_0_90_count": sum(value >= 0.90 for value in matched_ious),
        "iou_ge_0_90_rate": sum(value >= 0.90 for value in matched_ious) / max(1, len(predictions)),
        "oversized_box_count": sum(value > 1.25 for value in area_ratios),
        "oversized_box_rate_among_matches": sum(value > 1.25 for value in area_ratios)
        / max(1, len(area_ratios)),
        "undersized_box_count": sum(value < 0.80 for value in area_ratios),
        "undersized_box_rate_among_matches": sum(value < 0.80 for value in area_ratios)
        / max(1, len(area_ratios)),
        "predicted_to_target_area_ratio": _quantiles(area_ratios),
        "normalized_center_offset": _quantiles(center_offsets),
        "relative_width_error": _quantiles(width_errors),
        "relative_height_error": _quantiles(height_errors),
        "too_tight_crop_count": sum(
            record["too_tight_for_crop"] is True for record in sample_records
        ),
        "too_loose_crop_count": sum(
            record["too_loose_for_crop"] is True for record in sample_records
        ),
        "plate_area_ratio": _quantiles(
            [float(record["plate_area_ratio"]) for record in sample_records]
        ),
        "model_plate_width_px": _quantiles(
            [
                float(record["model_plate_width_px"])
                for record in sample_records
                if record["model_plate_width_px"] is not None
            ]
        ),
        "model_plate_height_px": _quantiles(
            [
                float(record["model_plate_height_px"])
                for record in sample_records
                if record["model_plate_height_px"] is not None
            ]
        ),
        "plate_size_slices": slices,
    }
    return localization, sample_records


def calibration_grid(
    predictions_by_nms: dict[float, tuple[SamplePredictions, ...]],
    confidence_thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for nms_threshold, predictions in predictions_by_nms.items():
        for confidence_threshold in confidence_thresholds:
            metrics = detection_metrics(
                predictions,
                confidence_threshold=confidence_threshold,
            )
            grid.append(
                {
                    "confidence_threshold": confidence_threshold,
                    "nms_iou_threshold": nms_threshold,
                    **metrics,
                }
            )
    return grid
