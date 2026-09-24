"""Single-class detection metrics with explicit IoU and confidence semantics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Any

import torch
from torch import Tensor

from anpr_engine.detection.model import DetectorOutput, box_iou, decode_predictions


@dataclass(frozen=True)
class SamplePredictions:
    sample_id: str
    target: Tensor
    boxes: Tensor
    scores: Tensor


def classify_sample_predictions(
    prediction: SamplePredictions,
    confidence_threshold: float,
    match_iou_threshold: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor, int | None]:
    selected = prediction.scores >= confidence_threshold
    boxes = prediction.boxes[selected]
    scores = prediction.scores[selected]
    ious = box_iou(prediction.target, boxes) if len(boxes) else torch.empty(0)
    matched_index: int | None = None
    if len(ious):
        best_index = int(ious.argmax())
        if float(ious[best_index]) >= match_iou_threshold:
            matched_index = best_index
    return boxes, scores, ious, matched_index


def _confidence_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "maximum": None,
        }
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "median": median(ordered),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "maximum": ordered[-1],
    }


def target_box_from_grid(target: Tensor) -> Tensor:
    if target.shape == (4,):
        return target.clamp(0.0, 1.0)
    positions = torch.nonzero(target[0] == 1.0, as_tuple=False)
    if positions.shape != (1, 2):
        raise ValueError("target must contain exactly one positive grid cell")
    cell_y, cell_x = positions[0]
    grid_height, grid_width = target.shape[1:]
    center_x = (cell_x + target[1, cell_y, cell_x]) / grid_width
    center_y = (cell_y + target[2, cell_y, cell_x]) / grid_height
    width = target[3, cell_y, cell_x]
    height = target[4, cell_y, cell_x]
    return torch.stack(
        (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
    ).clamp(0.0, 1.0)


def collect_predictions(
    output: DetectorOutput,
    targets: Tensor,
    sample_ids: tuple[str, ...],
    *,
    nms_iou_threshold: float,
    max_detections: int,
) -> tuple[SamplePredictions, ...]:
    decoded = decode_predictions(
        output,
        confidence_threshold=0.001,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )
    return tuple(
        SamplePredictions(
            sample_id=sample_id,
            target=target_box_from_grid(targets[index]).cpu(),
            boxes=boxes.cpu(),
            scores=scores.cpu(),
        )
        for index, (sample_id, (boxes, scores)) in enumerate(zip(sample_ids, decoded, strict=True))
    )


def _average_precision(predictions: tuple[SamplePredictions, ...], iou_threshold: float) -> float:
    ranked = sorted(
        (
            (float(score), prediction.sample_id, box, prediction.target)
            for prediction in predictions
            for box, score in zip(prediction.boxes, prediction.scores, strict=True)
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    matched: set[str] = set()
    true_positives: list[float] = []
    false_positives: list[float] = []
    for _, sample_id, box, target in ranked:
        iou = float(box_iou(box, target.unsqueeze(0))[0])
        is_match = iou >= iou_threshold and sample_id not in matched
        true_positives.append(float(is_match))
        false_positives.append(float(not is_match))
        if is_match:
            matched.add(sample_id)
    if not ranked:
        return 0.0
    cumulative_tp = torch.tensor(true_positives).cumsum(0)
    cumulative_fp = torch.tensor(false_positives).cumsum(0)
    recall = cumulative_tp / max(1, len(predictions))
    precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp(min=1e-8)
    recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([1.0]), precision, torch.tensor([0.0])))
    for index in range(precision.numel() - 2, -1, -1):
        precision[index] = torch.maximum(precision[index], precision[index + 1])
    changes = torch.nonzero(recall[1:] != recall[:-1], as_tuple=False).flatten()
    return float(torch.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def detection_metrics(
    predictions: tuple[SamplePredictions, ...],
    *,
    confidence_threshold: float,
    match_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    thresholds = tuple(round(0.5 + index * 0.05, 2) for index in range(10))
    average_precisions = {
        f"{threshold:.2f}": _average_precision(predictions, threshold) for threshold in thresholds
    }
    true_positives = 0
    false_positives = 0
    misses = 0
    confidences: list[float] = []
    matched_ious: list[float] = []
    detection_count = 0
    zero_detection_images = 0
    multiple_detection_images = 0
    false_positive_images = 0
    duplicate_false_positives = 0
    background_false_positives = 0
    false_positive_confidences: list[float] = []
    for prediction in predictions:
        boxes, scores, ious, matched_index = classify_sample_predictions(
            prediction, confidence_threshold, match_iou_threshold
        )
        detection_count += len(scores)
        zero_detection_images += int(len(scores) == 0)
        multiple_detection_images += int(len(scores) > 1)
        confidences.extend(float(score) for score in scores)
        if not scores.numel():
            misses += 1
            continue
        if matched_index is not None:
            true_positives += 1
            matched_ious.append(float(ious[matched_index]))
        else:
            misses += 1
        sample_false_positives = 0
        for index, score in enumerate(scores):
            if index == matched_index:
                continue
            sample_false_positives += 1
            false_positive_confidences.append(float(score))
            if matched_index is not None and float(ious[index]) >= match_iou_threshold:
                duplicate_false_positives += 1
            else:
                background_false_positives += 1
        false_positives += sample_false_positives
        false_positive_images += int(sample_false_positives > 0)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, len(predictions))
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    return {
        "population": len(predictions),
        "confidence_threshold": confidence_threshold,
        "match_iou_threshold": match_iou_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map_50": average_precisions["0.50"],
        "map_50_95": mean(average_precisions.values()),
        "average_precision_by_iou": average_precisions,
        "true_positive_count": true_positives,
        "false_positive_count": false_positives,
        "detection_miss_count": misses,
        "detections_per_image": detection_count / max(1, len(predictions)),
        "zero_detection_image_count": zero_detection_images,
        "multiple_detection_image_count": multiple_detection_images,
        "false_positive_image_count": false_positive_images,
        "duplicate_false_positive_count": duplicate_false_positives,
        "background_false_positive_count": background_false_positives,
        "duplicate_false_positive_rate": duplicate_false_positives / max(1, false_positives),
        "false_positive_confidence": _confidence_summary(false_positive_confidences),
        "confidence": {
            "count": len(confidences),
            "minimum": min(confidences, default=None),
            "median": median(confidences) if confidences else None,
            "maximum": max(confidences, default=None),
        },
        "matched_iou_mean": mean(matched_ious) if matched_ious else None,
    }
