import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.optim import AdamW

from anpr_engine.detection.config import DetectorConfig, detector_config_sha256
from anpr_engine.detection.example_inference import discover_images, review_flags
from anpr_engine.detection.inference import detect_images
from anpr_engine.detection.metrics import SamplePredictions, detection_metrics
from anpr_engine.detection.model import TinyYoloDetector, YoloXMultiScaleDetector
from anpr_engine.detection.post_training_analysis import (
    atomic_group_balanced_metrics,
    top1_analysis,
)
from anpr_engine.detection.training import (
    _device_memory,
    detector_loss,
    evaluate_checkpoint,
    evaluation_config,
    load_checkpoint,
    multiscale_detector_loss,
    save_checkpoint,
    verify_checkpoint_identity,
)
from anpr_engine.detection.validation_analysis import (
    localization_analysis,
    select_operating_points,
)
from anpr_engine.detection.validation_comparison import (
    compare_samples,
    sample_outcome,
    select_review_cases,
)

SHA256 = "a" * 64


def _config() -> DetectorConfig:
    return DetectorConfig.model_validate(
        {
            "dataset_content_sha256": SHA256,
            "model": {"image_size": 128, "base_channels": 8},
            "training": {"batch_size": 1, "epochs": 1},
        }
    )


def _multiscale_config() -> DetectorConfig:
    return DetectorConfig.model_validate(
        {
            "dataset_content_sha256": SHA256,
            "model": {
                "architecture": "yolox_multiscale_v1",
                "image_size": 256,
                "base_channels": 16,
                "head_channels": 64,
                "minimum_epochs": 1,
            },
            "training": {"batch_size": 1, "epochs": 1, "device": "cpu"},
            "inference": {"confidence_threshold": 0.99},
        }
    )


def test_detector_loss_backpropagates() -> None:
    model = TinyYoloDetector(base_channels=8)
    output = model(torch.zeros((1, 3, 128, 128)))
    target = torch.zeros_like(output)
    target[0, 0, 2, 2] = 1.0
    target[0, 1:, 2, 2] = torch.tensor([0.5, 0.5, 0.25, 0.1])

    total, objectness, box = detector_loss(
        output,
        target,
        objectness_positive_weight=20.0,
        box_loss_weight=5.0,
    )
    total.backward()  # type: ignore[no-untyped-call]

    assert total.isfinite()
    assert objectness.item() > 0
    assert box.item() >= 0


def test_multiscale_loss_assigns_positives_and_backpropagates() -> None:
    model = YoloXMultiScaleDetector(base_channels=16, head_channels=64)
    output = model(torch.zeros((2, 3, 256, 256)))
    targets = torch.tensor([[0.20, 0.35, 0.55, 0.52], [0.65, 0.60, 0.78, 0.67]])

    total, objectness, box, positive_count = multiscale_detector_loss(
        output,
        targets,
        focal_alpha=0.75,
        focal_gamma=2.0,
        box_loss_weight=5.0,
        center_radius=2.5,
        assignment_top_k=10,
    )
    total.backward()  # type: ignore[no-untyped-call]

    assert total.isfinite()
    assert objectness.item() > 0
    assert box.item() >= 0
    assert positive_count >= len(targets)


def test_detection_metrics_report_known_match_and_false_positive() -> None:
    predictions = (
        SamplePredictions(
            sample_id="one",
            target=torch.tensor([0.1, 0.2, 0.5, 0.6]),
            boxes=torch.tensor([[0.1, 0.2, 0.5, 0.6], [0.6, 0.6, 0.9, 0.9]]),
            scores=torch.tensor([0.9, 0.8]),
        ),
        SamplePredictions(
            sample_id="two",
            target=torch.tensor([0.2, 0.2, 0.4, 0.4]),
            boxes=torch.empty((0, 4)),
            scores=torch.empty(0),
        ),
    )

    metrics = detection_metrics(predictions, confidence_threshold=0.25)

    assert metrics["true_positive_count"] == 1
    assert metrics["false_positive_count"] == 1
    assert metrics["detection_miss_count"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["detections_per_image"] == 1.0
    assert metrics["zero_detection_image_count"] == 1
    assert metrics["multiple_detection_image_count"] == 1


def test_detection_metrics_separate_duplicate_and_background_false_positives() -> None:
    predictions = (
        SamplePredictions(
            sample_id="matched",
            target=torch.tensor([0.1, 0.1, 0.5, 0.5]),
            boxes=torch.tensor(
                [
                    [0.1, 0.1, 0.5, 0.5],
                    [0.12, 0.12, 0.48, 0.48],
                    [0.6, 0.6, 0.9, 0.9],
                ]
            ),
            scores=torch.tensor([0.9, 0.8, 0.7]),
        ),
        SamplePredictions(
            sample_id="missed",
            target=torch.tensor([0.1, 0.1, 0.3, 0.3]),
            boxes=torch.tensor([[0.7, 0.7, 0.9, 0.9]]),
            scores=torch.tensor([0.6]),
        ),
    )

    metrics = detection_metrics(predictions, confidence_threshold=0.5)

    assert metrics["false_positive_count"] == 3
    assert metrics["false_positive_image_count"] == 2
    assert metrics["duplicate_false_positive_count"] == 1
    assert metrics["background_false_positive_count"] == 2
    assert metrics["false_positive_confidence"]["count"] == 3
    assert metrics["false_positive_confidence"]["median"] == pytest.approx(0.7)


def test_top1_contract_and_atomic_group_balancing() -> None:
    predictions = (
        SamplePredictions(
            sample_id="correct",
            target=torch.tensor([0.1, 0.1, 0.5, 0.5]),
            boxes=torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
            scores=torch.tensor([0.8]),
        ),
        SamplePredictions(
            sample_id="wrong",
            target=torch.tensor([0.1, 0.1, 0.5, 0.5]),
            boxes=torch.tensor([[0.6, 0.6, 0.9, 0.9], [0.1, 0.1, 0.5, 0.5]]),
            scores=torch.tensor([0.9, 0.8]),
        ),
        SamplePredictions(
            sample_id="miss",
            target=torch.tensor([0.1, 0.1, 0.5, 0.5]),
            boxes=torch.empty((0, 4)),
            scores=torch.empty(0),
        ),
    )
    summary, records = top1_analysis(predictions, 0.5)

    assert summary["top1_correct_count"] == 1
    assert summary["top1_wrong_winner_count"] == 1
    assert summary["top1_miss_count"] == 1
    assert records["wrong"]["top1_status"] == "TOP1_WRONG_WINNER"

    balanced = atomic_group_balanced_metrics(
        [
            {
                "atomic_group_id": "a",
                "matched_at_iou_0_50": True,
                "top1_status": "TOP1_CORRECT",
                "best_iou": 0.9,
            },
            {
                "atomic_group_id": "a",
                "matched_at_iou_0_50": False,
                "top1_status": "TOP1_MISS",
                "best_iou": None,
            },
            {
                "atomic_group_id": "b",
                "matched_at_iou_0_50": True,
                "top1_status": "TOP1_CORRECT",
                "best_iou": 0.7,
            },
        ]
    )
    assert balanced["validation_group_count"] == 2
    assert balanced["group_balanced_detection_success"] == pytest.approx(0.75)


def test_example_inference_inventory_and_objective_review_flags(tmp_path: Path) -> None:
    image_path = tmp_path / "nested" / "plate.jpg"
    image_path.parent.mkdir()
    assert cv2.imwrite(image_path.as_posix(), np.zeros((12, 24, 3), dtype=np.uint8))
    (tmp_path / "notes.md").write_text("not an image", encoding="utf-8")

    inventory, unsupported = discover_images(tmp_path)

    assert inventory[0]["source_relative_path"] == "nested/plate.jpg"
    assert inventory[0]["decode_status"] == "VALID"
    assert inventory[0]["width"] == 24
    assert unsupported == ["notes.md"]
    assert review_flags(
        detection_count=0,
        scores=[],
        crop_width=None,
        crop_height=None,
        touches_boundary=False,
    ) == ["NO_DETECTION"]
    assert review_flags(
        detection_count=2,
        scores=[0.20, 0.70],
        crop_width=12,
        crop_height=4,
        touches_boundary=True,
    ) == [
        "LOW_TOP1_CONFIDENCE",
        "MULTIPLE_SURVIVING_DETECTIONS",
        "MULTIPLE_STRONG_CANDIDATES",
        "UNUSUALLY_SMALL_SELECTED_CROP",
        "TOP1_BBOX_TOUCHES_IMAGE_BOUNDARY",
    ]


def test_validation_analysis_selects_declared_points_and_localization() -> None:
    grid = [
        {
            "name": "recall",
            "precision": 0.2,
            "recall": 0.9,
            "f1": 0.32,
            "confidence_threshold": 0.1,
            "nms_iou_threshold": 0.5,
        },
        {
            "name": "balanced",
            "precision": 0.7,
            "recall": 0.7,
            "f1": 0.7,
            "confidence_threshold": 0.2,
            "nms_iou_threshold": 0.5,
        },
        {
            "name": "balanced-higher-threshold",
            "precision": 0.7,
            "recall": 0.7,
            "f1": 0.7,
            "confidence_threshold": 0.3,
            "nms_iou_threshold": 0.5,
        },
        {
            "name": "precise",
            "precision": 0.9,
            "recall": 0.5,
            "f1": 0.64,
            "confidence_threshold": 0.4,
            "nms_iou_threshold": 0.5,
        },
    ]
    selected = select_operating_points(grid)
    assert selected["maximum_f1"]["name"] == "balanced"
    assert selected["high_recall"]["name"] == "recall"
    assert selected["recommended_balanced"]["name"] == "balanced"

    predictions = (
        SamplePredictions(
            sample_id="one",
            target=torch.tensor([0.1, 0.1, 0.3, 0.2]),
            boxes=torch.tensor([[0.1, 0.1, 0.3, 0.2]]),
            scores=torch.tensor([0.9]),
        ),
        SamplePredictions(
            sample_id="two",
            target=torch.tensor([0.1, 0.1, 0.5, 0.3]),
            boxes=torch.empty((0, 4)),
            scores=torch.empty(0),
        ),
        SamplePredictions(
            sample_id="three",
            target=torch.tensor([0.1, 0.1, 0.7, 0.5]),
            boxes=torch.tensor([[0.1, 0.1, 0.7, 0.5]]),
            scores=torch.tensor([0.8]),
        ),
    )
    localization, samples = localization_analysis(predictions, 0.5)
    assert localization["matched_count"] == 2
    assert localization["iou_ge_0_75_rate"] == pytest.approx(2 / 3)
    assert localization["iou_ge_0_90_rate"] == pytest.approx(2 / 3)
    assert localization["normalized_center_offset"]["median"] == pytest.approx(0.0)
    assert localization["plate_size_slices"]["small"]["match_rate"] == 1.0
    assert localization["plate_size_slices"]["medium"]["miss_count"] == 1
    assert len(samples) == 3


def test_validation_comparison_uses_declared_order_and_cases() -> None:
    def sample(
        sample_id: str,
        *,
        size: str,
        matched: bool,
        false_positives: int,
        iou: float | None,
        area: float = 0.1,
    ) -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "plate_size_slice": size,
            "matched_at_iou_0_50": matched,
            "false_positive_count": false_positives,
            "detection_count": false_positives + int(matched),
            "best_iou": iou,
            "plate_area_ratio": area,
            "duplicate_false_positive_count": false_positives,
            "background_false_positive_count": 0,
            "best_normalized_center_offset": 0.1 if iou is not None else None,
        }

    baseline = [
        sample("ordinary", size="medium", matched=True, false_positives=0, iou=0.7),
        sample("small", size="small", matched=False, false_positives=0, iou=None, area=0.01),
        sample("fp-heavy", size="large", matched=True, false_positives=4, iou=0.8),
        sample("difficult", size="large", matched=False, false_positives=0, iou=None),
        sample("regression", size="large", matched=True, false_positives=0, iou=0.9),
    ]
    candidate = [
        sample("ordinary", size="medium", matched=True, false_positives=0, iou=0.72),
        sample("small", size="small", matched=False, false_positives=0, iou=None, area=0.01),
        sample("fp-heavy", size="large", matched=True, false_positives=1, iou=0.7),
        sample("difficult", size="large", matched=False, false_positives=0, iou=None),
        sample("regression", size="large", matched=False, false_positives=0, iou=None),
    ]
    summary, records = compare_samples(baseline, candidate)
    assert summary["outcome_counts"] == {
        "candidate_win": 1,
        "tie": 3,
        "candidate_regression": 1,
    }
    assert sample_outcome(baseline[0], candidate[0]) == "tie"
    selected = select_review_cases(records)
    assert selected["ordinary"]["sample_id"] == "ordinary"  # type: ignore[index]
    assert selected["small_plate"]["sample_id"] == "small"  # type: ignore[index]
    assert selected["baseline_1_fp_heavy"]["sample_id"] == "fp-heavy"  # type: ignore[index]
    assert selected["localization_failure"]["sample_id"] == "ordinary"  # type: ignore[index]
    assert selected["difficult_or_miss"]["sample_id"] in {"small", "difficult"}  # type: ignore[index]
    assert selected["regression"]["sample_id"] == "regression"  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        compare_samples([*baseline, baseline[0]], candidate)
    mismatched_slice = [*candidate]
    mismatched_slice[0] = {**mismatched_slice[0], "plate_size_slice": "large"}
    with pytest.raises(ValueError, match="size slice differs"):
        compare_samples(baseline, mismatched_slice)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    config = _config()
    model = TinyYoloDetector(base_channels=8)
    optimizer = AdamW(model.parameters())
    checkpoint = tmp_path / "detector.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        epoch=1,
        config=config,
        run_id="fixture-run",
        validation_metrics={
            "map_50": 0.0,
            "confidence": {"minimum": None, "median": None, "maximum": None},
            "matched_iou_mean": None,
        },
    )

    restored, metadata = load_checkpoint(checkpoint, torch.device("cpu"))

    assert isinstance(restored, TinyYoloDetector)
    assert metadata["run_id"] == "fixture-run"
    assert metadata["dataset_content_sha256"] == SHA256
    assert metadata["config_sha256"] == detector_config_sha256(config)
    verify_checkpoint_identity(metadata, config)

    changed_config = config.model_copy(
        update={"training": config.training.model_copy(update={"learning_rate": 0.01})}
    )
    with pytest.raises(ValueError, match="configuration"):
        verify_checkpoint_identity(metadata, changed_config)


def test_multiscale_checkpoint_and_cpu_inference_contract(tmp_path: Path) -> None:
    config = _multiscale_config()
    model = YoloXMultiScaleDetector(base_channels=16, head_channels=64)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    optimizer = AdamW(model.parameters())
    checkpoint = tmp_path / "multiscale.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        epoch=1,
        config=config,
        run_id="multiscale-fixture",
        validation_metrics={"map_50": 0.0},
    )

    restored, metadata = load_checkpoint(checkpoint, torch.device("cpu"))
    assert isinstance(restored, YoloXMultiScaleDetector)
    assert metadata["architecture"] == "yolox_multiscale_v1"
    verify_checkpoint_identity(metadata, config)

    image_path = tmp_path / "image.jpg"
    assert cv2.imwrite(image_path.as_posix(), np.zeros((100, 200, 3), dtype=np.uint8))
    output_path = detect_images(
        image_path,
        tmp_path / "output",
        checkpoint,
        config,
        confidence_threshold=0.7,
        nms_iou_threshold=0.4,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["operating_point"] == {
        "confidence_threshold": 0.7,
        "nms_iou_threshold": 0.4,
    }
    assert result["results"][0]["detections"] == []
    debug_path = tmp_path / "output" / result["results"][0]["debug"]
    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    assert [shape[-2:] for shape in debug["raw_grid_shapes"]] == [[32, 32], [16, 16], [8, 8]]


def test_locked_test_evaluation_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="locked-test access confirmation"):
        evaluate_checkpoint(
            tmp_path,
            tmp_path / "manifest.json",
            _config(),
            tmp_path / "detector.pt",
            "test",
            tmp_path / "metrics.json",
        )


def test_evaluation_config_applies_validated_inference_overrides() -> None:
    config = _config()

    resolved = evaluation_config(
        config,
        confidence_threshold=0.7,
        nms_iou_threshold=0.4,
    )

    assert resolved.inference.confidence_threshold == 0.7
    assert resolved.inference.nms_iou_threshold == 0.4
    assert config.inference.confidence_threshold == 0.25
    assert config.inference.nms_iou_threshold == 0.5


def test_evaluation_config_rejects_invalid_overrides() -> None:
    with pytest.raises(ValueError):
        evaluation_config(_config(), confidence_threshold=1.1)


def test_cpu_device_memory_is_not_reported() -> None:
    assert _device_memory(torch.device("cpu")) is None
