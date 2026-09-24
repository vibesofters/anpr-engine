import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from anpr_engine.detection.config import (
    DetectorConfig,
    detector_config_sha256,
)
from anpr_engine.detection.data import (
    DetectionDataset,
    LetterboxTransform,
    encode_grid_target,
    load_samples,
)
from anpr_engine.detection.metrics import target_box_from_grid
from anpr_engine.detection.model import (
    TinyYoloDetector,
    YoloXMultiScaleDetector,
    decode_predictions,
)

SHA256 = "a" * 64


def test_letterbox_bbox_round_trip() -> None:
    transform = LetterboxTransform(
        original_width=640,
        original_height=360,
        input_size=320,
        scale=0.5,
        pad_x=0,
        pad_y=70,
    )
    original = (100.0, 120.0, 300.0, 180.0)

    restored = transform.to_original_xyxy(transform.to_input_xyxy(original))

    assert restored == pytest.approx(original)


def test_dataset_loads_released_manifest_and_encodes_one_grid_target(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    label_path = tmp_path / "label.txt"
    assert cv2.imwrite(image_path.as_posix(), np.zeros((100, 200, 3), dtype=np.uint8))
    label_path.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "detection-v1",
                "samples": [
                    {
                        "sample_id": "sample-1",
                        "split": "train",
                        "canonical_image_path": "image.jpg",
                        "canonical_label_path": "label.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    samples = load_samples(manifest_path, tmp_path, "train")
    dataset = DetectionDataset(samples, image_size=320, grid_stride=32, training=False)

    image, target, sample_id = dataset[0]

    assert image.shape == (3, 320, 320)
    assert target.shape == (5, 10, 10)
    assert target[0].sum().item() == 1.0
    assert sample_id == "sample-1"

    multiscale_dataset = DetectionDataset(samples, image_size=320, grid_stride=None, training=False)
    _, normalized_target, _ = multiscale_dataset[0]
    assert normalized_target.tolist() == pytest.approx([0.3, 0.45, 0.7, 0.55])


def test_grid_target_round_trip() -> None:
    bbox = torch.tensor([0.1, 0.2, 0.5, 0.6])

    restored = target_box_from_grid(encode_grid_target(bbox, 10))

    assert restored.tolist() == pytest.approx(bbox.tolist())


def test_model_output_shape_matches_grid_dimensions() -> None:
    model = TinyYoloDetector(base_channels=8)
    output = model(torch.zeros((2, 3, 320, 320)))
    output[:, 0] = -20.0
    output[0, 0, 4, 5] = 20.0

    decoded = decode_predictions(
        output,
        confidence_threshold=0.5,
        nms_iou_threshold=0.5,
        max_detections=10,
    )

    assert output.shape == (2, 5, 10, 10)
    assert decoded[0][0].shape == (1, 4)
    assert decoded[1][0].shape == (0, 4)


def test_detector_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        DetectorConfig.model_validate(
            {
                "dataset_content_sha256": SHA256,
                "model": {"unknown": True},
            }
        )


def test_multiscale_model_outputs_stride_8_16_32_features() -> None:
    model = YoloXMultiScaleDetector(base_channels=16, head_channels=64)

    features = model.forward_features(torch.zeros((1, 3, 256, 256)))
    outputs = model(torch.zeros((1, 3, 256, 256)))

    assert [feature.shape[-2:] for feature in features] == [(32, 32), (16, 16), (8, 8)]
    assert [output.shape for output in outputs] == [
        (1, 5, 32, 32),
        (1, 5, 16, 16),
        (1, 5, 8, 8),
    ]


def test_multiscale_decoder_combines_levels_in_normalized_coordinates() -> None:
    outputs = (
        torch.zeros((1, 5, 32, 32)),
        torch.zeros((1, 5, 16, 16)),
        torch.zeros((1, 5, 8, 8)),
    )
    for output in outputs:
        output[:, 0] = -20.0
    outputs[0][0, 0, 4, 5] = 20.0

    decoded = decode_predictions(
        outputs,
        confidence_threshold=0.5,
        nms_iou_threshold=0.5,
        max_detections=10,
    )

    assert decoded[0][0].shape == (1, 4)
    assert decoded[0][0][0].tolist() == pytest.approx([5 / 32, 4 / 32, 6 / 32, 5 / 32])


def test_multiscale_decoder_caps_candidates_before_nms(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = (
        torch.zeros((1, 5, 32, 32)),
        torch.zeros((1, 5, 16, 16)),
        torch.zeros((1, 5, 8, 8)),
    )
    observed_counts: list[int] = []

    def fake_nms(boxes: torch.Tensor, scores: torch.Tensor, _: float) -> torch.Tensor:
        observed_counts.append(len(scores))
        return torch.arange(len(boxes))

    monkeypatch.setattr("anpr_engine.detection.model.non_maximum_suppression", fake_nms)
    decode_predictions(
        outputs,
        confidence_threshold=0.001,
        nms_iou_threshold=0.5,
        max_detections=10,
    )

    assert observed_counts == [200]


def test_tiny_config_hash_remains_checkpoint_compatible() -> None:
    config = DetectorConfig.model_validate(
        {
            "dataset_content_sha256": "5" * 64,
            "model": {"architecture": "tiny_yolo_grid_v1"},
        }
    )

    assert set(config.model.model_dump()) == {
        "architecture",
        "image_size",
        "base_channels",
        "grid_stride",
    }
    assert detector_config_sha256(config) == (
        "a6e29ea6ef360d945cc86fa0b73cbc2e900d6b96e526d1b7238a0f363feb3dc4"
    )
