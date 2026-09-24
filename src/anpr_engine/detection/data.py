"""Released Detection V1 manifest loader and YOLO-grid training dataset."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from anpr_engine.detection.config import DetectorAugmentationConfig

NORMALIZED_BBOX_EPSILON = 1e-12


def validate_normalized_yolo_bbox(values: tuple[float, float, float, float]) -> None:
    """Validate normalized xywh under the shared serialization-noise policy."""
    x, y, width, height = values
    if not all(math.isfinite(value) for value in values) or width <= 0 or height <= 0:
        raise ValueError("annotation has invalid normalized geometry")
    left, right, top, bottom = x - width / 2, x + width / 2, y - height / 2, y + height / 2
    if (
        min(x, y, left, top) < -NORMALIZED_BBOX_EPSILON
        or max(x, y, right, bottom) > 1 + NORMALIZED_BBOX_EPSILON
    ):
        raise ValueError("annotation is outside image bounds")


@dataclass(frozen=True)
class DetectionManifestSample:
    sample_id: str
    split: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class LetterboxTransform:
    original_width: int
    original_height: int
    input_size: int
    scale: float
    pad_x: int
    pad_y: int

    def to_input_xyxy(self, bbox: tuple[float, float, float, float]) -> Tensor:
        x1, y1, x2, y2 = bbox
        return torch.tensor(
            [
                (x1 * self.scale + self.pad_x) / self.input_size,
                (y1 * self.scale + self.pad_y) / self.input_size,
                (x2 * self.scale + self.pad_x) / self.input_size,
                (y2 * self.scale + self.pad_y) / self.input_size,
            ],
            dtype=torch.float32,
        )

    def to_original_xyxy(self, bbox: Tensor) -> tuple[float, float, float, float]:
        values = bbox.detach().cpu().tolist()
        x1 = max(
            0.0, min(self.original_width, (values[0] * self.input_size - self.pad_x) / self.scale)
        )
        y1 = max(
            0.0, min(self.original_height, (values[1] * self.input_size - self.pad_y) / self.scale)
        )
        x2 = max(
            0.0, min(self.original_width, (values[2] * self.input_size - self.pad_x) / self.scale)
        )
        y2 = max(
            0.0, min(self.original_height, (values[3] * self.input_size - self.pad_y) / self.scale)
        )
        return x1, y1, x2, y2


def load_samples(
    manifest_path: Path, repository: Path, split: str
) -> tuple[DetectionManifestSample, ...]:
    payload = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    if payload.get("dataset_id") not in {"detection-v1", "detection-v2", "detection-v3"}:
        raise ValueError("manifest is not a supported Detection release")
    samples = tuple(
        DetectionManifestSample(
            sample_id=str(record.get("sample_id", record.get("candidate_id"))),
            split=str(record["split"]),
            image_path=repository / str(record["canonical_image_path"]),
            label_path=repository / str(record["canonical_label_path"]),
        )
        for record in payload["samples"]
        if record["split"] == split
    )
    if not samples:
        raise ValueError(f"manifest contains no {split} samples")
    return samples


def load_yolo_bbox(path: Path, width: int, height: int) -> tuple[float, float, float, float]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 5 or rows[0][0] != "0":
        raise ValueError(f"expected exactly one class-0 YOLO annotation: {path}")
    center_x, center_y, box_width, box_height = (float(value) for value in rows[0][1:])
    validate_normalized_yolo_bbox((center_x, center_y, box_width, box_height))
    x1 = (center_x - box_width / 2.0) * width
    y1 = (center_y - box_height / 2.0) * height
    x2 = (center_x + box_width / 2.0) * width
    y2 = (center_y + box_height / 2.0) * height
    epsilon_pixels = NORMALIZED_BBOX_EPSILON * max(width, height)
    if not (
        -epsilon_pixels <= x1 < x2 <= width + epsilon_pixels
        and -epsilon_pixels <= y1 < y2 <= height + epsilon_pixels
    ):
        raise ValueError(f"annotation is outside image bounds: {path}")
    return x1, y1, x2, y2


def letterbox(
    image: NDArray[np.uint8], input_size: int
) -> tuple[NDArray[np.uint8], LetterboxTransform]:
    height, width = image.shape[:2]
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, LetterboxTransform(width, height, input_size, scale, pad_x, pad_y)


class DetectionDataset(Dataset[tuple[Tensor, Tensor, str]]):
    def __init__(
        self,
        samples: tuple[DetectionManifestSample, ...],
        *,
        image_size: int,
        grid_stride: int | None,
        training: bool,
        augmentation: DetectorAugmentationConfig | None = None,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.grid_size = image_size // grid_stride if grid_stride is not None else None
        self.training = training
        self.augmentation = augmentation or DetectorAugmentationConfig()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        sample = self.samples[index]
        decoded = cv2.imread(sample.image_path.as_posix(), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"cannot decode image: {sample.image_path}")
        image = cast(NDArray[np.uint8], decoded)
        height, width = image.shape[:2]
        bbox = load_yolo_bbox(sample.label_path, width, height)
        if self.training:
            image, bbox = self._augment(image, bbox)
        resized, transform = letterbox(image, self.image_size)
        normalized_bbox = transform.to_input_xyxy(bbox)
        target = (
            encode_grid_target(normalized_bbox, self.grid_size)
            if self.grid_size is not None
            else normalized_bbox
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
        return tensor, target, sample.sample_id

    def _augment(
        self, image: NDArray[np.uint8], bbox: tuple[float, float, float, float]
    ) -> tuple[NDArray[np.uint8], tuple[float, float, float, float]]:
        augmented_float = image.astype(np.float32)
        if self.augmentation.brightness_limit:
            factor = 1.0 + random.uniform(
                -self.augmentation.brightness_limit, self.augmentation.brightness_limit
            )
            augmented_float *= factor
        if self.augmentation.contrast_limit:
            factor = 1.0 + random.uniform(
                -self.augmentation.contrast_limit, self.augmentation.contrast_limit
            )
            mean = augmented_float.mean(axis=(0, 1), keepdims=True)
            augmented_float = (augmented_float - mean) * factor + mean
        augmented = cast(NDArray[np.uint8], np.clip(augmented_float, 0, 255).astype(np.uint8))
        if random.random() < self.augmentation.horizontal_flip_probability:
            width = augmented.shape[1]
            x1, y1, x2, y2 = bbox
            augmented = np.ascontiguousarray(augmented[:, ::-1])
            bbox = (width - x2, y1, width - x1, y2)
        return augmented, bbox


def encode_grid_target(bbox: Tensor, grid_size: int) -> Tensor:
    target = torch.zeros((5, grid_size, grid_size), dtype=torch.float32)
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    cell_x = min(grid_size - 1, int(center_x * grid_size))
    cell_y = min(grid_size - 1, int(center_y * grid_size))
    target[0, cell_y, cell_x] = 1.0
    target[1, cell_y, cell_x] = center_x * grid_size - cell_x
    target[2, cell_y, cell_x] = center_y * grid_size - cell_y
    target[3, cell_y, cell_x] = box_width
    target[4, cell_y, cell_x] = box_height
    return target
