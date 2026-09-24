"""Strict configuration for the project-owned Detection V1 baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator

from anpr_engine.configuration import StrictConfigModel


class TinyDetectorModelConfig(StrictConfigModel):
    architecture: Literal["tiny_yolo_grid_v1"] = "tiny_yolo_grid_v1"
    image_size: int = Field(default=320, ge=128, le=1280, multiple_of=32)
    base_channels: int = Field(default=16, ge=8, le=64)
    grid_stride: Literal[32] = 32


class MultiScaleDetectorModelConfig(StrictConfigModel):
    architecture: Literal["yolox_multiscale_v1"] = "yolox_multiscale_v1"
    image_size: int = Field(default=384, ge=256, le=1280, multiple_of=32)
    base_channels: int = Field(default=24, ge=16, le=64)
    head_channels: int = Field(default=96, ge=64, le=256)
    feature_strides: tuple[Literal[8], Literal[16], Literal[32]] = (8, 16, 32)
    focal_alpha: float = Field(default=0.75, gt=0.0, lt=1.0)
    focal_gamma: float = Field(default=2.0, ge=0.0, le=5.0)
    box_loss_weight: float = Field(default=5.0, gt=0.0)
    assignment_center_radius: float = Field(default=2.5, gt=0.0)
    assignment_top_k: int = Field(default=10, ge=1, le=50)
    minimum_epochs: int = Field(default=8, ge=1)
    early_stopping_patience: int = Field(default=4, ge=1)


DetectorModelConfig = TinyDetectorModelConfig | MultiScaleDetectorModelConfig


class DetectorAugmentationConfig(StrictConfigModel):
    horizontal_flip_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    brightness_limit: float = Field(default=0.15, ge=0.0, le=0.5)
    contrast_limit: float = Field(default=0.15, ge=0.0, le=0.5)


class DetectorTrainingConfig(StrictConfigModel):
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=8, ge=1)
    optimizer: Literal["adamw", "sgd"] = "adamw"
    learning_rate: float = Field(default=0.001, gt=0.0)
    weight_decay: float = Field(default=0.0001, ge=0.0)
    num_workers: int = Field(default=0, ge=0)
    seed: int = 20260823
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    objectness_positive_weight: float = Field(default=20.0, gt=0.0)
    box_loss_weight: float = Field(default=5.0, gt=0.0)
    augmentation: DetectorAugmentationConfig = Field(default_factory=DetectorAugmentationConfig)


class DetectorInferenceConfig(StrictConfigModel):
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections: int = Field(default=10, ge=1, le=100)
    crop_padding_x: float = Field(default=0.0, ge=0.0)
    crop_padding_y: float = Field(default=0.0, ge=0.0)


class DetectorConfig(StrictConfigModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["detection-v1", "detection-v2", "detection-v3"] = "detection-v1"
    dataset_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: DetectorModelConfig = Field(default_factory=TinyDetectorModelConfig)
    training: DetectorTrainingConfig = Field(default_factory=DetectorTrainingConfig)
    inference: DetectorInferenceConfig = Field(default_factory=DetectorInferenceConfig)

    @model_validator(mode="after")
    def image_size_must_match_stride(self) -> DetectorConfig:
        strides = (
            (self.model.grid_stride,)
            if isinstance(self.model, TinyDetectorModelConfig)
            else self.model.feature_strides
        )
        if any(self.model.image_size % stride for stride in strides):
            raise ValueError("image_size must be divisible by every detector stride")
        if (
            isinstance(self.model, MultiScaleDetectorModelConfig)
            and self.model.minimum_epochs > self.training.epochs
        ):
            raise ValueError("minimum_epochs cannot exceed configured training epochs")
        return self


def load_detector_config(path: Path) -> DetectorConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("detector configuration must be a YAML mapping")
    return DetectorConfig.model_validate(cast(dict[str, Any], payload))


def detector_config_sha256(config: DetectorConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()
