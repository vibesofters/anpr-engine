"""Typed Recognition V1 foundation configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator

from anpr_engine.configuration import StrictConfigModel


class RecognizerModelConfig(StrictConfigModel):
    architecture: Literal["crnn_bilstm_ctc_v1", "crnn_bilstm_dual_ctc_tr_v1"] = "crnn_bilstm_ctc_v1"
    input_channels: Literal[1, 3] = 3
    input_height: int = Field(default=32, ge=16, le=128, multiple_of=4)
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    lstm_hidden_size: int = Field(default=128, ge=16, le=1024)
    lstm_layers: int = Field(default=2, ge=1, le=6)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    initialization_seed: int = 20260824

    @model_validator(mode="after")
    def channels_must_be_positive(self) -> RecognizerModelConfig:
        if any(channel <= 0 for channel in self.cnn_channels):
            raise ValueError("all CNN channel counts must be positive")
        return self


class CropConfig(StrictConfigModel):
    padding_x: float = Field(default=2.0, ge=0.0, allow_inf_nan=False)
    padding_y: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    minimum_width: int = Field(default=8, ge=1)
    minimum_height: int = Field(default=4, ge=1)
    target_height: int = Field(default=32, ge=16, le=128)
    target_width: int = Field(default=160, ge=32, le=1024)
    resize_strategy: Literal["aspect_fit"] = "aspect_fit"
    padding_strategy: Literal["center", "right"] = "center"
    interpolation: Literal["linear", "area"] = "area"
    pad_value: int = Field(default=0, ge=0, le=255)
    input_color_space: Literal["BGR", "RGB"] = "BGR"
    output_color_space: Literal["RGB"] = "RGB"
    debug_crop_enabled: bool = False


class DecisionConfig(StrictConfigModel):
    # None is deliberately fail-closed: without calibration a structurally valid
    # plate remains LOW_CONFIDENCE rather than being accepted.
    acceptance_recognition_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RecognitionFoundationConfig(StrictConfigModel):
    schema_version: Literal["1.0"] = "1.0"
    charset_version: Literal["turkish-plate-ctc-v1"] = "turkish-plate-ctc-v1"
    model: RecognizerModelConfig = Field(default_factory=RecognizerModelConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)

    @model_validator(mode="after")
    def crop_height_must_match_model(self) -> RecognitionFoundationConfig:
        if self.crop.target_height != self.model.input_height:
            raise ValueError("crop target_height must match recognizer input_height")
        return self


def load_recognition_config(path: Path) -> RecognitionFoundationConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recognition configuration must be a YAML mapping")
    return RecognitionFoundationConfig.model_validate(cast(dict[str, Any], payload))


def recognition_config_sha256(config: RecognitionFoundationConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
