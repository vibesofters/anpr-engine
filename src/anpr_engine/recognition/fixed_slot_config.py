"""Governed foundation configuration for the permanent M1 fixed-slot recognizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator

from anpr_engine.configuration import StrictConfigModel
from anpr_engine.recognition.augmentation import DetectorAwareAugmentationConfig
from anpr_engine.recognition.config import CropConfig
from anpr_engine.recognition.decoder import TrDecoderConfig
from anpr_engine.recognition.enhanced_config import CountryLayoutSeamConfig, TrProfileConfig

M1_MODEL_ID = "crnn_bilstm_dual_slot_tr_v1"


class FixedSlotRecognizerModelConfig(StrictConfigModel):
    """M1 encoder dimensions, kept independent from the governed M0 config schema."""

    architecture: Literal["crnn_bilstm_dual_slot_tr_v1"]
    input_channels: Literal[3] = 3
    input_height: Literal[32] = 32
    input_width: Literal[160] = 160
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    lstm_hidden_size: int = Field(default=128, ge=16, le=1024)
    lstm_layers: int = Field(default=2, ge=1, le=6)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    initialization_seed: int = 20260827

    @model_validator(mode="after")
    def channels_must_be_positive(self) -> FixedSlotRecognizerModelConfig:
        if any(channel <= 0 for channel in self.cnn_channels):
            raise ValueError("all CNN channel counts must be positive")
        return self


class OrderedSlotExtractionConfig(StrictConfigModel):
    mechanism: Literal["learned_ordered_query_dot_product"]
    max_slots: int = Field(ge=1, le=32)


class FixedSlotCharacterHeadConfig(StrictConfigModel):
    vocabulary: Literal["0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    pad_symbol: Literal["PAD"] = "PAD"
    pad_index: Literal[36] = 36


class FixedSlotTypeHeadConfig(StrictConfigModel):
    classes: tuple[
        Literal["DIGIT", "LETTER", "PAD"],
        Literal["DIGIT", "LETTER", "PAD"],
        Literal["DIGIT", "LETTER", "PAD"],
    ]
    symbols: Literal["DLP"] = "DLP"
    pad_index: Literal[2] = 2

    @model_validator(mode="after")
    def exact_classes(self) -> FixedSlotTypeHeadConfig:
        if self.classes != ("DIGIT", "LETTER", "PAD"):
            raise ValueError("fixed-slot Type classes must be exactly DIGIT, LETTER, PAD")
        return self


class FixedSlotLossConfig(StrictConfigModel):
    name: Literal["character_slot_ce_plus_weighted_type_slot_ce"]
    lambda_type: float = Field(default=0.2, ge=0.0, le=1.0)
    character_pad_loss: Literal["equal_per_slot"] = "equal_per_slot"
    type_pad_loss: Literal["equal_per_slot"] = "equal_per_slot"
    metric_pad_policy: Literal["exclude_pad_positions"] = "exclude_pad_positions"

    @model_validator(mode="after")
    def frozen_lambda(self) -> FixedSlotLossConfig:
        if self.lambda_type != 0.20:
            raise ValueError("the initial M1 foundation requires lambda_type=0.20")
        return self


class FixedSlotArtifactConfig(StrictConfigModel):
    private_weights_included: Literal[False] = False
    runtime_weight_location: Literal["externally-configured"] = "externally-configured"


class FixedSlotRecognitionFoundationConfig(StrictConfigModel):
    schema_version: Literal["1.0"] = "1.0"
    model_id: Literal["crnn_bilstm_dual_slot_tr_v1"]
    architecture_version: Literal["1.0"] = "1.0"
    checkpoint_architecture_version: Literal["1.0"] = "1.0"
    model: FixedSlotRecognizerModelConfig
    crop: CropConfig
    slot_extraction: OrderedSlotExtractionConfig
    character_head: FixedSlotCharacterHeadConfig
    type_head: FixedSlotTypeHeadConfig
    loss: FixedSlotLossConfig
    profile: TrProfileConfig
    decoder: TrDecoderConfig
    augmentation: DetectorAwareAugmentationConfig
    country_layout_head: CountryLayoutSeamConfig
    artifacts: FixedSlotArtifactConfig

    @model_validator(mode="after")
    def frozen_geometry_and_slots(self) -> FixedSlotRecognitionFoundationConfig:
        if (self.crop.target_height, self.crop.target_width) != (
            self.model.input_height,
            self.model.input_width,
        ):
            raise ValueError("M1 crop geometry must match model input geometry")
        if self.slot_extraction.max_slots != 8:
            raise ValueError("the initial TR V1 M1 foundation requires max_slots=8")
        if self.character_head.pad_index != len(self.character_head.vocabulary):
            raise ValueError("Character PAD index must immediately follow the vocabulary")
        return self


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("M1 Recognition foundation must be a YAML mapping")
    return cast(dict[str, Any], payload)


def load_fixed_slot_foundation_config(path: Path) -> FixedSlotRecognitionFoundationConfig:
    return FixedSlotRecognitionFoundationConfig.model_validate(_yaml_mapping(path))


def fixed_slot_config_sha256(config: FixedSlotRecognitionFoundationConfig) -> str:
    content = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
