"""Governed configuration for the permanent M1.1 ordered-locality recognizer."""

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
from anpr_engine.recognition.fixed_slot_config import (
    FixedSlotCharacterHeadConfig,
    FixedSlotLossConfig,
    FixedSlotTypeHeadConfig,
)

M1_1_MODEL_ID = "crnn_bilstm_dual_slot_local_tr_v1"


class FixedSlotLocalRecognizerModelConfig(StrictConfigModel):
    """M1.1 encoder dimensions, intentionally matched to M1 v1."""

    architecture: Literal["crnn_bilstm_dual_slot_local_tr_v1"]
    input_channels: Literal[3] = 3
    input_height: Literal[32] = 32
    input_width: Literal[160] = 160
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    lstm_hidden_size: int = Field(default=128, ge=16, le=1024)
    lstm_layers: int = Field(default=2, ge=1, le=6)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    initialization_seed: int = 20260827

    @model_validator(mode="after")
    def channels_must_be_positive(self) -> FixedSlotLocalRecognizerModelConfig:
        if any(channel <= 0 for channel in self.cnn_channels):
            raise ValueError("all CNN channel counts must be positive")
        return self


class OrderedGaussianSlotExtractionConfig(StrictConfigModel):
    mechanism: Literal["learned_ordered_query_dot_product_plus_fixed_gaussian_locality"]
    max_slots: int = Field(ge=1, le=32)
    anchor_center_formula: Literal["(slot+0.5)*sequence_length/max_slots-0.5"]
    sigma_formula: Literal["sequence_length/max_slots"]
    locality_trainable_parameters: Literal[0] = 0


class FixedSlotLocalArtifactConfig(StrictConfigModel):
    private_weights_included: Literal[False] = False
    runtime_weight_location: Literal["externally-configured"] = "externally-configured"


class FixedSlotLocalRecognitionFoundationConfig(StrictConfigModel):
    schema_version: Literal["1.0"] = "1.0"
    model_id: Literal["crnn_bilstm_dual_slot_local_tr_v1"]
    architecture_version: Literal["1.1"] = "1.1"
    checkpoint_architecture_version: Literal["1.1"] = "1.1"
    model: FixedSlotLocalRecognizerModelConfig
    crop: CropConfig
    slot_extraction: OrderedGaussianSlotExtractionConfig
    character_head: FixedSlotCharacterHeadConfig
    type_head: FixedSlotTypeHeadConfig
    loss: FixedSlotLossConfig
    profile: TrProfileConfig
    decoder: TrDecoderConfig
    augmentation: DetectorAwareAugmentationConfig
    country_layout_head: CountryLayoutSeamConfig
    artifacts: FixedSlotLocalArtifactConfig

    @model_validator(mode="after")
    def frozen_geometry_and_slots(self) -> FixedSlotLocalRecognitionFoundationConfig:
        if (self.crop.target_height, self.crop.target_width) != (
            self.model.input_height,
            self.model.input_width,
        ):
            raise ValueError("M1.1 crop geometry must match model input geometry")
        if self.slot_extraction.max_slots != 8:
            raise ValueError("the initial TR V1 M1.1 foundation requires max_slots=8")
        if self.character_head.pad_index != len(self.character_head.vocabulary):
            raise ValueError("Character PAD index must immediately follow the vocabulary")
        return self


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("M1.1 Recognition configuration must be a YAML mapping")
    return cast(dict[str, Any], payload)


def load_fixed_slot_local_foundation_config(
    path: Path,
) -> FixedSlotLocalRecognitionFoundationConfig:
    return FixedSlotLocalRecognitionFoundationConfig.model_validate(_yaml_mapping(path))


def fixed_slot_local_config_sha256(config: StrictConfigModel) -> str:
    content = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
