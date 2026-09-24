"""Governed M3 configuration: permanent M2 encoder plus fixed ordered slots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import model_validator

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
from anpr_engine.recognition.fixed_slot_local_config import OrderedGaussianSlotExtractionConfig
from anpr_engine.recognition.transformer_config import TransformerEncoderModelConfig

M3_MODEL_ID = "cnn_transformer_dual_slot_tr_v1"


class TransformerFixedSlotRecognizerModelConfig(TransformerEncoderModelConfig):
    """M3 model geometry with the M2 encoder contract inherited unchanged."""

    architecture: Literal["cnn_transformer_dual_slot_tr_v1"]


class TransformerFixedSlotArtifactConfig(StrictConfigModel):
    private_weights_included: Literal[False] = False
    runtime_weight_location: Literal["externally-configured"] = "externally-configured"


class TransformerFixedSlotFoundationConfig(StrictConfigModel):
    schema_version: Literal["1.0"] = "1.0"
    model_id: Literal["cnn_transformer_dual_slot_tr_v1"]
    architecture_version: Literal["1.0"] = "1.0"
    checkpoint_architecture_version: Literal["1.0"] = "1.0"
    output_family: Literal["FIXED_SLOTS"] = "FIXED_SLOTS"
    model: TransformerFixedSlotRecognizerModelConfig
    crop: CropConfig
    slot_extraction: OrderedGaussianSlotExtractionConfig
    character_head: FixedSlotCharacterHeadConfig
    type_head: FixedSlotTypeHeadConfig
    loss: FixedSlotLossConfig
    profile: TrProfileConfig
    decoder: TrDecoderConfig
    augmentation: DetectorAwareAugmentationConfig
    country_layout_head: CountryLayoutSeamConfig
    artifacts: TransformerFixedSlotArtifactConfig

    @model_validator(mode="after")
    def frozen_m2_encoder_and_m1_slots(self) -> TransformerFixedSlotFoundationConfig:
        if (self.crop.target_height, self.crop.target_width) != (
            self.model.input_height,
            self.model.input_width,
        ):
            raise ValueError("M3 crop geometry must match model input geometry")
        if self.model.transformer.token_count != self.model.tokenizer.feature_map_width:
            raise ValueError("M3 Transformer token count must match tokenizer width")
        if self.slot_extraction.max_slots != 8:
            raise ValueError("TR V1 M3 requires exactly eight supervised slots")
        if self.character_head.pad_index != len(self.character_head.vocabulary):
            raise ValueError("Character PAD index must immediately follow the vocabulary")
        return self


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("M3 Recognition configuration must be a YAML mapping")
    return cast(dict[str, Any], payload)


def load_transformer_fixed_slot_foundation_config(
    path: Path,
) -> TransformerFixedSlotFoundationConfig:
    return TransformerFixedSlotFoundationConfig.model_validate(_yaml_mapping(path))


def transformer_fixed_slot_config_sha256(config: StrictConfigModel) -> str:
    content = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def transformer_fixed_slot_architecture_metadata(
    config: TransformerFixedSlotFoundationConfig,
) -> dict[str, object]:
    """Return all M3 fields that define strict checkpoint compatibility."""
    return {
        "tokenizer": config.model.tokenizer.model_dump(mode="json"),
        "transformer": config.model.transformer.model_dump(mode="json"),
        "slot_extraction": config.slot_extraction.model_dump(mode="json"),
        "character_head": config.character_head.model_dump(mode="json"),
        "type_head": config.type_head.model_dump(mode="json"),
        "loss": config.loss.model_dump(mode="json"),
    }
