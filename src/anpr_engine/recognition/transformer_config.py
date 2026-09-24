"""Governed configuration for the permanent M2 Transformer/dual-CTC recognizer."""

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
from anpr_engine.recognition.enhanced_config import (
    CharacterHeadConfig,
    CountryLayoutSeamConfig,
    TrProfileConfig,
    TypeHeadConfig,
)
from anpr_engine.recognition.type_ctc import CtcLossExecutionPolicy

M2_MODEL_ID = "cnn_transformer_dual_ctc_tr_v1"


class CnnWidthTokenizerConfig(StrictConfigModel):
    mechanism: Literal["cnn_vertical_mean_width_major"]
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    feature_map_height: Literal[4] = 4
    feature_map_width: Literal[40] = 40
    vertical_aggregation: Literal["mean"] = "mean"
    projection_dimension: Literal[160] = 160

    @model_validator(mode="after")
    def frozen_geometry(self) -> CnnWidthTokenizerConfig:
        if self.cnn_channels != (32, 64, 128):
            raise ValueError("the initial M2 tokenizer retains M0 CNN channels 32/64/128")
        return self


class CompactTransformerConfig(StrictConfigModel):
    embedding_dimension: Literal[160] = 160
    layers: Literal[3] = 3
    attention_heads: Literal[4] = 4
    feed_forward_dimension: Literal[320] = 320
    activation: Literal["gelu"] = "gelu"
    normalization: Literal["pre_norm_layer_norm"] = "pre_norm_layer_norm"
    positional_encoding: Literal["fixed_sinusoidal_1d"] = "fixed_sinusoidal_1d"
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    token_count: Literal[40] = 40

    @model_validator(mode="after")
    def dimensions_are_compatible(self) -> CompactTransformerConfig:
        if self.embedding_dimension % self.attention_heads:
            raise ValueError("Transformer embedding dimension must divide evenly across heads")
        if self.feed_forward_dimension != 2 * self.embedding_dimension:
            raise ValueError("the initial M2 Transformer freezes a 2x feed-forward expansion")
        if self.dropout != 0.1:
            raise ValueError("the initial M2 Transformer freezes dropout=0.1")
        return self


class TransformerEncoderModelConfig(StrictConfigModel):
    """Shared, frozen M2 CNN-tokenizer/Transformer encoder contract."""

    input_channels: Literal[3] = 3
    input_height: Literal[32] = 32
    input_width: Literal[160] = 160
    tokenizer: CnnWidthTokenizerConfig
    transformer: CompactTransformerConfig
    initialization_seed: Literal[20260827] = 20260827


class TransformerRecognizerModelConfig(TransformerEncoderModelConfig):
    architecture: Literal["cnn_transformer_dual_ctc_tr_v1"]


class TransformerCtcLossConfig(StrictConfigModel):
    name: Literal["character_ctc_plus_weighted_type_ctc"]
    lambda_type: float = Field(default=0.2, ge=0.0, le=1.0)
    execution_policy: Literal[CtcLossExecutionPolicy.MPS_MODEL_CPU_CTC] = (
        CtcLossExecutionPolicy.MPS_MODEL_CPU_CTC
    )

    @model_validator(mode="after")
    def frozen_lambda(self) -> TransformerCtcLossConfig:
        if self.lambda_type != 0.2:
            raise ValueError("the initial M2 foundation requires lambda_type=0.20")
        return self


class TransformerArtifactConfig(StrictConfigModel):
    private_weights_included: Literal[False] = False
    runtime_weight_location: Literal["externally-configured"] = "externally-configured"


class TransformerRecognitionFoundationConfig(StrictConfigModel):
    schema_version: Literal["1.0"] = "1.0"
    model_id: Literal["cnn_transformer_dual_ctc_tr_v1"]
    architecture_version: Literal["1.0"] = "1.0"
    checkpoint_architecture_version: Literal["1.0"] = "1.0"
    output_family: Literal["CTC"] = "CTC"
    model: TransformerRecognizerModelConfig
    crop: CropConfig
    character_head: CharacterHeadConfig
    type_head: TypeHeadConfig
    loss: TransformerCtcLossConfig
    profile: TrProfileConfig
    decoder: TrDecoderConfig
    augmentation: DetectorAwareAugmentationConfig
    country_layout_head: CountryLayoutSeamConfig
    artifacts: TransformerArtifactConfig

    @model_validator(mode="after")
    def frozen_encoder_ablation(self) -> TransformerRecognitionFoundationConfig:
        if (self.crop.target_height, self.crop.target_width) != (
            self.model.input_height,
            self.model.input_width,
        ):
            raise ValueError("M2 crop geometry must match model input geometry")
        if self.model.transformer.token_count != self.model.tokenizer.feature_map_width:
            raise ValueError("M2 Transformer token count must match tokenizer width")
        if self.character_head.blank_index != 0 or self.type_head.blank_index != 0:
            raise ValueError("M2 preserves Character and Type CTC blank index zero")
        return self


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("M2 Recognition configuration must be a YAML mapping")
    return cast(dict[str, Any], payload)


def load_transformer_foundation_config(path: Path) -> TransformerRecognitionFoundationConfig:
    return TransformerRecognitionFoundationConfig.model_validate(_yaml_mapping(path))


def transformer_config_sha256(config: StrictConfigModel) -> str:
    content = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def transformer_architecture_metadata(
    config: TransformerRecognitionFoundationConfig,
) -> dict[str, object]:
    """Return M2-specific metadata required before strict checkpoint loading."""
    return {
        "tokenizer": config.model.tokenizer.model_dump(mode="json"),
        "transformer": config.model.transformer.model_dump(mode="json"),
        "character_head": config.character_head.model_dump(mode="json"),
        "type_head": config.type_head.model_dump(mode="json"),
        "loss": config.loss.model_dump(mode="json"),
    }
