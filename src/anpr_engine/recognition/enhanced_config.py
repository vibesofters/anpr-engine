"""Strict public foundation for the project BiLSTM/CTC recognizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator

from anpr_engine.configuration import StrictConfigModel
from anpr_engine.recognition.augmentation import DetectorAwareAugmentationConfig
from anpr_engine.recognition.config import CropConfig, RecognizerModelConfig
from anpr_engine.recognition.decoder import TrDecoderConfig


class CharacterHeadConfig(StrictConfigModel):
    vocabulary: Literal["0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    blank_index: Literal[0] = 0


class TypeHeadConfig(StrictConfigModel):
    classes: tuple[Literal["DIGIT", "LETTER"], Literal["DIGIT", "LETTER"]]
    symbols: Literal["DL"] = "DL"
    blank_index: Literal[0] = 0

    @model_validator(mode="after")
    def exact_classes(self) -> TypeHeadConfig:
        if self.classes != ("DIGIT", "LETTER"):
            raise ValueError("type classes must be exactly DIGIT, LETTER")
        return self


class TrProfileConfig(StrictConfigModel):
    profile_id: Literal["TR"] = "TR"
    country_code: Literal["TR"] = "TR"
    supported_formats: tuple[str, ...]
    province_rule: Literal["existing-authoritative-01-through-81"]


class CountryLayoutSeamConfig(StrictConfigModel):
    enabled: Literal[False] = False
    source: Literal["shared_global_embedding"] = "shared_global_embedding"
    reason: Literal["one-class-TR-head-would-be-meaningless"]


class EnhancedRecognitionFoundationConfig(StrictConfigModel):
    schema_version: Literal["1.0"] = "1.0"
    model_id: Literal["crnn_bilstm_dual_ctc_tr_v1"]
    model: RecognizerModelConfig
    crop: CropConfig
    character_head: CharacterHeadConfig
    type_head: TypeHeadConfig
    lambda_type: float = Field(ge=0.0, le=1.0)
    profile: TrProfileConfig
    decoder: TrDecoderConfig
    augmentation: DetectorAwareAugmentationConfig
    country_layout_head: CountryLayoutSeamConfig


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("enhanced Recognition configuration must be a YAML mapping")
    return cast(dict[str, Any], payload)


def load_enhanced_foundation_config(path: Path) -> EnhancedRecognitionFoundationConfig:
    return EnhancedRecognitionFoundationConfig.model_validate(_yaml_mapping(path))


def enhanced_config_sha256(config: EnhancedRecognitionFoundationConfig) -> str:
    content = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
