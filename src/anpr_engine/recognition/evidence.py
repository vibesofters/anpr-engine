"""Immutable common evidence for Recognition output families.

This module deliberately preserves native output semantics.  CTC time steps and
fixed output slots are both ordered evidence, but neither is converted into the
other.  The shared decoder and evaluator consume this boundary instead of model-
specific wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import torch
from torch import Tensor


class RecognitionOutputFamily(StrEnum):
    """Implemented Recognition output representations."""

    CTC = "CTC"
    FIXED_SLOTS = "FIXED_SLOTS"


@dataclass(frozen=True)
class RecognitionModelCapability:
    """Checkpoint-relevant capabilities declared by a Recognition model."""

    model_id: str
    architecture_version: str
    output_family: RecognitionOutputFamily
    input_shape: tuple[int, int, int]
    character_symbols: str
    type_symbols: tuple[str, ...]
    profile_id: str
    checkpoint_architecture_version: str
    type_head_available: bool = True
    character_blank_index: int | None = None
    type_blank_index: int | None = None
    max_slots: int | None = None
    character_pad_index: int | None = None
    type_pad_index: int | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.architecture_version or not self.profile_id:
            raise ValueError("model capability identity fields must be non-empty")
        if self.input_shape != (3, 32, 160):
            raise ValueError("the frozen Recognition interface requires input shape (3, 32, 160)")
        if self.output_family is RecognitionOutputFamily.CTC:
            if self.character_blank_index is None or self.type_blank_index is None:
                raise ValueError("CTC capability requires Character and Type blank indexes")
            if any(
                value is not None
                for value in (self.max_slots, self.character_pad_index, self.type_pad_index)
            ):
                raise ValueError("CTC capability cannot declare fixed-slot PAD metadata")
        if self.output_family is RecognitionOutputFamily.FIXED_SLOTS:
            if self.max_slots is None or self.max_slots < 1:
                raise ValueError("fixed-slot capability requires max_slots")
            if self.character_pad_index is None or self.type_pad_index is None:
                raise ValueError("fixed-slot capability requires PAD indexes")
            if self.character_blank_index is not None or self.type_blank_index is not None:
                raise ValueError("fixed-slot capability cannot declare CTC blank metadata")

    @property
    def character_class_count(self) -> int:
        return len(self.character_symbols) + 1

    @property
    def type_class_count(self) -> int:
        return len(self.type_symbols) + (
            1 if self.output_family is RecognitionOutputFamily.CTC else 0
        )

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "architecture_version": self.architecture_version,
            "checkpoint_architecture_version": self.checkpoint_architecture_version,
            "output_family": self.output_family.value,
            "input_shape": list(self.input_shape),
            "character_symbols": self.character_symbols,
            "type_symbols": list(self.type_symbols),
            "profile_id": self.profile_id,
            "type_head_available": self.type_head_available,
            "character_blank_index": self.character_blank_index,
            "type_blank_index": self.type_blank_index,
            "max_slots": self.max_slots,
            "character_pad_index": self.character_pad_index,
            "type_pad_index": self.type_pad_index,
        }


@dataclass(frozen=True)
class RecognitionCandidateEvidence:
    """A family-native candidate before the shared TR profile is applied."""

    text: str
    character_log_probability: float
    native_position_count: int
    source: Literal["ctc_prefix_beam", "fixed_slot_beam"]


@dataclass(frozen=True)
class CtcDecoderEvidence:
    """Unbatched CTC evidence preserving the current decoder input tensors."""

    character_logits: Tensor
    type_logits: Tensor
    character_blank_index: int
    type_blank_index: int

    def __post_init__(self) -> None:
        if self.character_logits.ndim != 2 or self.type_logits.ndim != 2:
            raise ValueError("CTC decoder evidence requires unbatched [time, classes] logits")
        if self.character_logits.shape[0] != self.type_logits.shape[0]:
            raise ValueError("CTC Character and Type evidence must share the time axis")


@dataclass(frozen=True)
class FixedSlotDecoderEvidence:
    """Unbatched ordered slot evidence; PAD is a suffix-only semantic."""

    character_logits: Tensor
    type_logits: Tensor
    max_slots: int
    character_pad_index: int
    type_pad_index: int

    def __post_init__(self) -> None:
        if self.character_logits.ndim != 2 or self.type_logits.ndim != 2:
            raise ValueError("fixed-slot decoder evidence requires [slots, classes] logits")
        if self.character_logits.shape[0] != self.max_slots:
            raise ValueError("Character slot evidence must match configured max_slots")
        if self.type_logits.shape[0] != self.max_slots:
            raise ValueError("Type slot evidence must match configured max_slots")


DecoderEvidence = CtcDecoderEvidence | FixedSlotDecoderEvidence


@dataclass(frozen=True)
class CommonRecognitionOutput:
    """Lossless-enough common view of one unbatched Recognition prediction."""

    schema_version: str
    capability: RecognitionModelCapability
    raw_text: str
    normalized_raw_text: str
    predicted_length: int
    character_probabilities: Tensor
    type_probabilities: Tensor
    selected_character_symbols: tuple[str, ...]
    selected_type_symbols: tuple[str, ...]
    position_confidences: tuple[float, ...]
    type_position_confidences: tuple[float, ...]
    sequence_confidence: float
    confidence_semantics: str
    candidates: tuple[RecognitionCandidateEvidence, ...]
    decoder_evidence: DecoderEvidence
    raw_sequence_valid: bool
    invalid_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "common-recognition-output-v1":
            raise ValueError("unsupported common Recognition output schema")
        if self.predicted_length < 0:
            raise ValueError("predicted plate length must be non-negative")
        if self.character_probabilities.ndim != 2 or self.type_probabilities.ndim != 2:
            raise ValueError("common probability evidence must be rank two")
        if len(self.position_confidences) != self.character_probabilities.shape[0]:
            raise ValueError("Character confidence count must match native positions")
        if len(self.type_position_confidences) != self.type_probabilities.shape[0]:
            raise ValueError("Type confidence count must match native positions")
        if self.raw_sequence_valid and self.invalid_reasons:
            raise ValueError("valid RAW evidence cannot carry invalid reasons")
        if not self.raw_sequence_valid and not self.invalid_reasons:
            raise ValueError("invalid RAW evidence requires a reason")

    @property
    def output_family(self) -> RecognitionOutputFamily:
        return self.capability.output_family

    def serializable_dict(self) -> dict[str, object]:
        """Provide deterministic JSON-ready metadata without checkpoint tensors."""
        return {
            "schema_version": self.schema_version,
            "capability": self.capability.checkpoint_metadata(),
            "raw_text": self.raw_text,
            "normalized_raw_text": self.normalized_raw_text,
            "predicted_length": self.predicted_length,
            "character_probabilities": self.character_probabilities.tolist(),
            "type_probabilities": self.type_probabilities.tolist(),
            "selected_character_symbols": list(self.selected_character_symbols),
            "selected_type_symbols": list(self.selected_type_symbols),
            "position_confidences": list(self.position_confidences),
            "type_position_confidences": list(self.type_position_confidences),
            "sequence_confidence": self.sequence_confidence,
            "confidence_semantics": self.confidence_semantics,
            "candidates": [
                {
                    "text": item.text,
                    "character_log_probability": item.character_log_probability,
                    "native_position_count": item.native_position_count,
                    "source": item.source,
                }
                for item in self.candidates
            ],
            "raw_sequence_valid": self.raw_sequence_valid,
            "invalid_reasons": list(self.invalid_reasons),
        }


def frozen_cpu_float32(tensor: Tensor) -> Tensor:
    """Detach decoder/evaluator evidence without changing its float32 semantics."""
    if not tensor.is_floating_point():
        raise ValueError("common Recognition evidence requires floating-point tensors")
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()
