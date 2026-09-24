"""Project-owned Recognition capability registry and checkpoint compatibility checks."""

from __future__ import annotations

from collections.abc import Mapping

from anpr_engine.recognition.charset import V1_CHARSET
from anpr_engine.recognition.enhanced_model import ENHANCED_MODEL_ID
from anpr_engine.recognition.evidence import RecognitionModelCapability, RecognitionOutputFamily
from anpr_engine.recognition.fixed_slot import (
    FIXED_SLOT_CHARACTER_PAD_INDEX,
    FIXED_SLOT_TYPE_PAD_INDEX,
    FIXED_SLOT_TYPE_SYMBOLS,
)
from anpr_engine.recognition.fixed_slot_config import M1_MODEL_ID
from anpr_engine.recognition.fixed_slot_local_config import M1_1_MODEL_ID
from anpr_engine.recognition.transformer_config import M2_MODEL_ID
from anpr_engine.recognition.transformer_slot_config import M3_MODEL_ID
from anpr_engine.recognition.type_ctc import TYPE_CTC_CHARSET

M0_CAPABILITY = RecognitionModelCapability(
    model_id=ENHANCED_MODEL_ID,
    architecture_version="1.0",
    checkpoint_architecture_version="1.0",
    output_family=RecognitionOutputFamily.CTC,
    input_shape=(3, 32, 160),
    character_symbols=V1_CHARSET.symbols,
    type_symbols=("DIGIT", "LETTER"),
    profile_id="TR",
    character_blank_index=V1_CHARSET.blank_index,
    type_blank_index=TYPE_CTC_CHARSET.blank_index,
)

M1_CAPABILITY = RecognitionModelCapability(
    model_id=M1_MODEL_ID,
    architecture_version="1.0",
    checkpoint_architecture_version="1.0",
    output_family=RecognitionOutputFamily.FIXED_SLOTS,
    input_shape=(3, 32, 160),
    character_symbols=V1_CHARSET.symbols,
    type_symbols=FIXED_SLOT_TYPE_SYMBOLS,
    profile_id="TR",
    max_slots=8,
    character_pad_index=FIXED_SLOT_CHARACTER_PAD_INDEX,
    type_pad_index=FIXED_SLOT_TYPE_PAD_INDEX,
)

M1_1_CAPABILITY = RecognitionModelCapability(
    model_id=M1_1_MODEL_ID,
    architecture_version="1.1",
    checkpoint_architecture_version="1.1",
    output_family=RecognitionOutputFamily.FIXED_SLOTS,
    input_shape=(3, 32, 160),
    character_symbols=V1_CHARSET.symbols,
    type_symbols=FIXED_SLOT_TYPE_SYMBOLS,
    profile_id="TR",
    max_slots=8,
    character_pad_index=FIXED_SLOT_CHARACTER_PAD_INDEX,
    type_pad_index=FIXED_SLOT_TYPE_PAD_INDEX,
)

M2_CAPABILITY = RecognitionModelCapability(
    model_id=M2_MODEL_ID,
    architecture_version="1.0",
    checkpoint_architecture_version="1.0",
    output_family=RecognitionOutputFamily.CTC,
    input_shape=(3, 32, 160),
    character_symbols=V1_CHARSET.symbols,
    type_symbols=("DIGIT", "LETTER"),
    profile_id="TR",
    character_blank_index=V1_CHARSET.blank_index,
    type_blank_index=TYPE_CTC_CHARSET.blank_index,
)

M3_CAPABILITY = RecognitionModelCapability(
    model_id=M3_MODEL_ID,
    architecture_version="1.0",
    checkpoint_architecture_version="1.0",
    output_family=RecognitionOutputFamily.FIXED_SLOTS,
    input_shape=(3, 32, 160),
    character_symbols=V1_CHARSET.symbols,
    type_symbols=FIXED_SLOT_TYPE_SYMBOLS,
    profile_id="TR",
    max_slots=8,
    character_pad_index=FIXED_SLOT_CHARACTER_PAD_INDEX,
    type_pad_index=FIXED_SLOT_TYPE_PAD_INDEX,
)

_IMPLEMENTED_CAPABILITIES = {
    capability.model_id: capability
    for capability in (
        M0_CAPABILITY,
        M1_CAPABILITY,
        M1_1_CAPABILITY,
        M2_CAPABILITY,
        M3_CAPABILITY,
    )
}


class CheckpointCompatibilityError(ValueError):
    """Raised before an incompatible checkpoint can be treated as a model match."""


def implemented_capability(model_id: str) -> RecognitionModelCapability:
    try:
        return _IMPLEMENTED_CAPABILITIES[model_id]
    except KeyError as error:
        raise CheckpointCompatibilityError(
            f"Recognition model is not implemented: {model_id}"
        ) from error


def implemented_capabilities() -> tuple[RecognitionModelCapability, ...]:
    """Return all permanent implemented families without a mutable current-model alias."""
    return tuple(_IMPLEMENTED_CAPABILITIES.values())


def validate_checkpoint_capability(
    expected: RecognitionModelCapability,
    actual: RecognitionModelCapability | Mapping[str, object],
) -> None:
    """Fail closed on family, vocabulary, slot, profile, or input incompatibility."""
    observed = (
        actual.checkpoint_metadata() if isinstance(actual, RecognitionModelCapability) else actual
    )
    expected_metadata = expected.checkpoint_metadata()
    for field in (
        "model_id",
        "architecture_version",
        "checkpoint_architecture_version",
        "output_family",
        "input_shape",
        "character_symbols",
        "type_symbols",
        "profile_id",
        "type_head_available",
        "character_blank_index",
        "type_blank_index",
        "max_slots",
        "character_pad_index",
        "type_pad_index",
    ):
        expected_value = expected_metadata[field]
        actual_value = observed.get(field)
        if actual_value != expected_value:
            raise CheckpointCompatibilityError(
                f"checkpoint capability mismatch for {field}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
