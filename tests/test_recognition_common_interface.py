"""CPU-only contract tests for the Phase 0C common Recognition boundary."""

from __future__ import annotations

import json

import pytest
import torch

from anpr_engine.domain import ResultStatus
from anpr_engine.recognition.adapters import adapt_ctc_logits, adapt_fixed_slot_logits
from anpr_engine.recognition.charset import V1_CHARSET
from anpr_engine.recognition.decoder import (
    ProbabilityAwareTrDecoder,
    TrDecoderConfig,
    ctc_prefix_candidates,
)
from anpr_engine.recognition.evaluation import compute_common_recognition_evaluation_metrics
from anpr_engine.recognition.evidence import RecognitionModelCapability, RecognitionOutputFamily
from anpr_engine.recognition.profiles import TR_PROFILE
from anpr_engine.recognition.registry import (
    M0_CAPABILITY,
    CheckpointCompatibilityError,
    validate_checkpoint_capability,
)
from anpr_engine.recognition.type_ctc import TYPE_CTC_CHARSET, character_type_target


def _path_logits(text: str, *, symbols: str, blank_index: int, classes: int) -> torch.Tensor:
    indexes = {character: index for index, character in enumerate(symbols, start=1)}
    sequence: list[int] = []
    previous: str | None = None
    for character in text:
        if previous == character:
            sequence.append(blank_index)
        sequence.append(indexes[character])
        previous = character
    logits = torch.full((len(sequence), classes), -12.0, dtype=torch.float32)
    for index, value in enumerate(sequence):
        logits[index, value] = 12.0
    return logits


def _paired_ctc_logits(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Use a shared time axis with explicit blanks for Character and Type CTC."""
    character_indexes = {symbol: index for index, symbol in enumerate(V1_CHARSET.symbols, start=1)}
    type_indexes = {symbol: index for index, symbol in enumerate(TYPE_CTC_CHARSET.symbols, start=1)}
    steps = 2 * len(text) - 1
    character = torch.full((steps, 37), -12.0, dtype=torch.float32)
    types = torch.full((steps, 3), -12.0, dtype=torch.float32)
    for position in range(steps):
        if position % 2:
            character[position, 0] = 12.0
            types[position, 0] = 12.0
            continue
        character[position, character_indexes[text[position // 2]]] = 12.0
        type_symbol = character_type_target(text[position // 2])
        types[position, type_indexes[type_symbol]] = 12.0
    return character, types


def _slot_logits(text: str, *, max_slots: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    if len(text) > max_slots:
        raise ValueError("fixture text exceeds slots")
    character = torch.full((max_slots, 37), -12.0, dtype=torch.float32)
    character_indexes = {symbol: index for index, symbol in enumerate(V1_CHARSET.symbols)}
    types = torch.full((max_slots, 3), -12.0, dtype=torch.float32)
    for index in range(max_slots):
        if index < len(text):
            character[index, character_indexes[text[index]]] = 12.0
            types[index, 0 if text[index].isdigit() else 1] = 12.0
        else:
            character[index, 36] = 12.0
            types[index, 2] = 12.0
    return character, types


def _slot_capability(*, max_slots: int = 8) -> RecognitionModelCapability:
    return RecognitionModelCapability(
        model_id="fixture-fixed-slots",
        architecture_version="1.0",
        checkpoint_architecture_version="1.0",
        output_family=RecognitionOutputFamily.FIXED_SLOTS,
        input_shape=(3, 32, 160),
        character_symbols=V1_CHARSET.symbols,
        type_symbols=("DIGIT", "LETTER", "PAD"),
        profile_id="TR",
        max_slots=max_slots,
        character_pad_index=36,
        type_pad_index=2,
    )


def _decoder() -> ProbabilityAwareTrDecoder:
    return ProbabilityAwareTrDecoder(
        charset=V1_CHARSET,
        profile=TR_PROFILE,
        config=TrDecoderConfig(beam_width=32, character_top_k=3, minimum_decoder_confidence=0.0),
    )


def test_ctc_adapter_preserves_current_decoder_and_candidate_order() -> None:
    plate = "34AA111"
    character, types = _paired_ctc_logits(plate)
    decoder = _decoder()
    legacy = decoder.decode(character, types)
    common = adapt_ctc_logits(
        character,
        types,
        capability=M0_CAPABILITY,
        character_charset=V1_CHARSET,
        decoder_config=decoder.config,
    )
    through_common = decoder.decode_common(common)
    assert common.output_family is RecognitionOutputFamily.CTC
    assert common.raw_text == legacy.raw_text == plate
    assert through_common == legacy
    assert tuple(
        (item.text, item.character_log_probability) for item in common.candidates
    ) == ctc_prefix_candidates(
        character,
        V1_CHARSET,
        beam_width=decoder.config.beam_width,
        top_k=decoder.config.character_top_k,
    )
    assert json.loads(json.dumps(common.serializable_dict()))["raw_text"] == plate


@pytest.mark.parametrize("plate", ("34AA111", "34ABC123"))
def test_fixed_slot_adapter_handles_repeat_seven_and_eight_slot_targets(plate: str) -> None:
    character, types = _slot_logits(plate)
    common = adapt_fixed_slot_logits(
        character,
        types,
        capability=_slot_capability(),
        decoder_config=_decoder().config,
    )
    decoded = _decoder().decode_common(common)
    assert common.raw_sequence_valid
    assert common.raw_text == plate
    assert common.predicted_length == len(plate)
    assert common.selected_character_symbols[-1] == ("PAD" if len(plate) == 7 else "3")
    assert common.selected_type_symbols[-1] == ("PAD" if len(plate) == 7 else "D")
    assert decoded.decoded_text == plate
    assert decoded.status is ResultStatus.ACCEPTED
    assert all(item.source == "fixed_slot_beam" for item in common.candidates)


def test_fixed_slot_internal_pad_is_explicit_and_cannot_decode() -> None:
    character, types = _slot_logits("34ABC123")
    character[3].fill_(-12.0)
    character[3, 36] = 12.0
    common = adapt_fixed_slot_logits(
        character,
        types,
        capability=_slot_capability(),
        decoder_config=_decoder().config,
    )
    decoded = _decoder().decode_common(common)
    assert not common.raw_sequence_valid
    assert "INTERNAL_CHARACTER_PAD" in common.invalid_reasons
    assert decoded.decoded_text is None
    assert decoded.status is ResultStatus.LOW_CONFIDENCE


def test_common_metrics_preserve_plate_length_repeat_and_nonpad_type_semantics() -> None:
    character, types = _slot_logits("34AA111")
    slot = adapt_fixed_slot_logits(
        character,
        types,
        capability=_slot_capability(),
        decoder_config=_decoder().config,
    )
    ctc_character, ctc_type = _paired_ctc_logits("34ABC123")
    ctc = adapt_ctc_logits(
        ctc_character,
        ctc_type,
        capability=M0_CAPABILITY,
        character_charset=V1_CHARSET,
        decoder_config=_decoder().config,
    )
    metrics = compute_common_recognition_evaluation_metrics(
        targets=("34AA111", "34ABC123"),
        outputs=(slot, ctc),
        decoded_predictions=("34AA111", "34ABC123"),
        statuses=(ResultStatus.ACCEPTED, ResultStatus.ACCEPTED),
    )
    assert metrics.recognition.plate_length_accuracy == 1.0
    assert metrics.decoder.type_sequence_accuracy == 1.0
    assert metrics.repeated_character.population == 1
    assert metrics.repeated_digit.population == 1
    assert metrics.repeated_letter.population == 1
    assert (
        metrics.errors.deletions == metrics.errors.insertions == metrics.errors.substitutions == 0
    )


def test_checkpoint_capability_rejects_cross_family_slot_and_vocabulary_mismatch() -> None:
    validate_checkpoint_capability(M0_CAPABILITY, M0_CAPABILITY)
    wrong_family = M0_CAPABILITY.checkpoint_metadata() | {"output_family": "FIXED_SLOTS"}
    with pytest.raises(CheckpointCompatibilityError, match="output_family"):
        validate_checkpoint_capability(M0_CAPABILITY, wrong_family)
    wrong_slots = _slot_capability(max_slots=7)
    with pytest.raises(CheckpointCompatibilityError, match="max_slots"):
        validate_checkpoint_capability(_slot_capability(), wrong_slots)
    broken = M0_CAPABILITY.checkpoint_metadata() | {"character_symbols": "0123"}
    with pytest.raises(CheckpointCompatibilityError, match="character_symbols"):
        validate_checkpoint_capability(M0_CAPABILITY, broken)
