"""Adapters from native Recognition tensors to common output evidence."""

from __future__ import annotations

import math

from torch import Tensor

from anpr_engine.recognition.charset import PlateCharset
from anpr_engine.recognition.ctc import greedy_decode_logits
from anpr_engine.recognition.decoder import (
    TrDecoderConfig,
    ctc_prefix_candidates,
    fixed_slot_candidates,
)
from anpr_engine.recognition.evidence import (
    CommonRecognitionOutput,
    CtcDecoderEvidence,
    FixedSlotDecoderEvidence,
    RecognitionCandidateEvidence,
    RecognitionModelCapability,
    RecognitionOutputFamily,
    frozen_cpu_float32,
)
from anpr_engine.recognition.type_ctc import TYPE_CTC_CHARSET

COMMON_RECOGNITION_OUTPUT_SCHEMA = "common-recognition-output-v1"
FIXED_SLOT_CHARACTER_SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FIXED_SLOT_TYPE_SYMBOLS = ("DIGIT", "LETTER", "PAD")
FIXED_SLOT_TYPE_TEXT = ("D", "L", "PAD")


def _probabilities(logits: Tensor) -> Tensor:
    return frozen_cpu_float32(logits).softmax(dim=1)


def _confidence(probabilities: Tensor) -> tuple[tuple[float, ...], float]:
    values = tuple(float(value) for value in probabilities.max(dim=1).values.tolist())
    if not values:
        raise ValueError("native Recognition evidence must have at least one position")
    return values, float(
        math.exp(sum(math.log(max(value, 1e-30)) for value in values) / len(values))
    )


def adapt_ctc_logits(
    character_logits: Tensor,
    type_logits: Tensor,
    *,
    capability: RecognitionModelCapability,
    character_charset: PlateCharset,
    decoder_config: TrDecoderConfig,
) -> CommonRecognitionOutput:
    """Adapt one current M0/M2 sample without changing CTC decoding semantics."""
    if capability.output_family is not RecognitionOutputFamily.CTC:
        raise ValueError("CTC logits require a CTC model capability")
    if character_logits.ndim != 2 or type_logits.ndim != 2:
        raise ValueError("CTC adapter expects unbatched [time, classes] logits")
    if character_logits.shape[0] != type_logits.shape[0]:
        raise ValueError("CTC Character and Type logits must share the time axis")
    if character_logits.shape[1] != character_charset.class_count:
        raise ValueError("CTC Character class count does not match charset")
    if type_logits.shape[1] != TYPE_CTC_CHARSET.class_count:
        raise ValueError("CTC Type class count does not match type charset")
    if capability.character_blank_index != character_charset.blank_index:
        raise ValueError("CTC capability Character blank index does not match charset")
    if capability.type_blank_index != TYPE_CTC_CHARSET.blank_index:
        raise ValueError("CTC capability Type blank index does not match type charset")

    frozen_character_logits = frozen_cpu_float32(character_logits)
    frozen_type_logits = frozen_cpu_float32(type_logits)
    character_probabilities = frozen_character_logits.softmax(dim=1)
    type_probabilities = frozen_type_logits.softmax(dim=1)
    raw_text = greedy_decode_logits(frozen_character_logits.unsqueeze(1), character_charset)[0]
    type_text = greedy_decode_logits(frozen_type_logits.unsqueeze(1), TYPE_CTC_CHARSET)[0]
    character_confidences, sequence_confidence = _confidence(character_probabilities)
    type_confidences, _ = _confidence(type_probabilities)
    candidates = tuple(
        RecognitionCandidateEvidence(
            text=text,
            character_log_probability=score,
            native_position_count=frozen_character_logits.shape[0],
            source="ctc_prefix_beam",
        )
        for text, score in ctc_prefix_candidates(
            frozen_character_logits,
            character_charset,
            beam_width=decoder_config.beam_width,
            top_k=decoder_config.character_top_k,
        )
    )
    return CommonRecognitionOutput(
        schema_version=COMMON_RECOGNITION_OUTPUT_SCHEMA,
        capability=capability,
        raw_text=raw_text,
        normalized_raw_text=raw_text,
        predicted_length=len(raw_text),
        character_probabilities=character_probabilities,
        type_probabilities=type_probabilities,
        selected_character_symbols=tuple(raw_text),
        selected_type_symbols=tuple(type_text),
        position_confidences=character_confidences,
        type_position_confidences=type_confidences,
        sequence_confidence=sequence_confidence,
        confidence_semantics="ctc-timestep-max-geomean-v1",
        candidates=candidates,
        decoder_evidence=CtcDecoderEvidence(
            character_logits=frozen_character_logits,
            type_logits=frozen_type_logits,
            character_blank_index=character_charset.blank_index,
            type_blank_index=TYPE_CTC_CHARSET.blank_index,
        ),
        # Empty CTC text is an evaluable model prediction, not malformed CTC evidence.
        raw_sequence_valid=True,
    )


def adapt_fixed_slot_logits(
    character_logits: Tensor,
    type_logits: Tensor,
    *,
    capability: RecognitionModelCapability,
    decoder_config: TrDecoderConfig,
) -> CommonRecognitionOutput:
    """Adapt deterministic future fixed-slot evidence without constructing a model."""
    if capability.output_family is not RecognitionOutputFamily.FIXED_SLOTS:
        raise ValueError("fixed-slot logits require a fixed-slot model capability")
    if capability.character_symbols != FIXED_SLOT_CHARACTER_SYMBOLS:
        raise ValueError("fixed-slot Character vocabulary must be the frozen TR V1 vocabulary")
    if capability.type_symbols != FIXED_SLOT_TYPE_SYMBOLS:
        raise ValueError("fixed-slot Type vocabulary must be DIGIT, LETTER, PAD")
    if (
        capability.max_slots is None
        or capability.character_pad_index is None
        or capability.type_pad_index is None
    ):
        raise ValueError("fixed-slot capability must declare slot and PAD metadata")
    if character_logits.ndim != 2 or type_logits.ndim != 2:
        raise ValueError("fixed-slot adapter expects unbatched [slots, classes] logits")
    if character_logits.shape != (capability.max_slots, len(capability.character_symbols) + 1):
        raise ValueError("fixed-slot Character logits do not match capability")
    if type_logits.shape != (capability.max_slots, len(capability.type_symbols)):
        raise ValueError("fixed-slot Type logits do not match capability")

    frozen_character_logits = frozen_cpu_float32(character_logits)
    frozen_type_logits = frozen_cpu_float32(type_logits)
    character_probabilities = frozen_character_logits.softmax(dim=1)
    type_probabilities = frozen_type_logits.softmax(dim=1)
    character_indexes = tuple(
        int(index) for index in character_probabilities.argmax(dim=1).tolist()
    )
    type_indexes = tuple(int(index) for index in type_probabilities.argmax(dim=1).tolist())
    pad = capability.character_pad_index
    type_pad = capability.type_pad_index
    selected_characters = tuple(
        "PAD" if index == pad else capability.character_symbols[index]
        for index in character_indexes
    )
    selected_types = tuple(FIXED_SLOT_TYPE_TEXT[index] for index in type_indexes)
    character_pad_mask = tuple(index == pad for index in character_indexes)
    type_pad_mask = tuple(index == type_pad for index in type_indexes)
    first_pad = next((index for index, is_pad in enumerate(character_pad_mask) if is_pad), None)
    internal_pad = first_pad is not None and any(
        not item for item in character_pad_mask[first_pad:]
    )
    reasons: list[str] = []
    if all(character_pad_mask):
        reasons.append("EMPTY_SLOT_RAW")
    if internal_pad:
        reasons.append("INTERNAL_CHARACTER_PAD")
        reasons.append("NON_PAD_AFTER_PAD")
    if character_pad_mask != type_pad_mask:
        reasons.append("CHARACTER_TYPE_PAD_MISMATCH")
    raw_text = "".join(
        capability.character_symbols[index] for index in character_indexes if index != pad
    )
    character_confidences, sequence_confidence = _confidence(character_probabilities)
    type_confidences, _ = _confidence(type_probabilities)
    candidates = tuple(
        RecognitionCandidateEvidence(
            text=text,
            character_log_probability=score,
            native_position_count=capability.max_slots,
            source="fixed_slot_beam",
        )
        for text, score in fixed_slot_candidates(
            frozen_character_logits,
            character_symbols=capability.character_symbols,
            character_pad_index=pad,
            beam_width=decoder_config.beam_width,
            top_k=decoder_config.character_top_k,
        )
    )
    return CommonRecognitionOutput(
        schema_version=COMMON_RECOGNITION_OUTPUT_SCHEMA,
        capability=capability,
        raw_text=raw_text,
        normalized_raw_text=raw_text,
        predicted_length=sum(not item for item in character_pad_mask),
        character_probabilities=character_probabilities,
        type_probabilities=type_probabilities,
        selected_character_symbols=selected_characters,
        selected_type_symbols=selected_types,
        position_confidences=character_confidences,
        type_position_confidences=type_confidences,
        sequence_confidence=sequence_confidence,
        confidence_semantics="fixed-slot-all-slot-geomean-v1",
        candidates=candidates,
        decoder_evidence=FixedSlotDecoderEvidence(
            character_logits=frozen_character_logits,
            type_logits=frozen_type_logits,
            max_slots=capability.max_slots,
            character_pad_index=pad,
            type_pad_index=type_pad,
        ),
        raw_sequence_valid=not reasons,
        invalid_reasons=tuple(reasons),
    )


def adapt_fixed_slot_batch(
    character_logits: Tensor,
    type_logits: Tensor,
    *,
    capability: RecognitionModelCapability,
    decoder_config: TrDecoderConfig,
) -> tuple[CommonRecognitionOutput, ...]:
    """Convert native M1/M3 batch-first slots through the same unbatched adapter."""
    if character_logits.ndim != 3 or type_logits.ndim != 3:
        raise ValueError("fixed-slot batch adapter expects [batch, slots, classes] logits")
    if character_logits.shape[0] != type_logits.shape[0]:
        raise ValueError("fixed-slot Character and Type batches must be aligned")
    return tuple(
        adapt_fixed_slot_logits(
            character_logits[index],
            type_logits[index],
            capability=capability,
            decoder_config=decoder_config,
        )
        for index in range(character_logits.shape[0])
    )
