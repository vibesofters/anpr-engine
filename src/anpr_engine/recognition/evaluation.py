"""Transparent raw/grammar Recognition metrics and error ledgers."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from anpr_engine.domain import ResultStatus
from anpr_engine.recognition.evidence import CommonRecognitionOutput, RecognitionOutputFamily
from anpr_engine.recognition.metrics import (
    RecognitionErrorCounts,
    RecognitionMetrics,
    compute_recognition_error_counts,
    compute_recognition_metrics,
    edit_distance,
)
from anpr_engine.recognition.profiles import TR_PROFILE, TrFormat
from anpr_engine.recognition.type_ctc import character_type_target

CONFUSION_PAIRS = ("0/O", "1/I", "2/Z", "4/A", "5/S", "6/G", "8/B")


class DecoderEffect(StrEnum):
    GRAMMAR_FIXED_WRONG_RAW = "grammar_fixed_wrong_raw"
    GRAMMAR_DAMAGED_CORRECT_RAW = "grammar_damaged_correct_raw"
    GRAMMAR_CHANGED_BUT_STILL_WRONG = "grammar_changed_but_still_wrong"
    GRAMMAR_NO_EFFECT = "grammar_no_effect"
    GRAMMAR_REJECTED_INVALID = "grammar_rejected_invalid"
    LOW_CONFIDENCE_REJECTED = "low_confidence_rejected"


class ConfusionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pair: str
    raw_confusion_count: int = Field(ge=0)
    type_head_agreement: int = Field(ge=0)
    type_head_disagreement: int = Field(ge=0)
    decoder_correction_count: int = Field(ge=0)
    decoder_damage_count: int = Field(ge=0)


class PerFormatMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    population: int = Field(ge=0)
    exact_raw_accuracy: float = Field(ge=0.0, le=1.0)
    exact_tr_decoded_accuracy: float = Field(ge=0.0, le=1.0)
    character_error_rate: float = Field(ge=0.0)
    invalid_format_rate: float = Field(ge=0.0, le=1.0)
    low_confidence_rate: float = Field(ge=0.0, le=1.0)
    confusion_ledger: tuple[ConfusionEntry, ...]


class DecoderEvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_count: int = Field(ge=1)
    raw_exact_normalized_plate_accuracy: float = Field(ge=0.0, le=1.0)
    tr_decoded_exact_normalized_plate_accuracy: float = Field(ge=0.0, le=1.0)
    raw_valid_format_rate: float = Field(ge=0.0, le=1.0)
    decoded_valid_format_rate: float = Field(ge=0.0, le=1.0)
    type_sequence_accuracy: float = Field(ge=0.0, le=1.0)
    character_type_agreement_rate: float = Field(ge=0.0, le=1.0)
    type_ctc_loss: float | None = Field(default=None, ge=0.0)
    effect_counts: dict[DecoderEffect, int]
    per_format: dict[TrFormat, PerFormatMetrics]
    confusion_ledger: tuple[ConfusionEntry, ...]


class RepeatedSliceMetrics(BaseModel):
    """RAW exactness on a target-defined repeat slice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    population: int = Field(ge=0)
    raw_exact_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)


class CommonRecognitionEvaluationMetrics(BaseModel):
    """Family-neutral recognition metrics plus explicit native diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recognition: RecognitionMetrics
    decoder: DecoderEvaluationMetrics
    errors: RecognitionErrorCounts
    sequence_length_error_count: int = Field(ge=0)
    province_prefix_error_count: int = Field(ge=0)
    repeated_character: RepeatedSliceMetrics
    repeated_digit: RepeatedSliceMetrics
    repeated_letter: RepeatedSliceMetrics
    output_family_counts: dict[RecognitionOutputFamily, int]
    ctc_collapse_diagnostic_count: int | None = Field(default=None, ge=0)
    slot_internal_pad_count: int | None = Field(default=None, ge=0)


def decoder_effect(
    *, target: str, raw_text: str, decoded_text: str | None, status: ResultStatus
) -> DecoderEffect:
    if status is ResultStatus.LOW_CONFIDENCE:
        return DecoderEffect.LOW_CONFIDENCE_REJECTED
    if decoded_text is None:
        return DecoderEffect.GRAMMAR_REJECTED_INVALID
    raw_correct, decoded_correct = raw_text == target, decoded_text == target
    if not raw_correct and decoded_correct:
        return DecoderEffect.GRAMMAR_FIXED_WRONG_RAW
    if raw_correct and not decoded_correct:
        return DecoderEffect.GRAMMAR_DAMAGED_CORRECT_RAW
    if raw_text != decoded_text and not decoded_correct:
        return DecoderEffect.GRAMMAR_CHANGED_BUT_STILL_WRONG
    return DecoderEffect.GRAMMAR_NO_EFFECT


def _confusions(
    targets: tuple[str, ...],
    raw_predictions: tuple[str, ...],
    decoded_predictions: tuple[str | None, ...],
    type_predictions: tuple[str, ...],
) -> tuple[ConfusionEntry, ...]:
    entries: list[ConfusionEntry] = []
    for pair in CONFUSION_PAIRS:
        left, right = pair.split("/")
        raw_count = agreement = disagreement = corrections = damages = 0
        for target, raw, decoded, predicted_types in zip(
            targets, raw_predictions, decoded_predictions, type_predictions, strict=True
        ):
            if len(target) != len(raw):
                continue
            target_types = character_type_target(target)
            for index, (truth, prediction) in enumerate(zip(target, raw, strict=True)):
                if {truth, prediction} == {left, right}:
                    raw_count += 1
                    if (
                        index < len(predicted_types)
                        and predicted_types[index] == target_types[index]
                    ):
                        agreement += 1
                    else:
                        disagreement += 1
                    if (
                        decoded is not None
                        and len(decoded) == len(target)
                        and decoded[index] == truth
                    ):
                        corrections += 1
                if (
                    decoded is not None
                    and len(decoded) == len(target)
                    and raw[index] == truth
                    and {truth, decoded[index]} == {left, right}
                ):
                    damages += 1
        entries.append(
            ConfusionEntry(
                pair=pair,
                raw_confusion_count=raw_count,
                type_head_agreement=agreement,
                type_head_disagreement=disagreement,
                decoder_correction_count=corrections,
                decoder_damage_count=damages,
            )
        )
    return tuple(entries)


def compute_decoder_evaluation_metrics(
    *,
    targets: tuple[str, ...],
    raw_predictions: tuple[str, ...],
    decoded_predictions: tuple[str | None, ...],
    statuses: tuple[ResultStatus, ...],
    type_predictions: tuple[str, ...],
    type_ctc_loss: float | None = None,
) -> DecoderEvaluationMetrics:
    size = len(targets)
    if size < 1 or not all(
        len(values) == size
        for values in (raw_predictions, decoded_predictions, statuses, type_predictions)
    ):
        raise ValueError("decoder metric populations must be non-empty and aligned")
    raw_valid = tuple(TR_PROFILE.validate(text).valid for text in raw_predictions)
    decoded_valid = tuple(
        text is not None and TR_PROFILE.validate(text).valid for text in decoded_predictions
    )
    effects = Counter(
        decoder_effect(target=target, raw_text=raw, decoded_text=decoded, status=status)
        for target, raw, decoded, status in zip(
            targets, raw_predictions, decoded_predictions, statuses, strict=True
        )
    )
    per_format: dict[TrFormat, PerFormatMetrics] = {}
    for format_id in TrFormat:
        indexes = [
            index
            for index, target in enumerate(targets)
            if TR_PROFILE.validate(target).format_id is format_id
        ]
        if not indexes:
            per_format[format_id] = PerFormatMetrics(
                population=0,
                exact_raw_accuracy=0.0,
                exact_tr_decoded_accuracy=0.0,
                character_error_rate=0.0,
                invalid_format_rate=0.0,
                low_confidence_rate=0.0,
                confusion_ledger=_confusions((), (), (), ()),
            )
            continue
        target_characters = sum(len(targets[index]) for index in indexes)
        per_format[format_id] = PerFormatMetrics(
            population=len(indexes),
            exact_raw_accuracy=sum(raw_predictions[index] == targets[index] for index in indexes)
            / len(indexes),
            exact_tr_decoded_accuracy=sum(
                decoded_predictions[index] == targets[index] for index in indexes
            )
            / len(indexes),
            character_error_rate=sum(
                edit_distance(targets[index], raw_predictions[index]) for index in indexes
            )
            / target_characters,
            invalid_format_rate=sum(not decoded_valid[index] for index in indexes) / len(indexes),
            low_confidence_rate=sum(
                statuses[index] is ResultStatus.LOW_CONFIDENCE for index in indexes
            )
            / len(indexes),
            confusion_ledger=_confusions(
                tuple(targets[index] for index in indexes),
                tuple(raw_predictions[index] for index in indexes),
                tuple(decoded_predictions[index] for index in indexes),
                tuple(type_predictions[index] for index in indexes),
            ),
        )
    target_types = tuple(character_type_target(target) for target in targets)
    total_type_characters = sum(len(target) for target in target_types)
    agreed_type_characters = sum(
        sum(expected == predicted for expected, predicted in zip(target, prediction, strict=False))
        for target, prediction in zip(target_types, type_predictions, strict=True)
    )
    return DecoderEvaluationMetrics(
        sample_count=size,
        raw_exact_normalized_plate_accuracy=sum(
            raw == target for raw, target in zip(raw_predictions, targets, strict=True)
        )
        / size,
        tr_decoded_exact_normalized_plate_accuracy=sum(
            decoded == target for decoded, target in zip(decoded_predictions, targets, strict=True)
        )
        / size,
        raw_valid_format_rate=sum(raw_valid) / size,
        decoded_valid_format_rate=sum(decoded_valid) / size,
        type_sequence_accuracy=sum(
            prediction == target
            for prediction, target in zip(type_predictions, target_types, strict=True)
        )
        / size,
        character_type_agreement_rate=agreed_type_characters / total_type_characters,
        type_ctc_loss=type_ctc_loss,
        effect_counts={effect: effects[effect] for effect in DecoderEffect},
        per_format=per_format,
        confusion_ledger=_confusions(
            targets, raw_predictions, decoded_predictions, type_predictions
        ),
    )


def _repeat_slice(
    targets: tuple[str, ...], predictions: tuple[str, ...], *, kind: str
) -> RepeatedSliceMetrics:
    indexes: list[int] = []
    for index, target in enumerate(targets):
        adjacent = tuple(zip(target, target[1:], strict=False))
        if kind == "character" and any(left == right for left, right in adjacent):
            indexes.append(index)
        if kind == "digit" and any(left == right and left.isdigit() for left, right in adjacent):
            indexes.append(index)
        if kind == "letter" and any(left == right and left.isalpha() for left, right in adjacent):
            indexes.append(index)
    if not indexes:
        return RepeatedSliceMetrics(population=0, raw_exact_accuracy=None)
    return RepeatedSliceMetrics(
        population=len(indexes),
        raw_exact_accuracy=sum(predictions[index] == targets[index] for index in indexes)
        / len(indexes),
    )


def compute_common_recognition_evaluation_metrics(
    *,
    targets: tuple[str, ...],
    outputs: tuple[CommonRecognitionOutput, ...],
    decoded_predictions: tuple[str | None, ...],
    statuses: tuple[ResultStatus, ...],
) -> CommonRecognitionEvaluationMetrics:
    """Evaluate CTC and slot outputs by their shared plate-level semantics.

    Type/PAD positions are not credited: slot Type text is compacted to the
    corresponding non-PAD Character positions, while CTC uses its collapsed
    Type prediction directly.
    """
    if not targets or not (
        len(targets) == len(outputs) == len(decoded_predictions) == len(statuses)
    ):
        raise ValueError("common evaluation requires aligned non-empty populations")
    raw = tuple(output.raw_text for output in outputs)
    predicted_lengths = tuple(
        output.predicted_length if output.raw_sequence_valid else -1 for output in outputs
    )
    type_predictions: list[str] = []
    for output in outputs:
        if output.output_family is RecognitionOutputFamily.CTC:
            type_predictions.append("".join(output.selected_type_symbols))
            continue
        pairs = zip(
            output.selected_character_symbols,
            output.selected_type_symbols,
            strict=True,
        )
        type_predictions.append(
            "".join(
                type_symbol
                for character, type_symbol in pairs
                if character != "PAD" and type_symbol != "PAD"
            )
        )
    valid = tuple(
        output.raw_sequence_valid and TR_PROFILE.validate(output.raw_text).valid
        for output in outputs
    )
    recognition = compute_recognition_metrics(
        raw,
        targets,
        valid=valid,
        statuses=statuses,
        predicted_lengths=predicted_lengths,
    )
    decoder = compute_decoder_evaluation_metrics(
        targets=targets,
        raw_predictions=raw,
        decoded_predictions=decoded_predictions,
        statuses=statuses,
        type_predictions=tuple(type_predictions),
    )
    errors = compute_recognition_error_counts(raw, targets, valid=valid)
    families = {
        family: sum(output.output_family is family for output in outputs)
        for family in RecognitionOutputFamily
    }
    slot_outputs = tuple(
        output for output in outputs if output.output_family is RecognitionOutputFamily.FIXED_SLOTS
    )
    return CommonRecognitionEvaluationMetrics(
        recognition=recognition,
        decoder=decoder,
        errors=errors,
        sequence_length_error_count=sum(
            length != len(target) for length, target in zip(predicted_lengths, targets, strict=True)
        ),
        province_prefix_error_count=sum(
            prediction[:2] != target[:2] for prediction, target in zip(raw, targets, strict=True)
        ),
        repeated_character=_repeat_slice(targets, raw, kind="character"),
        repeated_digit=_repeat_slice(targets, raw, kind="digit"),
        repeated_letter=_repeat_slice(targets, raw, kind="letter"),
        output_family_counts=families,
        ctc_collapse_diagnostic_count=None,
        slot_internal_pad_count=(
            sum("INTERNAL_CHARACTER_PAD" in output.invalid_reasons for output in slot_outputs)
            if slot_outputs
            else None
        ),
    )
