"""Probability-aware CTC candidate search constrained by a configured country profile."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from anpr_engine.domain import ResultStatus
from anpr_engine.recognition.charset import PlateCharset
from anpr_engine.recognition.ctc import greedy_decode_logits
from anpr_engine.recognition.evidence import (
    CommonRecognitionOutput,
    CtcDecoderEvidence,
    FixedSlotDecoderEvidence,
    RecognitionOutputFamily,
)
from anpr_engine.recognition.profiles import CountryProfile, ProfileValidation
from anpr_engine.recognition.type_ctc import TYPE_CTC_CHARSET, character_type_target
from anpr_engine.recognition.validation import is_valid_turkish_province_prefix

TR_CTC_DECODER_VERSION = "ctc-prefix-beam-tr-profile-v1"
TR_FIXED_SLOT_DECODER_VERSION = "fixed-slot-beam-tr-profile-province-v2"
VALID_TR_PROVINCE_PREFIXES = tuple(f"{value:02d}" for value in range(1, 82))


class TrDecoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    beam_width: int = Field(default=12, ge=2, le=128)
    character_top_k: int = Field(default=12, ge=2, le=37)
    type_score_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    minimum_decoder_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_score_margin: float | None = Field(default=None, ge=0.0, le=1.0)


class DecodedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    character_log_probability: float
    type_log_probability: float
    combined_score: float
    confidence: float
    validation: ProfileValidation


class ProfiledRecognitionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_text: str
    raw_confidence: float
    decoded_text: str | None
    profile_id: str
    decoder_version: str
    decoder_score: float | None
    decoder_confidence: float | None
    decoding_changed_raw: bool
    structural_validation: ProfileValidation | None
    status: ResultStatus
    reason: str
    candidate_count: int


def decoder_cpu_logits(logits: Tensor) -> Tensor:
    """Normalize inference-only decoder evidence to CPU float32.

    Enhanced model logits originate as MPS float32. A direct request for CPU
    float64 asks MPS to perform an unsupported cast before the device copy.
    Decoder dynamic programming operates in log space over only 40 timesteps,
    so CPU float32 is sufficient and retains the model's native precision.
    """
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("decoder logits must be a floating-point [time, classes] tensor")
    return logits.detach().to(device="cpu", dtype=torch.float32)


def _logadd(*values: float) -> float:
    finite = tuple(value for value in values if not math.isinf(value))
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def _add_beam_probability(
    beams: dict[tuple[int, ...], tuple[float, float]],
    prefix: tuple[int, ...],
    *,
    blank: float | None = None,
    nonblank: float | None = None,
) -> None:
    old_blank, old_nonblank = beams.get(prefix, (-math.inf, -math.inf))
    beams[prefix] = (
        _logadd(old_blank, blank if blank is not None else -math.inf),
        _logadd(old_nonblank, nonblank if nonblank is not None else -math.inf),
    )


def ctc_prefix_candidates(
    logits: Tensor,
    charset: PlateCharset,
    *,
    beam_width: int,
    top_k: int,
) -> tuple[tuple[str, float], ...]:
    """Return candidate strings and exact prefix-beam log probabilities for one sample."""
    if logits.ndim != 2 or logits.shape[1] != charset.class_count:
        raise ValueError("single-sample CTC logits must have shape [time, classes]")
    log_probs = decoder_cpu_logits(logits).log_softmax(1)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
    for timestep in range(log_probs.shape[0]):
        row = log_probs[timestep]
        selected = torch.topk(row, k=min(top_k, row.numel())).indices.tolist()
        if charset.blank_index not in selected:
            selected.append(charset.blank_index)
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        for prefix, (prob_blank, prob_nonblank) in beams.items():
            total = _logadd(prob_blank, prob_nonblank)
            for index in selected:
                log_probability = float(row[index].item())
                if index == charset.blank_index:
                    _add_beam_probability(next_beams, prefix, blank=total + log_probability)
                    continue
                if prefix and index == prefix[-1]:
                    _add_beam_probability(
                        next_beams, prefix, nonblank=prob_nonblank + log_probability
                    )
                    _add_beam_probability(
                        next_beams,
                        prefix + (index,),
                        nonblank=prob_blank + log_probability,
                    )
                else:
                    _add_beam_probability(
                        next_beams, prefix + (index,), nonblank=total + log_probability
                    )
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: _logadd(*item[1]),
                reverse=True,
            )[:beam_width]
        )
    candidates = [
        (charset.decode_indices(prefix), _logadd(prob_blank, prob_nonblank))
        for prefix, (prob_blank, prob_nonblank) in beams.items()
        if prefix
    ]
    return tuple(sorted(candidates, key=lambda item: item[1], reverse=True))


def fixed_slot_candidates(
    logits: Tensor,
    *,
    character_symbols: str,
    character_pad_index: int,
    beam_width: int,
    top_k: int,
) -> tuple[tuple[str, float], ...]:
    """Return suffix-PAD fixed-slot candidates without importing TR grammar rules.

    Candidate generation is output-family-native.  The shared decoder validates
    and ranks those strings with the same profile as CTC candidates.
    """
    if logits.ndim != 2 or logits.shape[1] != len(character_symbols) + 1:
        raise ValueError("fixed-slot logits must have shape [slots, symbols plus PAD]")
    if not 0 <= character_pad_index < logits.shape[1]:
        raise ValueError("fixed-slot Character PAD index must be in range")
    log_probs = decoder_cpu_logits(logits).log_softmax(1)
    beams: list[tuple[tuple[int, ...], bool, float]] = [((), False, 0.0)]
    for slot in range(log_probs.shape[0]):
        row = log_probs[slot]
        ordered = sorted(range(row.numel()), key=lambda index: (-float(row[index].item()), index))[
            : min(top_k, row.numel())
        ]
        if character_pad_index not in ordered:
            ordered.append(character_pad_index)
        following: list[tuple[tuple[int, ...], bool, float]] = []
        for indexes, ended, score in beams:
            choices = (character_pad_index,) if ended else tuple(ordered)
            for index in choices:
                following.append(
                    (
                        indexes + (index,),
                        ended or index == character_pad_index,
                        score + float(row[index].item()),
                    )
                )
        following.sort(key=lambda item: (-item[2], item[0]))
        beams = following[:beam_width]
    candidates: list[tuple[str, float]] = []
    for indexes, _, score in beams:
        text = "".join(
            character_symbols[index] for index in indexes if index != character_pad_index
        )
        if text:
            candidates.append((text, score))
    return tuple(candidates)


def fixed_slot_type_log_probability(
    logits: Tensor,
    *,
    text: str,
    max_slots: int,
    type_pad_index: int,
) -> float:
    """Score a Character candidate's aligned D/L/PAD Type sequence."""
    if logits.ndim != 2 or logits.shape != (max_slots, 3):
        raise ValueError("fixed-slot Type logits must have shape [max_slots, 3]")
    if len(text) > max_slots:
        raise ValueError("fixed-slot candidate exceeds max_slots")
    log_probs = decoder_cpu_logits(logits).log_softmax(1)
    target = character_type_target(text)
    indexes = [0 if symbol == "D" else 1 for symbol in target]
    indexes.extend([type_pad_index] * (max_slots - len(indexes)))
    return sum(float(log_probs[position, index].item()) for position, index in enumerate(indexes))


def fixed_slot_text_log_probability(
    logits: Tensor,
    *,
    text: str,
    character_symbols: str,
    character_pad_index: int,
) -> float:
    """Score an exact fixed-slot text, including its required PAD suffix."""
    if logits.ndim != 2 or logits.shape[1] != len(character_symbols) + 1:
        raise ValueError("fixed-slot logits must have shape [slots, symbols plus PAD]")
    if len(text) > logits.shape[0] or any(symbol not in character_symbols for symbol in text):
        return -math.inf
    log_probs = decoder_cpu_logits(logits).log_softmax(1)
    indexes = [character_symbols.index(symbol) for symbol in text]
    indexes.extend([character_pad_index] * (logits.shape[0] - len(indexes)))
    return sum(float(log_probs[position, index].item()) for position, index in enumerate(indexes))


def province_constrained_fixed_slot_candidates(
    logits: Tensor,
    *,
    raw_text: str,
    candidates: Iterable[tuple[str, float]],
    character_symbols: str,
    character_pad_index: int,
) -> tuple[tuple[str, float], ...]:
    """Apply the v2 probability-aware 01–81 province constraint.

    A valid RAW province is stable.  For an invalid RAW province, every valid
    province is scored against the actual first-two-slot probabilities for each
    available suffix; numerical distance is never used.
    """
    base = tuple(candidates)
    raw_prefix_valid = len(raw_text) >= 2 and is_valid_turkish_province_prefix(raw_text[:2])
    suffixes = {text[2:] for text, _score in base if len(text) >= 2}
    if len(raw_text) >= 2:
        suffixes.add(raw_text[2:])
    texts: set[str]
    if raw_prefix_valid:
        texts = {text for text, _score in base if text.startswith(raw_text[:2])}
        texts.add(raw_text)
    else:
        texts = {prefix + suffix for prefix in VALID_TR_PROVINCE_PREFIXES for suffix in suffixes}
    scored = (
        (
            text,
            fixed_slot_text_log_probability(
                logits,
                text=text,
                character_symbols=character_symbols,
                character_pad_index=character_pad_index,
            ),
        )
        for text in texts
    )
    return tuple(sorted(scored, key=lambda item: (-item[1], item[0])))


def ctc_target_log_probability(logits: Tensor, target: Iterable[int], *, blank_index: int) -> float:
    """Exact forward probability for one CTC target, including repeated labels."""
    if logits.ndim != 2:
        raise ValueError("single-sample CTC logits must have shape [time, classes]")
    target_indexes = tuple(target)
    if not target_indexes:
        raise ValueError("CTC probability target cannot be empty")
    extended = [blank_index]
    for index in target_indexes:
        extended.extend((index, blank_index))
    log_probs = decoder_cpu_logits(logits).log_softmax(1)
    alpha = [-math.inf] * len(extended)
    alpha[0] = float(log_probs[0, blank_index].item())
    if len(extended) > 1:
        alpha[1] = float(log_probs[0, extended[1]].item())
    for timestep in range(1, log_probs.shape[0]):
        following = [-math.inf] * len(extended)
        for state, symbol in enumerate(extended):
            predecessors = [alpha[state]]
            if state > 0:
                predecessors.append(alpha[state - 1])
            if state > 1 and symbol != blank_index and symbol != extended[state - 2]:
                predecessors.append(alpha[state - 2])
            following[state] = _logadd(*predecessors) + float(log_probs[timestep, symbol].item())
        alpha = following
    return _logadd(alpha[-1], alpha[-2])


class ProbabilityAwareTrDecoder:
    def __init__(
        self,
        *,
        charset: PlateCharset,
        profile: CountryProfile,
        config: TrDecoderConfig,
    ) -> None:
        self.charset = charset
        self.profile = profile
        self.config = config

    def decode(self, character_logits: Tensor, type_logits: Tensor) -> ProfiledRecognitionResult:
        if character_logits.ndim != 2 or type_logits.ndim != 2:
            raise ValueError("decoder expects unbatched [time, classes] logits")
        if character_logits.shape[0] != type_logits.shape[0]:
            raise ValueError("character and type logits must share the time axis")
        raw_text = greedy_decode_logits(character_logits.unsqueeze(1), self.charset)[0]
        raw_log_probs = decoder_cpu_logits(character_logits).log_softmax(1)
        raw_confidence = float(torch.exp(raw_log_probs.max(1).values.mean()).item())
        beams = ctc_prefix_candidates(
            character_logits,
            self.charset,
            beam_width=self.config.beam_width,
            top_k=self.config.character_top_k,
        )
        return self._finalize_candidates(
            raw_text=raw_text,
            raw_confidence=raw_confidence,
            candidates=beams,
            native_position_count=character_logits.shape[0],
            type_log_probability=lambda text: ctc_target_log_probability(
                type_logits,
                TYPE_CTC_CHARSET.encode(character_type_target(text)),
                blank_index=TYPE_CTC_CHARSET.blank_index,
            ),
            decoder_version=TR_CTC_DECODER_VERSION,
        )

    def decode_common(self, output: CommonRecognitionOutput) -> ProfiledRecognitionResult:
        """Decode either common evidence family through one TR profile boundary."""
        if output.output_family is RecognitionOutputFamily.CTC:
            evidence = output.decoder_evidence
            if not isinstance(evidence, CtcDecoderEvidence):
                raise ValueError("CTC common output must contain CTC decoder evidence")
            # Deliberately preserve the established M0 path verbatim.
            return self.decode(evidence.character_logits, evidence.type_logits)
        if output.output_family is not RecognitionOutputFamily.FIXED_SLOTS:
            raise ValueError(f"unsupported Recognition output family: {output.output_family}")
        evidence = output.decoder_evidence
        if not isinstance(evidence, FixedSlotDecoderEvidence):
            raise ValueError("fixed-slot common output must contain fixed-slot decoder evidence")
        if not output.raw_sequence_valid:
            return ProfiledRecognitionResult(
                raw_text=output.raw_text,
                raw_confidence=output.sequence_confidence,
                decoded_text=None,
                profile_id=self.profile.profile_id,
                decoder_version=TR_FIXED_SLOT_DECODER_VERSION,
                decoder_score=None,
                decoder_confidence=None,
                decoding_changed_raw=False,
                structural_validation=None,
                status=ResultStatus.LOW_CONFIDENCE,
                reason="INVALID_SLOT_SEQUENCE:" + ",".join(output.invalid_reasons),
                candidate_count=0,
            )
        candidates = fixed_slot_candidates(
            evidence.character_logits,
            character_symbols=output.capability.character_symbols,
            character_pad_index=evidence.character_pad_index,
            beam_width=self.config.beam_width,
            top_k=self.config.character_top_k,
        )
        candidates = province_constrained_fixed_slot_candidates(
            evidence.character_logits,
            raw_text=output.raw_text,
            candidates=candidates,
            character_symbols=output.capability.character_symbols,
            character_pad_index=evidence.character_pad_index,
        )
        return self._finalize_candidates(
            raw_text=output.raw_text,
            raw_confidence=output.sequence_confidence,
            candidates=candidates,
            native_position_count=evidence.max_slots,
            type_log_probability=lambda text: fixed_slot_type_log_probability(
                evidence.type_logits,
                text=text,
                max_slots=evidence.max_slots,
                type_pad_index=evidence.type_pad_index,
            ),
            decoder_version=TR_FIXED_SLOT_DECODER_VERSION,
        )

    def _finalize_candidates(
        self,
        *,
        raw_text: str,
        raw_confidence: float,
        candidates: Iterable[tuple[str, float]],
        native_position_count: int,
        type_log_probability: Callable[[str], float],
        decoder_version: str,
    ) -> ProfiledRecognitionResult:
        valid: list[DecodedCandidate] = []
        for text, character_log_probability in candidates:
            validation = self.profile.validate(text)
            if not validation.valid:
                continue
            type_score = type_log_probability(text)
            combined = (
                character_log_probability + self.config.type_score_weight * type_score
            ) / float(native_position_count)
            confidence = math.exp(min(0.0, combined / (1.0 + self.config.type_score_weight)))
            valid.append(
                DecodedCandidate(
                    text=text,
                    character_log_probability=character_log_probability,
                    type_log_probability=type_score,
                    combined_score=combined,
                    confidence=confidence,
                    validation=validation,
                )
            )
        valid.sort(key=lambda candidate: candidate.combined_score, reverse=True)
        if not valid:
            return ProfiledRecognitionResult(
                raw_text=raw_text,
                raw_confidence=raw_confidence,
                decoded_text=None,
                profile_id=self.profile.profile_id,
                decoder_version=decoder_version,
                decoder_score=None,
                decoder_confidence=None,
                decoding_changed_raw=False,
                structural_validation=None,
                status=ResultStatus.LOW_CONFIDENCE,
                reason="NO_SUPPORTED_VALID_CANDIDATE",
                candidate_count=0,
            )
        best = valid[0]
        margin = best.confidence - valid[1].confidence if len(valid) > 1 else best.confidence
        threshold = self.config.minimum_decoder_confidence
        margin_threshold = self.config.minimum_score_margin
        calibrated = (
            threshold is not None
            and best.confidence >= threshold
            and (margin_threshold is None or margin >= margin_threshold)
        )
        return ProfiledRecognitionResult(
            raw_text=raw_text,
            raw_confidence=raw_confidence,
            decoded_text=best.text,
            profile_id=self.profile.profile_id,
            decoder_version=decoder_version,
            decoder_score=best.combined_score,
            decoder_confidence=best.confidence,
            decoding_changed_raw=best.text != raw_text,
            structural_validation=best.validation,
            status=ResultStatus.ACCEPTED if calibrated else ResultStatus.LOW_CONFIDENCE,
            reason="ACCEPTED_CALIBRATED" if calibrated else "UNCALIBRATED_OR_INSUFFICIENT_SUPPORT",
            candidate_count=len(valid),
        )
