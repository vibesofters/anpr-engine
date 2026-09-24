"""Probability-aware Turkish province constraint for fixed-slot Recognition."""

from __future__ import annotations

import pytest
import torch

from anpr_engine.recognition.decoder import (
    TR_FIXED_SLOT_DECODER_VERSION,
    fixed_slot_candidates,
    province_constrained_fixed_slot_candidates,
)

SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PAD = len(SYMBOLS)


def _logits(raw: str, preferred_valid_prefix: str | None = None) -> torch.Tensor:
    logits = torch.full((8, len(SYMBOLS) + 1), -12.0)
    for slot, symbol in enumerate(raw):
        logits[slot, SYMBOLS.index(symbol)] = 12.0
    logits[len(raw) :, PAD] = 12.0
    if preferred_valid_prefix is not None:
        for slot, symbol in enumerate(preferred_valid_prefix):
            logits[slot, SYMBOLS.index(symbol)] = 11.0
    return logits


@pytest.mark.parametrize(
    ("raw", "preferred", "expected"),
    (("00ABC123", "01", "01"), ("84ABC123", "34", "34"), ("99ABC123", "79", "79")),
)
def test_invalid_province_is_replaced_by_highest_probability_valid_pair(
    raw: str, preferred: str, expected: str
) -> None:
    logits = _logits(raw, preferred)
    base = fixed_slot_candidates(
        logits,
        character_symbols=SYMBOLS,
        character_pad_index=PAD,
        beam_width=12,
        top_k=12,
    )
    constrained = province_constrained_fixed_slot_candidates(
        logits,
        raw_text=raw,
        candidates=base,
        character_symbols=SYMBOLS,
        character_pad_index=PAD,
    )
    assert constrained[0][0][:2] == expected
    assert 1 <= int(constrained[0][0][:2]) <= 81


@pytest.mark.parametrize("prefix", ("01", "34", "81"))
def test_valid_raw_province_is_preserved(prefix: str) -> None:
    raw = prefix + "ABC123"
    logits = _logits(raw)
    base = fixed_slot_candidates(
        logits,
        character_symbols=SYMBOLS,
        character_pad_index=PAD,
        beam_width=12,
        top_k=12,
    )
    constrained = province_constrained_fixed_slot_candidates(
        logits,
        raw_text=raw,
        candidates=base,
        character_symbols=SYMBOLS,
        character_pad_index=PAD,
    )
    assert constrained[0][0].startswith(prefix)
    assert all(text.startswith(prefix) for text, _score in constrained)


def test_decoder_behavior_is_versioned() -> None:
    assert TR_FIXED_SLOT_DECODER_VERSION == "fixed-slot-beam-tr-profile-province-v2"
