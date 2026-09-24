"""Raw-preserving deterministic plate-text normalization."""

from __future__ import annotations

from dataclasses import dataclass

NORMALIZATION_VERSION = "turkish-plate-normalization-v1"
ALLOWED_SEPARATORS = frozenset({"-", " ", "\t", "\n", "\r", "\f", "\v"})


class PlateNormalizationError(ValueError):
    """Raised when raw text cannot be normalized without guessing."""


@dataclass(frozen=True)
class NormalizationResult:
    raw_text: str
    normalized_text: str
    version: str = NORMALIZATION_VERSION


def normalize_plate_text(raw_text: str) -> NormalizationResult:
    normalized: list[str] = []
    unsupported: list[str] = []
    for character in raw_text:
        if character in ALLOWED_SEPARATORS:
            continue
        if character.isascii() and character.isalnum():
            normalized.append(character.upper())
        else:
            unsupported.append(character)
    if unsupported:
        rendered = "".join(dict.fromkeys(unsupported))
        raise PlateNormalizationError(f"unsupported character(s): {rendered!r}")
    text = "".join(normalized)
    if not text:
        raise PlateNormalizationError("normalized plate text is empty")
    return NormalizationResult(raw_text=raw_text, normalized_text=text)
