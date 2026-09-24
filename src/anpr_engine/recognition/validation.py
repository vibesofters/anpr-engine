"""Conservative, non-correcting Turkish plate structure validation."""

from __future__ import annotations

import re

from anpr_engine.domain import PlateValidationReason, PlateValidationResult

TURKISH_PLATE_RULE_VERSION = "turkish-private-plate-structure-v1"
_ALLOWED = re.compile(r"^[0-9A-Z]+$")
_STRUCTURE = re.compile(r"^(?P<province>[0-9]{2})(?P<letters>[A-Z]{1,3})(?P<serial>[0-9]{2,4})$")
TURKISH_PROVINCE_MIN = 1
TURKISH_PROVINCE_MAX = 81


def is_valid_turkish_province_prefix(prefix: str) -> bool:
    """Authoritative V1 province rule shared by legacy and profile validators."""
    return (
        len(prefix) == 2
        and prefix.isascii()
        and prefix.isdigit()
        and TURKISH_PROVINCE_MIN <= int(prefix) <= TURKISH_PROVINCE_MAX
    )


def validate_turkish_plate(normalized_text: str) -> PlateValidationResult:
    reasons: list[PlateValidationReason] = []
    if not normalized_text:
        reasons.append(PlateValidationReason.EMPTY)
    if normalized_text and _ALLOWED.fullmatch(normalized_text) is None:
        reasons.append(PlateValidationReason.UNSUPPORTED_CHARACTER)
    if normalized_text and not 5 <= len(normalized_text) <= 9:
        reasons.append(PlateValidationReason.INVALID_LENGTH)

    match = _STRUCTURE.fullmatch(normalized_text)
    if match is None:
        reasons.append(PlateValidationReason.INVALID_STRUCTURE)
    else:
        if not is_valid_turkish_province_prefix(match.group("province")):
            reasons.append(PlateValidationReason.INVALID_PROVINCE)

    unique_reasons = tuple(dict.fromkeys(reasons))
    return PlateValidationResult(
        valid=not unique_reasons,
        rule_version=TURKISH_PLATE_RULE_VERSION,
        reason_codes=unique_reasons,
    )
