"""Country-specific grammar contracts kept outside the shared visual encoder."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from anpr_engine.recognition.charset import V1_CHARSET
from anpr_engine.recognition.normalization import normalize_plate_text
from anpr_engine.recognition.validation import (
    TURKISH_PLATE_RULE_VERSION,
    is_valid_turkish_province_prefix,
    validate_turkish_plate,
)

TR_PROFILE_ID = "TR"
TR_PROFILE_VERSION = "tr-standard-private-plate-profile-v1"


class TrFormat(StrEnum):
    PROVINCE_1L_4D = "2D-1L-4D"
    PROVINCE_1L_5D = "2D-1L-5D"
    PROVINCE_2L_3D = "2D-2L-3D"
    PROVINCE_2L_4D = "2D-2L-4D"
    PROVINCE_3L_2D = "2D-3L-2D"
    PROVINCE_3L_3D = "2D-3L-3D"


TR_FORMAT_PATTERNS: tuple[tuple[TrFormat, re.Pattern[str]], ...] = (
    (TrFormat.PROVINCE_1L_4D, re.compile(r"^[0-9]{2}[A-Z][0-9]{4}$")),
    (TrFormat.PROVINCE_1L_5D, re.compile(r"^[0-9]{2}[A-Z][0-9]{5}$")),
    (TrFormat.PROVINCE_2L_3D, re.compile(r"^[0-9]{2}[A-Z]{2}[0-9]{3}$")),
    (TrFormat.PROVINCE_2L_4D, re.compile(r"^[0-9]{2}[A-Z]{2}[0-9]{4}$")),
    (TrFormat.PROVINCE_3L_2D, re.compile(r"^[0-9]{2}[A-Z]{3}[0-9]{2}$")),
    (TrFormat.PROVINCE_3L_3D, re.compile(r"^[0-9]{2}[A-Z]{3}[0-9]{3}$")),
)


class ProfileReason(StrEnum):
    EMPTY = "EMPTY"
    UNSUPPORTED_CHARACTER = "UNSUPPORTED_CHARACTER"
    INVALID_PROVINCE = "INVALID_PROVINCE"
    UNSUPPORTED_TR_FORMAT = "UNSUPPORTED_TR_FORMAT"


class ProfileValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    profile_id: str
    profile_version: str
    format_id: TrFormat | None
    reason_codes: tuple[ProfileReason, ...]
    legacy_validator_version: str
    legacy_validator_valid: bool
    legacy_reason_codes: tuple[str, ...]
    governed_validator_disagreement: bool


class CountryProfile(Protocol):
    profile_id: str
    country_code: str
    script_family: str
    allowed_charset: str

    def normalize(self, raw_text: str) -> str: ...

    def validate(self, normalized_text: str) -> ProfileValidation: ...


class TrCountryProfile:
    """Exactly the owner-approved standard Turkish formats for Recognition V1."""

    profile_id = TR_PROFILE_ID
    country_code = "TR"
    script_family = "Latin"
    allowed_charset = V1_CHARSET.symbols
    allowed_formats = tuple(format_id for format_id, _pattern in TR_FORMAT_PATTERNS)

    def normalize(self, raw_text: str) -> str:
        return normalize_plate_text(raw_text).normalized_text

    def validate(self, normalized_text: str) -> ProfileValidation:
        reasons: list[ProfileReason] = []
        if not normalized_text:
            reasons.append(ProfileReason.EMPTY)
        elif any(character not in self.allowed_charset for character in normalized_text):
            reasons.append(ProfileReason.UNSUPPORTED_CHARACTER)
        format_id = next(
            (
                candidate
                for candidate, pattern in TR_FORMAT_PATTERNS
                if pattern.fullmatch(normalized_text)
            ),
            None,
        )
        if normalized_text and format_id is None:
            reasons.append(ProfileReason.UNSUPPORTED_TR_FORMAT)
        if len(normalized_text) >= 2 and not is_valid_turkish_province_prefix(normalized_text[:2]):
            reasons.append(ProfileReason.INVALID_PROVINCE)
        legacy = validate_turkish_plate(normalized_text)
        valid = not reasons
        return ProfileValidation(
            valid=valid,
            profile_id=self.profile_id,
            profile_version=TR_PROFILE_VERSION,
            format_id=format_id,
            reason_codes=tuple(dict.fromkeys(reasons)),
            legacy_validator_version=TURKISH_PLATE_RULE_VERSION,
            legacy_validator_valid=legacy.valid,
            legacy_reason_codes=tuple(reason.value for reason in legacy.reason_codes),
            governed_validator_disagreement=valid != legacy.valid,
        )


TR_PROFILE = TrCountryProfile()
