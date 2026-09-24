"""Versioned Turkish ANPR V1 model character set."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

V1_SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
V1_CHARSET_VERSION = "turkish-plate-ctc-v1"


class UnsupportedCharacterError(ValueError):
    """Raised instead of silently replacing an unsupported label character."""


class PlateCharset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    symbols: str = Field(min_length=1)
    blank_token: str = "<CTC_BLANK>"
    blank_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_mapping(self) -> PlateCharset:
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("charset symbols must be unique")
        if any(len(character) != 1 for character in self.symbols):
            raise ValueError("charset symbols must contain single characters")
        if self.blank_index > len(self.symbols):
            raise ValueError("blank_index must fit within the CTC class mapping")
        if self.blank_token in self.symbols:
            raise ValueError("blank token cannot be a model symbol")
        return self

    @property
    def class_count(self) -> int:
        return len(self.symbols) + 1

    @property
    def char_to_index(self) -> dict[str, int]:
        indexes = [index for index in range(self.class_count) if index != self.blank_index]
        return dict(zip(self.symbols, indexes, strict=True))

    @property
    def index_to_char(self) -> dict[int, str]:
        return {index: character for character, index in self.char_to_index.items()}

    def encode(self, text: str) -> tuple[int, ...]:
        mapping = self.char_to_index
        unsupported = tuple(character for character in text if character not in mapping)
        if unsupported:
            rendered = "".join(dict.fromkeys(unsupported))
            raise UnsupportedCharacterError(f"unsupported character(s): {rendered!r}")
        return tuple(mapping[character] for character in text)

    def decode_indices(self, indexes: Iterable[int]) -> str:
        mapping = self.index_to_char
        characters: list[str] = []
        for index in indexes:
            if index == self.blank_index:
                raise ValueError("blank index cannot be converted directly to text")
            try:
                characters.append(mapping[index])
            except KeyError as error:
                raise ValueError(f"invalid charset index: {index}") from error
        return "".join(characters)

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


V1_CHARSET = PlateCharset(
    version=V1_CHARSET_VERSION,
    symbols=V1_SYMBOLS,
    blank_index=0,
)
