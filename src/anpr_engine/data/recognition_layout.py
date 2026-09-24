"""Neutral manifest records for small project-owned synthetic fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RECOGNITION_LAYOUT_VERSION = "recognition-images-labels-layout-v1"


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value in {"", "."}:
        raise ValueError("path must be a non-empty portable relative POSIX path")
    return value


class RecognitionLayoutRecord(BaseModel):
    """One generated image/label pair in a portable synthetic-fixture layout."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["recognition-images-labels-layout-v1"] = (
        "recognition-images-labels-layout-v1"
    )
    sample_id: int = Field(gt=0)
    derived_image_path: str
    derived_label_path: str
    source_id: str
    source_version: str
    raw_image_relative_path: str
    raw_filename: str
    raw_transcription: str
    normalized_transcription: str = Field(pattern=r"^[0-9A-Z]+$")
    raw_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derived_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_license_verification: str
    source_provenance_state: str
    training_authorization_state: str
    group_id: str
    transformation_metadata: dict[str, object] | None = None

    @field_validator("derived_image_path", "derived_label_path", "raw_image_relative_path")
    @classmethod
    def paths_are_portable(cls, value: str) -> str:
        return _relative_path(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
