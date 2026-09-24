"""Immutable shared value contracts for inference consumers and adapters."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Confidence = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
DurationMilliseconds = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveInteger = Annotated[int, Field(gt=0)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SchemaVersion(DomainModel):
    major: Annotated[int, Field(ge=1)]
    minor: Annotated[int, Field(ge=0)]

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


class ArtifactIdentity(DomainModel):
    name: Annotated[str, Field(min_length=1)]
    version: Annotated[str, Field(min_length=1)]
    sha256: Sha256
    reference: str | None = None


class ImageRef(DomainModel):
    request_id: Annotated[str, Field(min_length=1)]
    relative_path: Annotated[str, Field(min_length=1)]
    extension: Literal[".jpg", ".jpeg", ".png"]
    content_sha256: Sha256 | None = None

    @field_validator("relative_path")
    @classmethod
    def relative_path_must_be_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must remain within the configured input root")
        return value


class ImageShape(DomainModel):
    width: PositiveInteger
    height: PositiveInteger
    channels: PositiveInteger
    dtype: Annotated[str, Field(min_length=1)]
    color_space: Annotated[str, Field(min_length=1)]


class PixelCrop(DomainModel):
    x1: Annotated[int, Field(ge=0)]
    y1: Annotated[int, Field(ge=0)]
    x2: PositiveInteger
    y2: PositiveInteger

    @model_validator(mode="after")
    def validate_nonempty(self) -> Self:
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("pixel crop must have positive width and height")
        return self


class BBoxXYXY(DomainModel):
    """Absolute half-open coordinates: [x1, x2) x [y1, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(coordinate) for coordinate in coordinates):
            raise ValueError("bbox coordinates must be finite")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("bbox must have positive width and height")
        return self

    def to_pixel_crop(
        self,
        *,
        image_width: int,
        image_height: int,
        padding_x: float = 0.0,
        padding_y: float = 0.0,
    ) -> PixelCrop:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if not math.isfinite(padding_x) or not math.isfinite(padding_y):
            raise ValueError("bbox padding must be finite")
        if padding_x < 0 or padding_y < 0:
            raise ValueError("bbox padding must be non-negative")

        x1 = max(0.0, min(float(image_width), self.x1 - padding_x))
        y1 = max(0.0, min(float(image_height), self.y1 - padding_y))
        x2 = max(0.0, min(float(image_width), self.x2 + padding_x))
        y2 = max(0.0, min(float(image_height), self.y2 + padding_y))
        return PixelCrop(
            x1=math.floor(x1),
            y1=math.floor(y1),
            x2=math.ceil(x2),
            y2=math.ceil(y2),
        )


class DetectionCandidate(DomainModel):
    bbox: BBoxXYXY
    detection_confidence: Confidence
    class_id: Literal["license_plate"] = "license_plate"
    detector: ArtifactIdentity


class RecognitionResult(DomainModel):
    raw_text: str
    normalized_text: str | None = None
    recognition_confidence: Confidence | None = None
    decoded_text: str | None = None
    decoder_confidence: Confidence | None = None
    profile_id: str | None = None
    decoding_changed_raw: bool = False
    structural_validation: dict[str, Any] | None = None
    decoder_status: Literal["ACCEPTED", "LOW_CONFIDENCE"] | None = None
    decoder_reason: str | None = None
    recognizer: ArtifactIdentity
    decoder_version: Annotated[str, Field(min_length=1)]
    charset_version: Annotated[str, Field(min_length=1)]
    preprocessing_signature: Annotated[str, Field(min_length=1)]


class PlateValidationReason(StrEnum):
    EMPTY = "EMPTY"
    UNSUPPORTED_CHARACTER = "UNSUPPORTED_CHARACTER"
    INVALID_LENGTH = "INVALID_LENGTH"
    INVALID_PROVINCE = "INVALID_PROVINCE"
    INVALID_STRUCTURE = "INVALID_STRUCTURE"


class PlateValidationResult(DomainModel):
    valid: bool
    rule_version: Annotated[str, Field(min_length=1)]
    reason_codes: tuple[PlateValidationReason, ...] = ()

    @model_validator(mode="after")
    def reasons_must_match_validity(self) -> Self:
        if self.valid and self.reason_codes:
            raise ValueError("valid plates cannot carry validation failure reasons")
        if not self.valid and not self.reason_codes:
            raise ValueError("invalid plates require at least one validation reason")
        return self


class CropReference(DomainModel):
    crop_id: Annotated[str, Field(min_length=1)]
    pixel_crop: PixelCrop
    source_bbox: BBoxXYXY
    preprocessing_signature: Annotated[str, Field(min_length=1)]
    relative_path: str | None = None

    @field_validator("relative_path")
    @classmethod
    def relative_path_must_be_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("crop relative_path must remain within the output root")
        return value


class DebugArtifacts(DomainModel):
    crop_path: str | None = None
    annotated_image_path: str | None = None

    @field_validator("crop_path", "annotated_image_path")
    @classmethod
    def paths_must_be_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("artifact paths must be portable relative paths")
        return value


class ResultStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    FAILED = "FAILED"


class ProcessingStage(StrEnum):
    INPUT = "input"
    DETECTION = "detection"
    CROP = "crop"
    RECOGNITION = "recognition"
    VALIDATION = "validation"
    OUTPUT = "output"


class ErrorCode(StrEnum):
    IMAGE_DECODE_ERROR = "IMAGE_DECODE_ERROR"
    DETECTION_MISS = "DETECTION_MISS"
    MULTIPLE_DETECTIONS = "MULTIPLE_DETECTIONS"
    INVALID_CROP = "INVALID_CROP"
    RECOGNIZER_ERROR = "RECOGNIZER_ERROR"
    PLATE_VALIDATION_FAILED = "PLATE_VALIDATION_FAILED"
    OUTPUT_WRITE_ERROR = "OUTPUT_WRITE_ERROR"


class EngineError(DomainModel):
    stage: ProcessingStage
    code: ErrorCode
    message: Annotated[str, Field(min_length=1)]
    recoverable: bool
    context: dict[str, Any] = Field(default_factory=dict)


class StageTimings(DomainModel):
    detection_ms: DurationMilliseconds | None = None
    preprocessing_ms: DurationMilliseconds | None = None
    recognition_ms: DurationMilliseconds | None = None
    postprocessing_ms: DurationMilliseconds | None = None
    serialization_ms: DurationMilliseconds | None = None
    total_ms: DurationMilliseconds


class InferenceResult(DomainModel):
    schema_version: SchemaVersion
    request_id: Annotated[str, Field(min_length=1)]
    run_id: str | None = None
    created_at: datetime
    image: ImageRef
    image_shape: ImageShape | None = None
    status: ResultStatus
    reasons: tuple[str, ...] = ()
    detection: DetectionCandidate | None = None
    crop: CropReference | None = None
    recognition: RecognitionResult | None = None
    validation: PlateValidationResult | None = None
    artifacts: DebugArtifacts | None = None
    errors: tuple[EngineError, ...] = ()
    timings: StageTimings

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a UTC offset")
        if value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("created_at must use UTC")
        return value
