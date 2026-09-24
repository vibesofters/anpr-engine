"""Recognition-side end-to-end skeleton using replaceable detector/recognizer ports."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
import numpy.typing as npt

from anpr_engine.domain import (
    CropReference,
    DetectionCandidate,
    EngineError,
    ErrorCode,
    ImageRef,
    ImageShape,
    InferenceResult,
    PlateValidationReason,
    PlateValidationResult,
    ProcessingStage,
    ResultStatus,
    SchemaVersion,
    StageTimings,
)
from anpr_engine.recognition.config import CropConfig, DecisionConfig
from anpr_engine.recognition.crop import InvalidCropError, generate_plate_crop
from anpr_engine.recognition.interfaces import Recognizer
from anpr_engine.recognition.normalization import PlateNormalizationError, normalize_plate_text
from anpr_engine.recognition.validation import validate_turkish_plate

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


class Detector(Protocol):
    def detect(self, image: npt.NDArray[np.uint8]) -> tuple[DetectionCandidate, ...]: ...


class RecognitionPipeline:
    def __init__(
        self,
        *,
        detector: Detector,
        recognizer: Recognizer,
        crop_config: CropConfig,
        decision_config: DecisionConfig,
    ) -> None:
        self._detector = detector
        self._recognizer = recognizer
        self._crop_config = crop_config
        self._decision_config = decision_config

    def process_file(self, path: Path, *, input_root: Path, request_id: str) -> InferenceResult:
        resolved_root = input_root.resolve()
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError("input image must remain within input_root")
        extension = resolved_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"unsupported image extension: {extension}")
        image_ref = ImageRef.model_validate(
            {
                "request_id": request_id,
                "relative_path": resolved_path.relative_to(resolved_root).as_posix(),
                "extension": extension,
                "content_sha256": hashlib.sha256(resolved_path.read_bytes()).hexdigest(),
            }
        )
        decoded = cv2.imread(resolved_path.as_posix(), cv2.IMREAD_COLOR)
        if decoded is None:
            return self._failed(
                image_ref,
                time.perf_counter(),
                stage=ProcessingStage.INPUT,
                code=ErrorCode.IMAGE_DECODE_ERROR,
                message="image decoding failed",
            )
        return self.process(cast(npt.NDArray[np.uint8], decoded), image_ref)

    def process(self, image: npt.NDArray[np.uint8], image_ref: ImageRef) -> InferenceResult:
        started = time.perf_counter()
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            return self._failed(
                image_ref,
                started,
                stage=ProcessingStage.INPUT,
                code=ErrorCode.IMAGE_DECODE_ERROR,
                message="decoded image must be uint8 BGR with three channels",
            )
        shape = ImageShape(
            width=image.shape[1],
            height=image.shape[0],
            channels=3,
            dtype="uint8",
            color_space="BGR",
        )
        detection_started = time.perf_counter()
        detections = self._detector.detect(image)
        detection_ms = _elapsed_ms(detection_started)
        if not detections:
            return self._failed(
                image_ref,
                started,
                image_shape=shape,
                stage=ProcessingStage.DETECTION,
                code=ErrorCode.DETECTION_MISS,
                message="no license plate candidate detected",
                detection_ms=detection_ms,
            )
        if len(detections) != 1:
            return self._failed(
                image_ref,
                started,
                image_shape=shape,
                stage=ProcessingStage.DETECTION,
                code=ErrorCode.MULTIPLE_DETECTIONS,
                message=f"expected one plate candidate, received {len(detections)}",
                detection_ms=detection_ms,
            )
        detection = detections[0]
        preprocess_started = time.perf_counter()
        try:
            crop = generate_plate_crop(
                image,
                detection.bbox,
                source_image_id=image_ref.request_id,
                config=self._crop_config,
            )
        except InvalidCropError as error:
            return self._failed(
                image_ref,
                started,
                image_shape=shape,
                detection=detection,
                stage=ProcessingStage.CROP,
                code=ErrorCode.INVALID_CROP,
                message=str(error),
                detection_ms=detection_ms,
                preprocessing_ms=_elapsed_ms(preprocess_started),
            )
        preprocessing_ms = _elapsed_ms(preprocess_started)
        recognition_started = time.perf_counter()
        try:
            recognition = self._recognizer.recognize(crop)
        except Exception as error:  # adapter boundary maps runtime failures to the domain code
            return self._failed(
                image_ref,
                started,
                image_shape=shape,
                detection=detection,
                crop=crop.reference,
                stage=ProcessingStage.RECOGNITION,
                code=ErrorCode.RECOGNIZER_ERROR,
                message=f"recognizer failed: {type(error).__name__}",
                detection_ms=detection_ms,
                preprocessing_ms=preprocessing_ms,
                recognition_ms=_elapsed_ms(recognition_started),
            )
        recognition_ms = _elapsed_ms(recognition_started)
        post_started = time.perf_counter()
        candidate_text = recognition.decoded_text or recognition.raw_text
        try:
            normalized = normalize_plate_text(candidate_text)
        except PlateNormalizationError:
            validation = _profile_validation(recognition.structural_validation) or (
                validate_turkish_plate(candidate_text)
            )
        else:
            recognition = recognition.model_copy(
                update={"normalized_text": normalized.normalized_text}
            )
            validation = _profile_validation(recognition.structural_validation) or (
                validate_turkish_plate(normalized.normalized_text)
            )
        postprocessing_ms = _elapsed_ms(post_started)
        errors: tuple[EngineError, ...]
        if recognition.decoder_status == ResultStatus.LOW_CONFIDENCE.value:
            status = ResultStatus.LOW_CONFIDENCE
            errors = ()
        elif not validation.valid:
            status = ResultStatus.FAILED
            errors = (
                EngineError(
                    stage=ProcessingStage.VALIDATION,
                    code=ErrorCode.PLATE_VALIDATION_FAILED,
                    message="recognized text failed Turkish plate validation",
                    recoverable=True,
                    context={"reason_codes": [reason.value for reason in validation.reason_codes]},
                ),
            )
        elif recognition.decoder_status == ResultStatus.ACCEPTED.value:
            status = ResultStatus.ACCEPTED
            errors = ()
        else:
            threshold = self._decision_config.acceptance_recognition_confidence
            accepted = (
                threshold is not None
                and recognition.recognition_confidence is not None
                and recognition.recognition_confidence >= threshold
            )
            status = ResultStatus.ACCEPTED if accepted else ResultStatus.LOW_CONFIDENCE
            errors = ()
        return InferenceResult(
            schema_version=SchemaVersion(major=1, minor=1),
            request_id=image_ref.request_id,
            created_at=datetime.now(UTC),
            image=image_ref,
            image_shape=shape,
            status=status,
            reasons=tuple(reason.value for reason in validation.reason_codes),
            detection=detection,
            crop=crop.reference,
            recognition=recognition,
            validation=validation,
            errors=errors,
            timings=StageTimings(
                detection_ms=detection_ms,
                preprocessing_ms=preprocessing_ms,
                recognition_ms=recognition_ms,
                postprocessing_ms=postprocessing_ms,
                total_ms=_elapsed_ms(started),
            ),
        )

    @staticmethod
    def _failed(
        image_ref: ImageRef,
        started: float,
        *,
        stage: ProcessingStage,
        code: ErrorCode,
        message: str,
        image_shape: ImageShape | None = None,
        detection: DetectionCandidate | None = None,
        crop: CropReference | None = None,
        detection_ms: float | None = None,
        preprocessing_ms: float | None = None,
        recognition_ms: float | None = None,
    ) -> InferenceResult:
        return InferenceResult(
            schema_version=SchemaVersion(major=1, minor=1),
            request_id=image_ref.request_id,
            created_at=datetime.now(UTC),
            image=image_ref,
            image_shape=image_shape,
            status=ResultStatus.FAILED,
            reasons=(code.value,),
            detection=detection,
            crop=crop,
            errors=(
                EngineError(
                    stage=stage,
                    code=code,
                    message=message,
                    recoverable=True,
                ),
            ),
            timings=StageTimings(
                detection_ms=detection_ms,
                preprocessing_ms=preprocessing_ms,
                recognition_ms=recognition_ms,
                total_ms=_elapsed_ms(started),
            ),
        )


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def _profile_validation(payload: dict[str, object] | None) -> PlateValidationResult | None:
    """Bridge a decoder profile result without reapplying the broader legacy grammar."""
    if payload is None or payload.get("profile_id") != "TR":
        return None
    reason_mapping = {
        "EMPTY": PlateValidationReason.EMPTY,
        "UNSUPPORTED_CHARACTER": PlateValidationReason.UNSUPPORTED_CHARACTER,
        "INVALID_PROVINCE": PlateValidationReason.INVALID_PROVINCE,
        "UNSUPPORTED_TR_FORMAT": PlateValidationReason.INVALID_STRUCTURE,
    }
    raw_reasons = payload.get("reason_codes", [])
    if not isinstance(raw_reasons, list):
        raise ValueError("profile reason_codes must be a list")
    reasons = tuple(reason_mapping[str(reason)] for reason in raw_reasons)
    return PlateValidationResult(
        valid=bool(payload.get("valid")),
        rule_version=str(payload.get("profile_version", "TR-profile-unknown")),
        reason_codes=reasons,
    )


def discover_images(input_root: Path) -> tuple[Path, ...]:
    if not input_root.is_dir():
        raise ValueError("input_root must be a directory")
    return tuple(
        candidate
        for candidate in sorted(input_root.iterdir(), key=lambda path: path.name.casefold())
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
    )
