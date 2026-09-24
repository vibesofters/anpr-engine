from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from anpr_engine.domain import (
    ArtifactIdentity,
    BBoxXYXY,
    DetectionCandidate,
    EngineError,
    ErrorCode,
    ImageRef,
    ImageShape,
    InferenceResult,
    ProcessingStage,
    ResultStatus,
    SchemaVersion,
    StageTimings,
)

SHA256 = "a" * 64


def _artifact() -> ArtifactIdentity:
    return ArtifactIdentity(name="fixture-detector", version="1.0", sha256=SHA256)


def test_bbox_uses_half_open_floor_ceil_clamp_and_padding() -> None:
    bbox = BBoxXYXY(x1=-0.2, y1=4.8, x2=99.2, y2=80.1)

    crop = bbox.to_pixel_crop(
        image_width=100,
        image_height=80,
        padding_x=2.0,
        padding_y=1.0,
    )

    assert crop.model_dump() == {"x1": 0, "y1": 3, "x2": 100, "y2": 80}


@pytest.mark.parametrize(
    "coordinates",
    [
        {"x1": 1.0, "y1": 1.0, "x2": 1.0, "y2": 2.0},
        {"x1": 1.0, "y1": 3.0, "x2": 2.0, "y2": 2.0},
        {"x1": float("nan"), "y1": 1.0, "x2": 2.0, "y2": 2.0},
    ],
)
def test_bbox_rejects_empty_inverted_or_nonfinite_values(coordinates: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        BBoxXYXY(**coordinates)


def test_detection_confidence_is_bounded_and_contract_is_immutable() -> None:
    with pytest.raises(ValidationError):
        DetectionCandidate(
            bbox=BBoxXYXY(x1=0, y1=0, x2=1, y2=1),
            detection_confidence=1.01,
            detector=_artifact(),
        )

    bbox = BBoxXYXY(x1=0, y1=0, x2=1, y2=1)
    with pytest.raises(ValidationError):
        bbox.x1 = 2


def test_result_schema_round_trips_without_vendor_types() -> None:
    image = ImageRef(request_id="request-1", relative_path="safe/example.jpg", extension=".jpg")
    result = InferenceResult(
        schema_version=SchemaVersion(major=1, minor=0),
        request_id=image.request_id,
        run_id="run-1",
        created_at=datetime(2026, 8, 22, 12, 0, tzinfo=UTC),
        image=image,
        image_shape=ImageShape(
            width=1920,
            height=1080,
            channels=3,
            dtype="uint8",
            color_space="RGB",
        ),
        status=ResultStatus.FAILED,
        reasons=("no detection",),
        errors=(
            EngineError(
                stage=ProcessingStage.DETECTION,
                code=ErrorCode.DETECTION_MISS,
                message="No license plate candidate was detected",
                recoverable=True,
            ),
        ),
        timings=StageTimings(detection_ms=5.25, total_ms=5.5),
    )

    restored = InferenceResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.detection is None
    assert restored.recognition is None


def test_image_paths_and_timestamps_must_be_safe() -> None:
    with pytest.raises(ValidationError):
        ImageRef(request_id="request-1", relative_path="../private.jpg", extension=".jpg")

    with pytest.raises(ValidationError):
        InferenceResult(
            schema_version=SchemaVersion(major=1, minor=0),
            request_id="request-1",
            created_at=datetime(2026, 8, 22, 12, 0),
            image=ImageRef(request_id="request-1", relative_path="safe.jpg", extension=".jpg"),
            status=ResultStatus.FAILED,
            timings=StageTimings(total_ms=1.0),
        )
