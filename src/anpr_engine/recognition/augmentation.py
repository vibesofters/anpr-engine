"""Deterministic detector/camera-aware Recognition augmentation with label safeguards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field


class DetectorAwareAugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    probability: float = Field(default=0.85, ge=0.0, le=1.0)
    max_translation_fraction: float = Field(default=0.025, ge=0.0, le=0.05)
    minimum_scale: float = Field(default=0.96, ge=0.90, le=1.0)
    maximum_tight_scale: float = Field(default=1.02, ge=1.0, le=1.05)
    max_rotation_degrees: float = Field(default=2.0, ge=0.0, le=4.0)
    max_perspective_fraction: float = Field(default=0.0125, ge=0.0, le=0.03)
    blur_probability: float = Field(default=0.20, ge=0.0, le=1.0)
    motion_blur_probability: float = Field(default=0.10, ge=0.0, le=1.0)
    brightness_delta: float = Field(default=0.16, ge=0.0, le=0.30)
    contrast_delta: float = Field(default=0.14, ge=0.0, le=0.30)
    jpeg_quality_minimum: int = Field(default=78, ge=60, le=100)
    degradation_scale_minimum: float = Field(default=0.72, ge=0.5, le=1.0)
    noise_stddev_maximum: float = Field(default=3.5, ge=0.0, le=8.0)
    glare_alpha_maximum: float = Field(default=0.10, ge=0.0, le=0.20)
    horizontal_flip: bool = False
    vertical_flip: bool = False


@dataclass(frozen=True)
class AugmentationResult:
    image: npt.NDArray[np.uint8]
    metadata: dict[str, object]
    label_preserved: bool


def _content_corners(
    bbox: tuple[float, float, float, float], matrix: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    x1, y1, x2, y2 = bbox
    points = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    return cast(npt.NDArray[np.float32], cv2.perspectiveTransform(points, matrix)[0])


def augment_detector_crop(
    image: npt.NDArray[np.uint8],
    *,
    seed: int,
    config: DetectorAwareAugmentationConfig,
    safe_content_bbox: tuple[float, float, float, float] | None = None,
) -> AugmentationResult:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("augmentation input must be uint8 HWC RGB")
    if config.horizontal_flip or config.vertical_flip:
        raise ValueError("Recognition augmentation forbids horizontal and vertical flips")
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    if float(rng.random()) > config.probability:
        return AugmentationResult(
            image=np.ascontiguousarray(image.copy()),
            metadata={"applied": False, "seed": seed, "horizontal_flip": False},
            label_preserved=True,
        )

    maximum_scale = config.maximum_tight_scale if safe_content_bbox is not None else 1.0
    scale = float(rng.uniform(config.minimum_scale, maximum_scale))
    angle = float(rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees))
    tx = float(
        rng.uniform(-config.max_translation_fraction, config.max_translation_fraction) * width
    )
    ty = float(
        rng.uniform(-config.max_translation_fraction, config.max_translation_fraction) * height
    )
    affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    affine[:, 2] = affine[:, 2] + np.asarray((tx, ty), dtype=np.float64)
    matrix = np.vstack((affine, np.array([0.0, 0.0, 1.0]))).astype(np.float32)
    perspective = config.max_perspective_fraction * min(width, height)
    offsets = rng.uniform(-perspective, perspective, size=(4, 2)).astype(np.float32)
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    perspective_matrix = cv2.getPerspectiveTransform(source, source + offsets)
    matrix = perspective_matrix @ matrix
    if safe_content_bbox is not None:
        corners = _content_corners(safe_content_bbox, matrix)
        if (
            corners[:, 0].min() < 0
            or corners[:, 1].min() < 0
            or corners[:, 0].max() >= width
            or corners[:, 1].max() >= height
        ):
            matrix = np.eye(3, dtype=np.float32)
            scale, angle, tx, ty, perspective = 1.0, 0.0, 0.0, 0.0, 0.0
    transformed = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    contrast = float(rng.uniform(1.0 - config.contrast_delta, 1.0 + config.contrast_delta))
    brightness = float(rng.uniform(-config.brightness_delta, config.brightness_delta) * 255.0)
    transformed = np.asarray(
        np.clip(transformed.astype(np.float32) * contrast + brightness, 0, 255),
        dtype=np.float32,
    )
    if float(rng.random()) < config.blur_probability:
        transformed = cv2.GaussianBlur(transformed, (3, 3), sigmaX=0.6)
    if float(rng.random()) < config.motion_blur_probability:
        kernel = np.zeros((3, 3), dtype=np.float32)
        kernel[1, :] = 1.0 / 3.0
        transformed = cv2.filter2D(transformed, -1, kernel)
    degradation = float(rng.uniform(config.degradation_scale_minimum, 1.0))
    if degradation < 0.995:
        small = cv2.resize(
            transformed,
            (max(1, round(width * degradation)), max(1, round(height * degradation))),
            interpolation=cv2.INTER_AREA,
        )
        transformed = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    noise_stddev = float(rng.uniform(0.0, config.noise_stddev_maximum))
    transformed = transformed + rng.normal(0.0, noise_stddev, transformed.shape).astype(np.float32)
    glare_alpha = float(rng.uniform(0.0, config.glare_alpha_maximum))
    if glare_alpha:
        glare = np.zeros_like(transformed)
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.circle(glare, center, max(2, height // 4), (255, 255, 255), thickness=-1)
        transformed = cv2.addWeighted(transformed, 1.0, glare, glare_alpha, 0.0)
    quality = int(rng.integers(config.jpeg_quality_minimum, 101))
    encoded, buffer = cv2.imencode(
        ".jpg", np.clip(transformed, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not encoded:
        raise RuntimeError("JPEG augmentation encoding failed")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG augmentation decoding failed")
    return AugmentationResult(
        image=np.ascontiguousarray(decoded),
        metadata={
            "applied": True,
            "seed": seed,
            "translation_pixels": [tx, ty],
            "scale": scale,
            "rotation_degrees": angle,
            "perspective_bound_pixels": perspective,
            "contrast": contrast,
            "brightness_delta_pixels": brightness,
            "degradation_scale": degradation,
            "noise_stddev": noise_stddev,
            "glare_alpha": glare_alpha,
            "jpeg_quality": quality,
            "horizontal_flip": False,
            "vertical_flip": False,
            "tight_crop_requires_safe_content_bbox": True,
        },
        label_preserved=True,
    )
