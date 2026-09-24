"""Detector-to-recognizer crop, resize, padding, and lineage contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from anpr_engine.domain import BBoxXYXY, CropReference
from anpr_engine.recognition.config import CropConfig


class InvalidCropError(ValueError):
    """Raised when detector geometry cannot produce a usable recognizer crop."""


@dataclass(frozen=True)
class PlateCrop:
    rgb: npt.NDArray[np.uint8]
    prepared_rgb: npt.NDArray[np.uint8]
    reference: CropReference


def crop_config_signature(config: CropConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"crop-v1-sha256-{hashlib.sha256(payload).hexdigest()}"


def generate_plate_crop(
    image: npt.NDArray[np.uint8],
    bbox: BBoxXYXY,
    *,
    source_image_id: str,
    config: CropConfig,
) -> PlateCrop:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise InvalidCropError("crop input must be a uint8 three-channel image")
    image_height, image_width = image.shape[:2]
    try:
        pixel_crop = bbox.to_pixel_crop(
            image_width=image_width,
            image_height=image_height,
            padding_x=config.padding_x,
            padding_y=config.padding_y,
        )
    except ValueError as error:
        raise InvalidCropError(str(error)) from error
    crop = image[pixel_crop.y1 : pixel_crop.y2, pixel_crop.x1 : pixel_crop.x2]
    if crop.shape[1] < config.minimum_width or crop.shape[0] < config.minimum_height:
        raise InvalidCropError(
            f"crop is smaller than minimum {config.minimum_width}x{config.minimum_height}"
        )
    rgb = (
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if config.input_color_space == "BGR" else crop.copy()
    )
    rgb = cast(npt.NDArray[np.uint8], rgb)
    prepared = _aspect_fit(rgb, config)
    signature = crop_config_signature(config)
    identity_payload = json.dumps(
        {
            "source_image_id": source_image_id,
            "source_bbox": bbox.model_dump(mode="json"),
            "pixel_crop": pixel_crop.model_dump(mode="json"),
            "preprocessing_signature": signature,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    crop_id = f"crop-sha256-{hashlib.sha256(identity_payload).hexdigest()}"
    reference = CropReference(
        crop_id=crop_id,
        pixel_crop=pixel_crop,
        source_bbox=bbox,
        preprocessing_signature=signature,
    )
    return PlateCrop(
        rgb=np.ascontiguousarray(rgb),
        prepared_rgb=np.ascontiguousarray(prepared),
        reference=reference,
    )


def _aspect_fit(image: npt.NDArray[np.uint8], config: CropConfig) -> npt.NDArray[np.uint8]:
    height, width = image.shape[:2]
    scale = min(config.target_width / width, config.target_height / height)
    resized_width = max(1, min(config.target_width, round(width * scale)))
    resized_height = max(1, min(config.target_height, round(height * scale)))
    interpolation = cv2.INTER_AREA if config.interpolation == "area" else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full(
        (config.target_height, config.target_width, 3),
        config.pad_value,
        dtype=np.uint8,
    )
    y_offset = (config.target_height - resized_height) // 2
    x_offset = (
        (config.target_width - resized_width) // 2 if config.padding_strategy == "center" else 0
    )
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return canvas
