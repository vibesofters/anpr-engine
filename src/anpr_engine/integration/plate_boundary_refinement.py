"""Image-evidence-only plate boundary refinement between Detection and Recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

METHOD_ID = "BLUE_BAND_GLYPH_QUAD_V1"


@dataclass(frozen=True)
class PlateBoundaryRefinement:
    rgb: NDArray[np.uint8]
    applied: bool
    provenance: dict[str, Any]


def _fallback(
    rgb: NDArray[np.uint8], reason: str, details: dict[str, Any]
) -> PlateBoundaryRefinement:
    return PlateBoundaryRefinement(
        rgb=np.ascontiguousarray(rgb.copy()),
        applied=False,
        provenance={
            "schema_version": "detector-recognizer-boundary-refinement-v1",
            "method_id": METHOD_ID,
            "status": "FALLBACK_ORIGINAL_DETECTION_CROP",
            "fallback_reason": reason,
            "runtime_selection_inputs": ["PIXELS", "DETECTION_CROP_GEOMETRY"],
            "forbidden_selection_inputs_used": [],
            **details,
        },
    )


def _character_components(gray: NDArray[np.uint8]) -> list[tuple[int, int, int, int, int]]:
    height, width = gray.shape
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[index])
        if (
            0.35 * height <= component_height <= 0.90 * height
            and 0.006 * width <= component_width <= 0.18 * width
            and area > 0.0025 * width * height
            and y < 0.40 * height
        ):
            candidates.append((x, y, component_width, component_height, area))
    if not candidates:
        return []
    median_height = float(np.median([item[3] for item in candidates]))
    candidates = sorted(
        item for item in candidates if 0.80 * median_height <= item[3] <= 1.20 * median_height
    )
    components: list[tuple[int, int, int, int, int]] = []
    for item in candidates:
        if components and item[0] < components[-1][0] + components[-1][2] * 0.60:
            if item[4] > components[-1][4]:
                components[-1] = item
        else:
            components.append(item)
    return components


def _blue_band(
    rgb: NDArray[np.uint8], components: list[tuple[int, int, int, int, int]]
) -> tuple[int, int, int, int, int] | None:
    height, width = rgb.shape[:2]
    median_height = float(np.median([item[3] for item in components]))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.asarray([85, 45, 25]), np.asarray([140, 255, 255]))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [
        tuple(map(int, stats[index]))
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] > 0.01 * width * height
        and stats[index, cv2.CC_STAT_LEFT] < components[0][0]
        and stats[index, cv2.CC_STAT_HEIGHT] > 0.40 * median_height
    ]
    return (
        cast(tuple[int, int, int, int, int], max(candidates, key=lambda item: item[4]))
        if candidates
        else None
    )


def refine_plate_boundary(rgb: NDArray[np.uint8]) -> PlateBoundaryRefinement:
    """Rectify a plate from visual components, or return the original crop unchanged.

    Constants are physical border allowances expressed relative to the median glyph height.
    Runtime selection never consumes text, grammar, Recognition output/confidence, or identity.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("boundary refinement requires uint8 RGB")
    height, width = rgb.shape[:2]
    if width < 80 or height < 20:
        return _fallback(rgb, "CROP_TOO_SMALL", {"input_dimensions": [width, height]})
    gray = cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    components = _character_components(gray)
    if not 4 <= len(components) <= 10:
        return _fallback(
            rgb,
            "CHARACTER_COMPONENT_QUALITY",
            {"input_dimensions": [width, height], "character_component_count": len(components)},
        )
    band = _blue_band(rgb, components)
    if band is None:
        return _fallback(
            rgb,
            "TR_BAND_NOT_RELIABLY_LOCALIZED",
            {"input_dimensions": [width, height], "character_component_count": len(components)},
        )
    median_height = float(np.median([item[3] for item in components]))
    left = max(0, band[0] - round(0.10 * median_height))
    right = min(
        width - 1,
        max(item[0] + item[2] for item in components) + round(0.35 * median_height),
    )
    centers = np.asarray([item[0] + item[2] / 2 for item in components])
    tops = np.asarray([item[1] for item in components])
    bottoms = np.asarray([item[1] + item[3] for item in components])
    top_slope, top_intercept = np.polyfit(centers, tops, 1)
    bottom_slope, bottom_intercept = np.polyfit(centers, bottoms, 1)
    quad = np.asarray(
        [
            [left, top_slope * left + top_intercept - 0.10 * median_height],
            [right, top_slope * right + top_intercept - 0.10 * median_height],
            [right, bottom_slope * right + bottom_intercept + 0.30 * median_height],
            [left, bottom_slope * left + bottom_intercept + 0.30 * median_height],
        ],
        dtype=np.float32,
    )
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    output_width = round(
        (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    )
    output_height = round(
        (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    )
    aspect_ratio = output_width / max(output_height, 1)
    component_left = min(item[0] for item in components)
    component_right = max(item[0] + item[2] for item in components)
    character_region_width = component_right - component_left
    details: dict[str, Any] = {
        "input_dimensions": [width, height],
        "character_component_count": len(components),
        "median_character_height": median_height,
        "character_region_width_before_resize": character_region_width,
        "quad_in_detection_crop_xy": quad.tolist(),
        "tr_band_bbox_xywh": list(band[:4]),
        "output_dimensions": [output_width, output_height],
        "output_aspect_ratio": aspect_ratio,
        "top_rotation_degrees": float(np.degrees(np.arctan(top_slope))),
        "bottom_rotation_degrees": float(np.degrees(np.arctan(bottom_slope))),
    }
    if (
        not 3.5 <= aspect_ratio <= 6.5
        or output_width < 0.65 * width
        or output_height < 0.45 * height
        or left > band[0]
        or right <= component_right
    ):
        return _fallback(rgb, "GEOMETRIC_QUALITY_GATE", details)
    target = np.asarray(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, target)
    refined = cast(
        NDArray[np.uint8],
        cv2.warpPerspective(
            rgb,
            matrix,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ),
    )
    resize_scale = min(160 / output_width, 32 / output_height)
    details.update(
        {
            "character_region_width_after_3x32x160_resize": character_region_width * resize_scale,
            "homography": matrix.tolist(),
            "status": "APPLIED",
            "schema_version": "detector-recognizer-boundary-refinement-v1",
            "method_id": METHOD_ID,
            "runtime_selection_inputs": ["PIXELS", "DETECTION_CROP_GEOMETRY"],
            "forbidden_selection_inputs_used": [],
            "tr_band_preserved": True,
            "dealer_strip_exclusion_policy": "glyph-bottom plus 0.30 median-glyph-height",
        }
    )
    return PlateBoundaryRefinement(np.ascontiguousarray(refined), True, details)
