from __future__ import annotations

import inspect

import cv2
import numpy as np
import numpy.typing as npt

from anpr_engine.integration.plate_boundary_refinement import (
    METHOD_ID,
    refine_plate_boundary,
)


def _plate(*, slope: int = 0, include_band: bool = True) -> npt.NDArray[np.uint8]:
    image = np.full((140, 560, 3), 205, dtype=np.uint8)
    image[10:112, 12:515] = 245
    image[112:140] = 30  # holder/dealer material below the plate face
    if include_band:
        image[14:106, 18:62] = (20, 70, 200)  # RGB blue band
    for index, x in enumerate((78, 132, 220, 274, 368, 420, 470)):
        top = 22 + round(index * slope / 6)
        cv2.rectangle(image, (x, top), (x + 30, top + 70), (25, 25, 25), -1)
    return image


def test_right_overhang_and_bottom_holder_are_removed() -> None:
    source = _plate()
    result = refine_plate_boundary(source)
    assert result.applied
    assert result.rgb.shape[1] < source.shape[1]
    assert result.rgb.shape[0] < source.shape[0]
    assert result.provenance["dealer_strip_exclusion_policy"]


def test_tr_band_is_preserved_and_provenance_is_complete() -> None:
    result = refine_plate_boundary(_plate())
    hsv = cv2.cvtColor(result.rgb, cv2.COLOR_RGB2HSV)
    blue_pixels = cv2.inRange(hsv, np.asarray([85, 45, 25]), np.asarray([140, 255, 255]))
    assert int(np.count_nonzero(blue_pixels)) > 100
    assert result.provenance["tr_band_preserved"] is True
    assert result.provenance["method_id"] == METHOD_ID
    assert result.provenance["forbidden_selection_inputs_used"] == []
    assert len(result.provenance["homography"]) == 3


def test_perspective_is_measured_and_rectified() -> None:
    result = refine_plate_boundary(_plate(slope=12))
    assert result.applied
    assert abs(result.provenance["top_rotation_degrees"]) > 0.5
    assert 3.5 <= result.provenance["output_aspect_ratio"] <= 6.5


def test_failed_geometry_falls_back_byte_exact() -> None:
    source = _plate(include_band=False)
    result = refine_plate_boundary(source)
    assert not result.applied
    assert result.provenance["status"] == "FALLBACK_ORIGINAL_DETECTION_CROP"
    assert np.array_equal(result.rgb, source)


def test_runtime_contains_no_per_image_or_text_selection() -> None:
    source = inspect.getsource(refine_plate_boundary)
    assert ".jpg" not in source
    assert list(inspect.signature(refine_plate_boundary).parameters) == ["rgb"]
    assert all(term not in source for term in ("decoded_text", "plate_text", "vehicle_id"))
