"""Small deterministic helpers for owner-provided detector example inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
ANNOTATION_EXTENSIONS = frozenset({".txt", ".xml", ".json"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_images(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read-only recursive inventory with explicit unsupported-file reporting."""
    images: list[dict[str, Any]] = []
    unsupported: list[str] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            unsupported.append(relative)
            continue
        image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        images.append(
            {
                "source_relative_path": relative,
                "source_sha256": sha256_file(path),
                "decode_status": "VALID" if image is not None else "DECODE_FAILURE",
                "width": int(image.shape[1]) if image is not None else None,
                "height": int(image.shape[0]) if image is not None else None,
            }
        )
    return images, unsupported


def possible_annotation_files(root: Path) -> list[str]:
    """Report possible sidecars without treating them as authoritative ground truth."""
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ANNOTATION_EXTENSIONS
    )


def review_flags(
    *,
    detection_count: int,
    scores: list[float],
    crop_width: int | None,
    crop_height: int | None,
    touches_boundary: bool,
) -> list[str]:
    """Objective flags only; they never claim a detection is visually wrong."""
    flags: list[str] = []
    if detection_count == 0:
        return ["NO_DETECTION"]
    if scores[0] < 0.25:
        flags.append("LOW_TOP1_CONFIDENCE")
    if detection_count > 1:
        flags.append("MULTIPLE_SURVIVING_DETECTIONS")
        if len(scores) > 1 and scores[1] >= 0.50:
            flags.append("MULTIPLE_STRONG_CANDIDATES")
    if crop_width is not None and crop_height is not None:
        if crop_width < 16 or crop_height < 8:
            flags.append("UNUSUALLY_SMALL_SELECTED_CROP")
        if crop_width / crop_height > 12 or crop_width / crop_height < 1.5:
            flags.append("UNUSUAL_SELECTED_CROP_ASPECT_RATIO")
    if touches_boundary:
        flags.append("TOP1_BBOX_TOUCHES_IMAGE_BOUNDARY")
    return flags
