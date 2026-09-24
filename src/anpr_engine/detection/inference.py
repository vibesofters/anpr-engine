"""User-testable image/folder inference with JSON, previews, crops, and debug evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from anpr_engine.detection.config import DetectorConfig
from anpr_engine.detection.data import letterbox
from anpr_engine.detection.model import decode_predictions
from anpr_engine.detection.training import (
    evaluation_config,
    load_checkpoint,
    resolve_device,
    sha256_file,
    verify_checkpoint_identity,
)
from anpr_engine.domain import ArtifactIdentity, BBoxXYXY, DetectionCandidate

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _inputs(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"unsupported image extension: {path.suffix}")
        return (path,)
    if path.is_dir():
        images = tuple(
            candidate
            for candidate in sorted(path.iterdir())
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            raise ValueError(f"input directory contains no supported images: {path}")
        return images
    raise ValueError(f"input path does not exist: {path}")


def detect_images(
    input_path: Path,
    output_root: Path,
    checkpoint_path: Path,
    config: DetectorConfig,
    *,
    confidence_threshold: float | None = None,
    nms_iou_threshold: float | None = None,
) -> Path:
    device = resolve_device(config.training.device)
    model, metadata = load_checkpoint(checkpoint_path, device)
    verify_checkpoint_identity(metadata, config)
    resolved_config = evaluation_config(
        config,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )
    identity = ArtifactIdentity(
        name=str(metadata["architecture"]).replace("_", "-"),
        version=str(metadata["run_id"]),
        sha256=sha256_file(checkpoint_path),
        reference=checkpoint_path.name,
    )
    annotated_root = output_root / "annotated"
    crop_root = output_root / "crops"
    debug_root = output_root / "debug"
    for directory in (annotated_root, crop_root, debug_root):
        directory.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for image_path in _inputs(input_path):
        decoded = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
        if decoded is None:
            results.append({"input": image_path.name, "error": "IMAGE_DECODE_ERROR"})
            continue
        image = cast(NDArray[np.uint8], decoded)
        prepared, transform = letterbox(image, resolved_config.model.image_size)
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        with torch.inference_mode():
            output = model(tensor)
        boxes, scores = decode_predictions(
            output,
            confidence_threshold=resolved_config.inference.confidence_threshold,
            nms_iou_threshold=resolved_config.inference.nms_iou_threshold,
            max_detections=resolved_config.inference.max_detections,
        )[0]
        annotated = image.copy()
        detections: list[DetectionCandidate] = []
        crop_paths: list[str] = []
        plate_index = 0
        for box, score in zip(boxes, scores, strict=True):
            x1, y1, x2, y2 = transform.to_original_xyxy(box)
            if x1 >= x2 or y1 >= y2:
                continue
            plate_index += 1
            candidate = DetectionCandidate(
                bbox=BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2),
                detection_confidence=float(score),
                detector=identity,
            )
            detections.append(candidate)
            crop = candidate.bbox.to_pixel_crop(
                image_width=image.shape[1],
                image_height=image.shape[0],
                padding_x=resolved_config.inference.crop_padding_x,
                padding_y=resolved_config.inference.crop_padding_y,
            )
            crop_path = crop_root / f"{image_path.stem}-plate-{plate_index:03d}.jpg"
            if not cv2.imwrite(crop_path.as_posix(), image[crop.y1 : crop.y2, crop.x1 : crop.x2]):
                raise OSError(f"failed to write plate crop: {crop_path}")
            crop_paths.append(crop_path.relative_to(output_root).as_posix())
            cv2.rectangle(
                annotated,
                (round(x1), round(y1)),
                (round(x2), round(y2)),
                (0, 220, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"license_plate {float(score):.3f}",
                (round(x1), max(18, round(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )
        annotated_path = annotated_root / f"{image_path.stem}-annotated.jpg"
        if not cv2.imwrite(annotated_path.as_posix(), annotated):
            raise OSError(f"failed to write annotated image: {annotated_path}")
        debug = {
            "input_shape": list(image.shape),
            "model_input_size": resolved_config.model.image_size,
            "letterbox_scale": transform.scale,
            "letterbox_pad_x": transform.pad_x,
            "letterbox_pad_y": transform.pad_y,
            "raw_grid_shapes": (
                [list(level.shape) for level in output]
                if isinstance(output, tuple)
                else [list(output.shape)]
            ),
        }
        debug_path = debug_root / f"{image_path.stem}.json"
        debug_path.write_text(json.dumps(debug, indent=2) + "\n", encoding="utf-8")
        results.append(
            {
                "input": image_path.name,
                "image_width": image.shape[1],
                "image_height": image.shape[0],
                "detections": [item.model_dump(mode="json") for item in detections],
                "annotated_image": annotated_path.relative_to(output_root).as_posix(),
                "plate_crops": crop_paths,
                "debug": debug_path.relative_to(output_root).as_posix(),
            }
        )
    output_path = output_root / "results.json"
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "checkpoint_sha256": identity.sha256,
                "operating_point": {
                    "confidence_threshold": resolved_config.inference.confidence_threshold,
                    "nms_iou_threshold": resolved_config.inference.nms_iou_threshold,
                },
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path
