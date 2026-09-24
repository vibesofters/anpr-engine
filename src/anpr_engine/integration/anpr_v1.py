"""Frozen V1 Detection-to-Recognition pipeline with fail-closed provenance."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from anpr_engine.detection.data import letterbox
from anpr_engine.detection.model import decode_multiscale_outputs, non_maximum_suppression
from anpr_engine.detection.runtime import (
    load_checkpoint,
    resolve_device,
    sha256_file,
)
from anpr_engine.domain import ArtifactIdentity, BBoxXYXY
from anpr_engine.integration.plate_boundary_refinement import refine_plate_boundary
from anpr_engine.recognition.adapters import adapt_fixed_slot_batch
from anpr_engine.recognition.charset import V1_CHARSET
from anpr_engine.recognition.checkpointing import load_recognition_checkpoint
from anpr_engine.recognition.crop import InvalidCropError, _aspect_fit, generate_plate_crop
from anpr_engine.recognition.decoder import ProbabilityAwareTrDecoder
from anpr_engine.recognition.profiles import TR_PROFILE
from anpr_engine.recognition.registry import M3_CAPABILITY
from anpr_engine.recognition.transformer_slot_config import (
    load_transformer_fixed_slot_foundation_config,
)
from anpr_engine.recognition.transformer_slot_model import (
    build_transformer_fixed_slot_recognizer,
)

DETECTOR_SHA = "0d5e9b52fae5638e0ef1badafdeb53fe0771d056e43c4953d13cceed520e71ea"
RECOGNIZER_SHA = "25b6e4af757b87da75162293a28274e3a0ecbb7f29bc55489b7132e353ad5ff7"
DETECTOR_VERSION = "detection-v3-20260827"
RECOGNIZER_VERSION = "m3-v4-stroke-20260914"
RECOGNIZER_CONFIG = Path("configs/recognition/cnn-transformer-dual-slot-tr-v1-foundation.yaml")
RESULT_SCHEMA = "anpr-v1-browser-result-v1"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class FrozenBundle:
    root: Path
    detector: Path
    recognizer: Path
    manifest: Path

    @classmethod
    def open(cls, root: Path) -> FrozenBundle:
        """Open a separately provisioned private bundle without modifying it."""
        resolved = root.expanduser().resolve()
        return cls(
            resolved,
            resolved / "detection.pt",
            resolved / "recognition.pt",
            resolved / "bundle-manifest.json",
        )

    def verify(self) -> None:
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "anpr-engine-private-model-bundle-v1":
            raise ValueError("private model bundle manifest mismatch")
        if payload.get("classification") != "PRIVATE_OWNER_ONLY":
            raise ValueError("private model bundle classification mismatch")
        models = {str(item.get("role")): item for item in payload.get("models", [])}
        if models.get("detection", {}).get("sha256") != DETECTOR_SHA:
            raise ValueError("detector manifest identity mismatch")
        if models.get("recognition", {}).get("sha256") != RECOGNIZER_SHA:
            raise ValueError("recognizer manifest identity mismatch")
        if sha256_file(self.detector) != DETECTOR_SHA:
            raise ValueError("detector bundle hash mismatch")
        if sha256_file(self.recognizer) != RECOGNIZER_SHA:
            raise ValueError("recognizer bundle hash mismatch")


def decode_upload(content: bytes) -> tuple[NDArray[np.uint8], str]:
    if not content:
        raise ValueError("empty image")
    try:
        with Image.open(__import__("io").BytesIO(content)) as opened:
            if opened.format not in {"JPEG", "PNG"}:
                raise ValueError("unsupported image content")
            image = ImageOps.exif_transpose(opened).convert("RGB")
            rgb = np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("corrupt or unsupported image") from error
    bgr = cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return bgr, opened.format or "UNKNOWN"


class AnprV1Pipeline:
    """Load the two frozen models once and execute the governed sequence."""

    def __init__(self, repository: Path, *, model_bundle: Path, device_name: str = "auto") -> None:
        started = time.perf_counter()
        self.repository = repository.resolve()
        self.bundle = FrozenBundle.open(model_bundle)
        self.bundle.verify()
        self.device = resolve_device(device_name)
        self.detector, detector_metadata = load_checkpoint(self.bundle.detector, self.device)
        if detector_metadata.get("architecture") != "yolox_multiscale_v1":
            raise ValueError("detector architecture identity mismatch")
        self.detector.eval()
        self.detector_identity = ArtifactIdentity(
            name="yolox_multiscale_v1",
            version=DETECTOR_VERSION,
            sha256=DETECTOR_SHA,
            reference="private-runtime:detection",
        )
        self.foundation = load_transformer_fixed_slot_foundation_config(
            self.repository / RECOGNIZER_CONFIG
        )
        if M3_CAPABILITY.max_slots is None:
            raise ValueError("M3 fixed-slot capability is missing max_slots")
        self.recognizer = build_transformer_fixed_slot_recognizer(
            self.foundation.model, max_slots=M3_CAPABILITY.max_slots
        )
        load_recognition_checkpoint(
            self.bundle.recognizer,
            model=self.recognizer,
            expected_capability=M3_CAPABILITY,
            expected_sha256=RECOGNIZER_SHA,
        )
        self.recognizer.to(self.device).eval()
        self.decoder = ProbabilityAwareTrDecoder(
            charset=V1_CHARSET, profile=TR_PROFILE, config=self.foundation.decoder
        )
        self.recognizer_identity = ArtifactIdentity(
            name="cnn_transformer_dual_slot_tr_v1",
            version=RECOGNIZER_VERSION,
            sha256=RECOGNIZER_SHA,
            reference="private-runtime:recognition",
        )
        self._lock = threading.Lock()
        self.load_count = 1
        self.startup_ms = (time.perf_counter() - started) * 1000

    def health(self) -> dict[str, Any]:
        self.bundle.verify()
        return {
            "status": "HEALTHY",
            "device": self.device.type,
            "model_load_count": self.load_count,
            "startup_model_load_ms": self.startup_ms,
            "detector": self.detector_identity.model_dump(mode="json"),
            "recognizer": self.recognizer_identity.model_dump(mode="json"),
            "freeze_state": "FROZEN_FOR_V1_INTEGRATION",
        }

    def infer(
        self,
        image: NDArray[np.uint8],
        *,
        batch_id: str,
        request_id: str,
        input_sha256: str,
        filename: str,
        output_directory: Path | None,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        timings: dict[str, float] = {}
        if output_directory is not None:
            output_directory.mkdir(parents=True, exist_ok=True)
        with self._lock, torch.inference_mode():
            stage = time.perf_counter()
            prepared, transform = letterbox(image, 384)
            tensor = (
                torch.from_numpy(
                    np.ascontiguousarray(
                        cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                    )
                )
                .float()
                .div(255)
                .unsqueeze(0)
                .to(self.device)
            )
            native = self.detector(tensor)
            objectness, boxes = decode_multiscale_outputs(native)
            scores = objectness[0].sigmoid().detach().cpu()
            candidate_boxes = boxes[0].clamp(0, 1).detach().cpu()
            mask = scores >= 0.15
            threshold_boxes, threshold_scores = candidate_boxes[mask], scores[mask]
            candidates: list[dict[str, Any]] = []
            if len(threshold_scores):
                keep = non_maximum_suppression(threshold_boxes, threshold_scores, 0.30)
                for index in keep.tolist():
                    x1, y1, x2, y2 = transform.to_original_xyxy(threshold_boxes[index])
                    candidates.append(
                        {
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "confidence": float(threshold_scores[index]),
                        }
                    )
            candidates.sort(key=lambda row: (-row["confidence"], row["bbox_xyxy"]))
            timings["detection_inference_ms"] = (time.perf_counter() - stage) * 1000
            if output_directory is not None:
                annotated = image.copy()
                for index, item in enumerate(candidates):
                    x1, y1, x2, y2 = (round(value) for value in item["bbox_xyxy"])
                    color = (0, 70, 255) if index == 0 else (0, 210, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3 if index == 0 else 2)
                    cv2.putText(
                        annotated,
                        f"{'TOP-1 ' if index == 0 else ''}{item['confidence']:.3f}",
                        (x1, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                annotated_path = output_directory / "annotated.jpg"
                cv2.imwrite(annotated_path.as_posix(), annotated)
            base: dict[str, Any] = {
                "schema_version": RESULT_SCHEMA,
                "batch_id": batch_id,
                "request_id": request_id,
                "input": {
                    "filename": filename,
                    "sha256": input_sha256,
                    "width": image.shape[1],
                    "height": image.shape[0],
                },
                "active_device": self.device.type,
                "models": {
                    "detector": self.detector_identity.model_dump(mode="json"),
                    "recognizer": self.recognizer_identity.model_dump(mode="json"),
                },
                "detection": {
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "selected": candidates[0] if candidates else None,
                },
                "annotated_image": "annotated.jpg" if output_directory is not None else None,
                "crop_reference": None,
                "recognition": None,
                "manual_review": {
                    "detection_assessment": "UNREVIEWED",
                    "recognition_assessment": "NOT_APPLICABLE" if not candidates else "UNREVIEWED",
                },
                "errors": [],
                "timings_ms": timings,
            }
            if not candidates:
                base["pipeline_status"] = "NO_PLATE_DETECTED"
                timings["total_request_ms"] = (time.perf_counter() - total_start) * 1000
                return base
            stage = time.perf_counter()
            selected = candidates[0]
            try:
                bbox = BBoxXYXY(
                    x1=selected["bbox_xyxy"][0],
                    y1=selected["bbox_xyxy"][1],
                    x2=selected["bbox_xyxy"][2],
                    y2=selected["bbox_xyxy"][3],
                )
                zero_crop = self.foundation.crop.model_copy(
                    update={"padding_x": 0.0, "padding_y": 0.0}
                )
                crop = generate_plate_crop(
                    image, bbox, source_image_id=request_id, config=zero_crop
                )
            except (ValueError, InvalidCropError) as error:
                base["pipeline_status"] = "INVALID_CROP"
                base["errors"] = [str(error)]
                timings["total_request_ms"] = (time.perf_counter() - total_start) * 1000
                return base
            if output_directory is not None:
                crop_path = output_directory / "crop.jpg"
                cv2.imwrite(crop_path.as_posix(), cv2.cvtColor(crop.rgb, cv2.COLOR_RGB2BGR))
            base["crop_reference"] = {
                **crop.reference.model_dump(mode="json"),
                "artifact": "crop.jpg" if output_directory is not None else None,
            }
            base["detection_crop_reference"] = base["crop_reference"]
            refinement = refine_plate_boundary(crop.rgb)
            if output_directory is not None:
                refined_path = output_directory / "refined-crop.jpg"
                cv2.imwrite(
                    refined_path.as_posix(), cv2.cvtColor(refinement.rgb, cv2.COLOR_RGB2BGR)
                )
            base["recognition_crop_reference"] = {
                "artifact": (
                    ("refined-crop.jpg" if refinement.applied else "crop.jpg")
                    if output_directory is not None
                    else None
                ),
                "refinement_applied": refinement.applied,
                **refinement.provenance,
            }
            timings["crop_preprocessing_ms"] = (time.perf_counter() - stage) * 1000
            stage = time.perf_counter()
            prepared_recognition_crop = (
                _aspect_fit(refinement.rgb, self.foundation.crop)
                if refinement.applied
                else crop.prepared_rgb
            )
            recognition_tensor = (
                torch.from_numpy(np.ascontiguousarray(prepared_recognition_crop.transpose(2, 0, 1)))
                .float()
                .div(255)
                .unsqueeze(0)
                .to(self.device)
            )
            native_recognition = self.recognizer(recognition_tensor)
            common = adapt_fixed_slot_batch(
                native_recognition.character_logits,
                native_recognition.type_logits,
                capability=M3_CAPABILITY,
                decoder_config=self.foundation.decoder,
            )[0]
            decoded = self.decoder.decode_common(common)
            timings["recognition_inference_ms"] = (time.perf_counter() - stage) * 1000
            base["recognition"] = {
                "raw_text": decoded.raw_text,
                "raw_confidence": decoded.raw_confidence,
                "tr_text": decoded.decoded_text,
                "decoder_confidence": decoded.decoder_confidence,
                "decoding_changed_raw": decoded.decoding_changed_raw,
                "structural_validation": decoded.structural_validation.model_dump(mode="json")
                if decoded.structural_validation
                else None,
                "profile_id": decoded.profile_id,
                "decoder_status": decoded.status.value,
                "decoder_reason": decoded.reason,
                "decoder_version": decoded.decoder_version,
            }
            base["pipeline_status"] = (
                "MULTIPLE_PLATE_CANDIDATES" if len(candidates) > 1 else decoded.status.value
            )
            timings["artifact_writing_ms"] = 0.0
            timings["total_request_ms"] = (time.perf_counter() - total_start) * 1000
            return base


def content_sha256(content: bytes) -> str:
    return _sha_bytes(content)
