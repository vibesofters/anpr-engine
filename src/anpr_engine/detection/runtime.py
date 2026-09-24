"""Inference-only Detection checkpoint identity and loading helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from anpr_engine.detection.model import TinyYoloDetector, YoloXMultiScaleDetector


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = cast(dict[str, Any], torch.load(path, map_location=device, weights_only=True))
    if payload.get("format_version") != "1.0":
        raise ValueError("unsupported detector checkpoint")
    architecture = payload.get("architecture")
    if architecture == "tiny_yolo_grid_v1":
        model: nn.Module = TinyYoloDetector(base_channels=int(payload["base_channels"]))
    elif architecture == "yolox_multiscale_v1":
        model = YoloXMultiScaleDetector(
            base_channels=int(payload["base_channels"]),
            head_channels=int(payload["head_channels"]),
        )
    else:
        raise ValueError("unsupported detector checkpoint")
    model.load_state_dict(cast(dict[str, Tensor], payload["model_state"]))
    model.to(device).eval()
    metadata = {key: value for key, value in payload.items() if not key.endswith("_state")}
    return model, metadata
