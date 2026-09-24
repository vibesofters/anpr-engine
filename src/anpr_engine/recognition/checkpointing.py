"""Model-family-safe Recognition checkpoint metadata and static load helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch
from torch import nn

from anpr_engine.recognition.evidence import RecognitionModelCapability
from anpr_engine.recognition.registry import validate_checkpoint_capability

RECOGNITION_CHECKPOINT_SCHEMA = "recognition-checkpoint-v1"


@dataclass(frozen=True)
class RecognitionCheckpointMetadata:
    """Identity required to keep all Recognition model families isolated."""

    capability: RecognitionModelCapability
    parameter_count: int
    config_sha256: str
    training_run_identity: str
    dataset_identity: str
    architecture_metadata: Mapping[str, object] | None = None
    checkpoint_sha256: str | None = None
    schema_version: str = RECOGNITION_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RECOGNITION_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported Recognition checkpoint schema")
        if self.parameter_count < 1:
            raise ValueError("checkpoint parameter count must be positive")
        for name, value in (
            ("config_sha256", self.config_sha256),
            ("checkpoint_sha256", self.checkpoint_sha256),
        ):
            if value is not None and (
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not self.training_run_identity or not self.dataset_identity:
            raise ValueError("checkpoint run and dataset identities must be non-empty")

    def embedded_dict(self) -> dict[str, object]:
        """Return metadata stored in the checkpoint; its own file SHA lives in the receipt."""
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "capability": self.capability.checkpoint_metadata(),
            "parameter_count": self.parameter_count,
            "config_sha256": self.config_sha256,
            "training_run_identity": self.training_run_identity,
            "dataset_identity": self.dataset_identity,
        }
        if self.architecture_metadata is not None:
            payload["architecture_metadata"] = dict(self.architecture_metadata)
        return payload

    def with_checkpoint_sha256(self, value: str) -> RecognitionCheckpointMetadata:
        return replace(self, checkpoint_sha256=value)


def checkpoint_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_recognition_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    metadata: RecognitionCheckpointMetadata,
) -> RecognitionCheckpointMetadata:
    """Save one explicit model namespace and return its external SHA receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata.embedded_dict(),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return metadata.with_checkpoint_sha256(checkpoint_file_sha256(path))


def load_recognition_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    expected_capability: RecognitionModelCapability,
    expected_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_architecture_metadata: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Reject the wrong model family before strict state-dict loading."""
    if expected_sha256 is not None and checkpoint_file_sha256(path) != expected_sha256:
        raise ValueError("Recognition checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Recognition checkpoint must contain a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Recognition checkpoint metadata is missing")
    if metadata.get("schema_version") != RECOGNITION_CHECKPOINT_SCHEMA:
        raise ValueError("Recognition checkpoint schema mismatch")
    capability = metadata.get("capability")
    if not isinstance(capability, dict):
        raise ValueError("Recognition checkpoint capability metadata is missing")
    validate_checkpoint_capability(expected_capability, cast(Mapping[str, object], capability))
    expected_parameters = sum(parameter.numel() for parameter in model.parameters())
    if metadata.get("parameter_count") != expected_parameters:
        raise ValueError(
            "Recognition checkpoint parameter-count mismatch: "
            f"expected {expected_parameters}, got {metadata.get('parameter_count')!r}"
        )
    if (
        expected_config_sha256 is not None
        and metadata.get("config_sha256") != expected_config_sha256
    ):
        raise ValueError("Recognition checkpoint config identity mismatch")
    if expected_architecture_metadata is not None:
        observed_architecture = metadata.get("architecture_metadata")
        if observed_architecture != dict(expected_architecture_metadata):
            raise ValueError("Recognition checkpoint architecture metadata mismatch")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Recognition checkpoint state dict is missing")
    model.load_state_dict(cast(dict[str, torch.Tensor], state_dict), strict=True)
    return cast(Mapping[str, object], metadata)
