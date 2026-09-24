"""Sanitized, family-neutral Recognition training primitives.

The module intentionally contains no experiment registry, private dataset identity,
detached-process control, or implicit checkpoint discovery.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import cv2
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor, nn
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import Dataset

from anpr_engine.recognition.charset import V1_CHARSET
from anpr_engine.recognition.checkpointing import (
    RecognitionCheckpointMetadata,
    load_recognition_checkpoint,
    save_recognition_checkpoint,
)
from anpr_engine.recognition.ctc import greedy_decode_logits
from anpr_engine.recognition.enhanced_config import (
    enhanced_config_sha256,
    load_enhanced_foundation_config,
)
from anpr_engine.recognition.enhanced_model import (
    EnhancedRecognizerOutput,
    build_enhanced_recognizer,
)
from anpr_engine.recognition.evidence import RecognitionModelCapability
from anpr_engine.recognition.fixed_slot import (
    FIXED_SLOT_CHARACTER_PAD_INDEX,
    FIXED_SLOT_CHARACTER_SYMBOLS,
    FixedSlotRecognizerOutput,
    build_fixed_slot_recognizer,
    encode_fixed_slot_targets,
    fixed_slot_loss,
)
from anpr_engine.recognition.fixed_slot_config import (
    fixed_slot_config_sha256,
    load_fixed_slot_foundation_config,
)
from anpr_engine.recognition.metrics import edit_distance
from anpr_engine.recognition.model import parameter_count
from anpr_engine.recognition.normalization import normalize_plate_text
from anpr_engine.recognition.profiles import TR_PROFILE
from anpr_engine.recognition.registry import (
    M0_CAPABILITY,
    M1_CAPABILITY,
    M2_CAPABILITY,
    M3_CAPABILITY,
)
from anpr_engine.recognition.transformer_config import (
    load_transformer_foundation_config,
    transformer_architecture_metadata,
    transformer_config_sha256,
)
from anpr_engine.recognition.transformer_model import (
    TransformerRecognizerOutput,
    build_transformer_recognizer,
)
from anpr_engine.recognition.transformer_slot_config import (
    load_transformer_fixed_slot_foundation_config,
    transformer_fixed_slot_architecture_metadata,
    transformer_fixed_slot_config_sha256,
)
from anpr_engine.recognition.transformer_slot_model import (
    build_transformer_fixed_slot_recognizer,
)
from anpr_engine.recognition.type_ctc import dual_ctc_loss

RecognitionFamily = Literal[
    "crnn_bilstm_dual_ctc_tr_v1",
    "crnn_bilstm_dual_slot_tr_v1",
    "cnn_transformer_dual_ctc_tr_v1",
    "cnn_transformer_dual_slot_tr_v1",
]
RecognitionOutput = (
    EnhancedRecognizerOutput | TransformerRecognizerOutput | FixedSlotRecognizerOutput
)


def _portable_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    lowered = value.casefold()
    forbidden = ("private", ".worktree", "artifacts/experiments", "users/")
    if (
        path.is_absolute()
        or value in {"", "."}
        or ".." in path.parts
        or "\\" in value
        or any(token in lowered for token in forbidden)
    ):
        raise ValueError(
            "public training paths must be portable and contain no private identifiers"
        )
    return value


class PublicTrainingConfig(BaseModel):
    """Strict, path-safe training configuration with no embedded dataset identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["public-recognition-training-v1"]
    family: RecognitionFamily
    foundation_config: str
    images_directory: str = "images"
    labels_directory: str = "labels"
    optimizer: Literal["adamw", "sgd"] = "adamw"
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.0001, ge=0.0, le=1.0)
    batch_size: int = Field(default=1, ge=1, le=256)
    epochs: int = Field(default=1, ge=1, le=10_000)
    scheduler_step_epochs: int = Field(default=1, ge=1)
    scheduler_gamma: float = Field(default=0.95, gt=0.0, le=1.0)
    reporting: Literal["percentage_only"] = "percentage_only"
    test_or_holdout_access: Literal[False] = False

    @field_validator("foundation_config", "images_directory", "labels_directory")
    @classmethod
    def paths_are_public_and_portable(cls, value: str) -> str:
        return _portable_relative_path(value)


def load_public_training_config(path: Path) -> PublicTrainingConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("public Recognition training configuration must be a YAML mapping")
    return PublicTrainingConfig.model_validate(payload)


class PublicRecognitionDataset(Dataset[tuple[Tensor, str]]):
    """Image/label pairs supplied by the user; filenames never serve as labels."""

    def __init__(
        self,
        root: Path,
        *,
        images_directory: str = "images",
        labels_directory: str = "labels",
        input_size: tuple[int, int] = (32, 160),
    ) -> None:
        self.root = root.resolve()
        self.images_root = self.root / _portable_relative_path(images_directory)
        self.labels_root = self.root / _portable_relative_path(labels_directory)
        self.input_size = input_size
        if not self.images_root.is_dir() or not self.labels_root.is_dir():
            raise ValueError("dataset must contain the configured image and label directories")
        self.images = tuple(
            path
            for path in sorted(self.images_root.iterdir())
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
        )
        if not self.images:
            raise ValueError("dataset contains no JPEG or PNG images")
        for image in self.images:
            if not (self.labels_root / f"{image.stem}.txt").is_file():
                raise ValueError(f"missing label for image stem {image.stem!r}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        image_path = self.images[index]
        image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode image {image_path.name!r}")
        target_height, target_width = self.input_size
        image = cv2.cvtColor(
            cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        text = (self.labels_root / f"{image_path.stem}.txt").read_text(encoding="utf-8").strip()
        normalized = normalize_plate_text(text).normalized_text
        if text != normalized or not TR_PROFILE.validate(text).valid:
            raise ValueError("labels must be normalized values accepted by the TR V1 profile")
        return tensor, text


@dataclass(frozen=True)
class PublicRecognitionFamily:
    family: RecognitionFamily
    model: nn.Module
    capability: RecognitionModelCapability
    config_sha256: str
    architecture_metadata: Mapping[str, object]
    lambda_type: float


def build_public_recognition_family(
    family: RecognitionFamily,
    *,
    foundation_path: Path,
) -> PublicRecognitionFamily:
    """Build one of the four disclosed families from a public foundation file."""
    if family == "crnn_bilstm_dual_ctc_tr_v1":
        enhanced_config = load_enhanced_foundation_config(foundation_path)
        enhanced_model = build_enhanced_recognizer(
            enhanced_config.model, character_class_count=V1_CHARSET.class_count
        )
        metadata: dict[str, object] = {"model": enhanced_config.model.model_dump(mode="json")}
        return PublicRecognitionFamily(
            family,
            enhanced_model,
            M0_CAPABILITY,
            enhanced_config_sha256(enhanced_config),
            metadata,
            enhanced_config.lambda_type,
        )
    if family == "crnn_bilstm_dual_slot_tr_v1":
        slot_config = load_fixed_slot_foundation_config(foundation_path)
        slot_model = build_fixed_slot_recognizer(
            slot_config.model, max_slots=slot_config.slot_extraction.max_slots
        )
        slot_metadata: dict[str, object] = {
            "model": slot_config.model.model_dump(mode="json"),
            "slot_extraction": slot_config.slot_extraction.model_dump(mode="json"),
        }
        return PublicRecognitionFamily(
            family,
            slot_model,
            M1_CAPABILITY,
            fixed_slot_config_sha256(slot_config),
            slot_metadata,
            slot_config.loss.lambda_type,
        )
    if family == "cnn_transformer_dual_ctc_tr_v1":
        transformer_config = load_transformer_foundation_config(foundation_path)
        transformer_model = build_transformer_recognizer(transformer_config.model)
        return PublicRecognitionFamily(
            family,
            transformer_model,
            M2_CAPABILITY,
            transformer_config_sha256(transformer_config),
            transformer_architecture_metadata(transformer_config),
            transformer_config.loss.lambda_type,
        )
    transformer_slot_config = load_transformer_fixed_slot_foundation_config(foundation_path)
    transformer_slot_model = build_transformer_fixed_slot_recognizer(
        transformer_slot_config.model,
        max_slots=transformer_slot_config.slot_extraction.max_slots,
    )
    return PublicRecognitionFamily(
        family,
        transformer_slot_model,
        M3_CAPABILITY,
        transformer_fixed_slot_config_sha256(transformer_slot_config),
        transformer_fixed_slot_architecture_metadata(transformer_slot_config),
        transformer_slot_config.loss.lambda_type,
    )


def validate_training_batch(
    bundle: PublicRecognitionFamily,
    images: Tensor,
    texts: tuple[str, ...],
) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != bundle.capability.input_shape:
        raise ValueError(
            f"training images must have shape [batch, {bundle.capability.input_shape}]"
        )
    if images.shape[0] != len(texts) or not texts:
        raise ValueError("training image and target populations must be non-empty and aligned")
    for text in texts:
        if (
            normalize_plate_text(text).normalized_text != text
            or not TR_PROFILE.validate(text).valid
        ):
            raise ValueError(
                "training targets must be normalized and valid under the TR V1 profile"
            )


def recognition_loss(
    bundle: PublicRecognitionFamily,
    output: RecognitionOutput,
    texts: tuple[str, ...],
) -> Tensor:
    if bundle.capability.max_slots is not None:
        slot_output = cast(FixedSlotRecognizerOutput, output)
        targets = encode_fixed_slot_targets(texts, max_slots=bundle.capability.max_slots).to(
            slot_output.character_logits.device
        )
        return fixed_slot_loss(slot_output, targets, lambda_type=bundle.lambda_type).total
    ctc_output = cast(EnhancedRecognizerOutput | TransformerRecognizerOutput, output)
    return dual_ctc_loss(
        ctc_output.character_logits,
        ctc_output.type_logits,
        texts=texts,
        character_charset=V1_CHARSET,
        lambda_type=bundle.lambda_type,
    ).total


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: PublicTrainingConfig,
) -> tuple[Optimizer, LRScheduler]:
    if config.optimizer == "sgd":
        optimizer: Optimizer = SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    scheduler = StepLR(
        optimizer, step_size=config.scheduler_step_epochs, gamma=config.scheduler_gamma
    )
    return optimizer, scheduler


def train_one_step(
    bundle: PublicRecognitionFamily,
    images: Tensor,
    texts: tuple[str, ...],
    optimizer: Optimizer,
) -> float:
    """Perform one bounded, explicit training step and return only the scalar loss."""
    validate_training_batch(bundle, images, texts)
    bundle.model.train()
    optimizer.zero_grad(set_to_none=True)
    output = cast(RecognitionOutput, bundle.model(images))
    loss = recognition_loss(bundle, output, texts)
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("Recognition training loss is not finite")
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    return float(loss.detach().cpu().item())


def _slot_predictions(output: FixedSlotRecognizerOutput) -> tuple[str, ...]:
    indexes = output.character_logits.argmax(dim=2).detach().cpu()
    predictions: list[str] = []
    for row in indexes.tolist():
        symbols: list[str] = []
        for index in row:
            if index == FIXED_SLOT_CHARACTER_PAD_INDEX:
                break
            symbols.append(FIXED_SLOT_CHARACTER_SYMBOLS[index])
        predictions.append("".join(symbols))
    return tuple(predictions)


def percentage_validation_metrics(
    targets: tuple[str, ...], predictions: tuple[str, ...]
) -> dict[str, float]:
    """Return percentages only; never return counts or denominators."""
    if not targets or len(targets) != len(predictions):
        raise ValueError("validation targets and predictions must be non-empty and aligned")
    character_total = sum(len(target) for target in targets)
    errors = sum(
        edit_distance(target, prediction)
        for target, prediction in zip(targets, predictions, strict=True)
    )
    return {
        "exact_accuracy_percent": 100.0
        * sum(target == prediction for target, prediction in zip(targets, predictions, strict=True))
        / len(targets),
        "character_accuracy_percent": 100.0 * max(0.0, 1.0 - errors / character_total),
        "length_accuracy_percent": 100.0
        * sum(
            len(target) == len(prediction)
            for target, prediction in zip(targets, predictions, strict=True)
        )
        / len(targets),
    }


def validate_one_batch(
    bundle: PublicRecognitionFamily,
    images: Tensor,
    texts: tuple[str, ...],
) -> dict[str, float]:
    predictions = predict_one_batch(bundle, images, texts)
    return percentage_validation_metrics(texts, predictions)


def predict_one_batch(
    bundle: PublicRecognitionFamily,
    images: Tensor,
    texts: tuple[str, ...],
) -> tuple[str, ...]:
    """Return family-native raw predictions after validating the public batch contract."""
    validate_training_batch(bundle, images, texts)
    bundle.model.eval()
    with torch.inference_mode():
        output = cast(RecognitionOutput, bundle.model(images))
    if bundle.capability.max_slots is None:
        ctc_output = cast(EnhancedRecognizerOutput | TransformerRecognizerOutput, output)
        return greedy_decode_logits(ctc_output.character_logits, V1_CHARSET)
    return _slot_predictions(cast(FixedSlotRecognizerOutput, output))


def save_public_training_checkpoint(
    path: Path,
    *,
    bundle: PublicRecognitionFamily,
) -> RecognitionCheckpointMetadata:
    metadata = RecognitionCheckpointMetadata(
        capability=bundle.capability,
        parameter_count=parameter_count(bundle.model),
        config_sha256=bundle.config_sha256,
        training_run_identity="user-controlled-local-run",
        dataset_identity="user-supplied-dataset",
        architecture_metadata=bundle.architecture_metadata,
    )
    return save_recognition_checkpoint(path, model=bundle.model, metadata=metadata)


def explicitly_resume_public_checkpoint(
    path: Path,
    *,
    bundle: PublicRecognitionFamily,
    expected_sha256: str | None = None,
) -> None:
    """Load only a path the caller explicitly selected; there is no auto-resume."""
    load_recognition_checkpoint(
        path,
        model=bundle.model,
        expected_capability=bundle.capability,
        expected_sha256=expected_sha256,
        expected_config_sha256=bundle.config_sha256,
        expected_architecture_metadata=bundle.architecture_metadata,
    )


def public_config_sha256(config: PublicTrainingConfig) -> str:
    content = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
