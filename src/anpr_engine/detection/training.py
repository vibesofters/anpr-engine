"""Config-driven training, checkpointing, and evaluation for the Detection V1 baseline."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections.abc import Sized
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import SGD, AdamW, Optimizer
from torch.utils.data import DataLoader

from anpr_engine.detection.config import (
    DetectorConfig,
    DetectorInferenceConfig,
    MultiScaleDetectorModelConfig,
    TinyDetectorModelConfig,
    detector_config_sha256,
)
from anpr_engine.detection.data import DetectionDataset, load_samples
from anpr_engine.detection.metrics import SamplePredictions, collect_predictions, detection_metrics
from anpr_engine.detection.model import (
    DetectorOutput,
    TinyYoloDetector,
    YoloXMultiScaleDetector,
    box_iou,
    decode_multiscale_outputs,
    multiscale_cell_geometry,
)


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def verify_dataset_identity(manifest_path: Path, config: DetectorConfig) -> None:
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("dataset_id") != config.dataset_id:
        raise ValueError("detector config dataset ID does not match the canonical manifest")
    if manifest.get("content_sha256") != config.dataset_content_sha256:
        raise ValueError("detector config dataset checksum does not match the canonical manifest")


def verify_checkpoint_identity(metadata: dict[str, Any], config: DetectorConfig) -> None:
    if (
        metadata.get("dataset_id") != config.dataset_id
        or metadata.get("dataset_content_sha256") != config.dataset_content_sha256
    ):
        raise ValueError("checkpoint dataset identity does not match detector config")
    if metadata.get("config_sha256") != detector_config_sha256(config):
        raise ValueError("checkpoint configuration does not match detector config")


def detector_loss(
    output: Tensor,
    target: Tensor,
    *,
    objectness_positive_weight: float,
    box_loss_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    objectness_target = target[:, 0]
    objectness_weight = torch.ones_like(objectness_target)
    objectness_weight[objectness_target == 1.0] = objectness_positive_weight
    objectness_loss = nn.functional.binary_cross_entropy_with_logits(
        output[:, 0], objectness_target, weight=objectness_weight
    )
    positive = objectness_target == 1.0
    box_prediction = output[:, 1:].sigmoid().permute(0, 2, 3, 1)[positive]
    box_target = target[:, 1:].permute(0, 2, 3, 1)[positive]
    box_loss = nn.functional.smooth_l1_loss(box_prediction, box_target)
    total = objectness_loss + box_loss_weight * box_loss
    return total, objectness_loss, box_loss


def _aligned_generalized_iou(prediction: Tensor, target: Tensor) -> Tensor:
    intersection_top_left = torch.maximum(prediction[:, :2], target[:, :2])
    intersection_bottom_right = torch.minimum(prediction[:, 2:], target[:, 2:])
    intersection = (intersection_bottom_right - intersection_top_left).clamp(min=0).prod(dim=1)
    prediction_area = (prediction[:, 2:] - prediction[:, :2]).clamp(min=0).prod(dim=1)
    target_area = (target[:, 2:] - target[:, :2]).clamp(min=0).prod(dim=1)
    union = (prediction_area + target_area - intersection).clamp(min=1e-8)
    enclosing_top_left = torch.minimum(prediction[:, :2], target[:, :2])
    enclosing_bottom_right = torch.maximum(prediction[:, 2:], target[:, 2:])
    enclosing_area = (enclosing_bottom_right - enclosing_top_left).clamp(min=0).prod(dim=1)
    return intersection / union - (enclosing_area - union) / enclosing_area.clamp(min=1e-8)


def _dynamic_positive_mask(
    objectness_logits: Tensor,
    decoded_boxes: Tensor,
    targets: Tensor,
    centers: Tensor,
    cell_sizes: Tensor,
    *,
    center_radius: float,
    top_k: int,
) -> Tensor:
    positive_mask = torch.zeros_like(objectness_logits, dtype=torch.bool)
    with torch.no_grad():
        for batch_index, target in enumerate(targets):
            inside_box = (
                (centers[:, 0] > target[0])
                & (centers[:, 1] > target[1])
                & (centers[:, 0] < target[2])
                & (centers[:, 1] < target[3])
            )
            target_center = (target[:2] + target[2:]) / 2.0
            inside_center = ((centers - target_center).abs() / cell_sizes).amax(
                dim=1
            ) < center_radius
            candidates = inside_box | inside_center
            candidate_indices = torch.nonzero(candidates, as_tuple=False).flatten()
            if not candidate_indices.numel():
                candidate_indices = ((centers - target_center).square().sum(dim=1)).argmin().view(1)
            candidate_boxes = decoded_boxes[batch_index, candidate_indices]
            candidate_ious = box_iou(target, candidate_boxes).clamp(min=0.0, max=1.0)
            candidate_scores = objectness_logits[batch_index, candidate_indices].sigmoid()
            costs = (
                -torch.log(candidate_scores.clamp(min=1e-8))
                + 3.0 * (1.0 - candidate_ious)
                + 0.05
                * ((centers[candidate_indices] - target_center) / cell_sizes[candidate_indices])
                .square()
                .sum(dim=1)
            )
            top_count = min(top_k, candidate_ious.numel())
            dynamic_count = max(1, int(candidate_ious.topk(top_count).values.sum().item()))
            selected = candidate_indices[
                costs.topk(min(dynamic_count, top_count), largest=False).indices
            ]
            positive_mask[batch_index, selected] = True
    return positive_mask


def multiscale_detector_loss(
    output: tuple[Tensor, Tensor, Tensor],
    targets: Tensor,
    *,
    focal_alpha: float,
    focal_gamma: float,
    box_loss_weight: float,
    center_radius: float,
    assignment_top_k: int,
) -> tuple[Tensor, Tensor, Tensor, int]:
    objectness_logits, decoded_boxes = decode_multiscale_outputs(output)
    centers, cell_sizes = multiscale_cell_geometry(output)
    positive = _dynamic_positive_mask(
        objectness_logits,
        decoded_boxes.detach(),
        targets,
        centers,
        cell_sizes,
        center_radius=center_radius,
        top_k=assignment_top_k,
    )
    objectness_target = positive.to(objectness_logits.dtype)
    probabilities = objectness_logits.sigmoid()
    cross_entropy = nn.functional.binary_cross_entropy_with_logits(
        objectness_logits, objectness_target, reduction="none"
    )
    target_probability = torch.where(positive, probabilities, 1.0 - probabilities)
    alpha = torch.where(positive, focal_alpha, 1.0 - focal_alpha)
    objectness_loss = (
        alpha * (1.0 - target_probability).pow(focal_gamma) * cross_entropy
    ).sum() / positive.sum().clamp(min=1)
    matched_targets = targets[:, None, :].expand_as(decoded_boxes)[positive]
    box_loss = (1.0 - _aligned_generalized_iou(decoded_boxes[positive], matched_targets)).mean()
    total = objectness_loss + box_loss_weight * box_loss
    return total, objectness_loss, box_loss, int(positive.sum().item())


def _model(config: DetectorConfig) -> nn.Module:
    if isinstance(config.model, TinyDetectorModelConfig):
        return TinyYoloDetector(config.model.base_channels)
    return YoloXMultiScaleDetector(config.model.base_channels, config.model.head_channels)


def _optimizer(config: DetectorConfig, model: nn.Module) -> Optimizer:
    if config.training.optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay,
        )
    return AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def _loader(
    repository: Path,
    manifest_path: Path,
    config: DetectorConfig,
    split: str,
    *,
    training: bool,
) -> DataLoader[tuple[Tensor, Tensor, str]]:
    dataset = DetectionDataset(
        load_samples(manifest_path, repository, split),
        image_size=config.model.image_size,
        grid_stride=(
            config.model.grid_stride if isinstance(config.model, TinyDetectorModelConfig) else None
        ),
        training=training,
        augmentation=config.training.augmentation,
    )
    generator = torch.Generator().manual_seed(config.training.seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=training,
        num_workers=config.training.num_workers,
        generator=generator,
        worker_init_fn=partial(_seed_worker, base_seed=config.training.seed),
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, str]],
    device: torch.device,
    config: DetectorConfig,
) -> dict[str, Any]:
    model.eval()
    predictions: list[SamplePredictions] = []
    with torch.inference_mode():
        for images, targets, sample_ids in loader:
            output = cast(DetectorOutput, model(images.to(device)))
            predictions.extend(
                collect_predictions(
                    output,
                    targets,
                    tuple(sample_ids),
                    nms_iou_threshold=config.inference.nms_iou_threshold,
                    max_detections=config.inference.max_detections,
                )
            )
    return detection_metrics(
        tuple(predictions), confidence_threshold=config.inference.confidence_threshold
    )


def evaluation_config(
    config: DetectorConfig,
    *,
    confidence_threshold: float | None = None,
    nms_iou_threshold: float | None = None,
) -> DetectorConfig:
    inference = config.inference.model_dump()
    if confidence_threshold is not None:
        inference["confidence_threshold"] = confidence_threshold
    if nms_iou_threshold is not None:
        inference["nms_iou_threshold"] = nms_iou_threshold
    return config.model_copy(
        update={"inference": DetectorInferenceConfig.model_validate(inference)}
    )


def _git_state(repository: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return revision, dirty


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "mps_available": torch.backends.mps.is_available(),
    }


def _device_memory(device: torch.device) -> dict[str, int] | None:
    if device.type == "mps":
        return {
            "current_allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "recommended_max_bytes": torch.mps.recommended_max_memory(),
        }
    if device.type == "cuda":
        return {
            "current_allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    return None


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    epoch: int,
    config: DetectorConfig,
    run_id: str,
    validation_metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": "1.0",
        "architecture": config.model.architecture,
        "base_channels": config.model.base_channels,
        "image_size": config.model.image_size,
        "dataset_id": config.dataset_id,
        "dataset_content_sha256": config.dataset_content_sha256,
        "config_sha256": detector_config_sha256(config),
        "run_id": run_id,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "validation_metrics": validation_metrics,
    }
    if isinstance(config.model, MultiScaleDetectorModelConfig):
        payload["head_channels"] = config.model.head_channels
    torch.save(payload, path)


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


def train_detector(
    repository: Path,
    manifest_path: Path,
    config_path: Path,
    config: DetectorConfig,
    artifact_root: Path,
) -> Path:
    seed_everything(config.training.seed)
    verify_dataset_identity(manifest_path, config)
    device = resolve_device(config.training.device)
    started_at = datetime.now(UTC)
    run_id = f"{config.dataset_id}-{started_at.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    run_root = artifact_root / "experiments" / run_id
    checkpoint_root = run_root / "checkpoints"
    run_root.mkdir(parents=True, exist_ok=False)
    train_loader = _loader(repository, manifest_path, config, "train", training=True)
    validation_loader = _loader(repository, manifest_path, config, "validation", training=False)
    model = _model(config).to(device)
    optimizer = _optimizer(config, model)
    revision, dirty = _git_state(repository)
    metrics_path = run_root / "metrics.jsonl"
    best_map = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    completed_epochs = 0
    stop_reason = "configured_epochs_completed"
    epoch_checkpoints: dict[str, dict[str, Any]] = {}
    best_path = checkpoint_root / "best.pt"
    for epoch in range(1, config.training.epochs + 1):
        completed_epochs = epoch
        epoch_started = time.monotonic()
        model.train()
        losses: list[float] = []
        objectness_losses: list[float] = []
        box_losses: list[float] = []
        positive_assignments: list[int] = []
        for images, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(images.to(device))
            if isinstance(config.model, TinyDetectorModelConfig):
                total, objectness, box = detector_loss(
                    cast(Tensor, output),
                    targets.to(device),
                    objectness_positive_weight=config.training.objectness_positive_weight,
                    box_loss_weight=config.training.box_loss_weight,
                )
            else:
                total, objectness, box, positive_count = multiscale_detector_loss(
                    cast(tuple[Tensor, Tensor, Tensor], output),
                    targets.to(device),
                    focal_alpha=config.model.focal_alpha,
                    focal_gamma=config.model.focal_gamma,
                    box_loss_weight=config.model.box_loss_weight,
                    center_radius=config.model.assignment_center_radius,
                    assignment_top_k=config.model.assignment_top_k,
                )
                positive_assignments.append(positive_count)
            total.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(total.detach().cpu()))
            objectness_losses.append(float(objectness.detach().cpu()))
            box_losses.append(float(box.detach().cpu()))
        validation = evaluate_model(model, validation_loader, device, config)
        epoch_record = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "train_objectness_loss": sum(objectness_losses) / len(objectness_losses),
            "train_box_loss": sum(box_losses) / len(box_losses),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_duration_seconds": time.monotonic() - epoch_started,
            "device_memory_after_epoch": _device_memory(device),
            "positive_assignments_per_batch_mean": (
                sum(positive_assignments) / len(positive_assignments)
                if positive_assignments
                else None
            ),
            "validation": validation,
        }
        save_checkpoint(
            checkpoint_root / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            config=config,
            run_id=run_id,
            validation_metrics=validation,
        )
        if isinstance(config.model, MultiScaleDetectorModelConfig):
            epoch_checkpoint = checkpoint_root / f"epoch-{epoch:03d}.pt"
            save_checkpoint(
                epoch_checkpoint,
                model,
                optimizer,
                epoch=epoch,
                config=config,
                run_id=run_id,
                validation_metrics=validation,
            )
            checkpoint_identity = {
                "path": epoch_checkpoint.relative_to(repository).as_posix(),
                "sha256": sha256_file(epoch_checkpoint),
            }
            epoch_record["checkpoint"] = checkpoint_identity
            epoch_checkpoints[str(epoch)] = checkpoint_identity
        if float(validation["map_50"]) > best_map:
            best_map = float(validation["map_50"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                config=config,
                run_id=run_id,
                validation_metrics=validation,
            )
        else:
            epochs_without_improvement += 1
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, sort_keys=True) + "\n")
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if (
            isinstance(config.model, MultiScaleDetectorModelConfig)
            and epoch >= config.model.minimum_epochs
            and epochs_without_improvement >= config.model.early_stopping_patience
        ):
            stop_reason = "validation_early_stopping"
            break
    completed_at = datetime.now(UTC)
    run_manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "COMPLETED",
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
        "source_revision": revision,
        "source_revision_dirty": dirty,
        "dataset_id": config.dataset_id,
        "dataset_content_sha256": config.dataset_content_sha256,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": detector_config_sha256(config),
        "config_source_sha256": sha256_file(config_path),
        "config": config.model_dump(mode="json"),
        "environment": _environment(device),
        "split_counts": {
            "train": len(cast(Sized, train_loader.dataset)),
            "validation": len(cast(Sized, validation_loader.dataset)),
        },
        "best_checkpoint": best_path.relative_to(repository).as_posix(),
        "best_checkpoint_sha256": sha256_file(best_path),
        "best_validation_map_50": best_map,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "stop_reason": stop_reason,
        "epoch_checkpoints": epoch_checkpoints,
        "selection_split": "validation",
        "test_accessed": False,
        "initialization": "random; no pretrained weights",
        "implementation": f"project-owned {config.model.architecture}",
    }
    (run_root / "run-manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return best_path


def evaluate_checkpoint(
    repository: Path,
    manifest_path: Path,
    config: DetectorConfig,
    checkpoint_path: Path,
    split: str,
    output_path: Path,
    *,
    confirm_locked_test_access: bool = False,
    confidence_threshold: float | None = None,
    nms_iou_threshold: float | None = None,
) -> dict[str, Any]:
    if split == "test" and not confirm_locked_test_access:
        raise ValueError("test evaluation requires explicit locked-test access confirmation")
    verify_dataset_identity(manifest_path, config)
    device = resolve_device(config.training.device)
    model, metadata = load_checkpoint(checkpoint_path, device)
    verify_checkpoint_identity(metadata, config)
    resolved_config = evaluation_config(
        config,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )
    loader = _loader(repository, manifest_path, resolved_config, split, training=False)
    metrics = evaluate_model(model, loader, device, resolved_config)
    result = {
        "schema_version": "1.0",
        "split": split,
        "dataset_id": config.dataset_id,
        "dataset_content_sha256": config.dataset_content_sha256,
        "checkpoint": checkpoint_path.relative_to(repository).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": metadata,
        "operating_point": {
            "confidence_threshold": resolved_config.inference.confidence_threshold,
            "nms_iou_threshold": resolved_config.inference.nms_iou_threshold,
        },
        "metrics": metrics,
        "environment": _environment(device),
        "evaluated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "test_accessed": split == "test",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def export_torchscript(checkpoint_path: Path, output_path: Path, config: DetectorConfig) -> str:
    model, metadata = load_checkpoint(checkpoint_path, torch.device("cpu"))
    verify_checkpoint_identity(metadata, config)
    example = torch.zeros((1, 3, config.model.image_size, config.model.image_size))
    traced = torch.jit.trace(model, example)  # type: ignore[no-untyped-call]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(output_path.as_posix())
    return sha256_file(output_path)
