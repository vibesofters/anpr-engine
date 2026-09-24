"""Train one public Recognition family on an explicitly supplied local dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from anpr_engine.recognition.public_training import (
    PublicRecognitionDataset,
    build_optimizer_and_scheduler,
    build_public_recognition_family,
    explicitly_resume_public_checkpoint,
    load_public_training_config,
    percentage_validation_metrics,
    predict_one_batch,
    save_public_training_checkpoint,
    train_one_step,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        help="explicitly selected compatible checkpoint; auto-resume is not supported",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    config = load_public_training_config(arguments.config)
    repository = Path(__file__).resolve().parents[1]
    foundation = repository / config.foundation_config
    bundle = build_public_recognition_family(config.family, foundation_path=foundation)
    if arguments.resume is not None:
        explicitly_resume_public_checkpoint(arguments.resume, bundle=bundle)

    training = PublicRecognitionDataset(
        arguments.dataset_root / "train",
        images_directory=config.images_directory,
        labels_directory=config.labels_directory,
    )
    validation = PublicRecognitionDataset(
        arguments.dataset_root / "validation",
        images_directory=config.images_directory,
        labels_directory=config.labels_directory,
    )
    generator = torch.Generator().manual_seed(20260827)
    training_loader = DataLoader(
        training, batch_size=config.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False)
    optimizer, scheduler = build_optimizer_and_scheduler(bundle.model, config)

    for _epoch in range(config.epochs):
        for images, texts in training_loader:
            train_one_step(bundle, images, tuple(texts), optimizer)
        scheduler.step()

    validation_targets: list[str] = []
    validation_predictions: list[str] = []
    for images, texts in validation_loader:
        batch_targets = tuple(texts)
        validation_targets.extend(batch_targets)
        validation_predictions.extend(predict_one_batch(bundle, images, batch_targets))
    metrics = percentage_validation_metrics(
        tuple(validation_targets), tuple(validation_predictions)
    )
    receipt = save_public_training_checkpoint(arguments.output, bundle=bundle)
    print(
        json.dumps(
            {
                "model_id": bundle.capability.model_id,
                "config_sha256": bundle.config_sha256,
                "checkpoint_sha256": receipt.checkpoint_sha256,
                "validation_percentages": metrics,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
