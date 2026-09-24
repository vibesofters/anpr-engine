"""Synthetic-only proof for the sanitized four-family Recognition training path."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from anpr_engine.recognition.public_training import (
    PublicRecognitionDataset,
    PublicTrainingConfig,
    build_optimizer_and_scheduler,
    build_public_recognition_family,
    explicitly_resume_public_checkpoint,
    save_public_training_checkpoint,
    train_one_step,
    validate_one_batch,
)
from anpr_engine.recognition.synthetic import TurkishPlateTextGenerator

ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONS = {
    "crnn_bilstm_dual_ctc_tr_v1": "crnn-bilstm-dual-ctc-tr-v1-foundation.yaml",
    "crnn_bilstm_dual_slot_tr_v1": "crnn-bilstm-dual-slot-tr-v1-foundation.yaml",
    "cnn_transformer_dual_ctc_tr_v1": "cnn-transformer-dual-ctc-tr-v1-foundation.yaml",
    "cnn_transformer_dual_slot_tr_v1": "cnn-transformer-dual-slot-tr-v1-foundation.yaml",
}


def _config(family: str) -> PublicTrainingConfig:
    return PublicTrainingConfig.model_validate(
        {
            "schema_version": "public-recognition-training-v1",
            "family": family,
            "foundation_config": f"configs/recognition/{FOUNDATIONS[family]}",
            "learning_rate": 0.0001,
        }
    )


@pytest.mark.parametrize("family", tuple(FOUNDATIONS))
def test_each_family_completes_a_finite_step_and_checkpoint_round_trip(
    family: str, tmp_path: Path
) -> None:
    config = _config(family)
    foundation = ROOT / config.foundation_config
    bundle = build_public_recognition_family(config.family, foundation_path=foundation)
    target = TurkishPlateTextGenerator(seed=91027).generate(1).normalized_text
    images = torch.linspace(0.0, 1.0, 3 * 32 * 160).reshape(1, 3, 32, 160)
    optimizer, _scheduler = build_optimizer_and_scheduler(bundle.model, config)

    loss = train_one_step(bundle, images, (target,), optimizer)
    metrics = validate_one_batch(bundle, images, (target,))

    assert np.isfinite(loss)
    assert set(metrics) == {
        "exact_accuracy_percent",
        "character_accuracy_percent",
        "length_accuracy_percent",
    }
    assert all(0.0 <= value <= 100.0 for value in metrics.values())

    checkpoint = tmp_path / f"{family}.pt"
    receipt = save_public_training_checkpoint(checkpoint, bundle=bundle)
    assert receipt.checkpoint_sha256 is not None
    reloaded = build_public_recognition_family(config.family, foundation_path=foundation)
    explicitly_resume_public_checkpoint(
        checkpoint,
        bundle=reloaded,
        expected_sha256=receipt.checkpoint_sha256,
    )
    for expected, actual in zip(
        bundle.model.state_dict().values(), reloaded.model.state_dict().values(), strict=True
    ):
        assert torch.equal(expected, actual)


def test_selected_transformer_slot_family_has_expected_output_contract() -> None:
    config = _config("cnn_transformer_dual_slot_tr_v1")
    bundle = build_public_recognition_family(
        config.family, foundation_path=ROOT / config.foundation_config
    )
    with torch.inference_mode():
        output = bundle.model(torch.zeros((1, 3, 32, 160)))
    assert output.character_logits.shape == (1, 8, 37)
    assert output.type_logits.shape == (1, 8, 3)
    assert output.slot_attention.shape == (1, 8, 40)


def test_public_dataset_loads_user_supplied_image_label_pairs(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    assert cv2.imwrite((images / "fixture.png").as_posix(), np.zeros((20, 80, 3), np.uint8))
    target = TurkishPlateTextGenerator(seed=81003).generate(1).normalized_text
    (labels / "fixture.txt").write_text(f"{target}\n", encoding="utf-8")

    dataset = PublicRecognitionDataset(tmp_path)
    image, loaded_target = dataset[0]

    assert image.shape == (3, 32, 160)
    assert image.dtype is torch.float32
    assert loaded_target == target


@pytest.mark.parametrize(
    "value",
    (
        "/absolute/foundation.yaml",
        "../outside.yaml",
        "configs/private/foundation.yaml",
        "artifacts/experiments/foundation.yaml",
        "Users/local-owner/foundation.yaml",
    ),
)
def test_public_configuration_rejects_unsafe_embedded_paths(value: str) -> None:
    with pytest.raises(ValueError, match="portable"):
        PublicTrainingConfig.model_validate(
            {
                "schema_version": "public-recognition-training-v1",
                "family": "cnn_transformer_dual_slot_tr_v1",
                "foundation_config": value,
            }
        )


def test_public_configuration_rejects_unknown_private_fields() -> None:
    with pytest.raises(ValueError):
        PublicTrainingConfig.model_validate(
            {
                "schema_version": "public-recognition-training-v1",
                "family": "cnn_transformer_dual_slot_tr_v1",
                "foundation_config": (
                    "configs/recognition/cnn-transformer-dual-slot-tr-v1-foundation.yaml"
                ),
                "dataset_identity": "not-accepted",
            }
        )
