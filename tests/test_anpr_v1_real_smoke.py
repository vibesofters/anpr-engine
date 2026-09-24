"""Private-bundle smoke checks using non-identifying in-memory tensors only."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from anpr_engine.integration.anpr_v1 import AnprV1Pipeline


def test_project_models_load_once_and_accept_synthetic_tensors() -> None:
    repository = Path(__file__).parents[1]
    configured = os.environ.get("ANPR_PRIVATE_MODEL_BUNDLE")
    if not configured:
        pytest.skip("ANPR_PRIVATE_MODEL_BUNDLE is required for the private-model smoke test")
    bundle = Path(configured)
    if not all((bundle / name).is_file() for name in ("detection.pt", "recognition.pt")):
        pytest.skip("private model bundle is intentionally not committed")

    pipeline = AnprV1Pipeline(repository, model_bundle=bundle, device_name="cpu")
    with torch.inference_mode():
        detector_output = pipeline.detector(torch.zeros((1, 3, 384, 384)))
        recognizer_output = pipeline.recognizer(torch.zeros((1, 3, 32, 160)))

    assert pipeline.load_count == 1
    assert detector_output is not None
    assert tuple(recognizer_output.character_logits.shape) == (1, 8, 37)
    assert tuple(recognizer_output.type_logits.shape) == (1, 8, 3)
    assert np.isfinite(recognizer_output.character_logits.detach().numpy()).all()
    assert pipeline.health()["status"] == "HEALTHY"
