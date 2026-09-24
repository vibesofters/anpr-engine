"""Synthetic smoke coverage for every retained project-owned model family."""

from pathlib import Path

import torch

from anpr_engine.recognition.charset import V1_CHARSET
from anpr_engine.recognition.enhanced_config import load_enhanced_foundation_config
from anpr_engine.recognition.enhanced_model import build_enhanced_recognizer
from anpr_engine.recognition.fixed_slot import build_fixed_slot_recognizer
from anpr_engine.recognition.fixed_slot_config import load_fixed_slot_foundation_config
from anpr_engine.recognition.transformer_config import load_transformer_foundation_config
from anpr_engine.recognition.transformer_model import build_transformer_recognizer
from anpr_engine.recognition.transformer_slot_config import (
    load_transformer_fixed_slot_foundation_config,
)
from anpr_engine.recognition.transformer_slot_model import (
    build_transformer_fixed_slot_recognizer,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "recognition"


def test_all_retained_recognition_architectures_run_on_synthetic_cpu_input() -> None:
    """Public architecture code must not depend on private files or fixtures."""
    image = torch.zeros((1, 3, 32, 160), dtype=torch.float32)

    ctc = load_enhanced_foundation_config(CONFIGS / "crnn-bilstm-dual-ctc-tr-v1-foundation.yaml")
    bilstm_ctc = build_enhanced_recognizer(
        ctc.model, character_class_count=V1_CHARSET.class_count
    ).eval()

    slot = load_fixed_slot_foundation_config(
        CONFIGS / "crnn-bilstm-dual-slot-tr-v1-foundation.yaml"
    )
    bilstm_slot = build_fixed_slot_recognizer(
        slot.model, max_slots=slot.slot_extraction.max_slots
    ).eval()

    transformer_ctc_config = load_transformer_foundation_config(
        CONFIGS / "cnn-transformer-dual-ctc-tr-v1-foundation.yaml"
    )
    transformer_ctc = build_transformer_recognizer(transformer_ctc_config.model).eval()

    transformer_slot_config = load_transformer_fixed_slot_foundation_config(
        CONFIGS / "cnn-transformer-dual-slot-tr-v1-foundation.yaml"
    )
    transformer_slot = build_transformer_fixed_slot_recognizer(
        transformer_slot_config.model,
        max_slots=transformer_slot_config.slot_extraction.max_slots,
    ).eval()

    with torch.inference_mode():
        outputs = (
            bilstm_ctc(image),
            bilstm_slot(image),
            transformer_ctc(image),
            transformer_slot(image),
        )

    for output in outputs:
        assert 1 in output.character_logits.shape[:2]
        assert 1 in output.type_logits.shape[:2]
        assert torch.isfinite(output.character_logits).all()
        assert torch.isfinite(output.type_logits).all()
