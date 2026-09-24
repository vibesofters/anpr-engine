"""Static and smoke contracts for permanent M1 fixed-slot Recognition."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from anpr_engine.domain import ResultStatus
from anpr_engine.recognition.adapters import adapt_fixed_slot_batch, adapt_fixed_slot_logits
from anpr_engine.recognition.charset import V1_CHARSET
from anpr_engine.recognition.checkpointing import (
    RecognitionCheckpointMetadata,
    load_recognition_checkpoint,
    save_recognition_checkpoint,
)
from anpr_engine.recognition.decoder import ProbabilityAwareTrDecoder
from anpr_engine.recognition.enhanced_config import load_enhanced_foundation_config
from anpr_engine.recognition.enhanced_model import ENHANCED_MODEL_ID, build_enhanced_recognizer
from anpr_engine.recognition.evaluation import compute_common_recognition_evaluation_metrics
from anpr_engine.recognition.fixed_slot import (
    FIXED_SLOT_CHARACTER_PAD_INDEX,
    FIXED_SLOT_TYPE_PAD_INDEX,
    CrnnBilstmDualSlotRecognizer,
    FixedSlotTargetError,
    build_fixed_slot_recognizer,
    encode_fixed_slot_targets,
    fixed_slot_loss,
    fixed_slot_parameter_breakdown,
)
from anpr_engine.recognition.fixed_slot_config import (
    M1_MODEL_ID,
    FixedSlotRecognitionFoundationConfig,
    fixed_slot_config_sha256,
    load_fixed_slot_foundation_config,
)
from anpr_engine.recognition.model import parameter_count
from anpr_engine.recognition.profiles import TR_PROFILE
from anpr_engine.recognition.registry import (
    M0_CAPABILITY,
    M1_1_CAPABILITY,
    M1_CAPABILITY,
    CheckpointCompatibilityError,
    implemented_capabilities,
    implemented_capability,
)

ROOT = Path(__file__).resolve().parents[1]
M0_FOUNDATION = ROOT / "configs/recognition/crnn-bilstm-dual-ctc-tr-v1-foundation.yaml"
M1_FOUNDATION = ROOT / "configs/recognition/crnn-bilstm-dual-slot-tr-v1-foundation.yaml"
LEDGER = ROOT / "docs/reference/recognition-four-family-summary.md"


def _m1() -> tuple[FixedSlotRecognitionFoundationConfig, CrnnBilstmDualSlotRecognizer]:
    config = load_fixed_slot_foundation_config(M1_FOUNDATION)
    return config, build_fixed_slot_recognizer(
        config.model,
        max_slots=config.slot_extraction.max_slots,
    )


def _slot_logits(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    character = torch.full((8, 37), -12.0)
    types = torch.full((8, 3), -12.0)
    lookup = {symbol: index for index, symbol in enumerate(V1_CHARSET.symbols)}
    for slot in range(8):
        if slot < len(text):
            character[slot, lookup[text[slot]]] = 12.0
            types[slot, 0 if text[slot].isdigit() else 1] = 12.0
        else:
            character[slot, FIXED_SLOT_CHARACTER_PAD_INDEX] = 12.0
            types[slot, FIXED_SLOT_TYPE_PAD_INDEX] = 12.0
    return character, types


def _decoder() -> ProbabilityAwareTrDecoder:
    config = load_fixed_slot_foundation_config(M1_FOUNDATION)
    return ProbabilityAwareTrDecoder(
        charset=V1_CHARSET,
        profile=TR_PROFILE,
        config=config.decoder.model_copy(update={"minimum_decoder_confidence": 0.0}),
    )


def test_m0_and_m1_registry_coexist_without_current_model_alias() -> None:
    capabilities = implemented_capabilities()
    assert tuple(item.model_id for item in capabilities)[:2] == (ENHANCED_MODEL_ID, M1_MODEL_ID)
    assert implemented_capability(ENHANCED_MODEL_ID) is M0_CAPABILITY
    assert implemented_capability(M1_MODEL_ID) is M1_CAPABILITY
    assert M0_CAPABILITY.model_id != M1_CAPABILITY.model_id
    assert M0_CAPABILITY.output_family.value == "CTC"
    assert M1_CAPABILITY.output_family.value == "FIXED_SLOTS"
    assert M0_CAPABILITY.checkpoint_metadata()["max_slots"] is None
    assert M1_CAPABILITY.checkpoint_metadata()["max_slots"] == 8
    assert M1_1_CAPABILITY.model_id not in (ENHANCED_MODEL_ID, M1_MODEL_ID)


def test_m1_foundation_is_independent_and_m0_foundation_is_unchanged() -> None:
    m0 = load_enhanced_foundation_config(M0_FOUNDATION)
    m1 = load_fixed_slot_foundation_config(M1_FOUNDATION)
    assert m0.model_id == "crnn_bilstm_dual_ctc_tr_v1"
    assert m0.model.architecture == "crnn_bilstm_dual_ctc_tr_v1"
    assert m1.model_id == "crnn_bilstm_dual_slot_tr_v1"
    assert m1.model.architecture == "crnn_bilstm_dual_slot_tr_v1"
    assert m1.slot_extraction.max_slots == 8
    assert m1.character_head.pad_index == 36
    assert m1.type_head.classes == ("DIGIT", "LETTER", "PAD")
    assert m1.type_head.pad_index == 2
    assert m1.loss.lambda_type == 0.20
    assert m1.crop.model_dump() == m0.crop.model_dump()
    assert m1.model.cnn_channels == m0.model.cnn_channels
    assert m1.model.lstm_hidden_size == m0.model.lstm_hidden_size
    assert m1.model.lstm_layers == m0.model.lstm_layers
    assert m1.model.dropout == m0.model.dropout
    assert m1.model.initialization_seed == m0.model.initialization_seed
    assert m1.artifacts.private_weights_included is False
    assert m1.artifacts.runtime_weight_location == "externally-configured"


def test_fixed_slot_targets_pad_types_repeats_and_order() -> None:
    targets = encode_fixed_slot_targets(("34ABC123", "34A1234", "34AA111", "34BB122"), max_slots=8)
    lookup = {symbol: index for index, symbol in enumerate(V1_CHARSET.symbols)}
    assert targets.character_indexes[0].tolist() == [lookup[symbol] for symbol in "34ABC123"]
    assert targets.character_indexes[1].tolist() == [
        *[lookup[symbol] for symbol in "34A1234"],
        FIXED_SLOT_CHARACTER_PAD_INDEX,
    ]
    assert targets.type_indexes[0].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert targets.type_indexes[1].tolist() == [0, 0, 1, 0, 0, 0, 0, 2]
    assert targets.lengths.tolist() == [8, 7, 7, 7]
    assert targets.character_indexes[2, 2].item() == targets.character_indexes[2, 3].item()
    assert targets.character_indexes[2, 4].item() == targets.character_indexes[2, 5].item()
    assert targets.character_indexes[3, 2].item() == targets.character_indexes[3, 3].item()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("34abc123", "already be normalized"),
        ("34ABÇ12", "unsupported character"),
        ("34ABC1234", "exceeds max_slots"),
        ("99AB12", "outside the governed TR V1 profile"),
    ],
)
def test_fixed_slot_targets_reject_instead_of_truncate_or_guess(text: str, message: str) -> None:
    with pytest.raises(FixedSlotTargetError, match=message):
        encode_fixed_slot_targets((text,), max_slots=8)


@pytest.mark.parametrize("batch", (1, 3))
def test_m1_cpu_forward_shapes_finiteness_and_common_adapter(batch: int) -> None:
    config, model = _m1()
    model.eval()
    images = torch.linspace(0.0, 1.0, batch * 3 * 32 * 160).reshape(batch, 3, 32, 160)
    with torch.inference_mode():
        output = model(images)
    assert output.character_logits.shape == (batch, 8, 37)
    assert output.type_logits.shape == (batch, 8, 3)
    assert output.global_embedding.shape == (batch, 256)
    assert output.slot_attention.shape == (batch, 8, 40)
    assert torch.isfinite(output.character_logits).all()
    assert torch.isfinite(output.type_logits).all()
    assert torch.allclose(output.slot_attention.sum(dim=2), torch.ones((batch, 8)))
    common = adapt_fixed_slot_batch(
        output.character_logits,
        output.type_logits,
        capability=M1_CAPABILITY,
        decoder_config=config.decoder,
    )
    assert len(common) == batch
    assert all(item.predicted_length <= 8 for item in common)


@pytest.mark.parametrize("plate", ("34AA111", "34BB122", "34ABC123"))
def test_m1_structural_raw_repeat_pad_decoder_and_common_metrics(plate: str) -> None:
    character, types = _slot_logits(plate)
    output = adapt_fixed_slot_logits(
        character,
        types,
        capability=M1_CAPABILITY,
        decoder_config=_decoder().config,
    )
    decoded = _decoder().decode_common(output)
    metrics = compute_common_recognition_evaluation_metrics(
        targets=(plate,),
        outputs=(output,),
        decoded_predictions=(decoded.decoded_text,),
        statuses=(decoded.status,),
    )
    assert output.raw_sequence_valid
    assert output.raw_text == plate
    assert output.predicted_length == len(plate)
    assert decoded.decoded_text == plate
    assert decoded.status is ResultStatus.ACCEPTED
    assert metrics.recognition.exact_normalized_plate_accuracy == 1.0
    assert metrics.recognition.plate_length_accuracy == 1.0
    assert metrics.decoder.type_sequence_accuracy == 1.0


def test_m1_non_pad_after_pad_is_visible_and_invalid() -> None:
    character, types = _slot_logits("34ABC123")
    character[3].fill_(-12.0)
    character[3, FIXED_SLOT_CHARACTER_PAD_INDEX] = 12.0
    output = adapt_fixed_slot_logits(
        character,
        types,
        capability=M1_CAPABILITY,
        decoder_config=_decoder().config,
    )
    assert output.raw_text == "34AC123"
    assert output.selected_character_symbols[3] == "PAD"
    assert "INTERNAL_CHARACTER_PAD" in output.invalid_reasons
    assert "NON_PAD_AFTER_PAD" in output.invalid_reasons
    assert _decoder().decode_common(output).decoded_text is None


def test_m1_cpu_loss_backward_reaches_encoder_slots_and_heads() -> None:
    config, model = _m1()
    model.train()
    images = torch.linspace(0.0, 1.0, 2 * 3 * 32 * 160).reshape(2, 3, 32, 160)
    targets = encode_fixed_slot_targets(("34AA111", "06ABC123"), max_slots=8)
    output = model(images)
    losses = fixed_slot_loss(output, targets, lambda_type=config.loss.lambda_type)
    assert torch.isfinite(losses.character_slot_cross_entropy)
    assert torch.isfinite(losses.type_slot_cross_entropy)
    assert torch.isfinite(losses.total)
    losses.total.backward()  # type: ignore[no-untyped-call]
    parameters = dict(model.named_parameters())
    for name in (
        "features.0.0.weight",
        "sequence_model.weight_ih_l0",
        "slot_extraction.queries",
        "character_head.weight",
        "type_head.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert float(gradient.norm().item()) > 0.0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_m1_native_mps_slot_loss_backward_smoke() -> None:
    config, model = _m1()
    model = model.to("mps")
    model.train()
    images = torch.linspace(0.0, 1.0, 3 * 32 * 160, device="mps").reshape(1, 3, 32, 160)
    targets = encode_fixed_slot_targets(("34AA111",), max_slots=8).to("mps")
    output = model(images)
    losses = fixed_slot_loss(output, targets, lambda_type=config.loss.lambda_type)
    assert losses.total.device.type == "mps"
    assert torch.isfinite(losses.total).cpu().item()
    losses.total.backward()  # type: ignore[no-untyped-call]
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all().cpu().item()


def test_m1_serialization_metadata_output_parity_and_cross_model_rejection(
    tmp_path: Path,
) -> None:
    config, model = _m1()
    model.eval()
    images = torch.linspace(0.0, 1.0, 3 * 32 * 160).reshape(1, 3, 32, 160)
    with torch.inference_mode():
        before = model(images)
    metadata = RecognitionCheckpointMetadata(
        capability=M1_CAPABILITY,
        parameter_count=parameter_count(model),
        config_sha256=fixed_slot_config_sha256(config),
        training_run_identity="STATIC_SMOKE_NOT_TRAINED",
        dataset_identity="NONE_STATIC_SMOKE",
    )
    checkpoint = tmp_path / M1_MODEL_ID / "static-smoke.pt"
    receipt = save_recognition_checkpoint(checkpoint, model=model, metadata=metadata)
    assert receipt.checkpoint_sha256 is not None

    _, reloaded = _m1()
    loaded = load_recognition_checkpoint(
        checkpoint,
        model=reloaded,
        expected_capability=M1_CAPABILITY,
        expected_sha256=receipt.checkpoint_sha256,
        expected_config_sha256=fixed_slot_config_sha256(config),
    )
    reloaded.eval()
    with torch.inference_mode():
        after = reloaded(images)
    assert loaded["training_run_identity"] == "STATIC_SMOKE_NOT_TRAINED"
    assert torch.equal(before.character_logits, after.character_logits)
    assert torch.equal(before.type_logits, after.type_logits)
    assert torch.equal(before.slot_attention, after.slot_attention)

    m0_config = load_enhanced_foundation_config(M0_FOUNDATION)
    m0 = build_enhanced_recognizer(
        m0_config.model,
        character_class_count=V1_CHARSET.class_count,
    )
    with pytest.raises(CheckpointCompatibilityError, match="model_id"):
        load_recognition_checkpoint(
            checkpoint,
            model=m0,
            expected_capability=M0_CAPABILITY,
        )

    _, wrong_config_model = _m1()
    with pytest.raises(ValueError, match="config identity"):
        load_recognition_checkpoint(
            checkpoint,
            model=wrong_config_model,
            expected_capability=M1_CAPABILITY,
            expected_config_sha256="0" * 64,
        )


def test_m1_parameter_isolation_and_onnx_environment_status() -> None:
    m0_config = load_enhanced_foundation_config(M0_FOUNDATION)
    m0 = build_enhanced_recognizer(
        m0_config.model,
        character_class_count=V1_CHARSET.class_count,
    )
    _, m1 = _m1()
    for m0_tensor, m1_tensor in zip(
        m0.features.state_dict().values(),
        m1.features.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(m0_tensor, m1_tensor)
    for m0_tensor, m1_tensor in zip(
        m0.sequence_model.state_dict().values(),
        m1.sequence_model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(m0_tensor, m1_tensor)
    breakdown = fixed_slot_parameter_breakdown(m1)
    assert parameter_count(m0) == 763_208
    assert parameter_count(m1) == 765_256
    assert parameter_count(m1) - parameter_count(m0) == 2_048
    assert breakdown == {
        "slot_extraction": 2_048,
        "character_slot_head": 9_509,
        "type_slot_head": 771,
    }
    assert importlib.util.find_spec("onnx") is None
    assert importlib.util.find_spec("onnxruntime") is None


def test_four_model_ledger_is_append_only_evidence_not_a_roadmap() -> None:
    content = LEDGER.read_text(encoding="utf-8")
    for family in (
        "CNN + BiLSTM + CTC",
        "CNN + BiLSTM + Fixed Slots",
        "CNN Tokenizer + Transformer + CTC",
        "CNN Tokenizer + Transformer + Fixed Slots",
    ):
        assert family in content
    assert "selected architecture" in content
    assert "percentage-only development-validation" in content
    assert "population sizes" in content
