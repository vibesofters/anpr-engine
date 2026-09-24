"""Deterministic digit/letter CTC supervision for the enhanced recognizer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from anpr_engine.recognition.charset import PlateCharset
from anpr_engine.recognition.ctc import CtcTargets, encode_transcriptions

TYPE_SYMBOLS = "DL"
TYPE_CLASSES = ("DIGIT", "LETTER")
TYPE_CTC_BLANK_INDEX = 0
TYPE_CTC_CHARSET = PlateCharset(
    version="character-type-ctc-v1",
    symbols=TYPE_SYMBOLS,
    blank_index=TYPE_CTC_BLANK_INDEX,
)


class CtcLossExecutionPolicy(StrEnum):
    """Where native CTC executes relative to the model's logits."""

    NATIVE_DEVICE = "NATIVE_DEVICE"
    MPS_MODEL_CPU_CTC = "MPS_MODEL_CPU_CTC"


class CtcLossExecutionMetadata(BaseModel):
    """Machine-readable runtime identity for a dual-CTC training step."""

    model_device: str
    character_ctc_device: str
    type_ctc_device: str
    policy: CtcLossExecutionPolicy
    autograd_preserving_device_bridge: bool
    description: str

    model_config = ConfigDict(extra="forbid", frozen=True)


def ctc_loss_execution_metadata(
    policy: CtcLossExecutionPolicy, *, model_device: torch.device
) -> CtcLossExecutionMetadata:
    """Describe the loss path and reject a hybrid bridge outside its safe scope."""
    if policy is CtcLossExecutionPolicy.MPS_MODEL_CPU_CTC:
        if model_device.type != "mps":
            raise ValueError("MPS_MODEL_CPU_CTC requires MPS-resident model logits")
        return CtcLossExecutionMetadata(
            model_device="mps",
            character_ctc_device="cpu",
            type_ctc_device="cpu",
            policy=policy,
            autograd_preserving_device_bridge=True,
            description=(
                "Model forward/backward on MPS; native CTC loss evaluated on CPU through "
                "an autograd-preserving device bridge."
            ),
        )
    return CtcLossExecutionMetadata(
        model_device=model_device.type,
        character_ctc_device=model_device.type,
        type_ctc_device=model_device.type,
        policy=policy,
        autograd_preserving_device_bridge=False,
        description="Native CTC loss executes on the model-logit device.",
    )


def character_type_target(text: str) -> str:
    if not text:
        raise ValueError("character type target cannot be empty")
    types: list[str] = []
    for character in text:
        if character.isascii() and character.isdigit():
            types.append("D")
        elif character.isascii() and character.isalpha() and character.isupper():
            types.append("L")
        else:
            raise ValueError(f"unsupported character for type target: {character!r}")
    return "".join(types)


def encode_type_targets(texts: tuple[str, ...]) -> CtcTargets:
    return encode_transcriptions(
        tuple(character_type_target(text) for text in texts), TYPE_CTC_CHARSET
    )


@dataclass(frozen=True)
class DualCtcLoss:
    total: Tensor
    character: Tensor
    character_type: Tensor
    lambda_type: float
    execution: CtcLossExecutionMetadata


def dual_ctc_loss(
    character_logits: Tensor,
    type_logits: Tensor,
    *,
    texts: tuple[str, ...],
    character_charset: PlateCharset,
    lambda_type: float,
    execution_policy: CtcLossExecutionPolicy = CtcLossExecutionPolicy.NATIVE_DEVICE,
) -> DualCtcLoss:
    if not 0.0 <= lambda_type <= 1.0:
        raise ValueError("lambda_type must be between zero and one")
    if character_logits.ndim != 3 or type_logits.ndim != 3:
        raise ValueError("dual CTC logits must have shape [time, batch, classes]")
    if character_logits.shape[:2] != type_logits.shape[:2]:
        raise ValueError("character and type logits must share time and batch axes")
    if character_logits.shape[1] != len(texts):
        raise ValueError("target count must match the logits batch axis")
    character_targets = encode_transcriptions(texts, character_charset)
    type_targets = encode_type_targets(texts)
    execution = ctc_loss_execution_metadata(execution_policy, model_device=character_logits.device)
    loss_device = torch.device(execution.character_ctc_device)
    character_loss_logits = character_logits.to(loss_device)
    type_loss_logits = type_logits.to(loss_device)
    input_lengths = torch.full(
        (len(texts),),
        character_loss_logits.shape[0],
        dtype=torch.long,
        device=loss_device,
    )
    ctc = nn.CTCLoss(blank=character_charset.blank_index, zero_infinity=False)
    character_loss = ctc(
        character_loss_logits.log_softmax(2),
        character_targets.values.to(loss_device),
        input_lengths,
        character_targets.lengths.to(loss_device),
    )
    type_loss = nn.CTCLoss(blank=TYPE_CTC_BLANK_INDEX, zero_infinity=False)(
        type_loss_logits.log_softmax(2),
        type_targets.values.to(loss_device),
        input_lengths,
        type_targets.lengths.to(loss_device),
    )
    return DualCtcLoss(
        total=character_loss + lambda_type * type_loss,
        character=character_loss,
        character_type=type_loss,
        lambda_type=lambda_type,
        execution=execution,
    )
