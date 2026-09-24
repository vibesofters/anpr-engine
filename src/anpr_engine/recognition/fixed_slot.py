"""Permanent M1 CNN/BiLSTM/fixed-slot model, targets, and loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from anpr_engine.recognition.charset import V1_CHARSET
from anpr_engine.recognition.fixed_slot_config import FixedSlotRecognizerModelConfig
from anpr_engine.recognition.model import ConvFeatureBlock
from anpr_engine.recognition.normalization import PlateNormalizationError, normalize_plate_text
from anpr_engine.recognition.profiles import TR_PROFILE
from anpr_engine.recognition.type_ctc import character_type_target

FIXED_SLOT_CHARACTER_SYMBOLS = V1_CHARSET.symbols
FIXED_SLOT_CHARACTER_PAD_INDEX = len(FIXED_SLOT_CHARACTER_SYMBOLS)
FIXED_SLOT_TYPE_SYMBOLS = ("DIGIT", "LETTER", "PAD")
FIXED_SLOT_TYPE_PAD_INDEX = 2


class FixedSlotTargetError(ValueError):
    """Raised rather than truncating or guessing an invalid fixed-slot target."""


@dataclass(frozen=True)
class FixedSlotTargets:
    character_indexes: Tensor
    type_indexes: Tensor
    lengths: Tensor
    non_pad_mask: Tensor

    def to(self, device: torch.device | str) -> FixedSlotTargets:
        return FixedSlotTargets(
            character_indexes=self.character_indexes.to(device),
            type_indexes=self.type_indexes.to(device),
            lengths=self.lengths.to(device),
            non_pad_mask=self.non_pad_mask.to(device),
        )


class FixedSlotRecognizerOutput(NamedTuple):
    character_logits: Tensor
    type_logits: Tensor
    global_embedding: Tensor
    slot_attention: Tensor


class FixedSlotLoss(NamedTuple):
    character_slot_cross_entropy: Tensor
    type_slot_cross_entropy: Tensor
    weighted_type_contribution: Tensor
    total: Tensor


class LearnedOrderedQueryPool(nn.Module):
    """Reduce ordered BiLSTM states to distinct supervised left-to-right slots."""

    def __init__(self, *, max_slots: int, embedding_size: int) -> None:
        super().__init__()
        if max_slots < 1 or embedding_size < 1:
            raise ValueError("ordered query dimensions must be positive")
        self.max_slots = max_slots
        self.embedding_size = embedding_size
        self.queries = nn.Parameter(torch.empty(max_slots, embedding_size))
        nn.init.normal_(self.queries, mean=0.0, std=embedding_size**-0.5)

    def forward(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        if encoded.ndim != 3 or encoded.shape[2] != self.embedding_size:
            raise ValueError("ordered query pool expects [time, batch, embedding] states")
        scores = torch.einsum("sd,tbd->bst", self.queries, encoded) / math.sqrt(self.embedding_size)
        attention = scores.softmax(dim=2)
        slots = torch.einsum("bst,tbd->bsd", attention, encoded)
        return slots, attention


class CrnnBilstmDualSlotRecognizer(nn.Module):
    """M1: M0-sized CNN/BiLSTM encoder with ordered Character/Type/PAD slots."""

    def __init__(self, config: FixedSlotRecognizerModelConfig, *, max_slots: int) -> None:
        super().__init__()
        self.config = config
        first, second, third = config.cnn_channels
        self.features = nn.Sequential(
            ConvFeatureBlock(config.input_channels, first, pool=(2, 2)),
            ConvFeatureBlock(first, second, pool=(2, 2)),
            ConvFeatureBlock(second, third, pool=(2, 1)),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.sequence_model = nn.LSTM(
            input_size=third,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        shared_size = config.lstm_hidden_size * 2
        self.slot_extraction = LearnedOrderedQueryPool(
            max_slots=max_slots,
            embedding_size=shared_size,
        )
        self.character_head = nn.Linear(shared_size, len(FIXED_SLOT_CHARACTER_SYMBOLS) + 1)
        self.type_head = nn.Linear(shared_size, len(FIXED_SLOT_TYPE_SYMBOLS))

    def forward(self, images: Tensor) -> FixedSlotRecognizerOutput:
        if images.ndim != 4:
            raise ValueError("recognizer input must have shape [batch, channels, height, width]")
        batch, channels, height, width = images.shape
        if batch < 1:
            raise ValueError("recognizer batch must be non-empty")
        if channels != self.config.input_channels:
            raise ValueError(
                f"expected {self.config.input_channels} input channels, got {channels}"
            )
        if (height, width) != (self.config.input_height, self.config.input_width):
            raise ValueError(
                "expected fixed M1 input geometry "
                f"{self.config.input_height}x{self.config.input_width}, got {height}x{width}"
            )
        if not images.is_floating_point():
            raise ValueError("recognizer input must be floating-point")
        features = self.features(images).squeeze(2)
        sequence = features.permute(2, 0, 1).contiguous()
        encoded, _ = self.sequence_model(sequence)
        slots, attention = self.slot_extraction(encoded)
        return FixedSlotRecognizerOutput(
            character_logits=cast(Tensor, self.character_head(slots)),
            type_logits=cast(Tensor, self.type_head(slots)),
            global_embedding=encoded.mean(dim=0),
            slot_attention=attention,
        )


def build_fixed_slot_recognizer(
    config: FixedSlotRecognizerModelConfig,
    *,
    max_slots: int,
) -> CrnnBilstmDualSlotRecognizer:
    """Construct M1 deterministically without changing the caller's CPU RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.initialization_seed)
        return CrnnBilstmDualSlotRecognizer(config, max_slots=max_slots)


def encode_fixed_slot_targets(
    texts: tuple[str, ...],
    *,
    max_slots: int,
) -> FixedSlotTargets:
    """Encode exact normalized TR V1 labels with explicit suffix PAD and no truncation."""
    if not texts:
        raise FixedSlotTargetError("fixed-slot target population must be non-empty")
    character_lookup = {symbol: index for index, symbol in enumerate(FIXED_SLOT_CHARACTER_SYMBOLS)}
    character_rows: list[list[int]] = []
    type_rows: list[list[int]] = []
    lengths: list[int] = []
    for text in texts:
        try:
            normalized = normalize_plate_text(text).normalized_text
        except PlateNormalizationError as error:
            raise FixedSlotTargetError(str(error)) from error
        if normalized != text:
            raise FixedSlotTargetError("fixed-slot target must already be normalized")
        unsupported = tuple(character for character in text if character not in character_lookup)
        if unsupported:
            raise FixedSlotTargetError(f"unsupported fixed-slot character: {unsupported[0]!r}")
        if len(text) > max_slots:
            raise FixedSlotTargetError(
                f"fixed-slot target length {len(text)} exceeds max_slots={max_slots}"
            )
        validation = TR_PROFILE.validate(text)
        if not validation.valid:
            raise FixedSlotTargetError("fixed-slot target is outside the governed TR V1 profile")
        character_indexes = [character_lookup[character] for character in text]
        character_indexes.extend(
            [FIXED_SLOT_CHARACTER_PAD_INDEX] * (max_slots - len(character_indexes))
        )
        type_indexes = [0 if symbol == "D" else 1 for symbol in character_type_target(text)]
        type_indexes.extend([FIXED_SLOT_TYPE_PAD_INDEX] * (max_slots - len(type_indexes)))
        character_rows.append(character_indexes)
        type_rows.append(type_indexes)
        lengths.append(len(text))
    character_tensor = torch.tensor(character_rows, dtype=torch.long)
    type_tensor = torch.tensor(type_rows, dtype=torch.long)
    return FixedSlotTargets(
        character_indexes=character_tensor,
        type_indexes=type_tensor,
        lengths=torch.tensor(lengths, dtype=torch.long),
        non_pad_mask=character_tensor.ne(FIXED_SLOT_CHARACTER_PAD_INDEX),
    )


def fixed_slot_loss(
    output: FixedSlotRecognizerOutput,
    targets: FixedSlotTargets,
    *,
    lambda_type: float,
) -> FixedSlotLoss:
    """Equal-per-slot CE teaches suffix termination without PAD-dominated weighting."""
    if not 0.0 <= lambda_type <= 1.0:
        raise ValueError("lambda_type must be between zero and one")
    expected = targets.character_indexes.shape
    if output.character_logits.shape[:2] != expected or output.type_logits.shape[:2] != expected:
        raise ValueError("fixed-slot logits and targets must share [batch, slots]")
    character = functional.cross_entropy(
        output.character_logits.reshape(-1, output.character_logits.shape[2]),
        targets.character_indexes.reshape(-1),
    )
    types = functional.cross_entropy(
        output.type_logits.reshape(-1, output.type_logits.shape[2]),
        targets.type_indexes.reshape(-1),
    )
    weighted_type = lambda_type * types
    return FixedSlotLoss(
        character_slot_cross_entropy=character,
        type_slot_cross_entropy=types,
        weighted_type_contribution=weighted_type,
        total=character + weighted_type,
    )


def fixed_slot_parameter_breakdown(model: CrnnBilstmDualSlotRecognizer) -> dict[str, int]:
    """Expose capacity added or replaced by the M1 output formulation."""
    return {
        "slot_extraction": sum(
            parameter.numel() for parameter in model.slot_extraction.parameters()
        ),
        "character_slot_head": sum(
            parameter.numel() for parameter in model.character_head.parameters()
        ),
        "type_slot_head": sum(parameter.numel() for parameter in model.type_head.parameters()),
    }
