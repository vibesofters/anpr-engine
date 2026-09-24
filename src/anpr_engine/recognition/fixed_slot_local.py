"""M1.1 fixed-slot recognizer with a parameter-free ordered Gaussian prior."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from anpr_engine.recognition.fixed_slot import (
    FIXED_SLOT_CHARACTER_SYMBOLS,
    FIXED_SLOT_TYPE_SYMBOLS,
    FixedSlotRecognizerOutput,
)
from anpr_engine.recognition.fixed_slot_local_config import (
    FixedSlotLocalRecognizerModelConfig,
)
from anpr_engine.recognition.model import ConvFeatureBlock


def ordered_slot_anchor_centers(
    sequence_length: int,
    max_slots: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return equal-cell centers spanning the dynamic sequence axis left to right."""
    if sequence_length < 1 or max_slots < 1:
        raise ValueError("sequence_length and max_slots must be positive")
    spacing = sequence_length / max_slots
    slots = torch.arange(max_slots, device=device, dtype=dtype)
    return (slots + 0.5) * spacing - 0.5


def ordered_gaussian_locality_bias(
    sequence_length: int,
    max_slots: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return [slots, time] fixed Gaussian logits with sigma equal to anchor spacing."""
    centers = ordered_slot_anchor_centers(
        sequence_length,
        max_slots,
        device=device,
        dtype=dtype,
    )
    positions = torch.arange(sequence_length, device=device, dtype=dtype)
    sigma = sequence_length / max_slots
    return -0.5 * ((positions.unsqueeze(0) - centers.unsqueeze(1)) / sigma).square()


class LearnedOrderedLocalityQueryPool(nn.Module):
    """M1 queries plus a soft, fixed, left-to-right Gaussian logit prior."""

    def __init__(self, *, max_slots: int, embedding_size: int) -> None:
        super().__init__()
        if max_slots < 1 or embedding_size < 1:
            raise ValueError("ordered locality query dimensions must be positive")
        self.max_slots = max_slots
        self.embedding_size = embedding_size
        self.queries = nn.Parameter(torch.empty(max_slots, embedding_size))
        nn.init.normal_(self.queries, mean=0.0, std=embedding_size**-0.5)

    def locality_bias(self, encoded: Tensor) -> Tensor:
        if encoded.ndim != 3 or encoded.shape[2] != self.embedding_size:
            raise ValueError("ordered locality pool expects [time, batch, embedding] states")
        return ordered_gaussian_locality_bias(
            encoded.shape[0],
            self.max_slots,
            device=encoded.device,
            dtype=encoded.dtype,
        )

    def forward(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        bias = self.locality_bias(encoded)
        content_scores = torch.einsum("sd,tbd->bst", self.queries, encoded) / math.sqrt(
            self.embedding_size
        )
        attention = (content_scores + bias.unsqueeze(0)).softmax(dim=2)
        slots = torch.einsum("bst,tbd->bsd", attention, encoded)
        return slots, attention


class CrnnBilstmDualSlotLocalRecognizer(nn.Module):
    """M1.1: frozen M1 encoder/heads with only the attention score changed."""

    def __init__(self, config: FixedSlotLocalRecognizerModelConfig, *, max_slots: int) -> None:
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
        self.slot_extraction = LearnedOrderedLocalityQueryPool(
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
                "expected fixed M1.1 input geometry "
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


def build_fixed_slot_local_recognizer(
    config: FixedSlotLocalRecognizerModelConfig,
    *,
    max_slots: int,
) -> CrnnBilstmDualSlotLocalRecognizer:
    """Construct M1.1 deterministically without changing the caller's CPU RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.initialization_seed)
        return CrnnBilstmDualSlotLocalRecognizer(config, max_slots=max_slots)
