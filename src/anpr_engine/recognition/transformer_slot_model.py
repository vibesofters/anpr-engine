"""M3 recognizer: unchanged M2 encoder followed by ordered fixed-slot supervision."""

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
from anpr_engine.recognition.fixed_slot_local import ordered_gaussian_locality_bias
from anpr_engine.recognition.transformer_model import build_transformer_encoder_modules
from anpr_engine.recognition.transformer_slot_config import (
    TransformerFixedSlotRecognizerModelConfig,
)


class LearnedOrderedTransformerSlotReducer(nn.Module):
    """Learned content queries with the frozen M1.1 left-to-right locality prior."""

    def __init__(self, *, max_slots: int, embedding_size: int) -> None:
        super().__init__()
        if max_slots < 1 or embedding_size < 1:
            raise ValueError("ordered Transformer slot dimensions must be positive")
        self.max_slots = max_slots
        self.embedding_size = embedding_size
        self.queries = nn.Parameter(torch.empty(max_slots, embedding_size))
        nn.init.normal_(self.queries, mean=0.0, std=embedding_size**-0.5)

    def locality_bias(self, encoded: Tensor) -> Tensor:
        if encoded.ndim != 3 or encoded.shape[2] != self.embedding_size:
            raise ValueError("M3 slot reducer expects [batch, tokens, embedding] states")
        return ordered_gaussian_locality_bias(
            encoded.shape[1],
            self.max_slots,
            device=encoded.device,
            dtype=encoded.dtype,
        )

    def forward(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        bias = self.locality_bias(encoded)
        content_scores = torch.einsum("sd,btd->bst", self.queries, encoded) / math.sqrt(
            self.embedding_size
        )
        attention = (content_scores + bias.unsqueeze(0)).softmax(dim=2)
        slots = torch.einsum("bst,btd->bsd", attention, encoded)
        return slots, attention


class CnnTransformerDualSlotRecognizer(nn.Module):
    """M3: permanent M2 representation mechanics with M1-compatible fixed slots."""

    def __init__(
        self,
        config: TransformerFixedSlotRecognizerModelConfig,
        *,
        max_slots: int,
    ) -> None:
        super().__init__()
        self.config = config
        (
            self.tokenizer,
            self.position,
            self.input_dropout,
            self.transformer,
        ) = build_transformer_encoder_modules(config)
        embedding = config.transformer.embedding_dimension
        self.slot_extraction = LearnedOrderedTransformerSlotReducer(
            max_slots=max_slots,
            embedding_size=embedding,
        )
        self.character_head = nn.Linear(embedding, len(FIXED_SLOT_CHARACTER_SYMBOLS) + 1)
        self.type_head = nn.Linear(embedding, len(FIXED_SLOT_TYPE_SYMBOLS))

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
                "expected fixed M3 input geometry "
                f"{self.config.input_height}x{self.config.input_width}, got {height}x{width}"
            )
        if not images.is_floating_point():
            raise ValueError("recognizer input must be floating-point")

        tokenized = self.tokenizer(images)
        positioned = self.input_dropout(self.position(tokenized.tokens))
        encoded = self.transformer(positioned)
        slots, attention = self.slot_extraction(encoded)
        return FixedSlotRecognizerOutput(
            character_logits=cast(Tensor, self.character_head(slots)),
            type_logits=cast(Tensor, self.type_head(slots)),
            global_embedding=encoded.mean(dim=1),
            slot_attention=attention,
        )


def build_transformer_fixed_slot_recognizer(
    config: TransformerFixedSlotRecognizerModelConfig,
    *,
    max_slots: int,
) -> CnnTransformerDualSlotRecognizer:
    """Construct M3 deterministically without changing the caller CPU RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.initialization_seed)
        return CnnTransformerDualSlotRecognizer(config, max_slots=max_slots)


def transformer_fixed_slot_parameter_breakdown(
    model: CnnTransformerDualSlotRecognizer,
) -> dict[str, int]:
    """Expose M3 capacity and the exact delta relative to M2."""
    return {
        "tokenizer": sum(parameter.numel() for parameter in model.tokenizer.parameters()),
        "transformer": sum(parameter.numel() for parameter in model.transformer.parameters()),
        "slot_extraction": sum(
            parameter.numel() for parameter in model.slot_extraction.parameters()
        ),
        "character_slot_head": sum(
            parameter.numel() for parameter in model.character_head.parameters()
        ),
        "type_slot_head": sum(parameter.numel() for parameter in model.type_head.parameters()),
        "positional_encoding": 0,
    }
