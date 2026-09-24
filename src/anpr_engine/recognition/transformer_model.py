"""Project-owned M2 CNN-tokenizer/Transformer recognizer with dual CTC heads."""

from __future__ import annotations

import math
from typing import NamedTuple, cast

import torch
from torch import Tensor, nn

from anpr_engine.recognition.model import ConvFeatureBlock
from anpr_engine.recognition.transformer_config import (
    TransformerEncoderModelConfig,
    TransformerRecognizerModelConfig,
)


class TransformerRecognizerOutput(NamedTuple):
    character_logits: Tensor
    type_logits: Tensor
    global_embedding: Tensor
    tokens: Tensor
    encoded_tokens: Tensor


class CnnWidthTokenizerOutput(NamedTuple):
    feature_map: Tensor
    tokens: Tensor


class FixedSinusoidalPositionEncoding(nn.Module):
    """Parameter-free deterministic one-dimensional horizontal positions."""

    def __init__(self, *, embedding_dimension: int, maximum_tokens: int) -> None:
        super().__init__()
        if embedding_dimension < 2 or maximum_tokens < 1:
            raise ValueError("positional encoding dimensions must be positive")
        positions = torch.arange(maximum_tokens, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, embedding_dimension, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / embedding_dimension)
        )
        encoding = torch.zeros((maximum_tokens, embedding_dimension), dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("positional encoding expects [batch, tokens, embedding]")
        encoding = cast(Tensor, self.encoding)
        if tokens.shape[1] > encoding.shape[1] or tokens.shape[2] != encoding.shape[2]:
            raise ValueError("token geometry exceeds the governed positional encoding")
        return tokens + encoding[:, : tokens.shape[1]].to(dtype=tokens.dtype)


class CnnWidthTokenizer(nn.Module):
    """Preserve 40 horizontal columns and aggregate only the vertical feature axis."""

    def __init__(self, config: TransformerEncoderModelConfig) -> None:
        super().__init__()
        first, second, third = config.tokenizer.cnn_channels
        self.expected_feature_shape = (
            third,
            config.tokenizer.feature_map_height,
            config.tokenizer.feature_map_width,
        )
        self.features = nn.Sequential(
            ConvFeatureBlock(config.input_channels, first, pool=(2, 2)),
            ConvFeatureBlock(first, second, pool=(2, 2)),
            ConvFeatureBlock(second, third, pool=(2, 1)),
        )
        self.projection = nn.Linear(third, config.tokenizer.projection_dimension)

    @staticmethod
    def width_major_sequence(feature_map: Tensor) -> Tensor:
        """Map `[B,C,H,W]` to `[B,W,C]` without reversing or rasterizing width."""
        if feature_map.ndim != 4:
            raise ValueError("CNN feature map must have shape [batch, channels, height, width]")
        return feature_map.mean(dim=2).transpose(1, 2).contiguous()

    def forward(self, images: Tensor) -> CnnWidthTokenizerOutput:
        feature_map = self.features(images)
        if tuple(feature_map.shape[1:]) != self.expected_feature_shape:
            raise ValueError(
                "M2 CNN tokenizer produced unexpected feature geometry: "
                f"expected {self.expected_feature_shape}, got {tuple(feature_map.shape[1:])}"
            )
        width_major = self.width_major_sequence(feature_map)
        return CnnWidthTokenizerOutput(
            feature_map=feature_map,
            tokens=cast(Tensor, self.projection(width_major)),
        )


def build_transformer_encoder_modules(
    config: TransformerEncoderModelConfig,
) -> tuple[CnnWidthTokenizer, FixedSinusoidalPositionEncoding, nn.Dropout, nn.TransformerEncoder]:
    """Build the permanent M2 encoder modules for reuse without semantic drift."""
    transformer = config.transformer
    tokenizer = CnnWidthTokenizer(config)
    position = FixedSinusoidalPositionEncoding(
        embedding_dimension=transformer.embedding_dimension,
        maximum_tokens=transformer.token_count,
    )
    input_dropout = nn.Dropout(transformer.dropout)
    layer = nn.TransformerEncoderLayer(
        d_model=transformer.embedding_dimension,
        nhead=transformer.attention_heads,
        dim_feedforward=transformer.feed_forward_dimension,
        dropout=transformer.dropout,
        activation=transformer.activation,
        batch_first=True,
        norm_first=True,
    )
    encoder = nn.TransformerEncoder(
        layer,
        num_layers=transformer.layers,
        norm=nn.LayerNorm(transformer.embedding_dimension),
        enable_nested_tensor=False,
    )
    reset_transformer_layers(encoder)
    return tokenizer, position, input_dropout, encoder


def reset_transformer_layers(transformer: nn.TransformerEncoder) -> None:
    """Deterministically break the identical cloned-layer initialization."""
    for layer in transformer.layers:
        nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
        if layer.self_attn.in_proj_bias is not None:
            nn.init.zeros_(layer.self_attn.in_proj_bias)
        nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
        if layer.self_attn.out_proj.bias is not None:
            nn.init.zeros_(layer.self_attn.out_proj.bias)
        nn.init.xavier_uniform_(layer.linear1.weight)
        nn.init.zeros_(layer.linear1.bias)
        nn.init.xavier_uniform_(layer.linear2.weight)
        nn.init.zeros_(layer.linear2.bias)
        nn.init.ones_(layer.norm1.weight)
        nn.init.zeros_(layer.norm1.bias)
        nn.init.ones_(layer.norm2.weight)
        nn.init.zeros_(layer.norm2.bias)
    if transformer.norm is not None:
        final_norm = cast(nn.LayerNorm, transformer.norm)
        nn.init.ones_(final_norm.weight)
        nn.init.zeros_(final_norm.bias)


class CnnTransformerDualCtcRecognizer(nn.Module):
    """M2 encoder ablation: CNN width tokens, compact Transformer, unchanged dual CTC."""

    def __init__(self, config: TransformerRecognizerModelConfig) -> None:
        super().__init__()
        self.config = config
        (
            self.tokenizer,
            self.position,
            self.input_dropout,
            self.transformer,
        ) = build_transformer_encoder_modules(config)
        transformer = config.transformer
        self.character_head = nn.Linear(transformer.embedding_dimension, 37)
        self.type_head = nn.Linear(transformer.embedding_dimension, 3)

    def _reset_transformer_layers(self) -> None:
        """Backward-compatible entrypoint for the shared deterministic reset."""
        reset_transformer_layers(self.transformer)

    def forward(self, images: Tensor) -> TransformerRecognizerOutput:
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
                "expected fixed M2 input geometry "
                f"{self.config.input_height}x{self.config.input_width}, got {height}x{width}"
            )
        if not images.is_floating_point():
            raise ValueError("recognizer input must be floating-point")

        tokenized = self.tokenizer(images)
        positioned = self.input_dropout(self.position(tokenized.tokens))
        encoded = self.transformer(positioned)
        character = self.character_head(encoded).transpose(0, 1).contiguous()
        types = self.type_head(encoded).transpose(0, 1).contiguous()
        return TransformerRecognizerOutput(
            character_logits=cast(Tensor, character),
            type_logits=cast(Tensor, types),
            global_embedding=encoded.mean(dim=1),
            tokens=tokenized.tokens,
            encoded_tokens=encoded,
        )


def build_transformer_recognizer(
    config: TransformerRecognizerModelConfig,
) -> CnnTransformerDualCtcRecognizer:
    """Construct M2 deterministically without changing the caller's CPU RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.initialization_seed)
        return CnnTransformerDualCtcRecognizer(config)


def transformer_parameter_breakdown(model: CnnTransformerDualCtcRecognizer) -> dict[str, int]:
    """Expose the governed M2 capacity by scientific component."""
    tokenizer = sum(parameter.numel() for parameter in model.tokenizer.parameters())
    transformer = sum(parameter.numel() for parameter in model.transformer.parameters())
    character_head = sum(parameter.numel() for parameter in model.character_head.parameters())
    type_head = sum(parameter.numel() for parameter in model.type_head.parameters())
    return {
        "tokenizer": tokenizer,
        "transformer": transformer,
        "character_ctc_head": character_head,
        "type_ctc_head": type_head,
        "positional_encoding": 0,
    }
