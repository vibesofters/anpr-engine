"""Project-owned shared CNN/BiLSTM recognizer with character and type CTC heads."""

from __future__ import annotations

from typing import NamedTuple, cast

import torch
from torch import Tensor, nn

from anpr_engine.recognition.config import RecognizerModelConfig
from anpr_engine.recognition.model import ConvFeatureBlock

ENHANCED_MODEL_ID = "crnn_bilstm_dual_ctc_tr_v1"


class EnhancedRecognizerOutput(NamedTuple):
    character_logits: Tensor
    type_logits: Tensor
    global_embedding: Tensor


class CountryLayoutHead(nn.Module):
    """Future multi-country seam; deliberately not instantiated for one-class TR V1."""

    def __init__(self, embedding_size: int, country_layout_count: int) -> None:
        super().__init__()
        if country_layout_count < 2:
            raise ValueError("CountryLayoutHead requires at least two country/layout classes")
        self.classifier = nn.Linear(embedding_size, country_layout_count)

    def forward(self, global_embedding: Tensor) -> Tensor:
        return cast(Tensor, self.classifier(global_embedding))


class EnhancedCrnnBilstmDualCtcRecognizer(nn.Module):
    """Return shared time-major character/type logits and a global embedding."""

    def __init__(self, config: RecognizerModelConfig, *, character_class_count: int) -> None:
        super().__init__()
        if character_class_count < 2:
            raise ValueError("character CTC requires at least one symbol plus blank")
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
        self.character_head = nn.Linear(shared_size, character_class_count)
        self.type_head = nn.Linear(shared_size, 3)

    def forward(self, images: Tensor) -> EnhancedRecognizerOutput:
        if images.ndim != 4:
            raise ValueError("recognizer input must have shape [batch, channels, height, width]")
        batch, channels, height, width = images.shape
        if batch < 1:
            raise ValueError("recognizer batch must be non-empty")
        if channels != self.config.input_channels:
            raise ValueError(
                f"expected {self.config.input_channels} input channels, got {channels}"
            )
        if height != self.config.input_height:
            raise ValueError(f"expected input height {self.config.input_height}, got {height}")
        if width < 8:
            raise ValueError("recognizer input width must be at least 8 pixels")
        if not images.is_floating_point():
            raise ValueError("recognizer input must be floating-point")
        features = self.features(images).squeeze(2)
        sequence = features.permute(2, 0, 1).contiguous()
        encoded, _ = self.sequence_model(sequence)
        return EnhancedRecognizerOutput(
            character_logits=cast(Tensor, self.character_head(encoded)),
            type_logits=cast(Tensor, self.type_head(encoded)),
            global_embedding=encoded.mean(dim=0),
        )


def build_enhanced_recognizer(
    config: RecognizerModelConfig, *, character_class_count: int
) -> EnhancedCrnnBilstmDualCtcRecognizer:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.initialization_seed)
        return EnhancedCrnnBilstmDualCtcRecognizer(
            config, character_class_count=character_class_count
        )
