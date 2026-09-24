"""Project-owned randomly initialized CRNN/BiLSTM/CTC recognizer."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from anpr_engine.recognition.config import RecognizerModelConfig


class ConvFeatureBlock(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        pool: tuple[int, int],
    ) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )


class CrnnBilstmCtcRecognizer(nn.Module):
    """Return time-major CTC logits with shape ``[time, batch, classes]``."""

    def __init__(self, config: RecognizerModelConfig, *, class_count: int) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("CTC recognizer requires at least one symbol plus blank")
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
        self.classifier = nn.Linear(config.lstm_hidden_size * 2, class_count)

    def forward(self, images: Tensor) -> Tensor:
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
            raise ValueError("recognizer input must be a floating-point tensor")

        features = self.features(images).squeeze(2)
        sequence = features.permute(2, 0, 1).contiguous()
        encoded, _ = self.sequence_model(sequence)
        return cast(Tensor, self.classifier(encoded))


def build_recognizer(
    config: RecognizerModelConfig,
    *,
    class_count: int,
) -> CrnnBilstmCtcRecognizer:
    """Construct deterministically without changing the caller's CPU RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.initialization_seed)
        return CrnnBilstmCtcRecognizer(config, class_count=class_count)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
