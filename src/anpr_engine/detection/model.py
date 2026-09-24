"""Small anchor-free YOLO-family model and project-owned decoding."""

from __future__ import annotations

import math
from typing import TypeAlias, cast

import torch
from torch import Tensor, nn


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = channels // 2
        self.convolution = nn.Sequential(
            ConvNormAct(channels, hidden_channels, kernel_size=1),
            ConvNormAct(hidden_channels, channels),
        )

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, features + self.convolution(features))


class DownStage(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, depth: int) -> None:
        super().__init__(
            ConvNormAct(input_channels, output_channels, stride=2),
            *(ResidualBlock(output_channels) for _ in range(depth)),
        )


class FeatureFusion(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            ConvNormAct(input_channels, output_channels, kernel_size=1),
            ResidualBlock(output_channels),
        )


class DecoupledHead(nn.Module):
    def __init__(self, input_channels: int, head_channels: int) -> None:
        super().__init__()
        self.stem = ConvNormAct(input_channels, head_channels, kernel_size=1)
        self.objectness = nn.Sequential(
            ConvNormAct(head_channels, head_channels),
            ConvNormAct(head_channels, head_channels),
            nn.Conv2d(head_channels, 1, 1),
        )
        self.regression = nn.Sequential(
            ConvNormAct(head_channels, head_channels),
            ConvNormAct(head_channels, head_channels),
            nn.Conv2d(head_channels, 4, 1),
        )
        objectness_predictor = cast(nn.Conv2d, self.objectness[-1])
        objectness_bias = objectness_predictor.bias
        if objectness_bias is None:
            raise ValueError("objectness predictor requires a bias")
        nn.init.constant_(objectness_bias, -math.log(99.0))

    def forward(self, features: Tensor) -> Tensor:
        stem = self.stem(features)
        return torch.cat((self.objectness(stem), self.regression(stem)), dim=1)


class TinyYoloDetector(nn.Module):
    """One-stage single-class detector with a stride-32 dense grid head."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 6),
            ConvBlock(base_channels * 6, base_channels * 8),
        )
        self.head = nn.Conv2d(base_channels * 8, 5, 1)

    def forward(self, images: Tensor) -> Tensor:
        return cast(Tensor, self.head(self.backbone(images)))


class YoloXMultiScaleDetector(nn.Module):
    """Project-owned anchor-free detector with stride-8/16/32 PAN/FPN features."""

    def __init__(self, base_channels: int = 24, head_channels: int = 96) -> None:
        super().__init__()
        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.stem = ConvNormAct(3, channels[0], stride=2)
        self.stage2 = DownStage(channels[0], channels[1], depth=1)
        self.stage3 = DownStage(channels[1], channels[2], depth=2)
        self.stage4 = DownStage(channels[2], channels[3], depth=2)
        self.stage5 = DownStage(channels[3], channels[4], depth=1)

        self.lateral5 = ConvNormAct(channels[4], head_channels, kernel_size=1)
        self.lateral4 = ConvNormAct(channels[3], head_channels, kernel_size=1)
        self.lateral3 = ConvNormAct(channels[2], head_channels, kernel_size=1)
        self.top4 = FeatureFusion(head_channels * 2, head_channels)
        self.top3 = FeatureFusion(head_channels * 2, head_channels)
        self.down3 = ConvNormAct(head_channels, head_channels, stride=2)
        self.bottom4 = FeatureFusion(head_channels * 2, head_channels)
        self.down4 = ConvNormAct(head_channels, head_channels, stride=2)
        self.bottom5 = FeatureFusion(head_channels * 2, head_channels)
        self.heads = nn.ModuleList(DecoupledHead(head_channels, head_channels) for _ in range(3))

    def forward_features(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stage2 = self.stage2(self.stem(images))
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)
        feature5 = self.lateral5(stage5)
        feature4 = self.top4(
            torch.cat(
                (
                    nn.functional.interpolate(feature5, scale_factor=2.0, mode="nearest"),
                    self.lateral4(stage4),
                ),
                dim=1,
            )
        )
        feature3 = self.top3(
            torch.cat(
                (
                    nn.functional.interpolate(feature4, scale_factor=2.0, mode="nearest"),
                    self.lateral3(stage3),
                ),
                dim=1,
            )
        )
        pan4 = self.bottom4(torch.cat((self.down3(feature3), feature4), dim=1))
        pan5 = self.bottom5(torch.cat((self.down4(pan4), feature5), dim=1))
        return feature3, pan4, pan5

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.forward_features(images)
        return cast(
            tuple[Tensor, Tensor, Tensor],
            tuple(head(feature) for head, feature in zip(self.heads, features, strict=True)),
        )


DetectorOutput: TypeAlias = Tensor | tuple[Tensor, Tensor, Tensor]


def decode_multiscale_outputs(output: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Return flattened objectness logits and normalized XYXY boxes."""
    logits: list[Tensor] = []
    boxes: list[Tensor] = []
    for level in output:
        batch_size, _, grid_height, grid_width = level.shape
        logits.append(level[:, 0].reshape(batch_size, -1))
        raw_boxes = level[:, 1:].permute(0, 2, 3, 1)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_height, device=level.device, dtype=level.dtype),
            torch.arange(grid_width, device=level.device, dtype=level.dtype),
            indexing="ij",
        )
        center_x = (grid_x + raw_boxes[..., 0].sigmoid()) / grid_width
        center_y = (grid_y + raw_boxes[..., 1].sigmoid()) / grid_height
        width = raw_boxes[..., 2].clamp(min=-4.0, max=4.0).exp() / grid_width
        height = raw_boxes[..., 3].clamp(min=-4.0, max=4.0).exp() / grid_height
        boxes.append(
            torch.stack(
                (
                    center_x - width / 2.0,
                    center_y - height / 2.0,
                    center_x + width / 2.0,
                    center_y + height / 2.0,
                ),
                dim=-1,
            ).reshape(batch_size, -1, 4)
        )
    return torch.cat(logits, dim=1), torch.cat(boxes, dim=1)


def multiscale_cell_geometry(
    output: tuple[Tensor, Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    centers: list[Tensor] = []
    cell_sizes: list[Tensor] = []
    for level in output:
        _, _, grid_height, grid_width = level.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_height, device=level.device, dtype=level.dtype),
            torch.arange(grid_width, device=level.device, dtype=level.dtype),
            indexing="ij",
        )
        centers.append(
            torch.stack(
                ((grid_x + 0.5) / grid_width, (grid_y + 0.5) / grid_height), dim=-1
            ).reshape(-1, 2)
        )
        cell_sizes.append(
            torch.tensor(
                (1.0 / grid_width, 1.0 / grid_height),
                device=level.device,
                dtype=level.dtype,
            ).expand(grid_height * grid_width, 2)
        )
    return torch.cat(centers), torch.cat(cell_sizes)


def box_iou(box: Tensor, boxes: Tensor) -> Tensor:
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / (box_area + boxes_area - intersection).clamp(min=1e-8)


def non_maximum_suppression(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    order = scores.argsort(descending=True)
    kept: list[Tensor] = []
    while order.numel():
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(boxes[current], boxes[remaining]) <= iou_threshold]
    return torch.stack(kept) if kept else torch.empty(0, dtype=torch.long, device=boxes.device)


def decode_predictions(
    output: DetectorOutput,
    *,
    confidence_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> list[tuple[Tensor, Tensor]]:
    if isinstance(output, tuple):
        objectness_logits, boxes = decode_multiscale_outputs(output)
        objectness = objectness_logits.sigmoid()
        boxes = boxes.clamp(0.0, 1.0)
        batch_size = objectness.shape[0]
        multiscale_decoded: list[tuple[Tensor, Tensor]] = []
        for batch_index in range(batch_size):
            mask = objectness[batch_index] >= confidence_threshold
            sample_scores = objectness[batch_index][mask]
            sample_boxes = boxes[batch_index][mask]
            if not sample_scores.numel():
                multiscale_decoded.append((sample_boxes.reshape(0, 4), sample_scores))
                continue
            candidate_count = min(sample_scores.numel(), max(100, max_detections * 20))
            sample_scores, candidate_indices = sample_scores.topk(candidate_count)
            sample_boxes = sample_boxes[candidate_indices]
            keep = non_maximum_suppression(sample_boxes, sample_scores, nms_iou_threshold)
            keep = keep[:max_detections]
            multiscale_decoded.append((sample_boxes[keep], sample_scores[keep]))
        return multiscale_decoded

    objectness = output[:, 0].sigmoid()
    box_values = output[:, 1:].sigmoid()
    batch_size, grid_height, grid_width = objectness.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_height, device=output.device),
        torch.arange(grid_width, device=output.device),
        indexing="ij",
    )
    center_x = (grid_x + box_values[:, 0]) / grid_width
    center_y = (grid_y + box_values[:, 1]) / grid_height
    width = box_values[:, 2]
    height = box_values[:, 3]
    boxes = torch.stack(
        (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)
    decoded: list[tuple[Tensor, Tensor]] = []
    for batch_index in range(batch_size):
        mask = objectness[batch_index] >= confidence_threshold
        sample_scores = objectness[batch_index][mask]
        sample_boxes = boxes[batch_index][mask]
        if not sample_scores.numel():
            decoded.append((sample_boxes.reshape(0, 4), sample_scores))
            continue
        keep = non_maximum_suppression(sample_boxes, sample_scores, nms_iou_threshold)
        keep = keep[:max_detections]
        decoded.append((sample_boxes[keep], sample_scores[keep]))
    return decoded
