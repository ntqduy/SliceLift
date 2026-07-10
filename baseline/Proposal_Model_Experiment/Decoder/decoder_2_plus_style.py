from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import FusionBlock2D, FusionBlock3D, PartialLoadMixin, resize2d, resize3d


class UNetPlusPlus3DDecoder(PartialLoadMixin, nn.Module):
    decoder_name = "nested_dense"

    def __init__(self, channels: list[int] | tuple[int, ...], num_classes: int = 2, deep_supervision: bool = False, normalization: str = "batchnorm", **_: Any) -> None:
        super().__init__()
        self.channels = [int(item) for item in channels]
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)
        self.up = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        self.deep_heads = nn.ModuleDict()
        for depth in range(1, 5):
            for level in range(0, 5 - depth):
                key = f"{level}_{depth}"
                self.up[key] = nn.ConvTranspose3d(self.channels[level + 1], self.channels[level], kernel_size=2, stride=2)
                self.blocks[key] = FusionBlock3D(self.channels[level] * (depth + 1), self.channels[level], normalization=normalization)
        if self.deep_supervision:
            for depth in range(1, 5):
                self.deep_heads[str(depth)] = nn.Conv3d(self.channels[0], self.num_classes, kernel_size=1)
        self.segmentation_head = nn.Conv3d(self.channels[0], self.num_classes, kernel_size=1)

    def forward(self, skips: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        nodes: dict[tuple[int, int], torch.Tensor] = {(level, 0): skips[level] for level in range(5)}
        for depth in range(1, 5):
            for level in range(0, 5 - depth):
                key = f"{level}_{depth}"
                target_size = tuple(int(item) for item in skips[level].shape[-3:])
                up = resize3d(self.up[key](nodes[(level + 1, depth - 1)]), target_size)
                dense = [nodes[(level, previous)] for previous in range(depth)]
                nodes[(level, depth)] = self.blocks[key](torch.cat([*dense, up], dim=1))
        logits = self.segmentation_head(nodes[(0, 4)])
        if not self.deep_supervision:
            return logits
        return tuple([logits] + [self.deep_heads[str(depth)](nodes[(0, depth)]) for depth in (1, 2, 3)])


class UNetPlusPlus2DDecoder(nn.Module):
    def __init__(self, channels: list[int] | tuple[int, ...], num_classes: int = 2, normalization: str = "batchnorm", **_: Any) -> None:
        super().__init__()
        self.channels = [int(item) for item in channels]
        self.blocks = nn.ModuleDict()
        for depth in range(1, 5):
            for level in range(0, 5 - depth):
                key = f"{level}_{depth}"
                self.blocks[key] = FusionBlock2D(self.channels[level] * depth + self.channels[level + 1], self.channels[level], normalization=normalization)
        self.segmentation_head = nn.Conv2d(self.channels[0], int(num_classes), kernel_size=1)

    def forward(self, skips: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        nodes: dict[tuple[int, int], torch.Tensor] = {(level, 0): skips[level] for level in range(5)}
        for depth in range(1, 5):
            for level in range(0, 5 - depth):
                key = f"{level}_{depth}"
                up = resize2d(nodes[(level + 1, depth - 1)], tuple(int(item) for item in skips[level].shape[-2:]))
                dense = [nodes[(level, previous)] for previous in range(depth)]
                nodes[(level, depth)] = self.blocks[key](torch.cat([*dense, up], dim=1))
        return self.segmentation_head(nodes[(0, 4)])


__all__ = ["UNetPlusPlus2DDecoder", "UNetPlusPlus3DDecoder"]
