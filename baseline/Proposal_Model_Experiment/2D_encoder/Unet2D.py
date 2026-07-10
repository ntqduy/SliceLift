from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import Base2DEncoder, DoubleConv2D, channels_from_config


class UNet2DEncoder(Base2DEncoder):
    encoder_name = "unet2d"

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] | tuple[int, ...] | None = None,
        base_channels: int = 32,
        normalization: str = "batchnorm",
        dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.channels = channels_from_config(channels, base_channels)
        stage_in = [int(in_channels)] + self.channels[:-1]
        self.stages = nn.ModuleList([DoubleConv2D(inp, out, normalization=normalization, dropout=float(dropout)) for inp, out in zip(stage_in, self.channels)])
        self.pools = nn.ModuleList([nn.MaxPool2d(2, 2, ceil_mode=True) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x, restore_shape = self._restore_slices(x)
        features = []
        current = x
        for index, stage in enumerate(self.stages):
            current = stage(current)
            features.append(current)
            if index < len(self.pools):
                current = self.pools[index](current)
        return self._reshape_features(features, restore_shape)


__all__ = ["UNet2DEncoder"]
