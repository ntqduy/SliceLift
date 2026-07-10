from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import Base2DEncoder, DoubleConv2D, NNUNetDown2D, channels_from_config


class NNUNet2DEncoder(Base2DEncoder):
    encoder_name = "nnunet2d"

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] | tuple[int, ...] | None = None,
        base_channels: int = 32,
        normalization: str = "instancenorm",
        dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.channels = channels_from_config(channels, base_channels)
        self.stem = DoubleConv2D(
            int(in_channels),
            self.channels[0],
            normalization=normalization,
            activation_name="leaky_relu",
            dropout=float(dropout),
        )
        self.stages = nn.ModuleList(
            [NNUNetDown2D(inp, out, normalization=normalization, dropout=float(dropout)) for inp, out in zip(self.channels[:-1], self.channels[1:])]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x, restore_shape = self._restore_slices(x)
        features = [self.stem(x)]
        current = features[0]
        for stage in self.stages:
            current = stage(current)
            features.append(current)
        return self._reshape_features(features, restore_shape)


__all__ = ["NNUNet2DEncoder"]
