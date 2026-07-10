from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from .common import DoubleConv3D, NNUNetDown3D, PartialLoadMixin, channels_from_config


class NNUNet3DEncoder(PartialLoadMixin, nn.Module):
    encoder_name = "nnunet3d"

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
        self.stem = DoubleConv3D(
            int(in_channels),
            self.channels[0],
            normalization=normalization,
            activation_name="leaky_relu",
            dropout=float(dropout),
        )
        self.stages = nn.ModuleList(
            [self.stem, *[NNUNetDown3D(inp, out, normalization=normalization, dropout=float(dropout)) for inp, out in zip(self.channels[:-1], self.channels[1:])]]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"3D encoder expects [B,C,D,H,W], got {tuple(x.shape)}")
        features = []
        current = x
        for stage in self.stages:
            current = stage(current)
            features.append(current)
        return features

    def forward_with_fusion(self, x: torch.Tensor, fuse_fn: Callable[[int, torch.Tensor], torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        raw_features = []
        fused_features = []
        current = x
        for index, stage in enumerate(self.stages):
            raw = stage(current)
            fused = fuse_fn(index, raw)
            raw_features.append(raw)
            fused_features.append(fused)
            current = fused
        return fused_features, raw_features


__all__ = ["NNUNet3DEncoder"]
