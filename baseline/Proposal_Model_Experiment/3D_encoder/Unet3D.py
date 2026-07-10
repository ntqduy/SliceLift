from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from .common import DoubleConv3D, PartialLoadMixin, channels_from_config


class UNet3DEncoder(PartialLoadMixin, nn.Module):
    encoder_name = "unet3d"

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] | tuple[int, ...] | None = None,
        base_channels: int = 32,
        normalization: str = "batchnorm",
        block_cls: type[nn.Module] = DoubleConv3D,
        dropout: float = 0.0,
        residual: str | None = None,
        conv_bias: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.channels = channels_from_config(channels, base_channels)
        stage_in = [int(in_channels)] + self.channels[:-1]
        self.stages = nn.ModuleList(
            [
                block_cls(inp, out, normalization=normalization, dropout=float(dropout), residual=residual, conv_bias=bool(conv_bias))
                for inp, out in zip(stage_in, self.channels)
            ]
        )
        self.pools = nn.ModuleList([nn.MaxPool3d(2, 2, ceil_mode=True) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"3D encoder expects [B,C,D,H,W], got {tuple(x.shape)}")
        features = []
        current = x
        for index, stage in enumerate(self.stages):
            current = stage(current)
            features.append(current)
            if index < len(self.pools):
                current = self.pools[index](current)
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
            if index < len(self.pools):
                current = self.pools[index](fused)
        return fused_features, raw_features


__all__ = ["UNet3DEncoder"]
