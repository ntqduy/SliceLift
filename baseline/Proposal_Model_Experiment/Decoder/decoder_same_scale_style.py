from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import FusionBlock2D, FusionBlock3D, PartialLoadMixin, resize2d, resize3d


class SameScale3DDecoder(PartialLoadMixin, nn.Module):
    decoder_name = "same_scale"

    def __init__(
        self,
        channels: list[int] | tuple[int, ...],
        num_classes: int = 2,
        deep_supervision: bool = False,
        normalization: str = "batchnorm",
        activation_name: str = "relu",
        dropout: float = 0.0,
        residual: str | None = None,
        conv_bias: bool = False,
        up_kernel_size: int = 2,
        **_: Any,
    ) -> None:
        super().__init__()
        self.channels = [int(item) for item in channels]
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)
        self.up = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        self.deep_heads = nn.ModuleDict()
        for index in range(4, 0, -1):
            kernel_size = int(up_kernel_size)
            if kernel_size == 3:
                self.up[str(index)] = nn.ConvTranspose3d(
                    self.channels[index],
                    self.channels[index - 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                )
            else:
                self.up[str(index)] = nn.ConvTranspose3d(self.channels[index], self.channels[index - 1], kernel_size=2, stride=2)
            self.blocks[str(index - 1)] = FusionBlock3D(
                self.channels[index - 1] * 2,
                self.channels[index - 1],
                normalization=normalization,
                activation_name=activation_name,
                dropout=float(dropout),
                residual=residual,
                conv_bias=bool(conv_bias),
            )
            if self.deep_supervision:
                self.deep_heads[str(index - 1)] = nn.Conv3d(self.channels[index - 1], self.num_classes, kernel_size=1)
        self.segmentation_head = nn.Conv3d(self.channels[0], self.num_classes, kernel_size=1)

    def forward(self, skips: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        decoded = {4: skips[4]}
        current = skips[4]
        for index in range(4, 0, -1):
            up = resize3d(self.up[str(index)](current), tuple(int(item) for item in skips[index - 1].shape[-3:]))
            current = self.blocks[str(index - 1)](torch.cat([skips[index - 1], up], dim=1))
            decoded[index - 1] = current
        logits = self.segmentation_head(decoded[0])
        if not self.deep_supervision:
            return logits
        full_size = tuple(int(item) for item in skips[0].shape[-3:])
        return tuple([logits] + [resize3d(self.deep_heads[str(index)](decoded[index]), full_size) for index in (1, 2, 3)])


class SameScale2DDecoder(nn.Module):
    def __init__(
        self,
        channels: list[int] | tuple[int, ...],
        num_classes: int = 2,
        normalization: str = "batchnorm",
        activation_name: str = "relu",
        dropout: float = 0.0,
        upsample_mode: str = "transpose",
        **_: Any,
    ) -> None:
        super().__init__()
        self.channels = [int(item) for item in channels]
        self.up = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        for index in range(4, 0, -1):
            if str(upsample_mode).lower() in {"bilinear", "interp", "interpolate"}:
                self.up[str(index)] = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(self.channels[index], self.channels[index - 1], kernel_size=1, bias=False),
                )
            else:
                self.up[str(index)] = nn.ConvTranspose2d(self.channels[index], self.channels[index - 1], kernel_size=2, stride=2)
            self.blocks[str(index - 1)] = FusionBlock2D(
                self.channels[index - 1] * 2,
                self.channels[index - 1],
                normalization=normalization,
                activation_name=activation_name,
                dropout=float(dropout),
            )
        self.segmentation_head = nn.Conv2d(self.channels[0], int(num_classes), kernel_size=1)

    def forward(self, skips: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        current = skips[4]
        for index in range(4, 0, -1):
            up = resize2d(self.up[str(index)](current), tuple(int(item) for item in skips[index - 1].shape[-2:]))
            current = self.blocks[str(index - 1)](torch.cat([skips[index - 1], up], dim=1))
        return self.segmentation_head(current)


__all__ = ["SameScale2DDecoder", "SameScale3DDecoder"]
