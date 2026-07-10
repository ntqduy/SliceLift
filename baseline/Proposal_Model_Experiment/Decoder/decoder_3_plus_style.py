from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import ConvNormAct2D, ConvNormAct3D, FusionBlock2D, FusionBlock3D, PartialLoadMixin, resize2d, resize3d


class UNet3Plus3DDecoder(PartialLoadMixin, nn.Module):
    decoder_name = "full_scale"

    def __init__(
        self,
        channels: list[int] | tuple[int, ...],
        num_classes: int = 2,
        fusion_channels: int | None = None,
        deep_supervision: bool = False,
        normalization: str = "batchnorm",
        **_: Any,
    ) -> None:
        super().__init__()
        self.channels = [int(item) for item in channels]
        self.num_classes = int(num_classes)
        self.fusion_channels = int(fusion_channels or self.channels[0])
        self.deep_supervision = bool(deep_supervision)
        self.projections = nn.ModuleDict()
        self.decoder_projections = nn.ModuleDict()
        self.fusion_blocks = nn.ModuleDict()
        self.deep_heads = nn.ModuleDict()
        for target_index in range(4):
            key = str(target_index)
            history = self.history_indices(target_index)
            self.projections[key] = nn.ModuleList([ConvNormAct3D(source, self.fusion_channels, kernel_size=1, padding=0, normalization=normalization) for source in self.channels])
            self.decoder_projections[key] = nn.ModuleList([ConvNormAct3D(self.channels[source], self.fusion_channels, kernel_size=1, padding=0, normalization=normalization) for source in history])
            self.fusion_blocks[key] = FusionBlock3D(self.fusion_channels * (len(self.channels) + len(history)), self.channels[target_index], normalization=normalization)
            if self.deep_supervision:
                self.deep_heads[key] = nn.Conv3d(self.channels[target_index], self.num_classes, kernel_size=1)
        self.segmentation_head = nn.Conv3d(self.channels[0], self.num_classes, kernel_size=1)

    @staticmethod
    def history_indices(target_index: int) -> list[int]:
        return list(range(3, int(target_index), -1))

    def forward(self, skips: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        decoded: dict[int, torch.Tensor] = {}
        for target_index in (3, 2, 1, 0):
            target_size = tuple(int(item) for item in skips[target_index].shape[-3:])
            key = str(target_index)
            projected = [projection(resize3d(source, target_size)) for projection, source in zip(self.projections[key], skips)]
            for projection, source_index in zip(self.decoder_projections[key], self.history_indices(target_index)):
                projected.append(projection(resize3d(decoded[source_index], target_size)))
            decoded[target_index] = self.fusion_blocks[key](torch.cat(projected, dim=1))
        logits = self.segmentation_head(decoded[0])
        if not self.deep_supervision:
            return logits
        full_size = tuple(int(item) for item in skips[0].shape[-3:])
        return tuple([logits] + [resize3d(self.deep_heads[str(index)](decoded[index]), full_size) for index in (1, 2, 3)])


class UNet3Plus2DDecoder(nn.Module):
    def __init__(self, channels: list[int] | tuple[int, ...], num_classes: int = 2, fusion_channels: int | None = None, normalization: str = "batchnorm", **_: Any) -> None:
        super().__init__()
        self.channels = [int(item) for item in channels]
        self.fusion_channels = int(fusion_channels or self.channels[0])
        self.projections = nn.ModuleDict()
        self.decoder_projections = nn.ModuleDict()
        self.fusion_blocks = nn.ModuleDict()
        for target_index in range(4):
            key = str(target_index)
            history = UNet3Plus3DDecoder.history_indices(target_index)
            self.projections[key] = nn.ModuleList([ConvNormAct2D(source, self.fusion_channels, kernel_size=1, padding=0, normalization=normalization) for source in self.channels])
            self.decoder_projections[key] = nn.ModuleList([ConvNormAct2D(self.channels[source], self.fusion_channels, kernel_size=1, padding=0, normalization=normalization) for source in history])
            self.fusion_blocks[key] = FusionBlock2D(self.fusion_channels * (len(self.channels) + len(history)), self.channels[target_index], normalization=normalization)
        self.segmentation_head = nn.Conv2d(self.channels[0], int(num_classes), kernel_size=1)

    def forward(self, skips: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        decoded: dict[int, torch.Tensor] = {}
        for target_index in (3, 2, 1, 0):
            key = str(target_index)
            target_size = tuple(int(item) for item in skips[target_index].shape[-2:])
            projected = [projection(resize2d(source, target_size)) for projection, source in zip(self.projections[key], skips)]
            for projection, source_index in zip(self.decoder_projections[key], UNet3Plus3DDecoder.history_indices(target_index)):
                projected.append(projection(resize2d(decoded[source_index], target_size)))
            decoded[target_index] = self.fusion_blocks[key](torch.cat(projected, dim=1))
        return self.segmentation_head(decoded[0])


__all__ = ["UNet3Plus2DDecoder", "UNet3Plus3DDecoder"]
