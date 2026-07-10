from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state", "state_dict", "model_state_dict", "network_weights"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def clean_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def normalization_3d(channels: int, name: str) -> nn.Module:
    normalized = str(name or "batchnorm").lower()
    if normalized in {"batch", "batchnorm", "bn"}:
        return nn.BatchNorm3d(channels)
    if normalized in {"instance", "instancenorm", "in"}:
        return nn.InstanceNorm3d(channels, affine=True)
    if normalized in {"group", "groupnorm", "gn"}:
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 3D normalization: {name}")


def normalization_2d(channels: int, name: str) -> nn.Module:
    normalized = str(name or "batchnorm").lower()
    if normalized in {"batch", "batchnorm", "bn"}:
        return nn.BatchNorm2d(channels)
    if normalized in {"instance", "instancenorm", "in"}:
        return nn.InstanceNorm2d(channels, affine=True)
    if normalized in {"group", "groupnorm", "gn"}:
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 2D normalization: {name}")


def activation(kind: str) -> nn.Module:
    if str(kind).lower() in {"leaky_relu", "lrelu", "nnunet"}:
        return nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    return nn.ReLU(inplace=True)


class ConvNormAct3D(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        normalization: str = "batchnorm",
        activation_name: str = "relu",
        bias: bool = False,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(int(in_channels), int(out_channels), kernel_size=kernel_size, padding=padding, bias=bool(bias)),
            normalization_3d(int(out_channels), normalization),
            activation(activation_name),
        )


class ConvNormAct2D(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        normalization: str = "batchnorm",
        activation_name: str = "relu",
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size, padding=padding, bias=False),
            normalization_2d(int(out_channels), normalization),
            activation(activation_name),
        )


class FusionBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str = "batchnorm",
        activation_name: str = "relu",
        dropout: float = 0.0,
        residual: str | None = None,
        conv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.residual = str(residual or "").lower() not in {"", "none", "false", "0"}
        layers: list[nn.Module] = [
            ConvNormAct3D(in_channels, out_channels, normalization=normalization, activation_name=activation_name, bias=conv_bias),
            ConvNormAct3D(out_channels, out_channels, normalization=normalization, activation_name=activation_name, bias=conv_bias),
        ]
        if float(dropout) > 0:
            layers.append(nn.Dropout3d(float(dropout)))
        self.block = nn.Sequential(*layers)
        self.residual_projection = (
            nn.Conv3d(int(in_channels), int(out_channels), kernel_size=1, bias=False)
            if self.residual and int(in_channels) != int(out_channels)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block(x)
        if not self.residual:
            return output
        return output + self.residual_projection(x)


class FusionBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm", activation_name: str = "relu", dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            ConvNormAct2D(in_channels, out_channels, normalization=normalization, activation_name=activation_name),
            ConvNormAct2D(out_channels, out_channels, normalization=normalization, activation_name=activation_name),
        ]
        if float(dropout) > 0:
            layers.append(nn.Dropout2d(float(dropout)))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PartialLoadMixin:
    def load_partial_state_dict(self, checkpoint: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        state_dict = checkpoint_state_dict(checkpoint)
        target_state = self.state_dict()
        updates = {}
        skipped = []
        prefix = str(prefix or "")
        for raw_key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = clean_key(str(raw_key))
            if prefix:
                if not key.startswith(prefix):
                    continue
                key = key[len(prefix) :]
            if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
                updates[key] = value
            elif key in target_state:
                skipped.append({"key": key, "reason": f"shape {tuple(value.shape)} != {tuple(target_state[key].shape)}"})
        merged = dict(target_state)
        merged.update(updates)
        self.load_state_dict(merged, strict=False)
        missing = [key for key in target_state if key not in updates]
        return {
            "loaded_keys": sorted(updates),
            "skipped_keys": skipped,
            "missing_keys": missing,
            "num_loaded_keys": len(updates),
            "num_skipped_keys": len(skipped),
            "num_missing_keys": len(missing),
        }


def resize3d(feature: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    if tuple(feature.shape[-3:]) == tuple(size):
        return feature
    return F.interpolate(feature, size=size, mode="trilinear", align_corners=False)


def resize2d(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(feature.shape[-2:]) == tuple(size):
        return feature
    return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
