from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state", "state_dict", "model_state_dict", "network_weights"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def clean_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def channels_from_config(channels: list[int] | tuple[int, ...] | None, base_channels: int) -> list[int]:
    values = list(channels or [int(base_channels) * (2**index) for index in range(5)])
    if len(values) != 5:
        raise ValueError(f"2D encoders expect 5 channel stages, got {values}")
    return [int(item) for item in values]


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


class ConvNormAct2D(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        normalization: str = "batchnorm",
        activation_name: str = "relu",
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            normalization_2d(int(out_channels), normalization),
            activation(activation_name),
        )


class DoubleConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str = "batchnorm",
        activation_name: str = "relu",
        dropout: float = 0.0,
    ) -> None:
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


class DenseConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.conv1 = ConvNormAct2D(in_channels, out_channels, normalization=normalization)
        self.conv2 = ConvNormAct2D(in_channels + out_channels, out_channels, normalization=normalization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        return self.conv2(torch.cat([x, y1], dim=1))


class NNUNetDown2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "instancenorm", dropout: float = 0.0) -> None:
        super().__init__()
        self.down = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, stride=2, padding=1, bias=False)
        self.block = DoubleConv2D(out_channels, out_channels, normalization=normalization, activation_name="leaky_relu", dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


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


class Base2DEncoder(PartialLoadMixin, nn.Module):
    def _restore_slices(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if x.ndim == 4:
            return x, None
        if x.ndim != 5:
            raise ValueError(f"2D encoder expects [B,C,H,W] or [B,K,C,H,W], got {tuple(x.shape)}")
        batch_size, num_slices, channels, height, width = x.shape
        return x.reshape(batch_size * num_slices, channels, height, width), (batch_size, num_slices)

    @staticmethod
    def _reshape_features(features: list[torch.Tensor], shape: tuple[int, int] | None) -> list[torch.Tensor]:
        if shape is None:
            return features
        batch_size, num_slices = shape
        return [feature.reshape(batch_size, num_slices, feature.shape[1], *feature.shape[-2:]) for feature in features]
