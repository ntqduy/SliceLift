from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def target_spatial_shape(target: torch.Tensor) -> tuple[int, ...]:
    """Return target spatial shape for [B, ...] or [B, 1, ...] masks."""
    if target.ndim < 3:
        raise ValueError(f"Target must have shape [B, ...] or [B, 1, ...], got {tuple(target.shape)}.")
    if target.ndim >= 4 and int(target.shape[1]) == 1:
        return tuple(int(size) for size in target.shape[2:])
    return tuple(int(size) for size in target.shape[1:])


def resize_to_shape(value: torch.Tensor, spatial_shape: Sequence[int], mode: str | None = None) -> torch.Tensor:
    """Resize a 2D/3D tensor to a target spatial shape."""
    spatial_shape = tuple(int(size) for size in spatial_shape)
    if tuple(value.shape[2:]) == spatial_shape:
        return value
    if mode is None:
        mode = "trilinear" if len(spatial_shape) == 3 else "bilinear"
    kwargs = {"size": spatial_shape, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(value, **kwargs)


def binary_target(target: torch.Tensor, spatial_shape: Sequence[int] | None = None) -> torch.Tensor:
    """Convert target masks to float binary masks with shape [B, 1, ...]."""
    if target.ndim < 3:
        raise ValueError(f"Target must have at least 3 dims, got {tuple(target.shape)}.")
    if target.ndim >= 4 and int(target.shape[1]) == 1:
        value = target[:, :1]
    else:
        value = target.unsqueeze(1)
    value = (value.float() > 0).float()
    if spatial_shape is not None and tuple(value.shape[2:]) != tuple(spatial_shape):
        value = resize_to_shape(value, spatial_shape, mode="nearest")
    return value


def foreground_logits(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one-channel or two-channel segmentation logits to foreground logits."""
    if logits.ndim not in {4, 5}:
        raise ValueError(f"Logits must be [B, C, H, W] or [B, C, D, H, W], got {tuple(logits.shape)}.")
    spatial_shape = target_spatial_shape(target)
    logits = resize_to_shape(logits.float(), spatial_shape)
    if int(logits.shape[1]) == 1:
        fg_logits = logits[:, :1]
    elif int(logits.shape[1]) >= 2:
        fg_logits = logits[:, 1:2] - logits[:, 0:1]
    else:
        raise ValueError(f"Logits channel dimension must be >=1, got {tuple(logits.shape)}.")
    return fg_logits, binary_target(target, spatial_shape=spatial_shape)


def single_channel_probability(value: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare probability-like inputs and binary target for dice-style losses."""
    spatial_shape = target_spatial_shape(target)
    value = resize_to_shape(value.float(), spatial_shape)
    if int(value.shape[1]) == 1:
        probability = value[:, :1]
    elif int(value.shape[1]) >= 2:
        probability = value[:, 1:2]
    else:
        raise ValueError(f"Probability channel dimension must be >=1, got {tuple(value.shape)}.")
    return probability.clamp(0.0, 1.0), binary_target(target, spatial_shape=spatial_shape)


def soft_dice_loss(probability: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice loss from probabilities and binary targets, fully tensor-based."""
    if probability.shape != target.shape:
        raise ValueError(f"Dice inputs must have matching shapes, got {tuple(probability.shape)} and {tuple(target.shape)}.")
    probability = probability.float()
    target = target.float()
    reduce_dims = tuple(range(2, probability.ndim))
    intersection = torch.sum(probability * target, dim=reduce_dims)
    denominator = torch.sum(probability + target, dim=reduce_dims)
    dice = (2.0 * intersection + float(smooth)) / (denominator + float(smooth))
    return 1.0 - dice.mean()


class DiceLoss(nn.Module):
    """Binary soft Dice loss for 2D/3D logits or probabilities."""

    def __init__(self, smooth: float = 1e-6, from_logits: bool = True) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.from_logits = bool(from_logits)

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            fg_logits, binary = foreground_logits(inputs, target)
            probability = torch.sigmoid(fg_logits)
        else:
            probability, binary = single_channel_probability(inputs, target)
        return soft_dice_loss(probability, binary, smooth=self.smooth)
