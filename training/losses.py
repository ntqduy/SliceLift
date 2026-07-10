from __future__ import annotations

"""Compatibility wrapper for the modular losses package."""

import torch

from losses import (
    BCEWithLogitsLoss,
    CompositeSegmentationLoss,
    DiceLoss,
    SegmentationCriterion,
    build_loss,
)
from losses.dice import foreground_logits, resize_to_shape, soft_dice_loss, target_spatial_shape


def resize_logits_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return resize_to_shape(logits, target_spatial_shape(target))


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    include_background: bool = False,
    smooth: float = 1e-6,
) -> torch.Tensor:
    del num_classes, include_background
    return DiceLoss(smooth=smooth, from_logits=True)(logits, target)


__all__ = [
    "BCEWithLogitsLoss",
    "CompositeSegmentationLoss",
    "DiceLoss",
    "SegmentationCriterion",
    "build_loss",
    "dice_loss",
    "foreground_logits",
    "resize_logits_to_target",
    "soft_dice_loss",
]
