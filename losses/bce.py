from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dice import foreground_logits


class BinaryBCEWithLogitsLoss(nn.Module):
    """BCEWithLogits for binary segmentation with one- or two-channel logits."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fg_logits, binary = foreground_logits(logits, target)
        return F.binary_cross_entropy_with_logits(fg_logits.float(), binary.float())


BCEWithLogitsLoss = BinaryBCEWithLogitsLoss
