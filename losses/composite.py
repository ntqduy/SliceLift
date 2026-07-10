from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .bce import BinaryBCEWithLogitsLoss
from .dice import DiceLoss


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalise_loss_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if "training" in config and isinstance(config.get("training"), Mapping):
        train_cfg = _as_mapping(config.get("training"))
        raw_loss = config.get("loss", train_cfg.get("loss", "ce_dice"))
    else:
        train_cfg = _as_mapping(config)
        raw_loss = train_cfg.get("loss", "ce_dice")

    if isinstance(raw_loss, Mapping):
        loss_cfg = dict(raw_loss)
    else:
        loss_cfg = {"name": str(raw_loss)}
    merged = dict(train_cfg)
    merged.update(loss_cfg)
    merged["name"] = str(merged.get("name", "ce_dice")).lower()
    return merged


def _tensor_zero_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().sum() * 0.0


class CompositeSegmentationLoss(nn.Module):
    """Binary segmentation loss configured as Dice, BCE, or Dice+BCE."""

    def __init__(self, config: Mapping[str, Any], num_classes: int = 2) -> None:
        super().__init__()
        cfg = _normalise_loss_config(config)
        self.loss_name = str(cfg.get("name", "dice_bce")).lower()
        original_loss_name = self.loss_name
        self.num_classes = int(num_classes)
        self.auxiliary_weight = float(cfg.get("auxiliary_weight", cfg.get("deep_supervision_weight", 1.0)))
        self.last_components: dict[str, torch.Tensor] = {}

        self.dice = DiceLoss(smooth=float(cfg.get("smooth", 1e-6)), from_logits=True)
        self.bce = BinaryBCEWithLogitsLoss()

        legacy_ce_dice = original_loss_name in {"ce_dice", "dice_ce", "combined"}
        default_dice_weight = 0.5 if legacy_ce_dice else 1.0
        default_bce_weight = 0.5 if legacy_ce_dice else 1.0
        self.lambda_dice = float(cfg.get("lambda_dice", cfg.get("dice_weight", default_dice_weight)))
        self.lambda_bce = float(cfg.get("lambda_bce", cfg.get("bce_weight", cfg.get("ce_weight", default_bce_weight))))

        aliases = {
            "ce_dice": "dice_bce",
            "dice_ce": "dice_bce",
            "combined": "dice_bce",
            "bce_dice": "dice_bce",
            "dice_bce": "dice_bce",
        }
        if self.loss_name in {"ce", "cross_entropy"}:
            self.loss_name = "bce"
        else:
            self.loss_name = aliases.get(self.loss_name, self.loss_name)
        if self.loss_name not in {
            "dice",
            "bce",
            "dice_bce",
        }:
            raise ValueError(
                "Unsupported loss '{}'. Available in this method: ce_dice/dice_bce, dice, bce.".format(cfg.get("name"))
            )
        self.requires_encoder_features = False

    def _component(self, key: str, value: torch.Tensor, components: dict[str, torch.Tensor]) -> torch.Tensor:
        components[key] = value
        return value

    def forward(self, logits: torch.Tensor, target: torch.Tensor, encoder_features: Any | None = None) -> torch.Tensor:
        components: dict[str, torch.Tensor] = {}
        total = _tensor_zero_like(logits)

        if self.loss_name == "dice":
            dice = self._component("loss_dice", self.dice(logits, target), components)
            total = total + self.lambda_dice * dice
        elif self.loss_name == "bce":
            bce = self._component("loss_bce", self.bce(logits, target), components)
            total = total + self.lambda_bce * bce
        else:
            dice = self._component("loss_dice", self.dice(logits, target), components)
            total = total + self.lambda_dice * dice

        if self.loss_name == "dice_bce":
            bce = self._component("loss_bce", self.bce(logits, target), components)
            total = total + self.lambda_bce * bce

        components["loss_total"] = total
        self.last_components = {key: value.detach() for key, value in components.items()}
        return total


class SegmentationCriterion(nn.Module):
    """Backward-compatible criterion wrapper for the existing training code."""

    def __init__(self, config: Mapping[str, Any], num_classes: int) -> None:
        super().__init__()
        cfg = _normalise_loss_config(config)
        self.loss = CompositeSegmentationLoss(config, num_classes=num_classes)
        self.auxiliary_weight = float(cfg.get("auxiliary_weight", cfg.get("deep_supervision_weight", 1.0)))
        self.requires_encoder_features = bool(getattr(self.loss, "requires_encoder_features", False))

    @property
    def last_components(self) -> dict[str, torch.Tensor]:
        return dict(getattr(self.loss, "last_components", {}))

    @property
    def loss_name(self) -> str:
        return str(getattr(self.loss, "loss_name", self.loss.__class__.__name__))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, encoder_features: Any | None = None) -> torch.Tensor:
        return self.loss(logits, target, encoder_features=encoder_features)


def build_loss(config: Mapping[str, Any], num_classes: int = 2) -> SegmentationCriterion:
    """Factory used by tests and the runner."""
    return SegmentationCriterion(config, num_classes=num_classes)
