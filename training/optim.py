from __future__ import annotations

from typing import Mapping

import torch


def build_optimizer(parameters, cfg: Mapping):
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", False)),
        )
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, cfg: Mapping):
    name = str(cfg.get("scheduler", "none")).lower()
    epochs = int(cfg.get("epochs", 1))
    if name in {"none", "null", "off"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=float(cfg.get("min_lr", 1e-6)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", max(1, epochs // 3))),
            gamma=float(cfg.get("gamma", 0.1)),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=int(cfg.get("scheduler_patience", 5)),
            factor=float(cfg.get("scheduler_factor", 0.5)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")
