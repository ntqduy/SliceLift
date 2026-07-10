from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from utils.model_output import BaseSegmentationModel
except Exception:  # pragma: no cover
    class BaseSegmentationModel(nn.Module):
        def set_architecture_config(self, **kwargs) -> None:
            self.architecture_config = dict(kwargs)

        def build_output(self, logits, features=None, aux=None):
            return logits, features


def import_proposal_module(module_path: str):
    package_root = "Proposal_Model_Experiment"
    try:
        return importlib.import_module(f"{package_root}.{module_path}")
    except ModuleNotFoundError:
        baseline_root = Path(__file__).resolve().parents[2]
        baseline_root_text = str(baseline_root)
        if baseline_root_text not in sys.path:
            sys.path.insert(0, baseline_root_text)
        return importlib.import_module(f"{package_root}.{module_path}")


_enc2d = import_proposal_module("2D_encoder.factory")
_enc3d = import_proposal_module("3D_encoder.factory")
_decoders = import_proposal_module("Decoder.factory")
_selector_module = import_proposal_module("selection_slice_2D")

build_2d_encoder = getattr(_enc2d, "build_2d_encoder")
normalise_2d_encoder_name = getattr(_enc2d, "normalise_2d_encoder_name")
build_3d_encoder = getattr(_enc3d, "build_3d_encoder")
normalise_3d_encoder_name = getattr(_enc3d, "normalise_3d_encoder_name")
build_2d_decoder = getattr(_decoders, "build_2d_decoder")
build_3d_decoder = getattr(_decoders, "build_3d_decoder")
normalise_decoder_model = getattr(_decoders, "normalise_decoder_model")
normalise_decoder_style = getattr(_decoders, "normalise_decoder_style")
SliceSelector = getattr(_selector_module, "SliceSelector")


def channels_from_config(channels: list[int] | tuple[int, ...] | None, base_channels: int) -> list[int]:
    values = list(channels or [int(base_channels) * (2**index) for index in range(5)])
    if len(values) != 5:
        raise ValueError(f"Expected 5 channel stages, got {values}")
    return [int(item) for item in values]


def checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state", "state_dict", "model_state_dict", "network_weights"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def normalise_encoder_fusion_mode(mode: str | None) -> str:
    value = str(mode or "concat").lower().replace("-", "_")
    if value in {"concat", "cat", "concat_conv", "concat_1x1", "concat_conv1x1"}:
        return "concat"
    if value in {"add", "sum", "add_conv", "add_1x1", "add_conv1x1"}:
        return "add"
    raise ValueError("encoder_fusion_mode must be concat or add.")


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


def main_logits(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def component_load_report(component: nn.Module, checkpoint: Mapping[str, Any], prefix: str, label: str, strict: bool) -> dict[str, Any]:
    if hasattr(component, "load_partial_state_dict") and not strict:
        report = component.load_partial_state_dict(checkpoint, prefix=prefix)
        report["label"] = label
        return report

    state_dict = checkpoint_state_dict(checkpoint)
    target = component.state_dict()
    updates = {}
    skipped = []
    for raw_key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key)
        if key.startswith("module."):
            key = key[7:]
        if prefix:
            if not key.startswith(prefix):
                continue
            key = key[len(prefix) :]
        if key in target and tuple(target[key].shape) == tuple(value.shape):
            updates[key] = value
        elif key in target:
            skipped.append({"key": key, "reason": f"shape {tuple(value.shape)} != {tuple(target[key].shape)}"})
    merged = dict(target)
    merged.update(updates)
    component.load_state_dict(merged, strict=False)
    missing = [key for key in target if key not in updates]
    return {
        "label": label,
        "loaded_keys": sorted(updates),
        "skipped_keys": skipped,
        "missing_keys": missing,
        "num_loaded_keys": len(updates),
        "num_skipped_keys": len(skipped),
        "num_missing_keys": len(missing),
    }


class SlicePositionEncoder(nn.Module):
    def __init__(self, channels: list[int] | tuple[int, ...], embedding_dim: int = 32, max_positions: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(max_positions), int(embedding_dim))
        self.projections = nn.ModuleList([nn.Linear(int(embedding_dim), int(channel)) for channel in channels])
        self.max_positions = int(max_positions)

    def forward_2d(self, features: list[torch.Tensor], slice_indices: torch.Tensor | None) -> list[torch.Tensor]:
        if not features:
            return features
        if slice_indices is None:
            batch_size = int(features[0].shape[0])
            slice_indices = torch.zeros(batch_size, device=features[0].device, dtype=torch.long)
        if slice_indices.ndim == 2:
            slice_indices = slice_indices[:, 0]
        slice_indices = slice_indices.to(device=features[0].device, dtype=torch.long).reshape(-1).clamp_(0, self.max_positions - 1)
        if slice_indices.numel() == 1 and int(features[0].shape[0]) > 1:
            slice_indices = slice_indices.expand(int(features[0].shape[0]))
        if slice_indices.numel() != int(features[0].shape[0]):
            raise ValueError(f"slice_indices must contain {int(features[0].shape[0])} values, got {int(slice_indices.numel())}.")
        embedding = self.embedding(slice_indices)
        encoded = []
        for feature, projection in zip(features, self.projections):
            bias = projection(embedding).to(dtype=feature.dtype, device=feature.device)
            encoded.append(feature + bias[..., None, None])
        return encoded

    def forward_selected(self, features: list[torch.Tensor], slice_indices: torch.Tensor) -> list[torch.Tensor]:
        clipped = slice_indices.clamp(0, self.max_positions - 1)
        embedding = self.embedding(clipped)
        encoded = []
        for feature, projection in zip(features, self.projections):
            bias = projection(embedding).to(dtype=feature.dtype, device=feature.device)
            encoded.append(feature + bias[..., None, None])
        return encoded
