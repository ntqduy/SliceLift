from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def ensure_model_on_device(model: nn.Module, device: torch.device) -> nn.Module:
    target_device = torch.device(device)
    model.to(target_device)
    mismatches: list[str] = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor is not None and tensor.device != target_device:
            mismatches.append(f"{name}={tensor.device}")
            if len(mismatches) >= 5:
                break
    if mismatches:
        joined = ", ".join(mismatches)
        raise RuntimeError(f"Model still has tensors outside {target_device}: {joined}")
    return model


def model_state_dict(model: nn.Module):
    return unwrap_model(model).state_dict()


def load_model_state(model: nn.Module, state_dict) -> None:
    unwrap_model(model).load_state_dict(state_dict)


def extract_logits(output: Any, num_classes: int | None = None) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict):
        for key in ("logits", "out", "output", "seg"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
    if isinstance(output, (tuple, list)):
        tensor_candidates = [item for item in output if isinstance(item, torch.Tensor)]
        if num_classes is not None:
            for item in tensor_candidates:
                if item.ndim >= 4 and item.shape[1] == num_classes:
                    return item
        if tensor_candidates:
            return tensor_candidates[0]
    raise TypeError(f"Cannot extract logits from model output of type {type(output)!r}")


def extract_auxiliary_logits(output: Any, num_classes: int | None = None) -> list[torch.Tensor]:
    if hasattr(output, "features") and isinstance(output.features, dict):
        deep_outputs = output.features.get("deep_outputs", [])
        return [item for item in deep_outputs if isinstance(item, torch.Tensor)]
    if isinstance(output, dict):
        deep_outputs = output.get("deep_outputs", output.get("aux_logits", []))
        if isinstance(deep_outputs, torch.Tensor):
            return [deep_outputs]
        if isinstance(deep_outputs, (tuple, list)):
            return [item for item in deep_outputs if isinstance(item, torch.Tensor)]
    if isinstance(output, (tuple, list)):
        tensor_candidates = [item for item in output if isinstance(item, torch.Tensor)]
        if not tensor_candidates:
            return []
        main = extract_logits(output, num_classes=num_classes)
        auxiliaries = tensor_candidates[1:]
        if tensor_candidates[0] is not main:
            auxiliaries = [item for item in tensor_candidates if item is not main]
        return auxiliaries
    return []


def extract_encoder_features(output: Any) -> list[torch.Tensor]:
    """Extract encoder feature tensors from model outputs used by attention supervision."""

    def _from_features(features: Any) -> list[torch.Tensor]:
        if features is None:
            return []
        if isinstance(features, torch.Tensor):
            return [features]
        if hasattr(features, "features"):
            return _from_features(features.features)
        if isinstance(features, Mapping):
            for key in ("encoder", "encoder_features", "skips"):
                if key in features:
                    return _from_features(features[key])
            if "bottleneck" in features and isinstance(features["bottleneck"], torch.Tensor):
                return [features["bottleneck"]]
            return [item for item in features.values() if isinstance(item, torch.Tensor)]
        if isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
            collected: list[torch.Tensor] = []
            for item in features:
                collected.extend(_from_features(item))
            return collected
        return []

    if hasattr(output, "features"):
        return _from_features(output.features)
    if isinstance(output, Mapping):
        for key in ("features", "encoder", "encoder_features", "skips"):
            if key in output:
                return _from_features(output[key])
        return []
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        for item in output[1:]:
            features = _from_features(item)
            if features:
                return features
    return []


def resize_logits(logits: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(logits.shape[2:]) == tuple(target_shape):
        return logits
    mode = "trilinear" if len(target_shape) == 3 else "bilinear"
    return F.interpolate(logits, size=target_shape, mode=mode, align_corners=False)


def predict_from_logits(logits: torch.Tensor, target_shape: tuple[int, ...] | None = None) -> torch.Tensor:
    if target_shape is not None:
        logits = resize_logits(logits, target_shape)
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits[:, 0]) > 0.5).long()
    return torch.argmax(torch.softmax(logits, dim=1), dim=1)


def _remove_profile_buffers(model: nn.Module) -> None:
    for module in unwrap_model(model).modules():
        for name in ("total_ops", "total_params"):
            module._buffers.pop(name, None)


def _snapshot_forward_hooks(model: nn.Module) -> dict[nn.Module, set[int]]:
    return {module: set(module._forward_hooks.keys()) for module in unwrap_model(model).modules()}


def _remove_new_forward_hooks(model: nn.Module, before: dict[nn.Module, set[int]]) -> None:
    for module in unwrap_model(model).modules():
        keep = before.get(module, set())
        for hook_id in list(module._forward_hooks.keys()):
            if hook_id not in keep:
                module._forward_hooks.pop(hook_id, None)


def estimate_flops(model: nn.Module, input_shape: tuple[int, ...], device: torch.device):
    try:
        from thop import profile
    except ImportError:
        return None, "thop is not installed"

    base_model = unwrap_model(model)
    was_training = base_model.training
    forward_hooks_before = _snapshot_forward_hooks(base_model)
    try:
        base_model.eval()
        dummy = torch.zeros(input_shape, device=device)
        flops, _ = profile(base_model, inputs=(dummy,), verbose=False)
        return int(flops), None
    except Exception as error:
        return None, str(error)
    finally:
        _remove_new_forward_hooks(base_model, forward_hooks_before)
        _remove_profile_buffers(base_model)
        base_model.train(was_training)
        base_model.to(device)
