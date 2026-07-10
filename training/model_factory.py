from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from typing import Any, Mapping

from torch import nn

from .utils import get_nested, project_root


@dataclass
class ModelBuildResult:
    model: nn.Module
    name: str
    backbone: str
    in_channels: int
    num_classes: int


_STAGE_TO_MODEL_TYPE = {
    "train_2d": "2D",
    "train_3d": "3D",
    "hybrid": "3D",
}

_PROPOSAL_2D_NAMES = {
    "proposal_model_experiment",
    "proposal_experiment_2d",
    "proposal_model_experiment_2d",
    "proposal_exp_2d",
    "full_unet3plus_2d",
    "full_unet_3_plus_2d",
    "fullunet3plus2d",
}

_PROPOSAL_3D_NAMES = {
    "proposal_experiment_3d",
    "proposal_model_experiment_3d",
    "proposal_exp_3d",
    "full_unet3d",
    "full_unet3d_3_plus",
    "full_unet3d_3plus",
    "full_unet_3d_3_plus",
    "fullunet3d",
    "fullunet3d_3plus",
}

_PROPOSAL_HYBRID_NAMES = {
    "proposal_model_experiment",
    "proposal_experiment_hybrid",
    "proposal_model_experiment_hybrid",
    "proposal_exp_hybrid",
    "hybrid_3d_2d",
    "hybrid3d2d",
    "hybrid_2d_3d",
    "hybrid_3d_2d_improve",
    "hybrid_3d_2d_slice_inject",
    "proposal_hybrid_3d_2d",
    "proposal_hybrid_3d_2d_unet3plus",
    "proposal_hybrid_3d_2d_improve",
    "proposal_hybrid_3d_2d_slice_inject",
}

_CLASS_BY_KIND = {
    "2d": "Experiment2DSegModel",
    "3d": "Experiment3DSegModel",
    "hybrid": "ExperimentHybridModel",
}


def _normalise_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _ensure_baseline_on_path() -> None:
    baseline_root = project_root() / "baseline"
    baseline_root_text = str(baseline_root)
    if baseline_root_text in sys.path:
        sys.path.remove(baseline_root_text)
    sys.path.insert(0, baseline_root_text)


def _proposal_module():
    _ensure_baseline_on_path()
    return importlib.import_module("Proposal_Model_Experiment.HybridModel.hybrid_model")


def _filter_kwargs(cls: type[nn.Module], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _constructor_kwargs(cls: type[nn.Module], model_args: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = _filter_kwargs(cls, model_args)
    kwargs.pop("__replace__", None)
    kwargs.pop("in_channels", None)
    kwargs.pop("num_classes", None)
    return kwargs


def _is_proposal_config(cfg: Mapping[str, Any], name: str) -> bool:
    experiment_name = _normalise_name(get_nested(cfg, "experiment.name", get_nested(cfg, "project.name", "")))
    return (
        "proposal_model_experiment" in experiment_name
        or "proposal_hybrid_3d_2d" in experiment_name
        or name in _PROPOSAL_2D_NAMES
        or name in _PROPOSAL_3D_NAMES
        or name in _PROPOSAL_HYBRID_NAMES
    )


def _resolve_model_type(cfg: Mapping[str, Any], name: str) -> str:
    stage = _normalise_name(get_nested(cfg, "experiment.stage", ""))
    configured_type = str(get_nested(cfg, "model.type", "2D")).upper()
    if _is_proposal_config(cfg, name) and stage in _STAGE_TO_MODEL_TYPE:
        return _STAGE_TO_MODEL_TYPE[stage]
    if configured_type not in {"2D", "3D"}:
        raise ValueError(f"model.type must be 2D or 3D, got {configured_type}")
    return configured_type


def _proposal_kind(cfg: Mapping[str, Any], model_type: str, name: str) -> str:
    stage = _normalise_name(get_nested(cfg, "experiment.stage", ""))
    if stage == "train_2d" or (model_type == "2D" and name in _PROPOSAL_2D_NAMES):
        return "2d"
    if stage == "train_3d" or name in _PROPOSAL_3D_NAMES:
        return "3d"
    if stage == "hybrid" or name in _PROPOSAL_HYBRID_NAMES:
        return "hybrid"
    if model_type == "2D":
        return "2d"
    raise ValueError(
        "Unsupported proposal model name "
        f"{name!r}. Expected a Stage 1 2D, Stage 2 3D, or Stage 3 hybrid proposal model."
    )


def _build_proposal_model(
    cfg: Mapping[str, Any],
    *,
    model_type: str,
    name: str,
    in_channels: int,
    num_classes: int,
    model_args: Mapping[str, Any],
) -> nn.Module:
    if name.startswith("proposal_method"):
        raise ValueError(
            "model.name uses Proposal_Method, but this repository only contains "
            "baseline/Proposal_Model_Experiment. Use proposal_model_experiment configs."
        )
    kind = _proposal_kind(cfg, model_type, name)
    cls = getattr(_proposal_module(), _CLASS_BY_KIND[kind])
    return cls(
        in_channels=in_channels,
        num_classes=num_classes,
        **_constructor_kwargs(cls, model_args),
    )


def _architecture_config_value(model: nn.Module, *keys: str) -> str | None:
    architecture_config = getattr(model, "architecture_config", None)
    if not isinstance(architecture_config, Mapping):
        return None
    for key in keys:
        value = architecture_config.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _resolved_backbone_name(model: nn.Module, configured_backbone: str) -> str:
    model_backbone = getattr(model, "backbone_name", None)
    if model_backbone:
        return str(model_backbone)
    architecture_backbone = _architecture_config_value(model, "backbone", "encoder_name", "encoder", "encoder_type")
    if architecture_backbone:
        return architecture_backbone
    return configured_backbone or model.__class__.__name__


def _model_args_for_type(cfg: Mapping[str, Any], model_type: str) -> dict[str, Any]:
    model_args = dict(get_nested(cfg, "model.args", {}) or {})
    model_args.pop("__replace__", None)
    typed_args = get_nested(cfg, f"model.args_{model_type.lower()}", None)
    if isinstance(typed_args, Mapping):
        typed_args = dict(typed_args)
        typed_args.pop("__replace__", None)
        model_args.update(typed_args)
    return model_args


def _validate_image_size(cfg: Mapping[str, Any], model_type: str) -> None:
    image_size = get_nested(cfg, "training.image_size", [256, 256])
    preserve_depth = bool(get_nested(cfg, "training.preserve_depth", get_nested(cfg, "dataset.preserve_depth", False)))
    if model_type == "2D" and len(image_size) < 2:
        raise ValueError("training.image_size must contain [height, width] for 2D models.")
    if model_type == "3D" and preserve_depth and len(image_size) < 2:
        raise ValueError("training.image_size must contain [height, width] when training.preserve_depth=true.")
    if model_type == "3D" and not preserve_depth and len(image_size) < 3:
        raise ValueError("training.image_size must contain [height, width, depth] for 3D models.")


def build_model(cfg: Mapping[str, Any], dataset_in_channels: int) -> ModelBuildResult:
    raw_name = str(get_nested(cfg, "model.name", "proposal_model_experiment"))
    name = _normalise_name(raw_name)
    model_type = _resolve_model_type(cfg, name)
    if not _is_proposal_config(cfg, name):
        raise ValueError(
            "This cleaned paper-release factory only supports the proposal architectures in "
            "baseline/Proposal_Model_Experiment. Set model.name to proposal_model_experiment "
            "and choose experiment.stage=train_2d, train_3d, or hybrid."
        )

    _validate_image_size(cfg, model_type)
    num_classes = int(get_nested(cfg, "model.num_classes", get_nested(cfg, "dataset.num_classes", 2)))
    requested_channels = get_nested(cfg, "model.in_channels", "auto")
    in_channels = int(dataset_in_channels if str(requested_channels).lower() == "auto" else requested_channels)
    model_args = _model_args_for_type(cfg, model_type)

    model = _build_proposal_model(
        cfg,
        model_type=model_type,
        name=name,
        in_channels=in_channels,
        num_classes=num_classes,
        model_args=model_args,
    )
    return ModelBuildResult(
        model=model,
        name=str(getattr(model, "model_name", name)),
        backbone=_resolved_backbone_name(model, str(get_nested(cfg, "model.backbone", "") or "")),
        in_channels=in_channels,
        num_classes=num_classes,
    )
