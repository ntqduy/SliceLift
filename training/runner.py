from __future__ import annotations

import copy
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import (
    build_2d_volume_eval_datasets,
    build_datasets,
    build_datasets_from_records,
    build_kfold_record_splits,
    stack_slices_as_volume,
)
from .losses import SegmentationCriterion
from .metrics import MetricAccumulator
from .model_factory import build_model
from .optim import build_optimizer, build_scheduler
from .torch_utils import (
    ensure_model_on_device,
    estimate_flops,
    extract_auxiliary_logits,
    extract_encoder_features,
    extract_logits,
    load_model_state,
    model_state_dict,
    predict_from_logits,
    unwrap_model,
)
from .utils import (
    count_parameters,
    describe_device,
    display_path,
    format_large_number,
    get_nested,
    sanitize_name,
    save_json,
    seed_worker,
    set_seed,
    setup_logger,
)
from .visualization import save_prediction_visuals


_PROPOSAL_EXPERIMENT_NAME = "proposal_hybrid_3d_2d"
_PROPOSAL_IMPROVE_EXPERIMENT_NAME = "proposal_hybrid_3d_2d_improve"
_PROPOSAL_MODEL_EXPERIMENT_NAME = "proposal_model_experiment"
_PROPOSAL_METHOD_NAME = "proposal_method"
_PROPOSAL_STAGES = {"train_2d", "train_3d", "hybrid"}


def _proposal_default_output_name(cfg: Mapping[str, Any]) -> str:
    if _proposal_uses_method(cfg):
        return "Proposal_Method"
    return "Proposal_Model_Experiment" if _proposal_uses_model_experiment(cfg) else "Proposal_hybrid_3D_2D"


def _is_proposal_hybrid_config(cfg: Mapping[str, Any]) -> bool:
    experiment_name = str(get_nested(cfg, "experiment.name", get_nested(cfg, "project.name", ""))).lower()
    model_name = str(get_nested(cfg, "model.name", "")).lower()
    return _PROPOSAL_EXPERIMENT_NAME in experiment_name or _PROPOSAL_IMPROVE_EXPERIMENT_NAME in experiment_name or model_name in {
        "proposal_hybrid_3d_2d",
        "proposal_hybrid_3d_2d_unet3plus",
        "proposal_hybrid_3d_2d_improve",
        "proposal_hybrid_3d_2d_slice_inject",
        "proposal_model_experiment",
        "proposal_experiment_2d",
        "proposal_experiment_3d",
        "proposal_experiment_hybrid",
        "proposal_method",
        "proposal_method_2d",
        "proposal_method_3d",
        "proposal_method_hybrid",
        "hybrid_3d_2d_improve",
        "hybrid_3d_2d_slice_inject",
        "hybrid_3d_2d",
    } or _PROPOSAL_MODEL_EXPERIMENT_NAME in experiment_name or _PROPOSAL_METHOD_NAME in experiment_name


def _proposal_uses_improve_model(cfg: Mapping[str, Any]) -> bool:
    experiment_name = str(get_nested(cfg, "experiment.name", get_nested(cfg, "project.name", ""))).lower()
    model_name = str(get_nested(cfg, "model.name", "")).lower()
    return "improve" in experiment_name or "slice_inject" in experiment_name or "improve" in model_name or "slice_inject" in model_name


def _proposal_uses_model_experiment(cfg: Mapping[str, Any]) -> bool:
    experiment_name = str(get_nested(cfg, "experiment.name", get_nested(cfg, "project.name", ""))).lower()
    model_name = str(get_nested(cfg, "model.name", "")).lower()
    return _PROPOSAL_MODEL_EXPERIMENT_NAME in experiment_name or model_name in {
        "proposal_model_experiment",
        "proposal_experiment_2d",
        "proposal_experiment_3d",
        "proposal_experiment_hybrid",
    }


def _proposal_uses_method(cfg: Mapping[str, Any]) -> bool:
    experiment_name = str(get_nested(cfg, "experiment.name", get_nested(cfg, "project.name", ""))).lower()
    model_name = str(get_nested(cfg, "model.name", "")).lower()
    return _PROPOSAL_METHOD_NAME in experiment_name or model_name in {
        "proposal_method",
        "proposal_method_2d",
        "proposal_method_3d",
        "proposal_method_hybrid",
    }


def _proposal_stage(cfg: Mapping[str, Any]) -> str:
    stage = str(get_nested(cfg, "experiment.stage", "train_2d")).lower()
    if stage not in _PROPOSAL_STAGES:
        raise ValueError(f"experiment.stage must be one of {sorted(_PROPOSAL_STAGES)}, got {stage!r}")
    return stage


def _proposal_slice_mode(cfg: Mapping[str, Any]) -> str:
    return str(get_nested(cfg, "model.slice_selection.mode", get_nested(cfg, "slice_2d.sampling_strategy", "uniform"))).lower().replace("-", "_")


def _is_group_based_slice_mode(mode: Any) -> bool:
    return str(mode or "").lower().replace("-", "_") == "group_based"


def _group_based_slice_config(slice_selection: Mapping[str, Any]) -> dict[str, Any]:
    group_cfg: dict[str, Any] = {}
    for key in ("group_based", "group-based"):
        value = slice_selection.get(key, {})
        if isinstance(value, Mapping):
            group_cfg.update(dict(value))
    return group_cfg


def _group_based_selection_count(slice_selection: Mapping[str, Any]) -> int:
    group_cfg = _group_based_slice_config(slice_selection)
    fallback_count = max(1, int(slice_selection.get("num_slices", 1)))
    num_groups = max(1, int(group_cfg.get("num_groups", slice_selection.get("num_groups", fallback_count))))
    samples_per_group = max(1, int(group_cfg.get("samples_per_group", slice_selection.get("samples_per_group", 1))))
    return num_groups * samples_per_group


def _proposal_is_auto_value(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "auto"}


def _proposal_is_nnunet_name(value: Any) -> bool:
    key = str(value or "").strip().lower().replace("-", "_").replace("+", "plus")
    return key in {"nnunet", "nnunet2d", "nnunet3d", "nn_unet", "nn_unet2d", "nn_unet3d"}


def _proposal_is_unet3d_name(value: Any) -> bool:
    key = str(value or "").strip().lower().replace("-", "_").replace("+", "plus")
    return key in {"unet", "unet3d", "unet_3d"}


def _proposal_default_normalization(configured: Any = None, *model_names: Any) -> str:
    if not _proposal_is_auto_value(configured):
        return str(configured)
    return "instancenorm" if any(_proposal_is_nnunet_name(name) for name in model_names) else "batchnorm"


def _proposal_decoder_style(configured: Any = None, default: str = "same_scale") -> str:
    key = str(configured or default).strip().lower().replace("-", "_").replace("+", "plus")
    if key == "auto":
        key = str(default or "same_scale").strip().lower().replace("-", "_").replace("+", "plus")
    aliases = {
        "same": "same_scale",
        "same_scale": "same_scale",
        "skip": "same_scale",
        "unet": "same_scale",
        "unet3d": "same_scale",
        "unet_3d": "same_scale",
        "unet_decoder": "same_scale",
        "nnunet": "same_scale",
        "nnunet3d": "same_scale",
        "nn_unet": "same_scale",
        "nn_unet3d": "same_scale",
        "nested": "nested_dense",
        "nested_dense": "nested_dense",
        "unetpp": "nested_dense",
        "unetplusplus": "nested_dense",
        "unet_plus_plus": "nested_dense",
        "unetpp3d": "nested_dense",
        "unetplusplus3d": "nested_dense",
        "unet_plus_plus3d": "nested_dense",
        "full": "full_scale",
        "full_scale": "full_scale",
        "unet3plus": "full_scale",
        "unet_3plus": "full_scale",
        "unet_3_plus": "full_scale",
        "unet3plus3d": "full_scale",
        "unet_3plus3d": "full_scale",
        "unet_3_plus3d": "full_scale",
        "full_encoder_single_decoder": "full_encoder_single_decoder",
        "all_encoder_single_decoder": "full_encoder_single_decoder",
        "full_encoder_one_decoder": "full_encoder_single_decoder",
        "all_encoder_one_decoder": "full_encoder_single_decoder",
        "single_encoder_full_decoder": "single_encoder_full_decoder",
        "single_encoder_full_decode": "single_encoder_full_decoder",
        "same_encoder_full_decoder": "single_encoder_full_decoder",
        "same_encoder_full_decode": "single_encoder_full_decoder",
        "same_scale_full_decoder": "single_encoder_full_decoder",
        "same_scale_full_decode": "single_encoder_full_decoder",
        "single_skip_full_decoder": "single_encoder_full_decoder",
        "single_skip_full_decode": "single_encoder_full_decoder",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported decoder style: {configured!r}")
    return aliases[key]


def _proposal_decoder_model(configured: Any = None, default: str = "unet3d") -> str:
    key = str(configured or default).strip().lower().replace("-", "_").replace("+", "plus")
    if key == "auto":
        key = str(default or "unet3d").strip().lower().replace("-", "_").replace("+", "plus")
    aliases = {
        "unet": "unet3d",
        "unet3d": "unet3d",
        "unet_3d": "unet3d",
        "unetpp": "unetpp3d",
        "unetplusplus": "unetpp3d",
        "unet_plus_plus": "unetpp3d",
        "unetpp3d": "unetpp3d",
        "unetplusplus3d": "unetpp3d",
        "unet_plus_plus3d": "unetpp3d",
        "unet_plus_plus_3d": "unetpp3d",
        "unet3plus": "unet3plus3d",
        "unet_3plus": "unet3plus3d",
        "unet_3_plus": "unet3plus3d",
        "unet3plus3d": "unet3plus3d",
        "unet_3plus3d": "unet3plus3d",
        "unet_3_plus3d": "unet3plus3d",
        "unet_3_plus_3d": "unet3plus3d",
        "nnunet": "nnunet3d",
        "nnunet3d": "nnunet3d",
        "nn_unet": "nnunet3d",
        "nn_unet3d": "nnunet3d",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported decoder model: {configured!r}")
    return aliases[key]


def _proposal_default_decoder_model_for_style(style: str, encoder_3d: Any = None) -> str:
    style = _proposal_decoder_style(style)
    if style == "nested_dense":
        return "unetpp3d"
    if style in {"full_scale", "full_encoder_single_decoder", "single_encoder_full_decoder"}:
        return "unet3plus3d"
    encoder = str(encoder_3d or "").strip().lower().replace("-", "_").replace("+", "plus")
    if encoder in {"nnunet", "nnunet3d", "nn_unet", "nn_unet3d"}:
        return "nnunet3d"
    return "unet3d"


def _proposal_2d_decoder_style(encoder_type: str, configured: Any = None) -> str:
    value = str(configured or "").strip()
    if value and value.lower() != "auto":
        return _proposal_decoder_style(value)
    encoder = str(encoder_type or "unet").lower().replace("-", "_").replace("+", "plus")
    if encoder in {"unetpp", "unetplusplus", "unet_plus_plus"}:
        return _proposal_decoder_style(default="nested_dense")
    if encoder in {"unet3plus", "unet_3_plus", "unet_3plus"}:
        return _proposal_decoder_style(default="full_scale")
    return _proposal_decoder_style(default="same_scale")


def _proposal_3d_decoder_style(encoder_type: str, configured: Any = None) -> str:
    value = str(configured or "").strip()
    if value and value.lower() != "auto":
        return _proposal_decoder_style(value)
    encoder = str(encoder_type or "unet3d").lower().replace("-", "_").replace("+", "plus")
    if encoder in {"unetpp3d", "unet3dpp", "unet_plus_plus3d", "unet_plusplus3d", "unetpp_3d", "unet_plus_plus_3d"}:
        return _proposal_decoder_style(default="nested_dense")
    if encoder in {"unet3plus3d", "unet_3_plus3d", "unet_3plus3d", "unet3plus_3d", "unet_3_plus_3d"}:
        return _proposal_decoder_style(default="full_scale")
    return _proposal_decoder_style(default="same_scale")


def _proposal_encoder_3d_type(cfg: Mapping[str, Any]) -> str:
    return str(get_nested(cfg, "model.encoder_3d.type", "unet3d"))


def _proposal_stage2_decoder_model(cfg: Mapping[str, Any]) -> str:
    configured = get_nested(cfg, "model.decoder_3d.model", None)
    configured_style = get_nested(cfg, "model.decoder_3d.style", get_nested(cfg, "model.decoder_3d.type", None))
    if not _proposal_is_auto_value(configured_style):
        default = _proposal_default_decoder_model_for_style(configured_style, _proposal_encoder_3d_type(cfg))
    else:
        default = _proposal_encoder_3d_type(cfg)
    return _proposal_decoder_model(configured, default=default)


def _proposal_stage2_decoder_style(cfg: Mapping[str, Any]) -> str:
    configured = get_nested(cfg, "model.decoder_3d.style", get_nested(cfg, "model.decoder_3d.type", None))
    return _proposal_3d_decoder_style(_proposal_stage2_decoder_model(cfg), configured)


def _proposal_hybrid_decoder_model(cfg: Mapping[str, Any]) -> str:
    configured_style = get_nested(cfg, "model.decoder.style", get_nested(cfg, "model.decoder.type", None))
    default = _proposal_default_decoder_model_for_style(configured_style or "full_scale", _proposal_encoder_3d_type(cfg))
    return _proposal_decoder_model(get_nested(cfg, "model.decoder.model", None), default=default)


def _proposal_hybrid_decoder_style(cfg: Mapping[str, Any]) -> str:
    configured = get_nested(cfg, "model.decoder.style", get_nested(cfg, "model.decoder.type", None))
    return _proposal_3d_decoder_style(_proposal_hybrid_decoder_model(cfg), configured)


def _proposal_decoder_id(decoder_model: str, decoder_style: str) -> str:
    return f"{sanitize_name(decoder_model)}_{sanitize_name(decoder_style)}"


def _proposal_seed_dir(cfg: Mapping[str, Any]) -> str:
    return f"seed_{int(get_nested(cfg, 'seed', 42))}"


def _proposal_encoder_2d_id(cfg: Mapping[str, Any]) -> str:
    return sanitize_name(str(get_nested(cfg, "model.encoder_2d.type", "unet")))


def _proposal_encoder_3d_id(cfg: Mapping[str, Any]) -> str:
    return sanitize_name(str(get_nested(cfg, "model.encoder_3d.type", "unet3d")))


def _proposal_stage1_run_dir(cfg: Mapping[str, Any]) -> Path:
    return (
        _proposal_stage_base_dir(cfg, "stage1_2d_dir", "1_2D_pretrain")
        / _proposal_encoder_2d_id(cfg)
        / _proposal_position_dir(cfg)
        / _proposal_slice_mode(cfg)
        / _proposal_seed_dir(cfg)
    )


def _proposal_stage2_decoder_id(cfg: Mapping[str, Any]) -> str:
    return _proposal_decoder_id(_proposal_stage2_decoder_model(cfg), _proposal_stage2_decoder_style(cfg))


def _proposal_hybrid_decoder_id(cfg: Mapping[str, Any]) -> str:
    return _proposal_decoder_id(_proposal_hybrid_decoder_model(cfg), _proposal_hybrid_decoder_style(cfg))


def _proposal_stage2_combo(cfg: Mapping[str, Any]) -> str:
    return f"{_proposal_encoder_3d_id(cfg)}_{_proposal_stage2_decoder_id(cfg)}"


def _proposal_hybrid_combo(cfg: Mapping[str, Any]) -> str:
    return f"{_proposal_encoder_2d_id(cfg)}_{_proposal_encoder_3d_id(cfg)}_{_proposal_hybrid_decoder_id(cfg)}"


def _proposal_stage2_run_dir(cfg: Mapping[str, Any]) -> Path:
    return _proposal_stage_base_dir(cfg, "stage2_3d_dir", "2_3D_pretrain") / _proposal_stage2_combo(cfg) / _proposal_seed_dir(cfg)


def _proposal_stage2_decoder_pretrain_run_dir(cfg: Mapping[str, Any]) -> Path:
    """Canonical Stage 2 run for the decoder architecture used by Stage 3."""
    decoder_model = _proposal_hybrid_decoder_model(cfg)
    combo = f"{sanitize_name(decoder_model)}_{_proposal_hybrid_decoder_id(cfg)}"
    return _proposal_stage_base_dir(cfg, "stage2_3d_dir", "2_3D_pretrain") / combo / _proposal_seed_dir(cfg)


def _proposal_hybrid_run_dir(cfg: Mapping[str, Any]) -> Path:
    return (
        _proposal_stage_base_dir(cfg, "stage3_hybrid_dir", "3_hybrid")
        / _proposal_seed_dir(cfg)
        / _proposal_hybrid_combo(cfg)
        / _proposal_position_dir(cfg)
        / _proposal_slice_mode(cfg)
    )


def _proposal_stage_model_name(stage: str, use_improve: bool = False, use_experiment: bool = False, use_method: bool = False) -> tuple[str, str, str]:
    if use_method:
        if stage == "train_2d":
            return "proposal_method_2d", "2D", "proposal_method_2d_encoder"
        if stage == "train_3d":
            return "proposal_method_3d", "3D", "proposal_method_3d_encoder"
        return "proposal_method_hybrid", "3D", "proposal_method_hybrid_encoder"
    if use_experiment:
        if stage == "train_2d":
            return "proposal_experiment_2d", "2D", "proposal_experiment_2d_encoder"
        if stage == "train_3d":
            return "proposal_experiment_3d", "3D", "proposal_experiment_3d_encoder"
        return "proposal_experiment_hybrid", "3D", "proposal_experiment_hybrid_encoder"
    if stage == "train_2d":
        return "full_unet3plus_2d", "2D", "unet3plus2d_encoder"
    if stage == "train_3d":
        return "full_unet3d_3_plus", "3D", "unet3d_encoder"
    if use_improve:
        return "hybrid_3d_2d_improve", "3D", "hybrid_unet3plus_2d3d_slice_inject_encoder"
    return "hybrid_3d_2d", "3D", "hybrid_unet3plus_2d3d_encoder"


def _prepare_proposal_stage_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if not _is_proposal_hybrid_config(cfg):
        return cfg

    patched = copy.deepcopy(dict(cfg))
    stage = _proposal_stage(patched)
    mode = _proposal_slice_mode(patched)
    experiment_cfg = dict(patched.get("experiment", {}) if isinstance(patched.get("experiment", {}), Mapping) else {})
    use_position = False
    use_method = _proposal_uses_method(patched)
    use_experiment = _proposal_uses_model_experiment(patched) and not use_method
    experiment_cfg["use_position_encoder"] = use_position
    patched["experiment"] = experiment_cfg
    model_name, model_type, backbone = _proposal_stage_model_name(
        stage,
        use_improve=_proposal_uses_improve_model(patched),
        use_experiment=use_experiment,
        use_method=use_method,
    )

    model_cfg = dict(patched.get("model", {}) if isinstance(patched.get("model", {}), Mapping) else {})
    model_cfg["name"] = model_name
    model_cfg["type"] = model_type
    model_cfg["backbone"] = backbone
    model_cfg["in_channels"] = int(model_cfg.get("in_channels", 1))
    model_cfg["num_classes"] = int(model_cfg.get("num_classes", get_nested(patched, "dataset.num_classes", 2)))

    encoder_2d_channels = list(get_nested(model_cfg, "encoder_2d.channels", [32, 64, 128, 256, 512]))
    encoder_3d_channels = list(get_nested(model_cfg, "encoder_3d.channels", [32, 64, 128, 256, 512]))
    encoder_2d_type = str(get_nested(model_cfg, "encoder_2d.type", "unet3plus_2d" if not use_experiment else "unet"))
    encoder_3d_type = str(get_nested(model_cfg, "encoder_3d.type", "unet3d"))
    decoder_2d_cfg = dict(model_cfg.get("decoder_2d", {}) if isinstance(model_cfg.get("decoder_2d", {}), Mapping) else {})
    decoder_3d_cfg = dict(model_cfg.get("decoder_3d", {}) if isinstance(model_cfg.get("decoder_3d", {}), Mapping) else {})
    decoder_cfg = dict(model_cfg.get("decoder", {}) if isinstance(model_cfg.get("decoder", {}), Mapping) else {})
    decoder_2d_style = _proposal_2d_decoder_style(encoder_2d_type, decoder_2d_cfg.get("style", decoder_2d_cfg.get("type")))
    hybrid_default_model = _proposal_default_decoder_model_for_style(decoder_cfg.get("style", decoder_cfg.get("type", "full_scale")), encoder_3d_type)
    hybrid_decoder_model = _proposal_decoder_model(decoder_cfg.get("model"), default=hybrid_default_model)
    hybrid_decoder_style = _proposal_3d_decoder_style(hybrid_decoder_model, decoder_cfg.get("style", decoder_cfg.get("type")))
    decoder_3d_model_raw = decoder_3d_cfg.get("model")
    decoder_3d_style_raw = decoder_3d_cfg.get("style", decoder_3d_cfg.get("type"))
    if not _proposal_is_auto_value(decoder_3d_style_raw):
        decoder_3d_default_model = _proposal_default_decoder_model_for_style(decoder_3d_style_raw, encoder_3d_type)
    else:
        decoder_3d_default_model = encoder_3d_type
    decoder_3d_model = _proposal_decoder_model(decoder_3d_model_raw, default=decoder_3d_default_model)
    decoder_3d_style = _proposal_3d_decoder_style(decoder_3d_model, decoder_3d_style_raw)
    active_3d_decoder_model = decoder_3d_model if stage == "train_3d" else hybrid_decoder_model
    active_3d_decoder_style = decoder_3d_style if stage == "train_3d" else hybrid_decoder_style
    model_cfg.pop("position_encoder", None)
    slice_selection = dict(model_cfg.get("slice_selection", {}) if isinstance(model_cfg.get("slice_selection", {}), Mapping) else {})
    slice_selection["mode"] = mode
    if _is_group_based_slice_mode(mode):
        group_cfg = _group_based_slice_config(slice_selection)
        slice_selection.pop("group-based", None)
        slice_selection["group_based"] = group_cfg
        slice_selection["num_slices"] = _group_based_selection_count(slice_selection)
    model_cfg["slice_selection"] = slice_selection
    fusion_cfg = dict(model_cfg.get("fusion", {}) if isinstance(model_cfg.get("fusion", {}), Mapping) else {})
    raw_fusion_type = str(fusion_cfg.get("type", fusion_cfg.get("encoder_fusion_mode", "concat"))).lower().replace("-", "_")
    encoder_fusion_mode = "add" if raw_fusion_type in {"add", "sum", "add_conv", "add_1x1", "add_conv1x1"} else "concat"
    fusion_cfg["type"] = encoder_fusion_mode
    model_cfg["fusion"] = fusion_cfg

    args_2d = dict(model_cfg.get("args_2d", {}) if isinstance(model_cfg.get("args_2d", {}), Mapping) else {})
    args_2d.pop("__replace__", None)
    args_2d_normalization = _proposal_default_normalization(args_2d.get("normalization"), encoder_2d_type)
    args_2d.update(
        {
            "encoder_type": encoder_2d_type,
            "encoder_channels": encoder_2d_channels,
            "decoder_style": decoder_2d_style,
            "normalization": args_2d_normalization,
            "deep_supervision": bool(decoder_2d_cfg.get("deep_supervision", args_2d.get("deep_supervision", False))),
            "use_position_encoder": use_position,
        }
    )
    args_3d = dict(model_cfg.get("args_3d", {}) if isinstance(model_cfg.get("args_3d", {}), Mapping) else {})
    args_3d.pop("__replace__", None)
    shared_normalization = args_3d.get("normalization", None)
    active_decoder_cfg = decoder_3d_cfg if stage == "train_3d" else decoder_cfg
    active_3d_deep_supervision = bool(active_decoder_cfg.get("deep_supervision", args_3d.get("deep_supervision", False)))
    args_3d_normalization_2d = _proposal_default_normalization(args_3d.get("normalization_2d", shared_normalization), encoder_2d_type)
    args_3d_normalization_3d = _proposal_default_normalization(args_3d.get("normalization_3d", shared_normalization), encoder_3d_type)
    args_3d_decoder_normalization = _proposal_default_normalization(
        args_3d.get("decoder_normalization", shared_normalization),
        active_3d_decoder_model,
    )
    args_3d_residual = args_3d.get("residual", None)
    if _proposal_is_auto_value(args_3d_residual):
        args_3d_residual = "conv" if _proposal_is_unet3d_name(encoder_3d_type) else "none"
    args_3d_conv_bias = args_3d.get("conv_bias", None)
    if _proposal_is_auto_value(args_3d_conv_bias):
        args_3d_conv_bias = bool(_proposal_is_unet3d_name(encoder_3d_type))
    if stage == "train_3d":
        decoder_3d_cfg["model"] = decoder_3d_model
        decoder_3d_cfg["style"] = decoder_3d_style
        decoder_3d_cfg["deep_supervision"] = active_3d_deep_supervision
        model_cfg["decoder_3d"] = decoder_3d_cfg
    else:
        decoder_3d_cfg["model"] = decoder_3d_model
        decoder_3d_cfg["style"] = decoder_3d_style
        model_cfg["decoder_3d"] = decoder_3d_cfg
        decoder_cfg["model"] = hybrid_decoder_model
        decoder_cfg["style"] = hybrid_decoder_style
        decoder_cfg["deep_supervision"] = active_3d_deep_supervision
        model_cfg["decoder"] = decoder_cfg
    args_3d.update(
        {
            "encoder_type": encoder_3d_type,
            "encoder_2d_type": encoder_2d_type,
            "encoder_3d_type": encoder_3d_type,
            "decoder_model": active_3d_decoder_model,
            "decoder_style": active_3d_decoder_style,
            "encoder_channels": encoder_3d_channels,
            "encoder_2d_channels": encoder_2d_channels,
            "encoder_3d_channels": encoder_3d_channels,
            "deep_supervision": active_3d_deep_supervision,
            "normalization": args_3d_normalization_3d,
            "normalization_2d": args_3d_normalization_2d,
            "normalization_3d": args_3d_normalization_3d,
            "decoder_normalization": args_3d_decoder_normalization,
            "residual": args_3d_residual,
            "conv_bias": bool(args_3d_conv_bias),
            "use_position_encoder": use_position,
            "slice_selection": slice_selection,
            "encoder_fusion_mode": encoder_fusion_mode,
        }
    )
    pretrain_cfg = dict(patched.get("pretrain", {}) if isinstance(patched.get("pretrain", {}), Mapping) else {})
    args_3d.update(
        {
            "freeze_2d_encoder": bool(pretrain_cfg.get("freeze_2d_encoder", False)),
            "freeze_3d_encoder": bool(pretrain_cfg.get("freeze_3d_encoder", False)),
            "freeze_3d_decoder": bool(pretrain_cfg.get("freeze_3d_decoder", False)),
        }
    )
    model_cfg["args_2d"] = args_2d
    model_cfg["args_3d"] = args_3d
    patched["model"] = model_cfg

    slice_cfg = dict(patched.get("slice_2d", {}) if isinstance(patched.get("slice_2d", {}), Mapping) else {})
    slice_cfg["sampling_strategy"] = mode
    slice_cfg["samples_per_volume"] = (
        _group_based_selection_count(slice_selection)
        if _is_group_based_slice_mode(mode)
        else int(slice_selection.get("num_slices", slice_cfg.get("samples_per_volume", 1)))
    )
    slice_cfg["num_slices"] = int(model_cfg.get("in_channels", slice_cfg.get("num_slices", 1)))
    if _is_group_based_slice_mode(mode):
        group_cfg = _group_based_slice_config(slice_selection)
        slice_cfg.pop("group-based", None)
        slice_cfg["group_based"] = group_cfg
    patched["slice_2d"] = slice_cfg

    training_cfg = dict(patched.get("training", {}) if isinstance(patched.get("training", {}), Mapping) else {})
    training_cfg.setdefault("depth_axis", int(slice_cfg.get("axis", 2)))
    if model_type == "3D":
        training_cfg.setdefault("volume_layout", "DHW")
    if stage == "hybrid":
        # Hybrid slice selection must happen on the source depth axis. Resize
        # only in-plane H/W before the model so middle/random/uniform/group-based
        # all choose indices from the raw volume depth.
        training_cfg["preserve_depth"] = True
        if "raw_depth_batch_size_3d" not in training_cfg:
            training_cfg["raw_depth_batch_size_3d"] = 1
        training_cfg["batch_size_3d"] = int(training_cfg["raw_depth_batch_size_3d"])
    elif stage == "train_3d":
        training_cfg.setdefault("preserve_depth", False)
        if bool(training_cfg.get("preserve_depth", False)):
            if "raw_depth_batch_size_3d" not in training_cfg:
                training_cfg["raw_depth_batch_size_3d"] = 1
            training_cfg["batch_size_3d"] = int(training_cfg["raw_depth_batch_size_3d"])
    patched["training"] = training_cfg
    return patched


def _proposal_output_dir(cfg: Mapping[str, Any]) -> Path | None:
    if not _is_proposal_hybrid_config(cfg):
        return None
    stage = _proposal_stage(cfg)
    mode = _proposal_slice_mode(cfg)
    position_dir = _proposal_position_dir(cfg)
    if _proposal_uses_model_experiment(cfg) or _proposal_uses_method(cfg):
        if stage == "train_2d":
            return _proposal_stage1_run_dir(cfg)
        if stage == "train_3d":
            return _proposal_stage2_run_dir(cfg)
        return _proposal_hybrid_run_dir(cfg)
    if stage == "train_2d":
        return _proposal_stage_base_dir(cfg, "stage1_2d_dir", "1_2D_model") / position_dir / mode / _proposal_seed_dir(cfg)
    if stage == "train_3d":
        return _proposal_stage_base_dir(cfg, "stage2_3d_dir", "2_3D_model")
    return _proposal_stage_base_dir(cfg, "stage3_hybrid_dir", "3_hybrid_model") / position_dir / mode


def _proposal_checkpoint_paths(cfg: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    mode = _proposal_slice_mode(cfg)
    position_dir = _proposal_position_dir(cfg)
    if _proposal_uses_model_experiment(cfg) or _proposal_uses_method(cfg):
        default_2d = _proposal_stage1_run_dir(cfg) / "checkpoints" / "best_model.pth"
        default_3d_encoder = _proposal_stage2_run_dir(cfg) / "checkpoints" / "best_model.pth"
        default_3d_decoder = _proposal_stage2_decoder_pretrain_run_dir(cfg) / "checkpoints" / "best_model.pth"
        legacy_experiment_2d = None
        legacy_experiment_3d = None
        legacy_experiment_3d_style = None
    else:
        stage1_root = _proposal_stage_base_dir(cfg, "stage1_2d_dir", "1_2D_model") / position_dir / mode
        default_2d = stage1_root / _proposal_seed_dir(cfg) / "checkpoints" / "best_model.pth"
        unseeded_2d = stage1_root / "checkpoints" / "best_model.pth"
        default_3d_encoder = _proposal_stage_base_dir(cfg, "stage2_3d_dir", "2_3D_model") / "checkpoints" / "best_model.pth"
        default_3d_decoder = default_3d_encoder
        legacy_experiment_2d = None
        legacy_experiment_3d = None
        legacy_experiment_3d_style = None
    raw_2d = get_nested(cfg, "pretrain.stage1_2d_ckpt", str(default_2d))
    raw_3d_encoder = get_nested(cfg, "pretrain.stage2_3d_ckpt", str(default_3d_encoder))
    raw_3d_decoder = get_nested(cfg, "pretrain.stage2_3d_decoder_ckpt", str(default_3d_decoder))
    legacy_no_pos_root = Path("outputs/Proposal_hybrid_3D_2D")
    legacy_2d = legacy_no_pos_root / "1_2D_model" / mode / "checkpoints" / "best_model.pth"
    legacy_3d = legacy_no_pos_root / "2_3D_model" / "checkpoints" / "best_model.pth"
    if _proposal_uses_model_experiment(cfg) or _proposal_uses_method(cfg):
        ckpt_2d_fallbacks = []
        ckpt_3d_fallbacks = []
    else:
        ckpt_2d_fallbacks = [candidate for candidate in (unseeded_2d, legacy_experiment_2d, legacy_2d) if candidate is not None]
        ckpt_3d_fallbacks = [candidate for candidate in (legacy_experiment_3d, legacy_experiment_3d_style, legacy_3d) if candidate is not None]
    ckpt_2d = _resolve_proposal_checkpoint(raw_2d, default_2d, ckpt_2d_fallbacks)
    ckpt_3d_encoder = _resolve_proposal_checkpoint(raw_3d_encoder, default_3d_encoder, ckpt_3d_fallbacks)
    ckpt_3d_decoder = _resolve_proposal_checkpoint(raw_3d_decoder, default_3d_decoder, ckpt_3d_fallbacks)
    return ckpt_2d, ckpt_3d_encoder, ckpt_3d_decoder


def _proposal_pretrain_load_flags(cfg: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    pretrain_cfg = get_nested(cfg, "pretrain", {}) or {}
    return (
        bool(pretrain_cfg.get("load_2d_encoder_from_stage1", True)),
        bool(pretrain_cfg.get("load_3d_encoder_from_stage2", True)),
        bool(pretrain_cfg.get("load_3d_decoder_from_stage2", True)),
    )


def _proposal_shared_output_root(cfg: Mapping[str, Any]) -> Path:
    default_output_name = _proposal_default_output_name(cfg)
    configured = get_nested(cfg, "paths.output_root", None)
    if configured:
        root = Path(str(configured))
        if str(root).replace("\\", "/").rstrip("/") in {"outputs", "./outputs"}:
            return root / default_output_name
        return root
    return Path("outputs") / default_output_name


def _proposal_stage_base_dir(cfg: Mapping[str, Any], key: str, default_name: str) -> Path:
    configured = get_nested(cfg, f"paths.{key}", None)
    if configured:
        path = Path(str(configured))
        if str(path).replace("\\", "/").rstrip("/") == f"outputs/{default_name}":
            return Path("outputs") / _proposal_default_output_name(cfg) / default_name
        return path
    return _proposal_shared_output_root(cfg) / default_name


def _proposal_position_dir(cfg: Mapping[str, Any]) -> str:
    return "no_pos"


def _resolve_proposal_checkpoint(raw_value: Any, default: Path, legacy_candidates: Sequence[Path]) -> Path:
    text = str(raw_value or "").strip()
    candidates = [default, *legacy_candidates]
    if text.lower() in {"", "auto", "none"}:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return default

    requested = Path(text)
    for legacy in legacy_candidates:
        if requested == legacy and not requested.exists():
            return default
    return requested


def _proposal_fold_checkpoint(path: Path, output_dir: Path) -> Path:
    fold_name = output_dir.name
    if not fold_name.startswith("fold_") or path.parent.name != "checkpoints":
        return path
    checkpoint_root = path.parent.parent
    if checkpoint_root.name == fold_name:
        return path
    if checkpoint_root.name.startswith("fold_"):
        checkpoint_root = checkpoint_root.parent
    return checkpoint_root / fold_name / path.parent.name / path.name


def _proposal_checkpoint_names(cfg: Mapping[str, Any]) -> tuple[str, str]:
    if _is_proposal_hybrid_config(cfg):
        return "last_model.pth", "best_model.pth"
    return "last.pth", "best.pth"


def _log_proposal_artifact_paths(output_dir: Path, checkpoint_dir: Path, cfg: Mapping[str, Any], logger) -> None:
    if not _is_proposal_hybrid_config(cfg):
        return
    last_name, best_name = _proposal_checkpoint_names(cfg)
    logger.info("Train log path: %s", display_path(output_dir / "logs" / "train.log"))
    logger.info("Train log text copy path: %s", display_path(output_dir / "logs.txt"))
    logger.info("Train CSV path: %s", display_path(output_dir / "logs" / "train.csv"))
    logger.info("Metrics CSV path: %s", display_path(output_dir / "logs" / "metrics.csv"))
    logger.info("Config snapshot path: %s", display_path(output_dir / "logs" / "config.yaml"))
    logger.info("Summary path: %s", display_path(output_dir / "summary.json"))
    logger.info("Last checkpoint path: %s", display_path(checkpoint_dir / last_name))
    logger.info("Best checkpoint path: %s", display_path(checkpoint_dir / best_name))
    if _proposal_stage(cfg) == "hybrid":
        ckpt_2d, ckpt_3d_encoder, ckpt_3d_decoder = _proposal_checkpoint_paths(cfg)
        ckpt_2d = _proposal_fold_checkpoint(ckpt_2d, output_dir)
        ckpt_3d_encoder = _proposal_fold_checkpoint(ckpt_3d_encoder, output_dir)
        ckpt_3d_decoder = _proposal_fold_checkpoint(ckpt_3d_decoder, output_dir)
        load_2d_encoder, load_3d_encoder, load_3d_decoder = _proposal_pretrain_load_flags(cfg)
        logger.info(
            "Stage 3 pretrained 2D checkpoint: %s",
            display_path(ckpt_2d) if load_2d_encoder else "<disabled>",
        )
        logger.info(
            "Stage 3 pretrained 3D encoder checkpoint: %s",
            display_path(ckpt_3d_encoder) if load_3d_encoder else "<disabled; initialized from model.encoder_3d>",
        )
        logger.info(
            "Stage 3 pretrained 3D decoder checkpoint: %s",
            display_path(ckpt_3d_decoder) if load_3d_decoder else "<disabled; initialized from model.decoder>",
        )
        logger.info("Pretrain loading log path: %s", display_path(output_dir / "logs" / "pretrain_loading.log"))


def _save_config_yaml(path: Path, cfg: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
    except Exception:
        save_json(path.with_suffix(".json"), {"config": cfg})


def _write_logs_txt(log_path: Path | None, output_dir: Path, logger) -> None:
    if log_path is None:
        return
    target = output_dir / "logs.txt"
    try:
        if log_path.resolve() == target.resolve():
            return
    except OSError:
        pass
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    if log_path.exists():
        target.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")


def _write_kfold_logs_txt(base_output_dir: Path, fold_outputs: Sequence[Path], rows: Sequence[Mapping[str, Any]]) -> None:
    splits = sorted({str(row.get("Split", "")) for row in rows if row.get("Split")}, key=_split_order)
    lines = [
        "K-fold run summary",
        f"Output: {display_path(base_output_dir)}",
        f"Metrics: {display_path(base_output_dir / 'metrics_kfold.csv')}",
        f"Summary: {display_path(base_output_dir / 'metrics_kfold_summary.csv')}",
        f"Splits: {', '.join(splits) if splits else 'N/A'}",
        f"Folds: {len(fold_outputs)}",
        f"Split file: {display_path(base_output_dir / 'kfold_splits.json')}",
        "",
        "Fold outputs:",
    ]
    lines.extend(f"- {display_path(path)}" for path in fold_outputs)
    (base_output_dir / "logs.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_proposal_pretrain_if_needed(model: torch.nn.Module, cfg: Mapping[str, Any], output_dir: Path, logger) -> dict[str, Any] | None:
    if not _is_proposal_hybrid_config(cfg) or _proposal_stage(cfg) != "hybrid":
        return None
    base_model = unwrap_model(model)
    if not hasattr(base_model, "load_pretrained_components"):
        raise TypeError("Hybrid stage model does not implement load_pretrained_components().")

    ckpt_2d, ckpt_3d_encoder, ckpt_3d_decoder = _proposal_checkpoint_paths(cfg)
    ckpt_2d = _proposal_fold_checkpoint(ckpt_2d, output_dir)
    ckpt_3d_encoder = _proposal_fold_checkpoint(ckpt_3d_encoder, output_dir)
    ckpt_3d_decoder = _proposal_fold_checkpoint(ckpt_3d_decoder, output_dir)
    pretrain_cfg = dict(get_nested(cfg, "pretrain", {}) or {})
    missing = []
    if bool(pretrain_cfg.get("load_2d_encoder_from_stage1", True)) and not ckpt_2d.exists():
        missing.append(f"Missing Stage 1 2D checkpoint. Please run experiment.stage=train_2d first: {ckpt_2d}")
    if bool(pretrain_cfg.get("load_3d_encoder_from_stage2", True)) and not ckpt_3d_encoder.exists():
        missing.append(f"Missing Stage 2 3D encoder checkpoint. Please run experiment.stage=train_3d first: {ckpt_3d_encoder}")
    if bool(pretrain_cfg.get("load_3d_decoder_from_stage2", True)) and not ckpt_3d_decoder.exists():
        missing.append(
            "Missing Stage 2 3D decoder checkpoint matching model.decoder. "
            f"Train that Stage 2 decoder first: {ckpt_3d_decoder}"
        )
    if missing:
        raise FileNotFoundError("\n".join(missing))

    report = base_model.load_pretrained_components(
        ckpt_2d=ckpt_2d,
        ckpt_3d=ckpt_3d_encoder,
        ckpt_3d_decoder=ckpt_3d_decoder,
        load_2d_encoder=bool(pretrain_cfg.get("load_2d_encoder_from_stage1", True)),
        load_3d_encoder=bool(pretrain_cfg.get("load_3d_encoder_from_stage2", True)),
        load_3d_decoder=bool(pretrain_cfg.get("load_3d_decoder_from_stage2", True)),
        strict=bool(pretrain_cfg.get("strict", False)),
        freeze_2d_encoder=bool(pretrain_cfg.get("freeze_2d_encoder", False)),
        freeze_3d_encoder=bool(pretrain_cfg.get("freeze_3d_encoder", False)),
        freeze_3d_decoder=bool(pretrain_cfg.get("freeze_3d_decoder", False)),
        log_path=output_dir / "logs" / "pretrain_loading.log",
    )
    logger.info("Stage 3 loaded 2D encoder: %s (%d keys)", report.get("loaded_2d_encoder"), report.get("num_loaded_2d_keys", 0))
    if "2d_position_encoder" in report.get("components", {}):
        logger.info(
            "Stage 3 loaded 2D position encoder: %s (%d keys)",
            report.get("loaded_2d_position_encoder"),
            report.get("num_loaded_2d_position_keys", 0),
        )
    logger.info(
        "Stage 3 loaded 3D encoder/decoder: %s/%s (%d/%d keys)",
        report.get("loaded_3d_encoder"),
        report.get("loaded_3d_decoder"),
        report.get("num_loaded_3d_encoder_keys", 0),
        report.get("num_loaded_3d_decoder_keys", 0),
    )
    return report


def _summary_metrics_from_rows(rows: list[Dict[str, Any]]) -> dict[str, Any]:
    preferred = None
    for row in rows:
        if str(row.get("Split", "")).lower() == "test":
            preferred = row
            break
    if preferred is None:
        preferred = rows[-1] if rows else {}
    return {
        "Dice": preferred.get("Dice", 0.0),
        "IoU": preferred.get("IoU", 0.0),
        "Accuracy": preferred.get("Accuracy", 0.0),
        "Recall": preferred.get("Recall", 0.0),
        "Precision": preferred.get("Precision", 0.0),
        "HD95": preferred.get("HD95", 0.0),
        "FPS": preferred.get("FPS", 0.0),
        "Params": preferred.get("Params", 0.0),
        "FLOPs": preferred.get("FLOPs", 0.0),
    }


def _write_proposal_summary(
    output_dir: Path,
    cfg: Mapping[str, Any],
    model_result,
    best_epoch: int,
    best_metric: float,
    best_checkpoint: Path,
    final_rows: list[Dict[str, Any]],
    pretrain_report: dict[str, Any] | None,
) -> None:
    if not _is_proposal_hybrid_config(cfg):
        return
    payload: dict[str, Any] = {
        "stage": _proposal_stage(cfg),
        "model_name": model_result.name,
        "use_position_encoder": bool(get_nested(cfg, "experiment.use_position_encoder", False)),
        "slice_selection_mode": _proposal_slice_mode(cfg),
        "output_dir": display_path(output_dir),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "best_checkpoint": display_path(best_checkpoint),
        "metrics": _summary_metrics_from_rows(final_rows),
    }
    if pretrain_report is not None:
        payload["pretrain"] = {
            "stage1_2d_ckpt": pretrain_report.get("stage1_2d_ckpt"),
            "stage2_3d_ckpt": pretrain_report.get("stage2_3d_ckpt"),
            "stage2_3d_encoder_ckpt": pretrain_report.get("stage2_3d_encoder_ckpt", pretrain_report.get("stage2_3d_ckpt")),
            "stage2_3d_decoder_ckpt": pretrain_report.get("stage2_3d_decoder_ckpt", pretrain_report.get("stage2_3d_ckpt")),
            "loaded_2d_encoder": pretrain_report.get("loaded_2d_encoder", False),
            "loaded_2d_position_encoder": pretrain_report.get("loaded_2d_position_encoder", False),
            "loaded_3d_encoder": pretrain_report.get("loaded_3d_encoder", False),
            "loaded_3d_decoder": pretrain_report.get("loaded_3d_decoder", False),
            "num_loaded_2d_keys": pretrain_report.get("num_loaded_2d_keys", 0),
            "num_loaded_2d_position_keys": pretrain_report.get("num_loaded_2d_position_keys", 0),
            "num_loaded_3d_encoder_keys": pretrain_report.get("num_loaded_3d_encoder_keys", 0),
            "num_loaded_3d_decoder_keys": pretrain_report.get("num_loaded_3d_decoder_keys", 0),
            "num_skipped_keys": pretrain_report.get("num_skipped_keys", 0),
        }
    save_json(output_dir / "summary.json", payload)


def _auto_train_missing_proposal_pretrains(cfg: Mapping[str, Any], config_path: Path | None = None) -> Mapping[str, Any]:
    if not _is_proposal_hybrid_config(cfg) or _proposal_stage(cfg) != "hybrid":
        return cfg
    if not bool(get_nested(cfg, "pretrain.auto_train_missing_pretrain", False)):
        return cfg

    ckpt_2d, ckpt_3d_encoder, ckpt_3d_decoder = _proposal_checkpoint_paths(cfg)
    missing_stage_cfgs: list[Mapping[str, Any]] = []
    if bool(get_nested(cfg, "pretrain.load_2d_encoder_from_stage1", True)) and not ckpt_2d.exists():
        stage_cfg = copy.deepcopy(dict(cfg))
        experiment_cfg = dict(stage_cfg.get("experiment", {}) if isinstance(stage_cfg.get("experiment", {}), Mapping) else {})
        experiment_cfg["stage"] = "train_2d"
        stage_cfg["experiment"] = experiment_cfg
        missing_stage_cfgs.append(stage_cfg)
    needs_encoder_source_run = (
        bool(get_nested(cfg, "pretrain.load_3d_encoder_from_stage2", True)) and not ckpt_3d_encoder.exists()
    ) or (
        bool(get_nested(cfg, "pretrain.load_3d_decoder_from_stage2", True))
        and ckpt_3d_decoder == ckpt_3d_encoder
        and not ckpt_3d_decoder.exists()
    )
    if needs_encoder_source_run:
        stage_cfg = copy.deepcopy(dict(cfg))
        experiment_cfg = dict(stage_cfg.get("experiment", {}) if isinstance(stage_cfg.get("experiment", {}), Mapping) else {})
        experiment_cfg["stage"] = "train_3d"
        stage_cfg["experiment"] = experiment_cfg
        missing_stage_cfgs.append(stage_cfg)
    if (
        bool(get_nested(cfg, "pretrain.load_3d_decoder_from_stage2", True))
        and not ckpt_3d_decoder.exists()
        and ckpt_3d_decoder != ckpt_3d_encoder
    ):
        stage_cfg = copy.deepcopy(dict(cfg))
        experiment_cfg = dict(stage_cfg.get("experiment", {}) if isinstance(stage_cfg.get("experiment", {}), Mapping) else {})
        experiment_cfg["stage"] = "train_3d"
        stage_cfg["experiment"] = experiment_cfg
        model_cfg = dict(stage_cfg.get("model", {}) if isinstance(stage_cfg.get("model", {}), Mapping) else {})
        hybrid_decoder_cfg = model_cfg.get("decoder", {})
        encoder_3d_cfg = dict(model_cfg.get("encoder_3d", {}) if isinstance(model_cfg.get("encoder_3d", {}), Mapping) else {})
        encoder_3d_cfg["type"] = _proposal_hybrid_decoder_model(cfg)
        model_cfg["encoder_3d"] = encoder_3d_cfg
        model_cfg["decoder_3d"] = copy.deepcopy(dict(hybrid_decoder_cfg)) if isinstance(hybrid_decoder_cfg, Mapping) else {}
        stage_cfg["model"] = model_cfg
        missing_stage_cfgs.append(stage_cfg)

    for stage_cfg in missing_stage_cfgs:
        pretrain_cfg = dict(stage_cfg.get("pretrain", {}) if isinstance(stage_cfg.get("pretrain", {}), Mapping) else {})
        pretrain_cfg["auto_train_missing_pretrain"] = False
        stage_cfg["pretrain"] = pretrain_cfg
        _run_single(_prepare_proposal_stage_config(stage_cfg), config_path=config_path)

    still_missing = []
    if bool(get_nested(cfg, "pretrain.load_2d_encoder_from_stage1", True)) and not ckpt_2d.exists():
        still_missing.append(str(ckpt_2d))
    if bool(get_nested(cfg, "pretrain.load_3d_encoder_from_stage2", True)) and not ckpt_3d_encoder.exists():
        still_missing.append(str(ckpt_3d_encoder))
    if bool(get_nested(cfg, "pretrain.load_3d_decoder_from_stage2", True)) and not ckpt_3d_decoder.exists():
        still_missing.append(str(ckpt_3d_decoder))
    if still_missing:
        raise FileNotFoundError("auto_train_missing_pretrain ran but these checkpoints are still missing: " + ", ".join(still_missing))
    return cfg


def run(cfg: Mapping[str, Any], config_path: Path | None = None) -> Path:
    cfg = _prepare_proposal_stage_config(cfg)
    cfg = _auto_train_missing_proposal_pretrains(cfg, config_path=config_path)
    if bool(get_nested(cfg, "k_fold.enabled", False)):
        return _run_kfold(cfg, config_path=config_path)
    return _run_single(cfg, config_path=config_path)


def visualize_from_checkpoint(
    cfg: Mapping[str, Any],
    checkpoint_path: Path,
    output_dir: Path | None = None,
    config_path: Path | None = None,
    fold_index: int | None = None,
) -> Path:
    cfg = _prepare_proposal_stage_config(cfg)
    seed = int(get_nested(cfg, "seed", 42))
    deterministic = bool(get_nested(cfg, "training.deterministic", True))
    set_seed(seed, deterministic=deterministic)

    records_metadata: Mapping[str, Any] | None = None
    if bool(get_nested(cfg, "k_fold.enabled", False)) and fold_index is not None:
        fold_records, records_metadata = build_kfold_record_splits(cfg)
        if fold_index < 0 or fold_index >= len(fold_records):
            raise ValueError(f"--fold must be between 1 and {len(fold_records)}, got {fold_index + 1}.")
        eval_datasets, records = build_datasets_from_records(cfg, fold_records[fold_index], augment_train=False)
    else:
        eval_datasets, records = build_datasets(cfg, augment_train=False)

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _strip_module_prefix(_checkpoint_state_dict(checkpoint))
    cfg = _cfg_with_checkpoint_shape_hints(cfg, state_dict)

    model_type = str(get_nested(cfg, "model.type", "2D")).upper()
    eval_2d_as_volume = model_type == "2D" and bool(get_nested(cfg, "evaluation.evaluate_2d_as_volume", True))
    visual_datasets = build_2d_volume_eval_datasets(cfg, records) if eval_2d_as_volume else eval_datasets
    model_result = build_model(cfg, dataset_in_channels=eval_datasets["train"].in_channels)

    if output_dir is None:
        output_dir = _default_visual_output_dir(cfg, model_result, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "visualize_logs.txt")

    device = torch.device("cuda:0" if bool(get_nested(cfg, "gpu.use_cuda", True)) and torch.cuda.is_available() else "cpu")
    load_model_state(model_result.model, state_dict)
    model = ensure_model_on_device(model_result.model, device)
    model.eval()

    logger.info("Config: %s", display_path(config_path) if config_path else "<in-memory>")
    logger.info("Checkpoint: %s", display_path(checkpoint_path))
    logger.info("Model: %s", model_result.name)
    logger.info("Backbone: %s", model_result.backbone)
    logger.info("GPU: %s", describe_device(device, str(get_nested(cfg, "gpu.ids", "0"))))
    logger.info("Visualization slice_position: %s", get_nested(cfg, "visualization.slice_position", "label_foreground"))
    logger.info("Visualization output: %s", display_path(output_dir))
    if records_metadata is not None:
        logger.info("K-fold visualization: fold %d/%d", int(fold_index or 0) + 1, int(records_metadata.get("num_folds", 0)))

    save_json(
        output_dir / "visualize_hyperparameter.json",
        {
            "config": cfg,
            "config_path": display_path(config_path) if config_path else None,
            "checkpoint": display_path(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None,
            "resolved": {
                "output_dir": display_path(output_dir),
                "model": model_result.name,
                "backbone": model_result.backbone,
                "architecture_config": getattr(model_result.model, "architecture_config", {}),
                "in_channels": model_result.in_channels,
                "num_classes": model_result.num_classes,
                "evaluation_mode": _evaluation_mode(model_type, eval_2d_as_volume),
                "evaluate_2d_as_volume": eval_2d_as_volume,
                "slice_selection": _slice_selection_metadata(cfg) if model_type == "2D" else None,
                "visualization_slice_position": get_nested(cfg, "visualization.slice_position", "label_foreground"),
                "fold": None if fold_index is None else fold_index + 1,
                "kfold": records_metadata,
            },
        },
    )

    save_prediction_visuals(
        model=model,
        datasets=visual_datasets,
        output_root=output_dir,
        cfg=cfg,
        device=device,
        num_classes=model_result.num_classes,
    )
    logger.info("Done. Visualization saved to %s", display_path(output_dir))
    return output_dir


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint)!r}.")
    for key in ("model_state", "state_dict", "model_state_dict", "network_weights"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    keys = [str(key) for key in state_dict.keys()]
    if keys and all(key.startswith("module.") for key in keys):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def _cfg_with_checkpoint_shape_hints(cfg: Mapping[str, Any], state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    model_name = str(get_nested(cfg, "model.name", "")).lower()
    if model_name not in {
        "unet_3_plus",
        "unet3plus",
        "unet_3plus",
        "unet_3_plus_cgm",
        "unet3plus_cgm",
        "unet3plus_hybrid_cgm",
        "unet_3_plus_hybrid_cgm",
    }:
        return cfg

    for key in (
        "model.outconv1.weight",
        "model.outconv2.weight",
        "model.outconv3.weight",
        "model.outconv4.weight",
        "model.outconv5.weight",
    ):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            output_channels = int(tensor.shape[0])
            model_cfg = dict(cfg.get("model", {}) if isinstance(cfg.get("model", {}), Mapping) else {})
            args_cfg = dict(model_cfg.get("args", {}) if isinstance(model_cfg.get("args", {}), Mapping) else {})
            args_2d_cfg = dict(model_cfg.get("args_2d", {}) if isinstance(model_cfg.get("args_2d", {}), Mapping) else {})
            if int(args_cfg.get("internal_num_classes", args_2d_cfg.get("internal_num_classes", -1))) == output_channels:
                return cfg
            args_cfg["internal_num_classes"] = output_channels
            model_cfg["args"] = args_cfg
            patched = dict(cfg)
            patched["model"] = model_cfg
            return patched
    return cfg


def _default_visual_output_dir(cfg: Mapping[str, Any], model_result, checkpoint_path: Path) -> Path:
    output_root = Path(get_nested(cfg, "project.output_root", "outputs"))
    model_type = str(get_nested(cfg, "model.type", "2D")).upper()
    checkpoint_parent = checkpoint_path.parent.parent.name if checkpoint_path.parent.name == "checkpoint" else checkpoint_path.parent.name
    checkpoint_tag = sanitize_name(f"{checkpoint_parent}_{checkpoint_path.stem}")
    slice_position = sanitize_name(get_nested(cfg, "visualization.slice_position", "label_foreground"))
    output_parts = [output_root, "visualize_from_weight", sanitize_name(model_result.name)]
    if model_type == "2D":
        output_parts.append(_slice_selection_name(cfg))
    return Path(*output_parts) / f"{sanitize_name(model_result.backbone)}_{checkpoint_tag}_slice_{slice_position}"


def _parse_id_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        parsed = [str(item).strip() for item in value if str(item).strip()]
    else:
        parsed = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return parsed


def _kfold_parallel_gpu_ids(cfg: Mapping[str, Any]) -> list[str]:
    configured = get_nested(cfg, "k_fold.parallel_gpu_ids", "auto")
    if str(configured).strip().lower() in {"", "auto", "none"}:
        configured = get_nested(cfg, "gpu.ids", "0")
    return _parse_id_list(configured)


def _kfold_parallel_workers(cfg: Mapping[str, Any], num_folds: int) -> int:
    if not bool(get_nested(cfg, "k_fold.run_parallel", False)):
        return 1
    raw = get_nested(cfg, "k_fold.max_parallel_folds", "auto")
    gpu_ids = _kfold_parallel_gpu_ids(cfg) if bool(get_nested(cfg, "gpu.use_cuda", True)) else []
    if str(raw).strip().lower() in {"", "auto", "none"}:
        workers = len(gpu_ids) if gpu_ids else min(num_folds, max(1, (os.cpu_count() or 1) // 2))
    else:
        workers = int(raw)
    if bool(get_nested(cfg, "k_fold.parallel_single_gpu_per_fold", True)) and gpu_ids:
        workers = min(workers, len(gpu_ids))
    return max(1, min(int(workers), int(num_folds)))


def _kfold_worker_cfg(cfg: Mapping[str, Any], fold_index: int) -> dict[str, Any]:
    worker_cfg = copy.deepcopy(dict(cfg))
    if (
        bool(get_nested(worker_cfg, "k_fold.run_parallel", False))
        and bool(get_nested(worker_cfg, "gpu.use_cuda", True))
        and bool(get_nested(worker_cfg, "k_fold.parallel_single_gpu_per_fold", True))
    ):
        gpu_ids = _kfold_parallel_gpu_ids(worker_cfg)
        if gpu_ids:
            gpu_id = gpu_ids[fold_index % len(gpu_ids)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            gpu_cfg = dict(worker_cfg.get("gpu", {}) if isinstance(worker_cfg.get("gpu", {}), Mapping) else {})
            gpu_cfg["ids"] = str(gpu_id)
            gpu_cfg["multi_gpu"] = False
            worker_cfg["gpu"] = gpu_cfg
    return worker_cfg


def _run_kfold_worker(payload: tuple[Mapping[str, Any], Mapping[str, Any], int, int, Mapping[str, Any], str | None]) -> str:
    cfg, records, fold_index, num_folds, metadata, config_path_text = payload
    worker_cfg = _kfold_worker_cfg(cfg, fold_index)
    output_dir = _run_single(
        worker_cfg,
        config_path=Path(config_path_text) if config_path_text else None,
        records_override=records,
        fold_index=fold_index,
        num_folds=num_folds,
        kfold_metadata=metadata,
    )
    return str(output_dir)


def _run_kfold(cfg: Mapping[str, Any], config_path: Path | None = None) -> Path:
    fold_records, metadata = build_kfold_record_splits(cfg)
    fold_outputs: list[Path] = []
    kfold_rows: list[Dict[str, Any]] = []
    base_output_dir: Path | None = None
    parallel_workers = _kfold_parallel_workers(cfg, len(fold_records))

    if parallel_workers > 1:
        ordered_outputs: list[Path | None] = [None] * len(fold_records)
        payloads = [
            (cfg, records, fold_index, len(fold_records), metadata, str(config_path) if config_path else None)
            for fold_index, records in enumerate(fold_records)
        ]
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=parallel_workers, mp_context=context) as executor:
            futures = {executor.submit(_run_kfold_worker, payload): payload[2] for payload in payloads}
            for future in as_completed(futures):
                fold_index = futures[future]
                ordered_outputs[fold_index] = Path(future.result())
        fold_outputs = [path for path in ordered_outputs if path is not None]
        if fold_outputs:
            base_output_dir = fold_outputs[0].parent
        for fold_index, output_dir in enumerate(fold_outputs):
            kfold_rows.extend(_read_fold_metrics(output_dir / "metrics.csv", fold_index=fold_index))
    else:
        for fold_index, records in enumerate(fold_records):
            output_dir = _run_single(
                _kfold_worker_cfg(cfg, fold_index),
                config_path=config_path,
                records_override=records,
                fold_index=fold_index,
                num_folds=len(fold_records),
                kfold_metadata=metadata,
            )
            fold_outputs.append(output_dir)
            base_output_dir = output_dir.parent
            kfold_rows.extend(_read_fold_metrics(output_dir / "metrics.csv", fold_index=fold_index))

    if base_output_dir is None:
        raise RuntimeError("No k-fold outputs were created.")

    _write_kfold_metrics_csv(base_output_dir / "metrics_kfold.csv", kfold_rows)
    _write_kfold_summary_csv(base_output_dir / "metrics_kfold_summary.csv", kfold_rows)
    save_json(
        base_output_dir / "kfold_splits.json",
        {
            "metadata": metadata,
            "folds": [
                {
                    "fold": index + 1,
                    "train_count": len(records["train"]),
                    "val_count": len(records["val"]),
                    "test_count": len(records["test"]),
                    "train": [record.case_id for record in records["train"]],
                    "val": [record.case_id for record in records["val"]],
                    "test": [record.case_id for record in records["test"]],
                }
                for index, records in enumerate(fold_records)
            ],
            "outputs": [display_path(path) for path in fold_outputs],
        },
    )
    _write_kfold_logs_txt(base_output_dir, fold_outputs, kfold_rows)
    return base_output_dir


def _run_single(
    cfg: Mapping[str, Any],
    config_path: Path | None = None,
    records_override: Mapping[str, Any] | None = None,
    fold_index: int | None = None,
    num_folds: int | None = None,
    kfold_metadata: Mapping[str, Any] | None = None,
) -> Path:
    cfg = _prepare_proposal_stage_config(cfg)
    seed = int(get_nested(cfg, "seed", 42))
    deterministic = bool(get_nested(cfg, "training.deterministic", True))
    set_seed(seed, deterministic=deterministic)

    if records_override is None:
        train_datasets, records = build_datasets(cfg, augment_train=True)
        eval_datasets, _ = build_datasets(cfg, augment_train=False)
    else:
        train_datasets, records = build_datasets_from_records(cfg, records_override, augment_train=True)
        eval_datasets, _ = build_datasets_from_records(cfg, records_override, augment_train=False)
    model_type = str(get_nested(cfg, "model.type", "2D")).upper()
    eval_2d_as_volume = model_type == "2D" and bool(get_nested(cfg, "evaluation.evaluate_2d_as_volume", True))
    evaluation_mode = _evaluation_mode(model_type, eval_2d_as_volume)
    volume_eval_datasets = build_2d_volume_eval_datasets(cfg, records) if eval_2d_as_volume else {}
    model_result = build_model(cfg, dataset_in_channels=train_datasets["train"].in_channels)

    epochs = int(get_nested(cfg, "training.epochs", 1))
    proposal_base_output_dir = _proposal_output_dir(cfg)
    if proposal_base_output_dir is None:
        output_root = Path(get_nested(cfg, "project.output_root", "outputs"))
        output_parts = [output_root, sanitize_name(model_result.name)]
        if model_type == "2D":
            output_parts.append(_slice_selection_name(cfg))
        output_suffix = sanitize_name(get_nested(cfg, "project.output_suffix", ""))
        output_name = sanitize_name(get_nested(cfg, "project.output_name", ""))
        if output_name:
            run_name = output_name
        else:
            run_name = f"{sanitize_name(model_result.backbone)}_epoch{epochs}"
            if output_suffix:
                run_name = f"{run_name}_{output_suffix}"
        base_output_dir = Path(*output_parts) / run_name
        checkpoint_dir_name = "checkpoint"
        log_path = None
    else:
        base_output_dir = proposal_base_output_dir
        checkpoint_dir_name = "checkpoints"
        log_path = None
    output_dir = base_output_dir / f"fold_{fold_index + 1:02d}" if fold_index is not None else base_output_dir
    checkpoint_dir = output_dir / checkpoint_dir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    proposal_pretrain_links = None
    if _is_proposal_hybrid_config(cfg) and _proposal_stage(cfg) == "hybrid":
        ckpt_2d_link, ckpt_3d_encoder_link, ckpt_3d_decoder_link = _proposal_checkpoint_paths(cfg)
        ckpt_2d_link = _proposal_fold_checkpoint(ckpt_2d_link, output_dir)
        ckpt_3d_encoder_link = _proposal_fold_checkpoint(ckpt_3d_encoder_link, output_dir)
        ckpt_3d_decoder_link = _proposal_fold_checkpoint(ckpt_3d_decoder_link, output_dir)
        load_2d_encoder, load_3d_encoder, load_3d_decoder = _proposal_pretrain_load_flags(cfg)
        proposal_pretrain_links = {
            "stage1_2d_ckpt": display_path(ckpt_2d_link) if load_2d_encoder else None,
            "stage2_3d_ckpt": display_path(ckpt_3d_encoder_link) if load_3d_encoder else None,
            "stage2_3d_encoder_ckpt": display_path(ckpt_3d_encoder_link) if load_3d_encoder else None,
            "stage2_3d_decoder_ckpt": display_path(ckpt_3d_decoder_link) if load_3d_decoder else None,
            "stage1_2d_expected_output": display_path(ckpt_2d_link.parent.parent) if load_2d_encoder else None,
            "stage2_3d_expected_output": display_path(ckpt_3d_encoder_link.parent.parent) if load_3d_encoder else None,
            "stage2_3d_encoder_expected_output": display_path(ckpt_3d_encoder_link.parent.parent) if load_3d_encoder else None,
            "stage2_3d_decoder_expected_output": display_path(ckpt_3d_decoder_link.parent.parent) if load_3d_decoder else None,
            "load_2d_encoder_from_stage1": load_2d_encoder,
            "load_3d_encoder_from_stage2": load_3d_encoder,
            "load_3d_decoder_from_stage2": load_3d_decoder,
        }

    if _is_proposal_hybrid_config(cfg):
        log_path = output_dir / "logs" / "train.log"
        _save_config_yaml(output_dir / "logs" / "config.yaml", cfg)
    logger = setup_logger(log_path or (output_dir / "logs.txt"))
    save_json(
        output_dir / "hyperparameter.json",
        {
            "config": cfg,
            "config_path": display_path(config_path) if config_path else None,
            "resolved": {
                "output_dir": display_path(output_dir),
                "model": model_result.name,
                "backbone": model_result.backbone,
                "architecture_config": getattr(model_result.model, "architecture_config", {}),
                "in_channels": model_result.in_channels,
                "num_classes": model_result.num_classes,
                "evaluation_mode": evaluation_mode,
                "evaluate_2d_as_volume": eval_2d_as_volume,
                "slice_selection": _slice_selection_metadata(cfg) if model_type == "2D" else None,
                "fold": None if fold_index is None else fold_index + 1,
                "num_folds": num_folds,
                "kfold": kfold_metadata,
                "pretrain_links": proposal_pretrain_links,
            },
        },
    )
    if proposal_pretrain_links is not None:
        save_json(output_dir / "logs" / "pretrain_links.json", proposal_pretrain_links)

    if _is_proposal_hybrid_config(cfg):
        logger.info("Stage: %s", _proposal_stage(cfg))
        logger.info("Output dir: %s", display_path(output_dir))
        logger.info("Position mode: no_pos")
        logger.info("Slice selection mode: %s", _proposal_slice_mode(cfg))
        _log_proposal_artifact_paths(output_dir, checkpoint_dir, cfg, logger)
    pretrain_report = _load_proposal_pretrain_if_needed(model_result.model, cfg, output_dir, logger)

    device = torch.device("cuda:0" if bool(get_nested(cfg, "gpu.use_cuda", True)) and torch.cuda.is_available() else "cpu")
    model = ensure_model_on_device(model_result.model, device)
    if device.type == "cuda" and bool(get_nested(cfg, "gpu.multi_gpu", False)) and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        model = ensure_model_on_device(model, device)

    logger.info("Config: %s", display_path(config_path) if config_path else "<in-memory>")
    if epochs < 5:
        logger.warning("training.epochs=%d is very low; this is usually only enough for a smoke test.", epochs)
    if fold_index is not None:
        logger.info("K-fold: fold %d/%d, source=%s, val ~= %.1f%%", fold_index + 1, num_folds, kfold_metadata.get("source_list") if kfold_metadata else "N/A", 100 / max(1, int(num_folds or 1)))
    logger.info("Model: %s", model_result.name)
    logger.info("Backbone: %s", model_result.backbone)
    if model_type == "2D":
        logger.info("Slice selection: %s", _slice_selection_name(cfg))
    logger.info("GPU: %s", describe_device(device, str(get_nested(cfg, "gpu.ids", "0"))))
    logger.info("Evaluation mode: %s", evaluation_mode)
    if eval_2d_as_volume:
        logger.info("2D evaluation: full-volume metrics from all slices")
    elif model_type == "2D":
        logger.info("2D evaluation: slice metrics from the configured slice sampler")
    for split in ("train", "val", "test"):
        logger.info(
            "%s samples: %d effective samples from %d volumes",
            split,
            len(train_datasets[split]),
            len(records[split]),
        )

    total_params, trainable_params = count_parameters(model)
    logger.info("Total parameters: %d (%s)", total_params, format_large_number(total_params))
    logger.info("Trainable parameters: %d (%s)", trainable_params, format_large_number(trainable_params))

    flops = None
    flops_error = None
    if bool(get_nested(cfg, "evaluation.compute_flops", True)):
        input_shape = _profile_input_shape(cfg, model_result.in_channels)
        flops, flops_error = estimate_flops(model, input_shape=input_shape, device=device)
        if flops is None:
            message = f"FLOPs profiling failed: {flops_error}"
            if bool(get_nested(cfg, "evaluation.fail_on_flops_error", True)):
                raise RuntimeError(message)
            logger.info("FLOPs: unavailable (%s)", flops_error)
        else:
            logger.info("FLOPs: %d (%s)", flops, format_large_number(flops))
        model = ensure_model_on_device(model, device)

    train_loader = _make_train_loader(train_datasets["train"], cfg, seed, device)
    eval_loaders = {split: _make_eval_loader(dataset, cfg, seed, device) for split, dataset in eval_datasets.items()}

    train_cfg = dict(get_nested(cfg, "training", {}) or {})
    optimizer = build_optimizer(model.parameters(), train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    criterion = SegmentationCriterion(cfg, num_classes=model_result.num_classes).to(device)
    amp_enabled = bool(get_nested(cfg, "training.amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    show_progress = bool(get_nested(cfg, "training.show_progress", True))

    history: list[Dict[str, Any]] = []
    best_metric = -math.inf
    best_epoch = 0
    patience_counter = 0
    last_checkpoint_name, best_checkpoint_name = _proposal_checkpoint_names(cfg)
    early_cfg = dict(get_nested(cfg, "early_stopping", {}) or {})
    early_enabled = bool(early_cfg.get("enabled", True))
    patience = int(early_cfg.get("patience", 20))
    min_delta = float(early_cfg.get("min_delta", 0.0))

    logger.info("Start training for %d epoch(s)", epochs)
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        _set_moe_router_epoch(model, epoch - 1)
        _reset_moe_router_stats(model)
        train_summary = _train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            num_classes=model_result.num_classes,
            amp_enabled=amp_enabled,
            grad_clip_norm=float(get_nested(cfg, "training.grad_clip_norm", 0.0)),
            moe_balance_weight=_moe_balance_weight(cfg),
            moe_entropy_weight=_moe_entropy_weight(cfg),
            progress_desc=f"Epoch {epoch:03d}/{epochs:03d} train",
            show_progress=show_progress,
        )
        _save_moe_router_heatmap(output_dir, model, epoch, cfg)
        if eval_2d_as_volume:
            val_summary = _evaluate_2d_volume(
                model=model,
                dataset=volume_eval_datasets["val"],
                criterion=criterion,
                device=device,
                num_classes=model_result.num_classes,
                compute_surface=False,
                slice_batch_size=int(get_nested(cfg, "evaluation.slice_batch_size", get_nested(cfg, "evaluation.batch_size", 1))),
                progress_desc=f"Epoch {epoch:03d}/{epochs:03d} val",
                show_progress=show_progress,
            )
        else:
            val_summary = _evaluate(
                model=model,
                loader=eval_loaders["val"],
                criterion=criterion,
                device=device,
                num_classes=model_result.num_classes,
                compute_surface=False,
                progress_desc=f"Epoch {epoch:03d}/{epochs:03d} val",
                show_progress=show_progress,
            )

        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_summary["Dice"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_summary["Loss"],
            "train_dice": train_summary["Dice"],
            "train_iou": train_summary["IoU"],
            "val_loss": val_summary["Loss"],
            "val_dice": val_summary["Dice"],
            "val_iou": val_summary["IoU"],
            "epoch_time_sec": epoch_seconds,
        }
        row.update(_prefixed_loss_components("train", train_summary))
        row.update(_prefixed_loss_components("val", val_summary))
        history.append(row)
        _write_train_csv(output_dir / "train.csv", history)
        if _is_proposal_hybrid_config(cfg):
            _write_train_csv(output_dir / "logs" / "train.csv", history)

        improved = val_summary["Dice"] > best_metric + min_delta
        if improved:
            best_metric = val_summary["Dice"]
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint_payload = _checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_metric,
            cfg=cfg,
        )
        torch.save(checkpoint_payload, checkpoint_dir / last_checkpoint_name)
        if last_checkpoint_name != "last.pth":
            torch.save(checkpoint_payload, checkpoint_dir / "last.pth")
        if improved:
            torch.save(checkpoint_payload, checkpoint_dir / best_checkpoint_name)
            if best_checkpoint_name != "best.pth":
                torch.save(checkpoint_payload, checkpoint_dir / "best.pth")

        logger.info(
            "Epoch %03d/%03d - train_loss %.5f train_dice %.5f val_loss %.5f val_dice %.5f lr %.6g%s",
            epoch,
            epochs,
            train_summary["Loss"],
            train_summary["Dice"],
            val_summary["Loss"],
            val_summary["Dice"],
            current_lr,
            " *best*" if improved else "",
        )

        if early_enabled and patience_counter >= patience:
            logger.info("Early stopping at epoch %d. Best epoch: %d, best val Dice: %.5f", epoch, best_epoch, best_metric)
            break

    best_checkpoint = checkpoint_dir / best_checkpoint_name
    if best_checkpoint.exists():
        state = torch.load(best_checkpoint, map_location=device)
        load_model_state(model, state["model_state"])
        logger.info("Loaded best checkpoint from epoch %d", state.get("epoch", best_epoch))

    compute_surface = bool(get_nested(cfg, "evaluation.compute_surface_metrics", False))
    final_rows = []
    for split, loader in eval_loaders.items():
        if eval_2d_as_volume:
            summary = _evaluate_2d_volume(
                model=model,
                dataset=volume_eval_datasets[split],
                criterion=criterion,
                device=device,
                num_classes=model_result.num_classes,
                compute_surface=compute_surface,
                slice_batch_size=int(get_nested(cfg, "evaluation.slice_batch_size", get_nested(cfg, "evaluation.batch_size", 1))),
                progress_desc=f"Final {split}",
                show_progress=show_progress,
            )
        else:
            summary = _evaluate(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                num_classes=model_result.num_classes,
                compute_surface=compute_surface,
                progress_desc=f"Final {split}",
                show_progress=show_progress,
            )
        final_rows.append(
            _metrics_row(
                split=split,
                summary=summary,
                params=total_params,
                flops=flops,
                evaluation_mode=evaluation_mode,
                model_name=model_result.name,
                model_type=model_type,
                encoder=model_result.backbone,
            )
        )
        logger.info(
            "Final %s - Dice %.5f IoU %.5f Acc %.5f Precision %.5f Recall %.5f GT+ %.6f Pred+ %.6f TP %d FP %d FN %d FPS %.3f",
            split,
            summary["Dice"],
            summary["IoU"],
            summary["Accuracy"],
            summary["Precision"],
            summary["Recall"],
            summary["GT Positive Ratio"],
            summary["Pred Positive Ratio"],
            summary["TP"],
            summary["FP"],
            summary["FN"],
            summary["FPS"],
        )

    _write_metrics_csv(output_dir / "metrics.csv", final_rows)
    if _is_proposal_hybrid_config(cfg):
        _write_metrics_csv(output_dir / "logs" / "metrics.csv", final_rows)
    _plot_curves(output_dir / "curve.pdf", history)
    visual_datasets = volume_eval_datasets if eval_2d_as_volume else eval_datasets
    save_prediction_visuals(
        model=model,
        datasets=visual_datasets,
        output_root=output_dir,
        cfg=cfg,
        device=device,
        num_classes=model_result.num_classes,
    )
    _write_proposal_summary(
        output_dir=output_dir,
        cfg=cfg,
        model_result=model_result,
        best_epoch=best_epoch,
        best_metric=best_metric,
        best_checkpoint=best_checkpoint,
        final_rows=final_rows,
        pretrain_report=pretrain_report,
    )
    if _is_proposal_hybrid_config(cfg):
        logger.info("Saved train history to %s", display_path(output_dir / "logs" / "train.csv"))
        logger.info("Saved final metrics to %s", display_path(output_dir / "logs" / "metrics.csv"))
        logger.info("Saved summary to %s", display_path(output_dir / "summary.json"))
        logger.info("Best epoch: %d, best val Dice: %.5f", best_epoch, best_metric)
        logger.info("Best checkpoint: %s", display_path(best_checkpoint))
    logger.info("Done. Output saved to %s", display_path(output_dir))
    _write_logs_txt(log_path, output_dir, logger)
    return output_dir


def _config_token(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(sanitize_name(item) for item in value) or "none"
    return sanitize_name(value)


def _slice_selection_metadata(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    slice_cfg = dict(get_nested(cfg, "slice_2d", {}) or {})
    metadata = {
        "num_slices": int(slice_cfg.get("num_slices", 1)),
        "axis": int(slice_cfg.get("axis", 2)),
        "position": slice_cfg.get("position", "center"),
        "sampling_strategy": str(slice_cfg.get("sampling_strategy", "center")).lower(),
        "samples_per_volume": int(slice_cfg.get("samples_per_volume", 1)),
    }
    group_cfg = _group_based_slice_config(slice_cfg)
    if group_cfg or _is_group_based_slice_mode(metadata["sampling_strategy"]):
        metadata["group_based"] = {
            "num_groups": int(group_cfg.get("num_groups", metadata["samples_per_volume"])),
            "samples_per_group": int(group_cfg.get("samples_per_group", 1)),
            "similarity_metric": str(group_cfg.get("similarity_metric", "mad")).lower(),
            "selection_order": str(group_cfg.get("selection_order", "closest")).lower(),
            "group_based": True,
        }
    return metadata


def _slice_selection_name(cfg: Mapping[str, Any]) -> str:
    metadata = _slice_selection_metadata(cfg)
    strategy = _config_token(metadata["sampling_strategy"])
    if _is_group_based_slice_mode(metadata["sampling_strategy"]):
        group_based = dict(metadata.get("group_based", {}) if isinstance(metadata.get("group_based", {}), Mapping) else {})
        num_groups = int(group_based.get("num_groups", metadata["samples_per_volume"]))
        samples_per_group = int(group_based.get("samples_per_group", 1))
        metric = _config_token(group_based.get("similarity_metric", "mad"))
        order = _config_token(group_based.get("selection_order", "closest"))
        return (
            f"sam_{strategy}"
            f"_spv{metadata['samples_per_volume']}"
            f"_g{num_groups}"
            f"_spg{samples_per_group}"
            f"_{metric}"
            f"_{order}"
            f"_sl{metadata['num_slices']}"
            f"_ax{metadata['axis']}"
        )
    position = _config_token(metadata["position"])
    return (
        f"pos_{position}"
        f"_sam_{strategy}"
        f"_spv{metadata['samples_per_volume']}"
        f"_sl{metadata['num_slices']}"
        f"_ax{metadata['axis']}"
    )


def _profile_input_shape(cfg: Mapping[str, Any], in_channels: int) -> tuple[int, ...]:
    model_type = str(get_nested(cfg, "model.type", "2D")).upper()
    image_size = [int(item) for item in get_nested(cfg, "training.image_size", [256, 256])]
    if model_type == "2D":
        if len(image_size) < 2:
            raise ValueError("training.image_size must contain [height, width] for 2D models.")
        return (1, in_channels, image_size[0], image_size[1])
    preserve_depth = bool(get_nested(cfg, "training.preserve_depth", get_nested(cfg, "dataset.preserve_depth", False)))
    if preserve_depth:
        if len(image_size) < 2:
            raise ValueError("training.image_size must contain [height, width] when training.preserve_depth=true.")
        profile_depth = int(get_nested(cfg, "training.profile_depth", image_size[2] if len(image_size) >= 3 else 64))
        if str(get_nested(cfg, "training.volume_layout", "HWD")).upper() == "DHW":
            return (1, in_channels, profile_depth, image_size[0], image_size[1])
        return (1, in_channels, image_size[0], image_size[1], profile_depth)
    if len(image_size) < 3:
        raise ValueError("training.image_size must contain [height, width, depth] for 3D models.")
    if str(get_nested(cfg, "training.volume_layout", "HWD")).upper() == "DHW":
        return (1, in_channels, image_size[2], image_size[0], image_size[1])
    return (1, in_channels, image_size[0], image_size[1], image_size[2])


def _evaluation_mode(model_type: str, eval_2d_as_volume: bool) -> str:
    if model_type == "2D":
        return "2d_stacked_volume" if eval_2d_as_volume else "2d_slice"
    return "3d_volume"


def _make_train_loader(dataset, cfg: Mapping[str, Any], seed: int, device: torch.device):
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_workers = int(get_nested(cfg, "training.num_workers", 0))
    kwargs = {
        "batch_size": int(_typed_config_value(cfg, "training", "batch_size", 1)),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": bool(get_nested(cfg, "training.pin_memory", True)) and device.type == "cuda",
        "worker_init_fn": seed_worker(seed),
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(get_nested(cfg, "training.persistent_workers", True))
        prefetch_factor = get_nested(cfg, "training.prefetch_factor", 2)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def _make_eval_loader(dataset, cfg: Mapping[str, Any], seed: int, device: torch.device):
    num_workers = int(get_nested(cfg, "evaluation.num_workers", get_nested(cfg, "training.num_workers", 0)))
    kwargs = {
        "batch_size": int(_typed_config_value(cfg, "evaluation", "batch_size", 1)),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(get_nested(cfg, "training.pin_memory", True)) and device.type == "cuda",
        "worker_init_fn": seed_worker(seed),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(get_nested(cfg, "training.persistent_workers", True))
        prefetch_factor = get_nested(cfg, "training.prefetch_factor", 2)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def _typed_config_value(cfg: Mapping[str, Any], section: str, key: str, default: Any = None):
    model_type = str(get_nested(cfg, "model.type", "2D")).lower()
    typed_value = get_nested(cfg, f"{section}.{key}_{model_type}", None)
    if typed_value is not None:
        return typed_value
    return get_nested(cfg, f"{section}.{key}", default)


def _move_batch(batch: Mapping[str, Any], device: torch.device):
    image = batch["image"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    return image, label


def _batch_slice_indices(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor | None:
    value = batch.get("slice_index")
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)


def _model_uses_slice_indices(model) -> bool:
    return bool(getattr(unwrap_model(model), "expects_slice_indices", False))


def _call_model(model, image: torch.Tensor, *, slice_indices: torch.Tensor | None = None, return_features: bool = False):
    kwargs: dict[str, Any] = {}
    if return_features:
        kwargs["return_features"] = True
    if slice_indices is not None and _model_uses_slice_indices(model):
        kwargs["slice_indices"] = slice_indices
    return model(image, **kwargs)


def _model_forward_for_loss(model, image: torch.Tensor, criterion, slice_indices: torch.Tensor | None = None):
    if not bool(getattr(criterion, "requires_encoder_features", False)):
        output = _call_model(model, image, slice_indices=slice_indices)
        return output, []
    feature_model = model
    if isinstance(model, torch.nn.DataParallel):
        device_count = max(1, len(getattr(model, "device_ids", [])))
        # DataParallel pads non-tensor kwargs across devices. With batch size
        # smaller than GPU count, return_features=True can reach a replica with
        # no image tensor and raise "forward() missing x".
        if int(image.shape[0]) < device_count:
            feature_model = unwrap_model(model)
    try:
        output = _call_model(feature_model, image, slice_indices=slice_indices, return_features=True)
    except TypeError as error:
        if isinstance(model, torch.nn.DataParallel) and feature_model is model:
            try:
                output = _call_model(unwrap_model(model), image, slice_indices=slice_indices, return_features=True)
            except TypeError:
                raise RuntimeError(
                    "The selected loss requires encoder attention supervision, but this model forward does not accept "
                    "return_features=True."
                ) from error
        else:
            raise RuntimeError(
                "The selected loss requires encoder attention supervision, but this model forward does not accept "
                "return_features=True."
            ) from error
    except RuntimeError as error:
        if isinstance(model, torch.nn.DataParallel) and "missing 1 required positional argument" in str(error):
            output = _call_model(unwrap_model(model), image, slice_indices=slice_indices, return_features=True)
        else:
            raise
    encoder_features = extract_encoder_features(output)
    if not encoder_features:
        raise RuntimeError(
            "The selected loss requires encoder features, but the model did not return encoder feature tensors. "
            "Use a U-Net/encoder-decoder model that exposes features={\"encoder\": [...]}."
        )
    return extract_logits(output), encoder_features


def _loss_components(criterion) -> dict[str, torch.Tensor]:
    components = getattr(criterion, "last_components", {})
    if not isinstance(components, Mapping):
        return {}
    return {str(key): value for key, value in components.items() if isinstance(value, torch.Tensor)}


def _accumulate_loss_components(component_sums: dict[str, float], criterion, batch_items: int) -> None:
    for key, value in _loss_components(criterion).items():
        if not key.startswith("loss_"):
            continue
        component_sums[key] = component_sums.get(key, 0.0) + float(value.detach().cpu()) * int(batch_items)


def _summarize_loss_components(component_sums: dict[str, float], items: int) -> dict[str, float]:
    if int(items) <= 0:
        return {}
    return {key: float(value) / float(items) for key, value in sorted(component_sums.items())}


def _prefixed_loss_components(prefix: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in summary.items() if str(key).startswith("loss_")}


def _moe_balance_weight(cfg: Mapping[str, Any]) -> float:
    nested = get_nested(cfg, "training.loss.moe_balance_weight", None)
    if nested is not None:
        return float(nested)
    return float(get_nested(cfg, "training.moe_balance_weight", 0.0))


def _moe_entropy_weight(cfg: Mapping[str, Any]) -> float:
    nested = get_nested(cfg, "training.loss.moe_entropy_weight", None)
    if nested is not None:
        return float(nested)
    return float(get_nested(cfg, "training.moe_entropy_weight", 0.0))


def _model_moe_balance_loss(model) -> torch.Tensor | None:
    base_model = unwrap_model(model)
    loss_fn = getattr(base_model, "moe_load_balance_loss", None)
    if not callable(loss_fn):
        loss_fn = getattr(base_model, "moe_balance_loss", None)
    if not callable(loss_fn):
        return None
    loss = loss_fn()
    return loss if isinstance(loss, torch.Tensor) else None


def _model_moe_entropy_loss(model) -> torch.Tensor | None:
    base_model = unwrap_model(model)
    loss_fn = getattr(base_model, "moe_entropy_loss", None)
    if not callable(loss_fn):
        return None
    loss = loss_fn()
    return loss if isinstance(loss, torch.Tensor) else None


def _set_moe_router_epoch(model, epoch: int) -> None:
    set_epoch_fn = getattr(unwrap_model(model), "set_moe_epoch", None)
    if callable(set_epoch_fn):
        set_epoch_fn(int(epoch))


def _reset_moe_router_stats(model) -> None:
    reset_fn = getattr(unwrap_model(model), "reset_moe_router_stats", None)
    if callable(reset_fn):
        reset_fn()


def _kernel_label(kernel) -> str:
    if isinstance(kernel, str):
        return kernel
    return "x".join(str(int(item)) for item in kernel)


def _save_moe_router_heatmap(output_dir: Path, model, epoch: int, cfg: Mapping[str, Any]) -> None:
    if not bool(get_nested(cfg, "training.save_moe_router_heatmap", True)):
        return
    every = max(1, int(get_nested(cfg, "training.moe_router_heatmap_every", 1)))
    if int(epoch) % every != 0:
        return

    stats_fn = getattr(unwrap_model(model), "moe_router_stats", None)
    if not callable(stats_fn):
        return
    stats = list(stats_fn())
    if not stats:
        return

    router_dir = output_dir / "logs" / "router"
    router_dir.mkdir(parents=True, exist_ok=True)
    csv_path = router_dir / "router_stats.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "layer", "expert", "kernel", "importance", "load", "samples", "top_k"],
        )
        if write_header:
            writer.writeheader()
        for item in stats:
            kernels = list(item.get("kernels", []))
            expert_labels = list(item.get("expert_labels", []))
            importance = item.get("importance")
            load = item.get("load")
            for expert_index, kernel in enumerate(kernels):
                label = expert_labels[expert_index] if expert_index < len(expert_labels) else kernel
                writer.writerow(
                    {
                        "epoch": int(epoch),
                        "layer": str(item.get("layer", "")),
                        "expert": expert_index,
                        "kernel": _kernel_label(label),
                        "importance": float(importance[expert_index]),
                        "load": float(load[expert_index]),
                        "samples": int(item.get("samples", 0)),
                        "top_k": int(item.get("top_k", 0)),
                    }
                )

    rows_by_layer: dict[str, list[dict[str, Any]]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_layer.setdefault(str(row.get("layer", "")), []).append(row)

    for layer, rows in rows_by_layer.items():
        epochs = sorted({int(row["epoch"]) for row in rows})
        kernels = sorted(
            {int(row["expert"]): str(row["kernel"]) for row in rows}.items(),
            key=lambda item: item[0],
        )
        expert_indices = [index for index, _ in kernels]
        expert_labels = [label for _, label in kernels]
        epoch_to_col = {value: index for index, value in enumerate(epochs)}
        expert_to_row = {value: index for index, value in enumerate(expert_indices)}
        importance_grid = np.full((len(expert_indices), len(epochs)), np.nan, dtype=np.float32)
        load_grid = np.full_like(importance_grid, np.nan)

        for row in rows:
            row_index = expert_to_row[int(row["expert"])]
            col_index = epoch_to_col[int(row["epoch"])]
            importance_grid[row_index, col_index] = float(row["importance"])
            load_grid[row_index, col_index] = float(row["load"])

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(max(10, len(epochs) * 0.45), max(4, len(expert_labels) * 0.55)),
            sharey=True,
        )
        for axis, title, grid in (
            (axes[0], "importance", importance_grid),
            (axes[1], "load", load_grid),
        ):
            finite = grid[np.isfinite(grid)]
            vmax = max(1e-6, float(finite.max())) if finite.size else 1.0
            im = axis.imshow(grid, aspect="auto", vmin=0.0, vmax=vmax)
            axis.set_title(f"{layer} router {title}")
            axis.set_xlabel("epoch")
            axis.set_xticks(np.arange(len(epochs)))
            axis.set_xticklabels([str(item) for item in epochs], rotation=45, ha="right")
            axis.set_yticks(np.arange(len(expert_labels)))
            axis.set_yticklabels(expert_labels)
            axis.set_ylabel("expert kernel")
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(router_dir / f"router_heatmap_{sanitize_name(layer)}.png", dpi=160)
        plt.close(fig)


def _metadata_preview(value: Any, max_items: int = 4) -> str:
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1).tolist()
        items = flat[:max_items]
        suffix = ", ..." if len(flat) > max_items else ""
        return f"{items}{suffix}"
    if isinstance(value, (list, tuple)):
        items = [str(item) for item in value[:max_items]]
        suffix = ", ..." if len(value) > max_items else ""
        return "[" + ", ".join(items) + suffix + "]"
    if value is None:
        return "<missing>"
    return str(value)


def _tensor_debug_stats(name: str, tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return f"{name}: <missing>"
    detached = tensor.detach()
    numel = int(detached.numel())
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    nonfinite_count = numel - finite_count
    parts = [
        f"{name}: shape={tuple(detached.shape)}",
        f"dtype={detached.dtype}",
        f"device={detached.device}",
        f"finite={finite_count}/{numel}",
    ]
    if numel and finite_count:
        finite_values = detached[finite_mask].float()
        parts.extend(
            [
                f"min={float(finite_values.min().item()):.6g}",
                f"max={float(finite_values.max().item()):.6g}",
                f"mean={float(finite_values.mean().item()):.6g}",
            ]
        )
    if nonfinite_count and detached.ndim > 0:
        per_sample = (~finite_mask).reshape(int(detached.shape[0]), -1).sum(dim=1).detach().cpu().tolist()
        parts.append(f"nonfinite_per_sample={per_sample[:8]}")
    return ", ".join(parts)


def _loss_component_debug(criterion) -> list[str]:
    rows = []
    for key, value in _loss_components(criterion).items():
        rows.append(_tensor_debug_stats(key, value))
    return rows


def _raise_nonfinite_training_loss(
    *,
    batch_index: int,
    batch: Mapping[str, Any],
    image: torch.Tensor,
    label: torch.Tensor,
    logits: torch.Tensor,
    loss: torch.Tensor,
    criterion,
    optimizer,
    amp_enabled: bool,
) -> None:
    lr_values = [float(group.get("lr", math.nan)) for group in getattr(optimizer, "param_groups", [])]
    message = [
        f"Non-finite training loss detected at train batch {batch_index}.",
        f"amp_enabled={amp_enabled}, lr={lr_values}",
        "Batch metadata:",
        f"  case_id={_metadata_preview(batch.get('case_id'))}",
        f"  image_path={_metadata_preview(batch.get('image_path'))}",
        f"  mask_path={_metadata_preview(batch.get('mask_path'))}",
        f"  slice_index={_metadata_preview(batch.get('slice_index'))}",
        "Tensor stats:",
        f"  {_tensor_debug_stats('image', image)}",
        f"  {_tensor_debug_stats('label', label)}",
        f"  {_tensor_debug_stats('logits', logits)}",
        f"  {_tensor_debug_stats('loss', loss)}",
    ]
    component_rows = _loss_component_debug(criterion)
    if component_rows:
        message.append("Loss components:")
        message.extend(f"  {row}" for row in component_rows)
    message.extend(
        [
            "Suggested checks: if image has non-finite values, inspect/fix the listed NIfTI file; "
            "if logits are non-finite while image is finite, disable training.amp or lower training.lr.",
        ]
    )
    raise FloatingPointError("\n".join(message))


def _progress(iterable, desc: str, show_progress: bool, total: int | None = None):
    if not show_progress:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True)


def _train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    num_classes: int,
    amp_enabled: bool,
    grad_clip_norm: float,
    moe_balance_weight: float,
    moe_entropy_weight: float,
    progress_desc: str,
    show_progress: bool,
):
    model.train()
    accumulator = MetricAccumulator(compute_surface=False)
    component_sums: dict[str, float] = {}
    component_items = 0
    progress = _progress(loader, desc=progress_desc, show_progress=show_progress, total=len(loader))
    for batch_index, batch in enumerate(progress, start=1):
        image, label = _move_batch(batch, device)
        slice_indices = _batch_slice_indices(batch, device)
        batch_items = int(image.shape[0])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            model_output, encoder_features = _model_forward_for_loss(model, image, criterion, slice_indices=slice_indices)
            logits = extract_logits(model_output, num_classes=num_classes)
            loss = criterion(logits, label, encoder_features=encoder_features or None)
            moe_balance_loss = _model_moe_balance_loss(model)
            if moe_balance_loss is not None and float(moe_balance_weight) > 0:
                loss = loss + float(moe_balance_weight) * moe_balance_loss
                component_sums["loss_moe_balance"] = component_sums.get("loss_moe_balance", 0.0) + float(moe_balance_loss.detach().cpu()) * batch_items
            moe_entropy_loss = _model_moe_entropy_loss(model)
            if moe_entropy_loss is not None and float(moe_entropy_weight) > 0:
                loss = loss + float(moe_entropy_weight) * moe_entropy_loss
                component_sums["loss_moe_entropy"] = component_sums.get("loss_moe_entropy", 0.0) + float(moe_entropy_loss.detach().cpu()) * batch_items
            _accumulate_loss_components(component_sums, criterion, batch_items)
            component_items += batch_items
            auxiliary_logits = extract_auxiliary_logits(model_output, num_classes=num_classes)
            if auxiliary_logits:
                auxiliary_losses = [criterion(aux_logits, label) for aux_logits in auxiliary_logits]
                auxiliary_loss = torch.stack(auxiliary_losses).mean()
                auxiliary_weight = float(getattr(criterion, "auxiliary_weight", 1.0))
                if auxiliary_weight > 0:
                    loss = (loss + auxiliary_weight * auxiliary_loss) / (1.0 + auxiliary_weight)
        if not torch.isfinite(loss):
            _raise_nonfinite_training_loss(
                batch_index=batch_index,
                batch=batch,
                image=image,
                label=label,
                logits=logits,
                loss=loss,
                criterion=criterion,
                optimizer=optimizer,
                amp_enabled=amp_enabled,
            )
        scaler.scale(loss).backward()
        if float(grad_clip_norm) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()

        prediction = predict_from_logits(logits.detach(), target_shape=tuple(label.shape[1:]))
        accumulator.update(prediction, label, loss=float(loss.detach().cpu()))
        if show_progress:
            summary = accumulator.summary()
            progress.set_postfix(loss=f"{summary['Loss']:.4f}", dice=f"{summary['Dice']:.4f}")
    summary = accumulator.summary()
    summary.update(_summarize_loss_components(component_sums, component_items))
    return summary


@torch.no_grad()
def _evaluate(model, loader, criterion, device, num_classes: int, compute_surface: bool, progress_desc: str, show_progress: bool):
    model.eval()
    accumulator = MetricAccumulator(compute_surface=compute_surface)
    component_sums: dict[str, float] = {}
    component_items = 0
    progress = _progress(loader, desc=progress_desc, show_progress=show_progress, total=len(loader))
    for batch in progress:
        image, label = _move_batch(batch, device)
        slice_indices = _batch_slice_indices(batch, device)
        batch_items = int(image.shape[0])
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = extract_logits(_call_model(model, image, slice_indices=slice_indices), num_classes=num_classes)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        loss = criterion(logits, label)
        _accumulate_loss_components(component_sums, criterion, batch_items)
        component_items += batch_items
        prediction = predict_from_logits(logits, target_shape=tuple(label.shape[1:]))
        accumulator.update(prediction, label, loss=float(loss.detach().cpu()))
        accumulator.add_inference_time(elapsed, int(image.shape[0]))
        if show_progress:
            summary = accumulator.summary()
            progress.set_postfix(loss=f"{summary['Loss']:.4f}", dice=f"{summary['Dice']:.4f}")
    summary = accumulator.summary()
    summary.update(_summarize_loss_components(component_sums, component_items))
    return summary


@torch.no_grad()
def _evaluate_2d_volume(
    model,
    dataset,
    criterion,
    device,
    num_classes: int,
    compute_surface: bool,
    slice_batch_size: int,
    progress_desc: str,
    show_progress: bool,
):
    model.eval()
    accumulator = MetricAccumulator(compute_surface=compute_surface)
    component_sums: dict[str, float] = {}
    component_items = 0
    slice_batch_size = max(1, int(slice_batch_size))
    axis = int(getattr(dataset, "axis", 2))

    progress = _progress(dataset, desc=progress_desc, show_progress=show_progress, total=len(dataset))
    for sample in progress:
        image_slices = sample["image_slices"].to(device, non_blocking=True)
        label_slices = sample["label_slices"].to(device, non_blocking=True)
        predictions = []
        elapsed_total = 0.0
        loss_sum = 0.0
        loss_items = 0

        for start_index in range(0, int(image_slices.shape[0]), slice_batch_size):
            image_batch = image_slices[start_index : start_index + slice_batch_size]
            label_batch = label_slices[start_index : start_index + slice_batch_size]
            slice_indices = torch.arange(
                start_index,
                start_index + int(image_batch.shape[0]),
                device=device,
                dtype=torch.long,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = extract_logits(_call_model(model, image_batch, slice_indices=slice_indices), num_classes=num_classes)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_total += time.perf_counter() - start

            loss = criterion(logits, label_batch)
            prediction = predict_from_logits(logits, target_shape=tuple(label_batch.shape[1:]))
            predictions.append(prediction.cpu().numpy())

            items = int(image_batch.shape[0])
            loss_sum += float(loss.detach().cpu()) * items
            loss_items += items
            _accumulate_loss_components(component_sums, criterion, items)
            component_items += items

        pred_slices = np.concatenate(predictions, axis=0)
        pred_volume = stack_slices_as_volume(pred_slices, axis)
        target_volume = sample["label"].numpy()
        case_loss = loss_sum / max(1, loss_items)

        accumulator.update(pred_volume[np.newaxis, ...], target_volume[np.newaxis, ...], loss=case_loss)
        accumulator.add_inference_time(elapsed_total, 1)
        if show_progress:
            summary = accumulator.summary()
            progress.set_postfix(loss=f"{summary['Loss']:.4f}", dice=f"{summary['Dice']:.4f}")

    summary = accumulator.summary()
    summary.update(_summarize_loss_components(component_sums, component_items))
    return summary


def _checkpoint_payload(epoch, model, optimizer, scheduler, best_metric, cfg):
    payload = {
        "epoch": epoch,
        "model_state": model_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": best_metric,
        "config": cfg,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    return payload


def _write_train_csv(path: Path, history: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fieldnames = ["epoch", "lr", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou", "epoch_time_sec"]
    extra_fieldnames = sorted({key for row in history for key in row if key not in base_fieldnames})
    fieldnames = base_fieldnames + extra_fieldnames
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _metrics_row(
    split: str,
    summary: Mapping[str, Any],
    params: int,
    flops: int | None,
    evaluation_mode: str,
    model_name: str,
    model_type: str,
    encoder: str,
) -> Dict[str, Any]:
    row = {
        "Split": split,
        "Model": model_name,
        "Type": model_type,
        "Encoder": encoder,
        "Evaluation Mode": evaluation_mode,
    }
    for key in [
        "Dice",
        "IoU",
        "Accuracy",
        "Precision",
        "Recall",
        "Pred Positives",
        "GT Positives",
        "Pred Positive Ratio",
        "GT Positive Ratio",
        "Specificity",
        "F1",
        "Loss",
        "HD95",
        "ASD",
        "Inference Time",
        "FPS",
        "TP",
        "TN",
        "FP",
        "FN",
    ]:
        row[key] = summary.get(key, math.nan)
    row["Params"] = params
    row["FLOPs"] = "" if flops is None else flops
    return row


def _write_metrics_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Split",
        "Model",
        "Type",
        "Encoder",
        "Evaluation Mode",
        "Dice",
        "IoU",
        "Accuracy",
        "Precision",
        "Recall",
        "Pred Positives",
        "GT Positives",
        "Pred Positive Ratio",
        "GT Positive Ratio",
        "Specificity",
        "F1",
        "Loss",
        "HD95",
        "ASD",
        "Params",
        "FLOPs",
        "Inference Time",
        "FPS",
        "TP",
        "TN",
        "FP",
        "FN",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_fold_metrics(path: Path, fold_index: int) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({"Fold": fold_index + 1, **row})
        return rows


def _write_kfold_metrics_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = ["Fold"] + [key for key in rows[0].keys() if key != "Fold"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


_KFOLD_SUMMARY_METRICS = [
    "Dice",
    "IoU",
    "Accuracy",
    "Precision",
    "Recall",
    "Pred Positive Ratio",
    "GT Positive Ratio",
    "Specificity",
    "F1",
    "Loss",
    "HD95",
    "ASD",
    "Params",
    "FLOPs",
    "Inference Time",
    "FPS",
]


def _float_or_nan(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite_values = np.asarray([value for value in values if not math.isnan(value)], dtype=float)
    if finite_values.size == 0:
        return math.nan, math.nan
    mean = float(np.mean(finite_values))
    std = float(np.std(finite_values, ddof=1)) if finite_values.size > 1 else 0.0
    return mean, std


def _format_mean_std(mean: float, std: float) -> str:
    if math.isnan(mean) and math.isnan(std):
        return "nan"
    return f"{mean:.6g} +/- {std:.6g}"


def _split_order(split: str) -> int:
    return {"train": 0, "val": 1, "test": 2}.get(str(split).lower(), 99)


def _write_kfold_summary_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    group_keys = ["Split", "Model", "Type", "Encoder", "Evaluation Mode"]
    grouped: dict[tuple[Any, ...], list[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_keys)
        grouped.setdefault(key, []).append(row)

    fieldnames = group_keys + ["Folds"]
    for metric in _KFOLD_SUMMARY_METRICS:
        fieldnames.extend([f"{metric} Mean", f"{metric} Std", f"{metric} Mean+/-Std"])

    summary_rows: list[Dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: (_split_order(str(item[0][0])), item[0])):
        summary: Dict[str, Any] = {field: value for field, value in zip(group_keys, key)}
        summary["Folds"] = len({row.get("Fold") for row in group_rows})
        for metric in _KFOLD_SUMMARY_METRICS:
            mean, std = _mean_std([_float_or_nan(row.get(metric)) for row in group_rows])
            summary[f"{metric} Mean"] = mean
            summary[f"{metric} Std"] = std
            summary[f"{metric} Mean+/-Std"] = _format_mean_std(mean, std)
        summary_rows.append(summary)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def _plot_curves(path: Path, history: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    epochs = [row["epoch"] for row in history]

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(epochs, [row["train_dice"] for row in history], label="train dice")
    axes[1].plot(epochs, [row["val_dice"] for row in history], label="val dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
