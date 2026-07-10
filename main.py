from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Mapping

warnings.filterwarnings(
    "ignore",
    message=r"The cuda\.cudart module is deprecated.*",
    category=FutureWarning,
)

try:
    import yaml
except ImportError as error:
    raise SystemExit("PyYAML is required. Activate the project environment or install it with `pip install pyyaml`.") from error


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and bool(value.get("__replace__", False)):
            merged[key] = {nested_key: nested_value for nested_key, nested_value in value.items() if nested_key != "__replace__"}
            continue
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> Mapping[str, Any]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    base_config = payload.get("extends", payload.get("base_config"))
    if not base_config:
        return payload
    base_path = Path(str(base_config))
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    override = dict(payload)
    override.pop("extends", None)
    override.pop("base_config", None)
    return _deep_merge(_load_yaml(base_path), override)


def _visible_gpu_ids(gpu_cfg: Mapping[str, Any]) -> str:
    ids = gpu_cfg.get("ids", "0")
    if isinstance(ids, (list, tuple)):
        parsed = [str(item).strip() for item in ids if str(item).strip()]
    else:
        parsed = [item.strip() for item in str(ids).split(",") if item.strip()]
    if not parsed:
        parsed = ["0"]
    if not bool(gpu_cfg.get("multi_gpu", False)):
        parsed = parsed[:1]
    return ",".join(parsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate cyst segmentation models from a YAML config.")
    parser.add_argument("--config", type=str, default=os.environ.get("CONFIG", "config/cyst.yaml"), help="Path to YAML config.")
    parser.add_argument("--visualize-only", action="store_true", help="Load a checkpoint and regenerate prediction/Grad-CAM visualizations only.")
    parser.add_argument("--checkpoint", type=str, default=os.environ.get("CHECKPOINT"), help="Checkpoint path used with --visualize-only.")
    parser.add_argument("--use-checkpoint-config", action="store_true", help="Use checkpoint['config'] for --visualize-only, then apply CLI visualization overrides.")
    parser.add_argument("--output-dir", type=str, default=os.environ.get("OUTPUT_DIR"), help="Output directory override for --visualize-only.")
    parser.add_argument("--fold", type=int, default=None, help="Optional 1-based k-fold index used with --visualize-only.")
    parser.add_argument("--slice-position", type=str, default=None, help="Override visualization.slice_position, for example label_foreground or center.")
    parser.add_argument("--visual-selection", type=str, default=None, help="Override visualization.selection, for example per_source or fixed.")
    parser.add_argument("--visual-seed", type=int, default=None, help="Override visualization.seed for reproducible sample selection.")
    parser.add_argument("--samples-per-source", type=int, default=None, help="Override visualization.samples_per_source.")
    parser.add_argument("--num-visuals", type=int, default=None, help="Override evaluation.save_predictions_per_split for fixed selection.")
    parser.add_argument("--evaluate-2d-as-volume", action="store_true", help="Force 2D visualization/evaluation datasets to use full-volume case records.")
    parser.add_argument("--gpu-ids", type=str, default=os.environ.get("GPU_IDS"), help="Override gpu.ids, for example 0 or 0,1.")
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true", default=None, help="Force gpu.use_cuda=true.")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", help="Force gpu.use_cuda=false.")
    parser.add_argument("--multi-gpu", dest="multi_gpu", action="store_true", default=None, help="Force gpu.multi_gpu=true.")
    parser.add_argument("--single-gpu", dest="multi_gpu", action="store_false", help="Force gpu.multi_gpu=false.")
    return parser.parse_args()


def _load_checkpoint_config(checkpoint_path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise SystemExit("PyTorch is required to load checkpoint config. Activate the project environment first.") from error
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("config"), Mapping):
        raise ValueError(f"Checkpoint does not contain a saved config mapping: {checkpoint_path}")
    state_dict = _checkpoint_state_dict(checkpoint)
    return _config_with_unet3plus_checkpoint_head(dict(checkpoint["config"]), state_dict)


def _checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state", "state_dict", "model_state_dict", "network_weights"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def _config_with_unet3plus_checkpoint_head(cfg: dict[str, Any], state_dict: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = dict(cfg.get("model", {}) if isinstance(cfg.get("model", {}), Mapping) else {})
    model_name = str(model_cfg.get("name", "")).lower()
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

    for key, tensor in state_dict.items():
        clean_key = str(key)[7:] if str(key).startswith("module.") else str(key)
        if clean_key in {
            "model.outconv1.weight",
            "model.outconv2.weight",
            "model.outconv3.weight",
            "model.outconv4.weight",
            "model.outconv5.weight",
        } and hasattr(tensor, "shape") and len(tensor.shape) >= 1:
            args_cfg = dict(model_cfg.get("args", {}) if isinstance(model_cfg.get("args", {}), Mapping) else {})
            args_cfg["internal_num_classes"] = int(tensor.shape[0])
            model_cfg["args"] = args_cfg
            cfg["model"] = model_cfg
            return cfg
    return cfg


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if args.visualize_only and args.use_checkpoint_config:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when using --use-checkpoint-config.")
        cfg = dict(_load_checkpoint_config(Path(args.checkpoint)))
        config_path_for_log = None
    else:
        cfg = dict(_load_yaml(config_path))
        config_path_for_log = config_path
    visualization_cfg = dict(cfg.get("visualization", {}) if isinstance(cfg.get("visualization", {}), Mapping) else {})
    evaluation_cfg = dict(cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation", {}), Mapping) else {})
    gpu_cfg = dict(cfg.get("gpu", {}) if isinstance(cfg.get("gpu", {}), Mapping) else {})
    if args.slice_position is not None:
        visualization_cfg["slice_position"] = args.slice_position
    if args.visual_selection is not None:
        visualization_cfg["selection"] = args.visual_selection
    if args.visual_seed is not None:
        visualization_cfg["seed"] = int(args.visual_seed)
    if args.samples_per_source is not None:
        visualization_cfg["samples_per_source"] = int(args.samples_per_source)
    if args.num_visuals is not None:
        evaluation_cfg["save_predictions_per_split"] = int(args.num_visuals)
    if args.evaluate_2d_as_volume:
        evaluation_cfg["evaluate_2d_as_volume"] = True
    if args.gpu_ids is not None:
        gpu_cfg["ids"] = args.gpu_ids
        gpu_cfg["use_cuda"] = True
    if args.use_cuda is not None:
        gpu_cfg["use_cuda"] = bool(args.use_cuda)
    if args.multi_gpu is not None:
        gpu_cfg["multi_gpu"] = bool(args.multi_gpu)
    if visualization_cfg:
        cfg["visualization"] = visualization_cfg
    if evaluation_cfg:
        cfg["evaluation"] = evaluation_cfg
    if gpu_cfg:
        cfg["gpu"] = gpu_cfg

    if bool(gpu_cfg.get("use_cuda", True)):
        os.environ["CUDA_VISIBLE_DEVICES"] = _visible_gpu_ids(gpu_cfg)

    from training.runner import run, visualize_from_checkpoint

    if args.visualize_only:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when using --visualize-only.")
        output_dir = visualize_from_checkpoint(
            cfg,
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            config_path=config_path_for_log,
            fold_index=None if args.fold is None else int(args.fold) - 1,
        )
    else:
        output_dir = run(cfg, config_path=config_path)
    print(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()
