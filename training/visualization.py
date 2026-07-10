from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .torch_utils import extract_logits, predict_from_logits, resize_logits, unwrap_model
from .utils import get_nested, sanitize_name


def fixed_indices(length: int, count: int, seed: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    rng = np.random.default_rng(seed)
    count = min(int(count), int(length))
    return sorted(int(item) for item in rng.choice(length, count, replace=False))


def source_from_case_id(case_id: str) -> str:
    match = re.match(r"([A-Za-z]+)", str(case_id).strip())
    return match.group(1).upper() if match else "UNKNOWN"


def _case_id_for_index(dataset, index: int) -> str:
    records = getattr(dataset, "records", None)
    if records is not None:
        record_index = int(index)
        dataset_index = getattr(dataset, "index", None)
        if dataset_index is not None:
            record_index = int(dataset_index[int(index)][0])
        if 0 <= record_index < len(records):
            return str(getattr(records[record_index], "case_id", f"sample_{index}"))
    return f"sample_{index}"


def per_source_indices(dataset, count_per_source: int, seed: int) -> list[int]:
    length = len(dataset)
    if length <= 0 or count_per_source <= 0:
        return []

    grouped: dict[str, list[tuple[str, int]]] = {}
    for index in range(length):
        case_id = _case_id_for_index(dataset, index)
        grouped.setdefault(source_from_case_id(case_id), []).append((case_id, index))

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for source in sorted(grouped):
        entries = sorted(grouped[source], key=lambda item: (item[0], item[1]))
        take = min(int(count_per_source), len(entries))
        chosen_positions = rng.choice(len(entries), take, replace=False)
        selected.extend(int(entries[int(position)][1]) for position in np.atleast_1d(chosen_positions))
    return sorted(selected)


def _normalise_for_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        low, high = float(np.min(image)), float(np.max(image))
    return np.clip((image - low) / (high - low + 1e-8), 0, 1)


def _position_to_index(position, depth: int) -> int:
    if isinstance(position, str):
        key = position.lower()
        if key == "center":
            return depth // 2
        if key == "first":
            return 0
        if key == "last":
            return depth - 1
        position = float(position)
    if isinstance(position, float) and 0.0 <= position <= 1.0:
        return int(round(position * (depth - 1)))
    return int(np.clip(int(position), 0, depth - 1))


def _axis_candidate(value: Any, ndim: int) -> int | None:
    if value is None:
        return None
    try:
        return int(np.clip(int(value), 0, ndim - 1))
    except (TypeError, ValueError):
        return None


def _visual_slice_axis(cfg: Mapping, ndim: int, model_type: str) -> int:
    axis = _axis_candidate(get_nested(cfg, "visualization.slice_axis", 2), ndim)
    if axis is None:
        return 0

    if str(model_type).upper() != "3D":
        return axis
    if str(get_nested(cfg, "training.volume_layout", "HWD")).upper() != "DHW":
        return axis
    axis_layout = str(get_nested(cfg, "visualization.slice_axis_layout", "source")).lower()
    if axis_layout in {"tensor", "model", "dhw"}:
        return axis

    depth_axis = _axis_candidate(get_nested(cfg, "training.depth_axis", get_nested(cfg, "slice_2d.axis", 2)), ndim)
    if depth_axis is None:
        return axis
    source_to_tensor = [depth_axis, *[item for item in range(ndim) if item != depth_axis]]
    if axis in source_to_tensor:
        return int(source_to_tensor.index(axis))
    return axis


def _foreground_slice_index(label: np.ndarray, pred: np.ndarray, axis: int) -> int:
    label_positive = np.asarray(label) > 0
    pred_positive = np.asarray(pred) > 0
    positive = np.logical_or(label_positive, pred_positive)
    if not np.any(positive):
        return int(label.shape[axis] // 2)
    reduce_axes = tuple(item for item in range(positive.ndim) if item != axis)
    scores = positive.sum(axis=reduce_axes)
    return int(np.argmax(scores))


def _positive_slice_index(values: np.ndarray, axis: int) -> int:
    positive = np.asarray(values) > 0
    if not np.any(positive):
        return int(values.shape[axis] // 2)
    reduce_axes = tuple(item for item in range(positive.ndim) if item != axis)
    scores = positive.sum(axis=reduce_axes)
    return int(np.argmax(scores))


def _display_slice_index(label: np.ndarray, pred: np.ndarray | None, axis: int, position: Any) -> int:
    depth = int(label.shape[axis])
    key = str(position).lower()
    if key in {"label_foreground", "gt_foreground", "gt", "label"}:
        return _positive_slice_index(label, axis)
    if key in {"prediction_foreground", "pred_foreground", "prediction", "pred"} and pred is not None:
        return _positive_slice_index(pred, axis)
    if key in {"foreground", "auto", "best"}:
        return _foreground_slice_index(label, pred if pred is not None else np.zeros_like(label), axis)
    return _position_to_index(position, depth)


def _first_spatial_tensor(value, expected_ndim: int | None = None) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor) and value.ndim in {4, 5}:
        if expected_ndim is not None and int(value.ndim) != int(expected_ndim):
            return None
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_spatial_tensor(item, expected_ndim=expected_ndim)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_spatial_tensor(item, expected_ndim=expected_ndim)
            if tensor is not None:
                return tensor
    return None


def _model_uses_slice_indices(model) -> bool:
    return bool(getattr(unwrap_model(model), "expects_slice_indices", False))


def _call_model(model, image: torch.Tensor, slice_index: int | None = None):
    if slice_index is None or not _model_uses_slice_indices(model):
        return model(image)
    indices = torch.full((int(image.shape[0]),), int(slice_index), device=image.device, dtype=torch.long)
    return model(image, slice_indices=indices)


def _is_encoder_name(name: str) -> bool:
    lower = name.lower()
    tail = lower.split(".")[-1]
    if tail in {"conv1", "conv2", "conv3", "conv4", "conv5"}:
        return True
    return any(
        keyword in lower
        for keyword in (
            "stem",
            "encoder",
            "enc",
            "down",
            "conv0_0",
            "conv1_0",
            "conv2_0",
            "conv3_0",
            "conv4_0",
            "conv_blk",
            "block_one",
            "block_two",
            "block_three",
            "block_four",
            "block_five",
        )
    )


def _is_decoder_name(name: str) -> bool:
    lower = name.lower()
    tail = lower.split(".")[-1]
    if tail in {"conv4d_1", "conv3d_1", "conv2d_1", "conv1d_1"}:
        return True
    return any(keyword in lower for keyword in ("decoder", "dec", "up", "_pt_hd", "_ut_hd", "_cat_hd"))


def _should_hook_feature(name: str) -> bool:
    return bool(name) and (_is_encoder_name(name) or _is_decoder_name(name) or "bottleneck" in name.lower())


def _feature_spatial_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(item) for item in tensor.shape[2:])


def _channel_mean_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().mean(dim=1).cpu()


def _select_unique_features(records: Sequence[Mapping[str, Any]], max_items: int) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for record in records:
        shape = tuple(record["shape"])
        if shape in seen:
            continue
        seen.add(shape)
        selected.append(record)
        if len(selected) >= max_items:
            break
    return selected


def _collect_visual_features(
    model,
    image: torch.Tensor,
    num_classes: int,
    model_type: str,
    slice_index: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_records: list[dict[str, Any]] = []
    skip_records: list[dict[str, Any]] = []
    handles = []
    expected_ndim = 5 if str(model_type).upper() == "3D" else 4

    def forward_hook(name: str):
        def hook(_module, _inputs, output):
            tensor = _first_spatial_tensor(output, expected_ndim=expected_ndim)
            if tensor is None:
                return
            feature_records.append(
                {
                    "name": name,
                    "tensor": _channel_mean_tensor(tensor),
                    "shape": _feature_spatial_shape(tensor),
                }
            )

        return hook

    def pre_hook(name: str):
        def hook(_module, inputs):
            if not isinstance(inputs, tuple) or len(inputs) < 2:
                return
            for item in inputs[1:]:
                tensor = _first_spatial_tensor(item, expected_ndim=expected_ndim)
                if tensor is None:
                    continue
                skip_records.append(
                    {
                        "name": name,
                        "tensor": _channel_mean_tensor(tensor),
                        "shape": _feature_spatial_shape(tensor),
                    }
                )
                break

        return hook

    for name, module in model.named_modules():
        if _should_hook_feature(name):
            handles.append(module.register_forward_hook(forward_hook(name)))
        if _is_decoder_name(name):
            handles.append(module.register_forward_pre_hook(pre_hook(name)))

    try:
        _ = extract_logits(_call_model(model, image, slice_index=slice_index), num_classes=num_classes)
    finally:
        for handle in handles:
            handle.remove()

    return feature_records, skip_records


def _foreground_logit_from_logits(logits: torch.Tensor, target_shape: tuple[int, ...], num_classes: int) -> np.ndarray:
    resized = resize_logits(logits, target_shape)
    class_index = 1 if int(num_classes) > 1 and resized.shape[1] > 1 else 0
    return resized[:, class_index].squeeze(0).detach().cpu().numpy()


def _take_display_slice(image: np.ndarray, label: np.ndarray, pred: np.ndarray, logit: np.ndarray, model_type: str, cfg: Mapping):
    if model_type == "2D":
        channel = image.shape[0] // 2
        return image[channel], label, pred, logit, None

    axis = _visual_slice_axis(cfg, label.ndim, model_type=model_type)
    position = get_nested(cfg, "visualization.slice_position", "center")
    image_volume = image[0]
    index = _display_slice_index(label, pred, axis=axis, position=position)
    return (
        np.take(image_volume, index, axis=axis),
        np.take(label, index, axis=axis),
        np.take(pred, index, axis=axis),
        np.take(logit, index, axis=axis),
        int(index),
    )


def _scale_slice_index(slice_index: int | None, target_shape: tuple[int, ...], feature_shape: tuple[int, ...], axis: int) -> int:
    axis = int(np.clip(axis, 0, len(feature_shape) - 1))
    if slice_index is None or len(target_shape) <= axis or target_shape[axis] <= 1:
        return int(feature_shape[axis] // 2)
    ratio = float(slice_index) / float(max(1, int(target_shape[axis]) - 1))
    return int(np.clip(round(ratio * (int(feature_shape[axis]) - 1)), 0, int(feature_shape[axis]) - 1))


def _feature_mean_map(feature: torch.Tensor, model_type: str, cfg: Mapping, target_shape: tuple[int, ...], slice_index: int | None) -> np.ndarray:
    values = feature[0].float().numpy()
    if values.ndim == 2 or model_type == "2D":
        return np.asarray(values)

    axis = _visual_slice_axis(cfg, values.ndim, model_type=model_type)
    index = _scale_slice_index(slice_index, target_shape, tuple(values.shape), axis)
    return np.take(values, index, axis=axis)


def _save_feature_record(path: Path, record: Mapping[str, Any], model_type: str, cfg: Mapping, target_shape: tuple[int, ...], slice_index: int | None) -> None:
    heatmap = _feature_mean_map(record["tensor"], model_type=model_type, cfg=cfg, target_shape=target_shape, slice_index=slice_index)
    _save_heatmap(path, heatmap, cmap="magma")


def _target_module_name(model, feature_records: Sequence[Mapping[str, Any]], kind: str) -> str | None:
    modules = dict(model.named_modules())
    if kind == "decoder":
        for record in reversed(feature_records):
            name = str(record["name"])
            if _is_decoder_name(name) and name in modules:
                return name
    candidates = [record for record in feature_records if str(record["name"]) in modules]
    if not candidates:
        return None
    if kind == "bottleneck":
        return str(min(candidates, key=lambda item: int(np.prod(item["shape"])))["name"])
    return str(candidates[-1]["name"])


def _resolve_gradcam_model_and_module(model, target_module_name: str | None):
    """Use the real module for Grad-CAM so DataParallel does not replicate a backward pass."""

    if not target_module_name:
        return None, None
    gradcam_model = unwrap_model(model)
    modules = dict(gradcam_model.named_modules())
    candidates = [str(target_module_name)]
    if str(target_module_name).startswith("module."):
        candidates.append(str(target_module_name)[len("module.") :])
    else:
        candidates.append(f"module.{target_module_name}")
    for name in candidates:
        if name in modules:
            return gradcam_model, modules[name]
    return None, None


def _crop_gradcam_volume(
    image: torch.Tensor,
    target_shape: tuple[int, ...],
    cfg: Mapping,
    slice_index: int | None,
) -> tuple[torch.Tensor, tuple[int, ...], int | None]:
    """Crop the 3D Grad-CAM input around the displayed slice to keep backward memory bounded."""

    if image.ndim != 5 or len(target_shape) != 3 or slice_index is None:
        return image, target_shape, slice_index
    window = int(get_nested(cfg, "visualization.gradcam_depth_window", 1))
    if window <= 0:
        return image, target_shape, slice_index

    axis = _visual_slice_axis(cfg, len(target_shape), model_type="3D")
    image_axis = axis + 2
    axis_length = int(image.shape[image_axis])
    if axis_length <= window:
        return image, target_shape, slice_index

    center = int(np.clip(slice_index, 0, axis_length - 1))
    start = max(0, center - window // 2)
    end = min(axis_length, start + window)
    start = max(0, end - window)

    slices = [slice(None)] * image.ndim
    slices[image_axis] = slice(start, end)
    cropped_image = image[tuple(slices)].contiguous()
    cropped_shape = list(target_shape)
    cropped_shape[axis] = int(end - start)
    return cropped_image, tuple(cropped_shape), int(center - start)


def _compute_gradcam(
    model,
    image: torch.Tensor,
    target_shape: tuple[int, ...],
    num_classes: int,
    target_module_name: str | None,
    cfg: Mapping,
    slice_index: int | None = None,
) -> np.ndarray | None:
    gradcam_model, target_module = _resolve_gradcam_model_and_module(model, target_module_name)
    if gradcam_model is None or target_module is None:
        return None

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    tensor_hook_handles = []
    parameter_grad_flags: list[tuple[torch.nn.Parameter, bool]] = []

    def activation_gradient_hook(gradient):
        gradients["value"] = gradient

    def forward_hook(_module, _inputs, output):
        expected_ndim = len(target_shape) + 2
        tensor = _first_spatial_tensor(output, expected_ndim=expected_ndim)
        if tensor is not None:
            activations["value"] = tensor
            if tensor.requires_grad:
                tensor_hook_handles.append(tensor.register_hook(activation_gradient_hook))

    forward_handle = target_module.register_forward_hook(forward_hook)
    try:
        original_target_shape = tuple(target_shape)
        grad_image, grad_target_shape, grad_slice_index = _crop_gradcam_volume(
            image.detach(),
            target_shape=original_target_shape,
            cfg=cfg,
            slice_index=slice_index,
        )
        gradcam_model.zero_grad(set_to_none=True)
        parameter_grad_flags = [(parameter, bool(parameter.requires_grad)) for parameter in gradcam_model.parameters()]
        for parameter, _requires_grad in parameter_grad_flags:
            parameter.requires_grad_(False)
        grad_image = grad_image.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = extract_logits(_call_model(gradcam_model, grad_image, slice_index=slice_index), num_classes=num_classes)
            resized = resize_logits(logits, grad_target_shape)
            class_index = 1 if int(num_classes) > 1 and resized.shape[1] > 1 else 0
            score_map = resized[:, class_index]
            if len(grad_target_shape) == 3 and grad_slice_index is not None:
                axis = _visual_slice_axis(cfg, len(grad_target_shape), model_type="3D")
                index = int(np.clip(grad_slice_index, 0, int(grad_target_shape[axis]) - 1))
                score_map = torch.index_select(score_map, dim=axis + 1, index=torch.tensor([index], device=score_map.device))
            score = score_map.amax()
            score.backward()
        activation = activations.get("value")
        gradient = gradients.get("value")
        if activation is None or gradient is None:
            return None
        reduce_dims = tuple(range(2, gradient.ndim))
        weights = gradient.mean(dim=reduce_dims, keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        if cam.ndim != len(grad_target_shape) + 2:
            return None
        mode = "trilinear" if len(grad_target_shape) == 3 else "bilinear"
        cam = F.interpolate(cam, size=grad_target_shape, mode=mode, align_corners=False).squeeze(0).squeeze(0)
        cam_min = torch.min(cam)
        cam_max = torch.max(cam)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        if len(grad_target_shape) == 3 and grad_slice_index is not None and bool(get_nested(cfg, "visualization.gradcam_slice_only", True)):
            axis = _visual_slice_axis(cfg, len(grad_target_shape), model_type="3D")
            index = int(np.clip(grad_slice_index, 0, int(grad_target_shape[axis]) - 1))
            cam = torch.index_select(cam, dim=axis, index=torch.tensor([index], device=cam.device)).squeeze(axis)
        return cam.detach().cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if bool(get_nested(cfg, "visualization.fail_on_gradcam_error", False)):
            raise
        warnings.warn(
            "Grad-CAM skipped because CUDA ran out of memory. "
            "Try lowering visualization.gradcam_depth_window, using one GPU, or setting visualization.save_diagnostics=false.",
            RuntimeWarning,
        )
        return None
    except RuntimeError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if bool(get_nested(cfg, "visualization.fail_on_gradcam_error", False)):
            raise
        detail = "CUDA ran out of memory" if "out of memory" in str(exc).lower() else f"runtime error: {exc}"
        warnings.warn(
            f"Grad-CAM skipped because {detail}. "
            "Try adjusting visualization.gradcam_depth_window or setting visualization.save_diagnostics=false.",
            RuntimeWarning,
        )
        return None
    finally:
        forward_handle.remove()
        for handle in tensor_hook_handles:
            handle.remove()
        for parameter, requires_grad in parameter_grad_flags:
            parameter.requires_grad_(requires_grad)
        gradcam_model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _save_gradcam(path: Path, cam: np.ndarray | None, model_type: str, cfg: Mapping, target_shape: tuple[int, ...], slice_index: int | None) -> None:
    if cam is None:
        cam = np.zeros(target_shape, dtype=np.float32)
    if cam.ndim == 3 and model_type == "3D":
        axis = _visual_slice_axis(cfg, cam.ndim, model_type=model_type)
        index = _scale_slice_index(slice_index, target_shape, tuple(cam.shape), axis)
        cam = np.take(cam, index, axis=axis)
    _save_heatmap(path, cam, cmap="jet", vmin=0.0, vmax=1.0)


@torch.no_grad()
def save_prediction_visuals(model, datasets, output_root: Path, cfg: Mapping, device: torch.device, num_classes: int) -> None:
    model_type = str(get_nested(cfg, "model.type", "2D")).upper()
    seed = int(get_nested(cfg, "visualization.seed", get_nested(cfg, "seed", 42)))
    count = int(get_nested(cfg, "evaluation.save_predictions_per_split", 7))
    selection = str(get_nested(cfg, "visualization.selection", "per_source")).lower()
    save_source_visuals = bool(get_nested(cfg, "visualization.save_per_source", True))
    source_count = int(get_nested(cfg, "visualization.samples_per_source", 1))

    model.eval()
    for split, dataset in datasets.items():
        if selection in {"per_source", "source", "by_source"}:
            prediction_indices = per_source_indices(dataset, source_count, seed)
        else:
            prediction_indices = fixed_indices(len(dataset), count, seed)
        source_indices = per_source_indices(dataset, source_count, seed)
        _save_visual_indices(
            model=model,
            dataset=dataset,
            split=split,
            indices=prediction_indices,
            split_dir=output_root / "predictions" / split,
            cfg=cfg,
            device=device,
            num_classes=num_classes,
            model_type=model_type,
        )
        if save_source_visuals:
            _save_visual_indices(
                model=model,
                dataset=dataset,
                split=split,
                indices=source_indices,
                split_dir=output_root / "visualize" / split,
                cfg=cfg,
                device=device,
                num_classes=num_classes,
                model_type=model_type,
            )


def _slice_axis_for_volume_sample(sample: Mapping[str, Any], label: np.ndarray, cfg: Mapping) -> int:
    image_slices = sample.get("image_slices")
    slice_count = int(image_slices.shape[0]) if hasattr(image_slices, "shape") and len(image_slices.shape) > 0 else None
    candidates = (
        sample.get("slice_axis"),
        get_nested(cfg, "slice_2d.axis", None),
        get_nested(cfg, "visualization.slice_axis", 2),
    )
    for candidate in candidates:
        axis = _axis_candidate(candidate, label.ndim)
        if axis is not None and (slice_count is None or int(label.shape[axis]) == slice_count):
            return axis

    if slice_count is not None:
        matching_axes = [axis for axis, size in enumerate(label.shape) if int(size) == slice_count]
        if matching_axes:
            return int(matching_axes[0])

    axis = _axis_candidate(get_nested(cfg, "visualization.slice_axis", 2), label.ndim)
    return 0 if axis is None else axis


def _slice_index_for_volume_sample(sample: Mapping[str, Any], cfg: Mapping) -> int:
    label = sample["label"].numpy()
    axis = _slice_axis_for_volume_sample(sample, label, cfg)
    position = get_nested(cfg, "visualization.slice_position", "label_foreground")
    slice_index = _display_slice_index(label, pred=None, axis=axis, position=position)
    image_slices = sample.get("image_slices")
    if hasattr(image_slices, "shape") and len(image_slices.shape) > 0:
        slice_index = int(np.clip(slice_index, 0, int(image_slices.shape[0]) - 1))
    return int(slice_index)


def _prepare_visual_sample(sample: Mapping[str, Any], cfg: Mapping, model_type: str, device: torch.device) -> tuple[torch.Tensor, np.ndarray, np.ndarray, tuple[int, ...], int | None]:
    if model_type == "2D" and "image_slices" in sample:
        slice_index = _slice_index_for_volume_sample(sample, cfg)
        image = sample["image_slices"][slice_index].unsqueeze(0).to(device, non_blocking=True)
        label = sample["label_slices"][slice_index].numpy()
        image_for_display = sample["image_slices"][slice_index].numpy()
        return image, label, image_for_display, tuple(label.shape), int(slice_index)

    image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
    label = sample["label"].numpy()
    return image, label, sample["image"].numpy(), tuple(sample["label"].shape), None


def _save_visual_indices(
    model,
    dataset,
    split: str,
    indices: Sequence[int],
    split_dir: Path,
    cfg: Mapping,
    device: torch.device,
    num_classes: int,
    model_type: str,
) -> None:
    panel_dir = split_dir / "panel"
    image_dir = split_dir / "image"
    predict_dir = split_dir / "predict"
    gt_dir = split_dir / "gt"
    logit_dir = split_dir / "logit"
    feature_mean_dir = split_dir / "feature_mean"
    encoder_activation_dir = split_dir / "encoder_activation"
    skip_dir = split_dir / "skip_connection"
    gradcam_bottleneck_dir = split_dir / "gradcam_bottleneck"
    gradcam_decoder_dir = split_dir / "gradcam_decoder"
    diagnostic_dirs = (feature_mean_dir, encoder_activation_dir, skip_dir, gradcam_bottleneck_dir, gradcam_decoder_dir)
    save_diagnostics = bool(get_nested(cfg, "visualization.save_diagnostics", True))
    save_summary_pdf = bool(get_nested(cfg, "visualization.save_summary_pdf", True))
    summary_pdf_path = split_dir / str(get_nested(cfg, "visualization.summary_pdf_name", "image_ground_truth_predict.pdf"))
    max_stages = int(get_nested(cfg, "visualization.max_diagnostic_stages", 5))
    for directory in (panel_dir, image_dir, predict_dir, gt_dir, logit_dir, *(diagnostic_dirs if save_diagnostics else ())):
        directory.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for order, index in enumerate(indices):
        sample = dataset[int(index)]
        image, label, image_for_display, target_shape, selected_slice_index = _prepare_visual_sample(sample, cfg, model_type=model_type, device=device)
        selected_slice_axis = (
            _slice_axis_for_volume_sample(sample, sample["label"].numpy(), cfg)
            if model_type == "2D" and "image_slices" in sample
            else (_visual_slice_axis(cfg, label.ndim, model_type=model_type) if label.ndim == 3 else "")
        )
        visual_slice_index = selected_slice_index if selected_slice_index is not None else sample.get("slice_index")
        if isinstance(visual_slice_index, torch.Tensor):
            visual_slice_index = int(visual_slice_index.reshape(-1)[0].item())
        elif visual_slice_index is not None:
            visual_slice_index = int(visual_slice_index)
        logits = extract_logits(_call_model(model, image, slice_index=visual_slice_index), num_classes=num_classes)
        pred = predict_from_logits(logits, target_shape=target_shape).squeeze(0).cpu().numpy()
        logit = _foreground_logit_from_logits(logits, target_shape=target_shape, num_classes=num_classes)
        del logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

        image_2d, label_2d, pred_2d, logit_2d, display_slice_index = _take_display_slice(
            image_for_display,
            label,
            pred,
            logit,
            model_type=model_type,
            cfg=cfg,
        )
        raw_case_id = str(sample.get("case_id", _case_id_for_index(dataset, int(index))))
        case_id = sanitize_name(raw_case_id)
        source = sanitize_name(source_from_case_id(raw_case_id))
        slice_index = selected_slice_index if selected_slice_index is not None else (sample.get("slice_index", -1) if display_slice_index is None else display_slice_index)
        base_name = f"{source}_{order:02d}_{case_id}_idx{int(index)}_slice{slice_index}"
        _save_panel(panel_dir / f"{base_name}.pdf", image_2d, label_2d, pred_2d, logit_2d)
        _save_slice(image_dir / f"{base_name}.png", image_2d, normalise=True)
        _save_slice(predict_dir / f"{base_name}.png", pred_2d)
        _save_slice(gt_dir / f"{base_name}.png", label_2d)
        _save_logit(logit_dir / f"{base_name}.png", logit_2d)
        summary_rows.append(
            {
                "image": np.asarray(image_2d).copy(),
                "label": np.asarray(label_2d).copy(),
                "pred": np.asarray(pred_2d).copy(),
                "case_id": raw_case_id,
                "slice_index": slice_index,
            }
        )

        gradcam_bottleneck_path = ""
        gradcam_decoder_path = ""
        if save_diagnostics:
            feature_records, skip_records = _collect_visual_features(
                model,
                image,
                num_classes=num_classes,
                model_type=model_type,
                slice_index=visual_slice_index,
            )
            selected_features = _select_unique_features(feature_records, max_stages)
            selected_encoder = _select_unique_features(
                [record for record in feature_records if _is_encoder_name(str(record["name"]))],
                max_stages,
            )
            selected_skips = _select_unique_features(skip_records, max_stages)
            if not selected_skips and selected_encoder:
                selected_skips = selected_encoder[:-1] or selected_encoder

            for stage_index, record in enumerate(selected_features):
                name = sanitize_name(str(record["name"]))
                _save_feature_record(
                    feature_mean_dir / f"{base_name}_feature{stage_index}_{name}.png",
                    record,
                    model_type=model_type,
                    cfg=cfg,
                    target_shape=target_shape,
                    slice_index=slice_index if isinstance(slice_index, int) else None,
                )
            for stage_index, record in enumerate(selected_encoder):
                name = sanitize_name(str(record["name"]))
                _save_feature_record(
                    encoder_activation_dir / f"{base_name}_encoder{stage_index}_{name}.png",
                    record,
                    model_type=model_type,
                    cfg=cfg,
                    target_shape=target_shape,
                    slice_index=slice_index if isinstance(slice_index, int) else None,
                )
            for stage_index, record in enumerate(selected_skips):
                name = sanitize_name(str(record["name"]))
                _save_feature_record(
                    skip_dir / f"{base_name}_skip{stage_index}_{name}.png",
                    record,
                    model_type=model_type,
                    cfg=cfg,
                    target_shape=target_shape,
                    slice_index=slice_index if isinstance(slice_index, int) else None,
                )

            bottleneck_module_name = _target_module_name(model, feature_records, kind="bottleneck")
            decoder_module_name = _target_module_name(model, feature_records, kind="decoder")
            gradcam_slice_index = slice_index if isinstance(slice_index, int) else None
            bottleneck_cam = _compute_gradcam(
                model,
                image,
                target_shape=target_shape,
                num_classes=num_classes,
                target_module_name=bottleneck_module_name,
                cfg=cfg,
                slice_index=gradcam_slice_index,
            )
            decoder_cam = _compute_gradcam(
                model,
                image,
                target_shape=target_shape,
                num_classes=num_classes,
                target_module_name=decoder_module_name,
                cfg=cfg,
                slice_index=gradcam_slice_index,
            )
            gradcam_bottleneck_path = str(gradcam_bottleneck_dir / f"{base_name}.png")
            gradcam_decoder_path = str(gradcam_decoder_dir / f"{base_name}.png")
            _save_gradcam(Path(gradcam_bottleneck_path), bottleneck_cam, model_type=model_type, cfg=cfg, target_shape=target_shape, slice_index=slice_index if isinstance(slice_index, int) else None)
            _save_gradcam(Path(gradcam_decoder_path), decoder_cam, model_type=model_type, cfg=cfg, target_shape=target_shape, slice_index=slice_index if isinstance(slice_index, int) else None)

        index_rows.append(
            {
                "split": split,
                "order": order,
                "index": int(index),
                "case_id": raw_case_id,
                "source": source,
                "slice_index": slice_index,
                "slice_axis": selected_slice_axis,
                "panel_path": str(panel_dir / f"{base_name}.pdf"),
                "image_path": str(image_dir / f"{base_name}.png"),
                "predict_path": str(predict_dir / f"{base_name}.png"),
                "gt_path": str(gt_dir / f"{base_name}.png"),
                "logit_path": str(logit_dir / f"{base_name}.png"),
                "summary_pdf_path": str(summary_pdf_path) if save_summary_pdf else "",
                "feature_mean_dir": str(feature_mean_dir),
                "encoder_activation_dir": str(encoder_activation_dir),
                "skip_connection_dir": str(skip_dir),
                "gradcam_bottleneck_path": gradcam_bottleneck_path,
                "gradcam_decoder_path": gradcam_decoder_path,
            }
        )
    if save_summary_pdf:
        _save_image_gt_predict_pdf(summary_pdf_path, summary_rows, cfg)
    _write_visual_index(split_dir / "visualization_index.csv", index_rows)


def _write_visual_index(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "split",
        "order",
        "index",
        "case_id",
        "source",
        "slice_index",
        "slice_axis",
        "panel_path",
        "image_path",
        "predict_path",
        "gt_path",
        "logit_path",
        "summary_pdf_path",
        "feature_mean_dir",
        "encoder_activation_dir",
        "skip_connection_dir",
        "gradcam_bottleneck_path",
        "gradcam_decoder_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _save_panel(path: Path, image: np.ndarray, label: np.ndarray, pred: np.ndarray, logit: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=150)
    titles = ["Image", "Ground Truth", "Predict", "Logit"]
    arrays = [image, label, pred, logit]
    for axis, title, array in zip(axes, titles, arrays):
        if title == "Logit":
            _show_logit(axis, array)
        else:
            _show_slice(axis, array, normalise=title == "Image")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_image_gt_predict_pdf(path: Path, rows: Sequence[Mapping[str, Any]], cfg: Mapping) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_per_page = max(1, int(get_nested(cfg, "visualization.summary_pdf_rows_per_page", 4)))
    columns = [("Image", "image", True), ("Ground Truth", "label", False), ("Predict", "pred", False)]

    with PdfPages(path) as pdf:
        for start in range(0, len(rows), rows_per_page):
            page_rows = rows[start : start + rows_per_page]
            fig_height = max(2.6, 2.45 * len(page_rows))
            fig, axes = plt.subplots(len(page_rows), len(columns), figsize=(9, fig_height), dpi=150, squeeze=False)
            for row_index, row in enumerate(page_rows):
                for col_index, (title, key, normalise) in enumerate(columns):
                    axis = axes[row_index][col_index]
                    _show_slice(axis, np.asarray(row[key]), normalise=normalise)
                    if row_index == 0:
                        axis.set_title(title)
                    axis.axis("off")
                label = f"{row.get('case_id', '')} | slice {row.get('slice_index', '')}"
                axes[row_index][0].text(
                    0.02,
                    0.98,
                    label,
                    transform=axes[row_index][0].transAxes,
                    ha="left",
                    va="top",
                    fontsize=6,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
                )
            fig.tight_layout(pad=0.45)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _save_slice(path: Path, array: np.ndarray, normalise: bool = False) -> None:
    Image.fromarray(_slice_to_uint8(array, normalise=normalise)).save(path)


def _save_logit(path: Path, logit: np.ndarray) -> None:
    Image.fromarray(_logit_to_rgb(logit)).save(path)


def _save_heatmap(path: Path, heatmap: np.ndarray, cmap: str = "magma", vmin: float | None = None, vmax: float | None = None) -> None:
    values = np.asarray(heatmap, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        values = np.zeros_like(values, dtype=np.float32)
        vmin = 0.0 if vmin is None else vmin
        vmax = 1.0 if vmax is None else vmax
    elif vmin is None or vmax is None:
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            low, high = 0.0, 1.0
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax
    Image.fromarray(_values_to_rgb(values, cmap=cmap, vmin=float(vmin), vmax=float(vmax))).save(path)


def _slice_to_uint8(array: np.ndarray, normalise: bool) -> np.ndarray:
    values = _normalise_for_display(array) if normalise else np.asarray(array, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if not normalise:
        finite = values[np.isfinite(values)]
        vmax = max(1.0, float(np.max(finite))) if finite.size else 1.0
        values = np.clip(values / (vmax + 1e-8), 0.0, 1.0)
    return np.round(values * 255.0).astype(np.uint8)


def _values_to_rgb(values: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=vmax, neginf=vmin)
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = np.clip((values - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    rgb = plt.get_cmap(cmap)(scaled)[..., :3]
    return np.round(rgb * 255.0).astype(np.uint8)


def _logit_to_rgb(logit: np.ndarray) -> np.ndarray:
    values = np.asarray(logit, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        values = np.zeros_like(values, dtype=np.float32)
        limit = 1.0
    else:
        limit = float(np.percentile(np.abs(finite), 99))
        if limit <= 1e-8:
            limit = max(1.0, float(np.max(np.abs(finite))))
    return _values_to_rgb(values, cmap="coolwarm", vmin=-limit, vmax=limit)


def _show_slice(axis, array: np.ndarray, normalise: bool) -> None:
    image = _normalise_for_display(array) if normalise else np.asarray(array)
    vmax = 1.0 if normalise else max(1.0, float(np.max(image)))
    axis.imshow(image, cmap="gray", vmin=0.0, vmax=vmax, aspect="equal", interpolation="nearest")
    axis.set_aspect("equal", adjustable="box")


def _show_logit(axis, logit: np.ndarray) -> None:
    values = np.asarray(logit, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        values = np.zeros_like(values, dtype=np.float32)
        limit = 1.0
    else:
        limit = float(np.percentile(np.abs(finite), 99))
        if limit <= 1e-8:
            limit = max(1.0, float(np.max(np.abs(finite))))
    axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="equal", interpolation="nearest")
    axis.set_aspect("equal", adjustable="box")
