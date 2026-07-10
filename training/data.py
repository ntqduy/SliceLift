from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from .utils import display_path, get_nested, save_json


_CENTER_SLICE_TOKENS = {"center", "centre", "middle", "mid", "single", "single_center", "center_slice", "middle_slice"}
_GROUP_BASED_LARGEST_ORDER_TOKENS = {"largest", "high", "higher", "different", "most_different", "diverse", "farthest"}
_GROUP_BASED_SMALLEST_ORDER_TOKENS = {"smallest", "low", "lower", "closest", "similar", "most_similar", "mean", "nearest_mean"}


def _is_group_based_strategy(strategy: Any) -> bool:
    return str(strategy or "").lower().replace("-", "_") == "group_based"


@dataclass(frozen=True)
class CystRecord:
    image_path: Path
    mask_path: Path
    case_id: str


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def read_records(root: Path, list_file: str) -> List[CystRecord]:
    path = root / list_file
    records: List[CystRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_value = row.get("image_path") or row.get("image") or row.get("image_file")
            mask_value = row.get("mask_path") or row.get("mask") or row.get("label_path") or row.get("label")
            if not image_value or not mask_value:
                raise ValueError(f"{path} must contain image_path and mask_path columns.")
            image_path = _resolve_path(root, image_value.strip())
            mask_path = _resolve_path(root, mask_value.strip())
            records.append(CystRecord(image_path=image_path, mask_path=mask_path, case_id=image_path.stem.replace(".nii", "")))
    return records


def _record_key(record: CystRecord) -> str:
    return str(record.image_path.resolve()).lower()


def _safe_split_token(value: str) -> str:
    text = Path(str(value)).stem or str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _list_file_sha1(root: Path, list_file: str) -> str:
    path = root / list_file
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kfold_split_dir(root: Path, source_list: str, test_list: str, num_folds: int, seed: int, shuffle: bool) -> Path:
    source_token = _safe_split_token(source_list)
    test_token = _safe_split_token(test_list)
    shuffle_token = "shuffle" if shuffle else "ordered"
    return root / "data_fold" / f"{source_token}__{test_token}__k{int(num_folds)}__seed{int(seed)}__{shuffle_token}"


def _save_fold_splits_to_disk(
    fold_dir: Path, 
    splits: List[Dict[str, Any]], 
    metadata: Dict[str, Any],
    root: Path
) -> None:
    """Save k-fold splits to disk in data_fold/ directory."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata_file = fold_dir / "config_metadata.json"
    save_json(metadata_file, metadata)
    
    # Save each fold's train/val lists
    for fold_index, fold_data in enumerate(splits):
        fold_path = fold_dir / f"fold_{fold_index}"
        fold_path.mkdir(parents=True, exist_ok=True)
        
        # Save train.txt
        train_file = fold_path / "train.txt"
        train_records = fold_data["train"]
        _save_records_to_csv(train_file, train_records, root)
        
        # Save val.txt
        val_file = fold_path / "val.txt"
        val_records = fold_data["val"]
        _save_records_to_csv(val_file, val_records, root)
    
    # Save test.txt (shared for all folds)
    if splits:
        test_file = fold_dir / "test.txt"
        test_records = splits[0]["test"]
        _save_records_to_csv(test_file, test_records, root)


def _save_records_to_csv(file_path: Path, records: List[CystRecord], root: Path) -> None:
    """Save records to CSV file (compatible with read_records format)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "mask_path"], lineterminator="\n")
        writer.writeheader()
        for record in records:
            try:
                image_rel = record.image_path.relative_to(root)
                mask_rel = record.mask_path.relative_to(root)
            except ValueError:
                image_rel = record.image_path
                mask_rel = record.mask_path
            writer.writerow({
                "image_path": str(image_rel).replace("\\", "/"),
                "mask_path": str(mask_rel).replace("\\", "/"),
            })


def _load_fold_splits_from_disk(
    fold_dir: Path,
    root: Path,
    num_folds: int
) -> tuple[List[Dict[str, Any]], Dict[str, Any]] | None:
    """Load k-fold splits from disk. Returns None if not found or invalid."""
    metadata_file = fold_dir / "config_metadata.json"

    try:
        saved_metadata: Dict[str, Any] = {}
        if metadata_file.exists():
            with metadata_file.open("r", encoding="utf-8") as handle:
                loaded_metadata = json.load(handle)
            if isinstance(loaded_metadata, dict):
                saved_metadata = loaded_metadata
        
        # Check if all required fold files exist
        for fold_index in range(num_folds):
            fold_path = fold_dir / f"fold_{fold_index}"
            if not (fold_path / "train.txt").exists() or not (fold_path / "val.txt").exists():
                return None
        
        # Load all folds
        splits = []
        test_records = None
        
        for fold_index in range(num_folds):
            fold_path = fold_dir / f"fold_{fold_index}"
            train_records = read_records(root, str((fold_path / "train.txt").relative_to(root)))
            val_records = read_records(root, str((fold_path / "val.txt").relative_to(root)))
            
            # Load test records (same for all folds)
            if test_records is None:
                test_file = fold_dir / "test.txt"
                if test_file.exists():
                    test_records = read_records(root, str(test_file.relative_to(root)))
                else:
                    test_records = []
            
            splits.append({
                "train": train_records,
                "val": val_records,
                "test": test_records
            })
        
        return splits, saved_metadata
    except Exception:
        return None


def _config_matches_metadata(cfg: Mapping[str, Any], metadata: Dict[str, Any]) -> bool:
    """Check if current k-fold config matches saved metadata."""
    kfold_cfg = dict(get_nested(cfg, "k_fold", {}) or {})
    source_list = str(kfold_cfg.get("source_list", get_nested(cfg, "dataset.all_train_list", "all_train.txt")))
    test_list = str(get_nested(cfg, "dataset.test_list", "test.txt"))
    num_folds = int(kfold_cfg.get("num_folds", 10))
    shuffle = bool(kfold_cfg.get("shuffle", True))
    seed = int(kfold_cfg.get("seed", get_nested(cfg, "seed", 42)))
    root = Path(get_nested(cfg, "dataset.root", "data"))
    expected_source_sha1 = _list_file_sha1(root, source_list)
    expected_test_sha1 = _list_file_sha1(root, test_list)
    
    return (
        metadata.get("source_list") == source_list
        and metadata.get("test_list") == test_list
        and metadata.get("num_folds") == num_folds
        and metadata.get("shuffle") == shuffle
        and metadata.get("seed") == seed
        and metadata.get("source_sha1", expected_source_sha1) == expected_source_sha1
        and metadata.get("test_sha1", expected_test_sha1) == expected_test_sha1
    )


def build_kfold_record_splits(cfg: Mapping[str, Any]):
    root = Path(get_nested(cfg, "dataset.root", "data"))
    kfold_cfg = dict(get_nested(cfg, "k_fold", {}) or {})
    source_list = str(kfold_cfg.get("source_list", get_nested(cfg, "dataset.all_train_list", "all_train.txt")))
    test_list = str(get_nested(cfg, "dataset.test_list", "test.txt"))
    num_folds = int(kfold_cfg.get("num_folds", 10))
    seed = int(kfold_cfg.get("seed", get_nested(cfg, "seed", 42)))
    reuse_existing = bool(kfold_cfg.get("reuse_existing", False))
    
    if num_folds < 2:
        raise ValueError("k_fold.num_folds must be >= 2.")

    shuffle = bool(kfold_cfg.get("shuffle", True))
    fold_dir = _kfold_split_dir(root, source_list, test_list, num_folds, seed, shuffle)
    legacy_fold_dir = root / "data_fold" / str(seed)
    
    # Try to load a saved split first so all models use the same folds.
    for candidate_dir in (fold_dir, legacy_fold_dir):
        if not candidate_dir.exists():
            continue
        loaded = _load_fold_splits_from_disk(candidate_dir, root, num_folds)
        if loaded is not None:
            splits, saved_metadata = loaded
            metadata_matches = _config_matches_metadata(cfg, saved_metadata)
            if metadata_matches or reuse_existing:
                metadata = dict(saved_metadata)
                metadata.setdefault("source_list", source_list)
                metadata.setdefault("test_list", test_list)
                metadata.setdefault("num_folds", num_folds)
                metadata.setdefault("shuffle", shuffle)
                metadata.setdefault("seed", seed)
                metadata.setdefault("records", len(splits[0]["train"]) + len(splits[0]["val"]) if splits else 0)
                metadata.setdefault("test_records", len(splits[0]["test"]) if splits else 0)
                metadata["split_dir"] = str(candidate_dir)
                metadata["reused_existing_split"] = True
                metadata["metadata_matched"] = bool(metadata_matches)
                return splits, metadata
    
    # Calculate new splits if not found or config mismatch
    all_records = read_records(root, source_list)
    if len(all_records) < num_folds:
        raise ValueError(f"k-fold needs at least {num_folds} records, got {len(all_records)} from {source_list}.")

    test_records = read_records(root, test_list)
    train_keys = {_record_key(record) for record in all_records}
    test_keys = {_record_key(record) for record in test_records}
    overlap = sorted(train_keys & test_keys)
    if overlap:
        raise ValueError(f"Test leakage detected: {len(overlap)} case(s) appear in both {source_list} and {test_list}.")

    indices = np.arange(len(all_records))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    folds = [np.asarray(items, dtype=int) for items in np.array_split(indices, num_folds)]
    splits = []
    for fold_index, val_indices in enumerate(folds):
        val_set = set(int(item) for item in val_indices.tolist())
        train_records = [record for index, record in enumerate(all_records) if index not in val_set]
        val_records = [all_records[int(index)] for index in val_indices.tolist()]
        splits.append({"train": train_records, "val": val_records, "test": list(test_records)})

    metadata = {
        "source_list": source_list,
        "test_list": test_list,
        "num_folds": num_folds,
        "shuffle": shuffle,
        "seed": seed,
        "records": len(all_records),
        "test_records": len(test_records),
        "source_sha1": _list_file_sha1(root, source_list),
        "test_sha1": _list_file_sha1(root, test_list),
        "split_dir": str(fold_dir),
    }
    
    # Save splits to data/data_fold/<source>__<test>__k<num_folds>__seed<seed>__<shuffle>/ for future use.
    _save_fold_splits_to_disk(fold_dir, splits, metadata, root)
    
    return splits, metadata


def _normalize_image(image: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "foreground_zscore").lower()
    image = image.astype(np.float32, copy=False)
    if mode in {"none", "identity"}:
        return image
    if mode in {"minmax", "min_max"}:
        min_value = float(np.min(image))
        max_value = float(np.max(image))
        return (image - min_value) / (max_value - min_value + 1e-8)

    if mode in {"foreground_zscore", "foreground"}:
        foreground = image > 0
        values = image[foreground] if np.any(foreground) else image.reshape(-1)
    elif mode in {"zscore", "standard"}:
        values = image.reshape(-1)
    else:
        raise ValueError(f"Unsupported intensity_normalization: {mode}")

    mean = float(np.mean(values))
    std = float(np.std(values))
    return (image - mean) / (std + 1e-8)


def _prepare_label(label: np.ndarray, num_classes: int, binarize: bool) -> np.ndarray:
    if binarize and num_classes == 2:
        return (label > 0).astype(np.int64)
    label = np.rint(label).astype(np.int64)
    return np.clip(label, 0, num_classes - 1)


def _resize_array(array: np.ndarray, output_size: Sequence[int], order: int) -> np.ndarray:
    output_size = tuple(int(item) for item in output_size)
    if tuple(array.shape) == output_size:
        return array
    factors = [target / current for target, current in zip(output_size, array.shape)]
    return zoom(array, factors, order=order)


def _resize_volume_inplane(array: np.ndarray, output_size: Sequence[int], depth_axis: int, order: int) -> np.ndarray:
    """Resize only in-plane H/W and return a [D, H, W] volume."""
    output_size = tuple(int(item) for item in output_size[:2])
    if len(output_size) != 2:
        raise ValueError("In-plane resize expects image_size [height, width].")
    depth_axis = int(np.clip(int(depth_axis), 0, array.ndim - 1))
    slices = np.moveaxis(array, depth_axis, 0)
    resized = [_resize_array(slice_item, output_size, order=order) for slice_item in slices]
    return np.stack(resized, axis=0)


def _slice_to_index(position: Any, depth: int) -> int:
    if isinstance(position, str):
        key = position.lower()
        if key in _CENTER_SLICE_TOKENS:
            return depth // 2
        if key == "first":
            return 0
        if key == "last":
            return depth - 1
        try:
            position = float(position)
        except ValueError as error:
            raise ValueError(f"Unsupported slice position: {position}") from error
    if isinstance(position, float) and 0.0 <= position <= 1.0:
        return int(round(position * (depth - 1)))
    return int(np.clip(int(position), 0, depth - 1))


def _group_based_slice_cfg(slice_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    group_cfg: Dict[str, Any] = {}
    for key in ("group_based", "group-based"):
        value = slice_cfg.get(key, {})
        if isinstance(value, Mapping):
            group_cfg.update(dict(value))
    return group_cfg


def _group_based_positions_from_volume(volume: np.ndarray, slice_cfg: Mapping[str, Any], axis: int) -> List[int]:
    depth = int(volume.shape[axis])
    if depth <= 0:
        raise ValueError("Cannot select group-based slices from a volume with depth=0.")
    group_cfg = _group_based_slice_cfg(slice_cfg)
    fallback_count = max(1, int(slice_cfg.get("samples_per_volume", slice_cfg.get("num_slices", 1))))
    num_groups = max(1, int(group_cfg.get("num_groups", slice_cfg.get("num_groups", fallback_count))))
    samples_per_group = max(1, int(group_cfg.get("samples_per_group", slice_cfg.get("samples_per_group", 1))))
    samples_per_volume = num_groups * samples_per_group
    similarity_metric = str(group_cfg.get("similarity_metric", slice_cfg.get("similarity_metric", "mad"))).lower()
    selection_order = str(group_cfg.get("selection_order", slice_cfg.get("selection_order", "closest"))).lower().replace("-", "_")
    if selection_order in _GROUP_BASED_LARGEST_ORDER_TOKENS:
        select_largest = True
    elif selection_order in _GROUP_BASED_SMALLEST_ORDER_TOKENS:
        select_largest = False
    else:
        raise ValueError(f"Unsupported group-based selection_order: {selection_order!r}")
    slices = np.moveaxis(np.asarray(volume, dtype=np.float32), int(axis), 0)
    boundaries = np.rint(np.linspace(0, depth, num=min(num_groups, depth) + 1)).astype(np.int64)
    chosen: List[int] = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            continue
        group = slices[start_i:end_i]
        if group.shape[0] == 1:
            scores = np.ones(1, dtype=np.float32)
        elif similarity_metric in {"cos", "cosine", "cosine_similarity"}:
            flattened = group.reshape(group.shape[0], -1).astype(np.float32)
            norms = np.linalg.norm(flattened, axis=1, keepdims=True)
            normalized = flattened / np.maximum(norms, 1e-6)
            scores = -(normalized @ normalized.T).mean(axis=1)
        elif similarity_metric in {"mad", "mean_absolute_deviation"}:
            center = group.mean(axis=0, keepdims=True)
            scores = np.abs(group - center).reshape(group.shape[0], -1).mean(axis=1)
        else:
            raise ValueError(f"Unsupported group-based similarity_metric: {similarity_metric}")
        take = min(samples_per_group, end_i - start_i)
        local = np.argsort(-scores if select_largest else scores)[:take]
        group_chosen = [start_i + int(item) for item in local]
        while len(group_chosen) < samples_per_group:
            group_chosen.append(group_chosen[-1] if group_chosen else (start_i + end_i - 1) // 2)
        chosen.extend(group_chosen)

    if len(chosen) < samples_per_volume:
        fallback = np.rint(np.linspace(0, depth - 1, num=samples_per_volume)).astype(np.int64).tolist()
        for item in fallback:
            index = int(np.clip(item, 0, depth - 1))
            if index not in chosen:
                chosen.append(index)
            if len(chosen) >= samples_per_volume:
                break
        for item in fallback:
            if len(chosen) >= samples_per_volume:
                break
            chosen.append(int(np.clip(item, 0, depth - 1)))

    if not chosen:
        chosen = [depth // 2]
    while len(chosen) < samples_per_volume:
        chosen.append(chosen[-1])
    return sorted(int(np.clip(item, 0, depth - 1)) for item in chosen[:samples_per_volume])


def _positions_from_config(
    slice_cfg: Mapping[str, Any],
    depth: int,
    rng: np.random.Generator,
    volume: np.ndarray | None = None,
    axis: int = 2,
) -> List[int]:
    strategy = str(slice_cfg.get("sampling_strategy", "center")).lower().replace("-", "_")
    samples_per_volume = max(1, int(slice_cfg.get("samples_per_volume", 1)))
    position = slice_cfg.get("position", "center")

    if strategy == "all":
        return list(range(depth))
    if strategy == "uniform":
        if samples_per_volume >= depth:
            return list(range(depth))
        return sorted({int(round(item)) for item in np.linspace(0, depth - 1, samples_per_volume)})
    if strategy == "random":
        replace = samples_per_volume > depth
        return sorted(int(item) for item in rng.choice(depth, samples_per_volume, replace=replace))
    if strategy in {"fixed", "positions"}:
        positions = position if isinstance(position, list) else [position]
        return sorted({_slice_to_index(item, depth) for item in positions})
    if strategy in _CENTER_SLICE_TOKENS:
        return [_slice_to_index(position, depth)]
    if _is_group_based_strategy(strategy):
        if volume is None:
            raise ValueError("2D group-based sampling requires the image volume when building slice indices.")
        return _group_based_positions_from_volume(volume, slice_cfg, axis=axis)
    raise ValueError(f"Unsupported 2D sampling_strategy: {strategy}")


def _channel_window(center: int, depth: int, num_slices: int) -> List[int]:
    num_slices = max(1, int(num_slices))
    start = center - (num_slices // 2)
    return [int(np.clip(start + offset, 0, depth - 1)) for offset in range(num_slices)]


def _load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def _extract_2d_sample(
    image: np.ndarray,
    label: np.ndarray,
    slice_index: int,
    axis: int,
    in_channels: int,
    image_size: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    depth = image.shape[axis]
    channel_indices = _channel_window(slice_index, depth, in_channels)
    image_slice = np.take(image, channel_indices, axis=axis)
    image_slice = np.moveaxis(image_slice, axis, 0)
    label_slice = np.take(label, slice_index, axis=axis)

    image_slice = np.stack([_resize_array(channel, image_size, order=1) for channel in image_slice], axis=0)
    label_slice = _resize_array(label_slice, image_size, order=0)
    return image_slice, label_slice


def stack_slices_as_volume(slices: Sequence[np.ndarray] | np.ndarray, axis: int) -> np.ndarray:
    return np.stack([np.asarray(item) for item in slices], axis=int(axis))


class CystSliceDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CystRecord],
        split: str,
        image_size: Sequence[int],
        slice_cfg: Mapping[str, Any],
        num_classes: int,
        binarize_masks: bool,
        normalization: str,
        seed: int,
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.split = split
        self.image_size = tuple(int(item) for item in image_size[:2])
        self.slice_cfg = dict(slice_cfg)
        self.num_classes = int(num_classes)
        self.binarize_masks = bool(binarize_masks)
        self.normalization = normalization
        self.seed = int(seed)
        self.augment = bool(augment)
        self.axis = int(self.slice_cfg.get("axis", 2))
        self.in_channels = max(1, int(self.slice_cfg.get("num_slices", 1)))
        self.index: List[tuple[int, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        rng = np.random.default_rng(self.seed + {"train": 0, "val": 1, "test": 2}.get(self.split, 3))
        for record_index, record in enumerate(self.records):
            shape = nib.load(str(record.image_path)).shape
            depth = int(shape[self.axis])
            volume = None
            if _is_group_based_strategy(self.slice_cfg.get("sampling_strategy", "center")):
                volume = _normalize_image(_load_nifti(record.image_path), self.normalization)
            for slice_index in _positions_from_config(self.slice_cfg, depth, rng, volume=volume, axis=self.axis):
                self.index.append((record_index, slice_index))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        record_index, slice_index = self.index[item]
        record = self.records[record_index]
        image = _normalize_image(_load_nifti(record.image_path), self.normalization)
        label = _prepare_label(_load_nifti(record.mask_path), self.num_classes, self.binarize_masks)

        image, label = _extract_2d_sample(
            image=image,
            label=label,
            slice_index=slice_index,
            axis=self.axis,
            in_channels=self.in_channels,
            image_size=self.image_size,
        )

        if self.augment:
            image, label = self._augment(image, label)

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "case_id": record.case_id,
            "image_path": display_path(record.image_path),
            "mask_path": display_path(record.mask_path),
            "slice_index": int(slice_index),
            "slice_axis": int(self.axis),
            "split": self.split,
        }

    @staticmethod
    def _augment(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=0)
        if random.random() < 0.5:
            image = np.flip(image, axis=2)
            label = np.flip(label, axis=1)
        if random.random() < 0.5:
            turns = random.randint(0, 3)
            image = np.rot90(image, turns, axes=(1, 2))
            label = np.rot90(label, turns, axes=(0, 1))
        return image.copy(), label.copy()


class CystSliceVolumeDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CystRecord],
        split: str,
        image_size: Sequence[int],
        slice_cfg: Mapping[str, Any],
        num_classes: int,
        binarize_masks: bool,
        normalization: str,
    ) -> None:
        self.records = list(records)
        self.split = split
        self.image_size = tuple(int(item) for item in image_size[:2])
        self.slice_cfg = dict(slice_cfg)
        self.num_classes = int(num_classes)
        self.binarize_masks = bool(binarize_masks)
        self.normalization = normalization
        self.axis = int(self.slice_cfg.get("axis", 2))
        self.in_channels = max(1, int(self.slice_cfg.get("num_slices", 1)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        record = self.records[item]
        image = _normalize_image(_load_nifti(record.image_path), self.normalization)
        label = _prepare_label(_load_nifti(record.mask_path), self.num_classes, self.binarize_masks)

        image_slices: List[np.ndarray] = []
        label_slices: List[np.ndarray] = []
        for slice_index in range(int(image.shape[self.axis])):
            image_slice, label_slice = _extract_2d_sample(
                image=image,
                label=label,
                slice_index=slice_index,
                axis=self.axis,
                in_channels=self.in_channels,
                image_size=self.image_size,
            )
            image_slices.append(image_slice.astype(np.float32))
            label_slices.append(label_slice.astype(np.int64))

        label_volume = stack_slices_as_volume(label_slices, self.axis)
        return {
            "image_slices": torch.from_numpy(np.stack(image_slices, axis=0).astype(np.float32)),
            "label_slices": torch.from_numpy(np.stack(label_slices, axis=0).astype(np.int64)),
            "label": torch.from_numpy(label_volume.astype(np.int64)),
            "case_id": record.case_id,
            "image_path": display_path(record.image_path),
            "mask_path": display_path(record.mask_path),
            "slice_index": -1,
            "slice_axis": int(self.axis),
            "split": self.split,
        }


class CystVolumeDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CystRecord],
        split: str,
        image_size: Sequence[int],
        num_classes: int,
        binarize_masks: bool,
        normalization: str,
        preserve_depth: bool = False,
        depth_axis: int = 2,
        volume_layout: str = "HWD",
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.split = split
        self.preserve_depth = bool(preserve_depth)
        self.depth_axis = int(depth_axis)
        self.volume_layout = str(volume_layout or "HWD").upper()
        if self.volume_layout not in {"HWD", "DHW"}:
            raise ValueError("training.volume_layout must be HWD or DHW.")
        self.image_size = tuple(int(item) for item in image_size[:2]) if self.preserve_depth else tuple(int(item) for item in image_size[:3])
        self.num_classes = int(num_classes)
        self.binarize_masks = bool(binarize_masks)
        self.normalization = normalization
        self.augment = bool(augment)
        self.in_channels = 1

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        record = self.records[item]
        image = _normalize_image(_load_nifti(record.image_path), self.normalization)
        label = _prepare_label(_load_nifti(record.mask_path), self.num_classes, self.binarize_masks)
        if self.preserve_depth:
            image = _resize_volume_inplane(image, self.image_size, depth_axis=self.depth_axis, order=1)
            label = _resize_volume_inplane(label, self.image_size, depth_axis=self.depth_axis, order=0)
            if self.volume_layout == "HWD":
                image = np.moveaxis(image, 0, self.depth_axis)
                label = np.moveaxis(label, 0, self.depth_axis)
        else:
            image = _resize_array(image, self.image_size, order=1)
            label = _resize_array(label, self.image_size, order=0)
            if self.volume_layout == "DHW":
                image = np.moveaxis(image, self.depth_axis, 0)
                label = np.moveaxis(label, self.depth_axis, 0)
        if self.augment:
            image, label = self._augment(image, label)
        image = np.expand_dims(image, axis=0)
        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "case_id": record.case_id,
            "image_path": display_path(record.image_path),
            "mask_path": display_path(record.mask_path),
            "slice_index": -1,
            "split": self.split,
        }

    @staticmethod
    def _augment(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for axis in range(3):
            if random.random() < 0.5:
                image = np.flip(image, axis=axis)
                label = np.flip(label, axis=axis)
        return image.copy(), label.copy()


def _build_datasets_from_records(cfg: Mapping[str, Any], records: Mapping[str, Sequence[CystRecord]], augment_train: bool = True):
    model_type = str(get_nested(cfg, "model.type", "2D")).upper()
    num_classes = int(get_nested(cfg, "model.num_classes", get_nested(cfg, "dataset.num_classes", 2)))
    binarize_masks = bool(get_nested(cfg, "dataset.binarize_masks", num_classes == 2))
    normalization = str(get_nested(cfg, "dataset.intensity_normalization", "foreground_zscore"))
    image_size = get_nested(cfg, "training.image_size", [256, 256])
    preserve_depth = bool(get_nested(cfg, "training.preserve_depth", get_nested(cfg, "dataset.preserve_depth", False)))
    if model_type == "2D" and len(image_size) < 2:
        raise ValueError("training.image_size must contain [height, width] for 2D models.")
    if model_type == "3D" and preserve_depth and len(image_size) < 2:
        raise ValueError("training.image_size must contain [height, width] when training.preserve_depth=true.")
    if model_type == "3D" and not preserve_depth and len(image_size) < 3:
        raise ValueError("training.image_size must contain [height, width, depth] for 3D models.")
    seed = int(get_nested(cfg, "seed", 42))
    augmentation_enabled = bool(get_nested(cfg, "augmentation.enabled", True))

    datasets = {}
    for split, split_records in records.items():
        split_augment = split == "train" and augment_train and augmentation_enabled
        if model_type == "2D":
            datasets[split] = CystSliceDataset(
                split_records,
                split=split,
                image_size=image_size,
                slice_cfg=get_nested(cfg, "slice_2d", {}),
                num_classes=num_classes,
                binarize_masks=binarize_masks,
                normalization=normalization,
                seed=seed,
                augment=split_augment,
            )
        elif model_type == "3D":
            datasets[split] = CystVolumeDataset(
                split_records,
                split=split,
                image_size=image_size,
                num_classes=num_classes,
                binarize_masks=binarize_masks,
                normalization=normalization,
                preserve_depth=preserve_depth,
                depth_axis=int(get_nested(cfg, "training.depth_axis", get_nested(cfg, "slice_2d.axis", 2))),
                volume_layout=str(get_nested(cfg, "training.volume_layout", "HWD")),
                augment=split_augment,
            )
        else:
            raise ValueError(f"model.type must be 2D or 3D, got {model_type}")
    return datasets, records


def build_datasets_from_records(cfg: Mapping[str, Any], records: Mapping[str, Sequence[CystRecord]], augment_train: bool = True):
    return _build_datasets_from_records(cfg, records, augment_train=augment_train)


def build_datasets(cfg: Mapping[str, Any], augment_train: bool = True):
    root = Path(get_nested(cfg, "dataset.root", "data"))
    split_files = {
        "train": str(get_nested(cfg, "dataset.train_list", "train.txt")),
        "val": str(get_nested(cfg, "dataset.val_list", "val.txt")),
        "test": str(get_nested(cfg, "dataset.test_list", "test.txt")),
    }
    records = {split: read_records(root, filename) for split, filename in split_files.items()}
    return _build_datasets_from_records(cfg, records, augment_train=augment_train)


def build_2d_volume_eval_datasets(cfg: Mapping[str, Any], records: Mapping[str, Sequence[CystRecord]]):
    num_classes = int(get_nested(cfg, "model.num_classes", get_nested(cfg, "dataset.num_classes", 2)))
    binarize_masks = bool(get_nested(cfg, "dataset.binarize_masks", num_classes == 2))
    normalization = str(get_nested(cfg, "dataset.intensity_normalization", "foreground_zscore"))
    image_size = get_nested(cfg, "training.image_size", [256, 256])
    if len(image_size) < 2:
        raise ValueError("training.image_size must contain [height, width] for 2D volume evaluation.")
    slice_cfg = get_nested(cfg, "slice_2d", {})

    return {
        split: CystSliceVolumeDataset(
            split_records,
            split=split,
            image_size=image_size,
            slice_cfg=slice_cfg,
            num_classes=num_classes,
            binarize_masks=binarize_masks,
            normalization=normalization,
        )
        for split, split_records in records.items()
    }
