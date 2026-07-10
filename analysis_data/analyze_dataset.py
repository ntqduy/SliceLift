from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class SampleRecord:
    split: str
    index: int
    image_path: Path
    mask_path: Path
    case_id: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cyst segmentation split files and NIfTI masks.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Dataset root containing train/val/test txt files.")
    parser.add_argument("--output-root", type=Path, default=Path("outpus_analysis"), help="Analysis output root.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional output subfolder name.")
    parser.add_argument("--splits", nargs="+", default=["train_new", "val_new", "test"], help="Split names to analyze.")
    parser.add_argument("--num-visuals", type=int, default=7, help="Number of 2D previews per split when --visual-selection=fixed.")
    parser.add_argument(
        "--visual-selection",
        choices=["per_source", "fixed"],
        default="per_source",
        help="Visualization selection mode. per_source saves one sample per source in each split.",
    )
    parser.add_argument("--visuals-per-source", type=int, default=1, help="Number of visualization samples per source.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for fixed visualization sample selection.")
    parser.add_argument("--slice-axis", type=int, default=2, help="Axis used for 2D slice visualization.")
    parser.add_argument("--label-threshold", type=float, default=0.0, help="Mask values above this threshold are treated as label.")
    parser.add_argument("--skip-visualization", action="store_true", help="Only write CSV/JSON summaries.")
    return parser.parse_args()


def case_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def source_from_case_id(case_id: str) -> str:
    match = re.match(r"([A-Za-z]+)", str(case_id).strip())
    return match.group(1).upper() if match else "UNKNOWN"


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value.strip())
    if path.is_absolute():
        return path
    return root / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def read_split_file(data_root: Path, split: str) -> List[SampleRecord]:
    split_file = data_root / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")

    records: List[SampleRecord] = []
    with split_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            image_value = row.get("image_path") or row.get("image") or row.get("image_file")
            mask_value = row.get("mask_path") or row.get("mask") or row.get("label_path") or row.get("label")
            if not image_value or not mask_value:
                raise ValueError(f"{split_file} must contain image_path and mask_path columns.")
            image_path = resolve_path(data_root, image_value)
            mask_path = resolve_path(data_root, mask_value)
            case_id = case_id_from_path(image_path)
            records.append(
                SampleRecord(
                    split=split,
                    index=index,
                    image_path=image_path,
                    mask_path=mask_path,
                    case_id=case_id,
                    source=source_from_case_id(case_id),
                )
            )
    return records


def shape_to_text(shape: Sequence[int] | None) -> str:
    if not shape:
        return ""
    return "x".join(str(int(item)) for item in shape)


def spatial_shape(shape: Sequence[int] | None) -> tuple[int, ...]:
    if not shape:
        return ()
    values = tuple(int(item) for item in shape)
    while len(values) > 3 and values[-1] == 1:
        values = values[:-1]
    return values[:3]


def shape_voxels(shape: Sequence[int] | None) -> int:
    if not shape:
        return 0
    return int(np.prod([int(item) for item in shape]))


def spacing_to_text(zooms: Sequence[float] | None, ndim: int | None) -> str:
    if not zooms or not ndim:
        return ""
    values = zooms[:ndim]
    return "x".join(f"{float(item):.6g}" for item in values)


def load_mask_array(mask_img: nib.Nifti1Image) -> np.ndarray:
    mask = np.asarray(mask_img.dataobj)
    mask = np.squeeze(mask)
    if mask.ndim > 3:
        mask = mask[..., 0]
    return mask


def load_image_array(image_img: nib.Nifti1Image) -> np.ndarray:
    image = np.asarray(image_img.dataobj, dtype=np.float32)
    image = np.squeeze(image)
    if image.ndim > 3:
        image = image[..., 0]
    return image


def analyze_record(record: SampleRecord, label_threshold: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "split": record.split,
        "index": record.index,
        "case_id": record.case_id,
        "source": record.source,
        "image_path": display_path(record.image_path),
        "mask_path": display_path(record.mask_path),
        "image_exists": record.image_path.exists(),
        "mask_exists": record.mask_path.exists(),
        "image_shape": "",
        "mask_shape": "",
        "image_dim0": "",
        "image_dim1": "",
        "image_dim2": "",
        "mask_dim0": "",
        "mask_dim1": "",
        "mask_dim2": "",
        "image_voxels": 0,
        "mask_voxels": 0,
        "image_spacing": "",
        "mask_spacing": "",
        "shape_match": False,
        "has_label": False,
        "positive_voxels": 0,
        "total_voxels": 0,
        "positive_ratio": 0.0,
        "error": "",
    }

    try:
        if row["image_exists"]:
            image_img = nib.load(str(record.image_path))
            image_shape = tuple(int(item) for item in image_img.shape)
            image_spatial_shape = spatial_shape(image_shape)
            row["image_shape"] = shape_to_text(image_spatial_shape)
            for dim_index, dim_value in enumerate(image_spatial_shape[:3]):
                row[f"image_dim{dim_index}"] = int(dim_value)
            row["image_voxels"] = shape_voxels(image_spatial_shape)
            row["image_spacing"] = spacing_to_text(image_img.header.get_zooms(), len(image_spatial_shape))

        if row["mask_exists"]:
            mask_img = nib.load(str(record.mask_path))
            mask_shape = tuple(int(item) for item in mask_img.shape)
            mask_spatial_shape = spatial_shape(mask_shape)
            row["mask_shape"] = shape_to_text(mask_spatial_shape)
            for dim_index, dim_value in enumerate(mask_spatial_shape[:3]):
                row[f"mask_dim{dim_index}"] = int(dim_value)
            row["mask_voxels"] = shape_voxels(mask_spatial_shape)
            row["mask_spacing"] = spacing_to_text(mask_img.header.get_zooms(), len(mask_spatial_shape))
            mask = load_mask_array(mask_img)
            positive_voxels = int(np.count_nonzero(mask > label_threshold))
            total_voxels = int(mask.size)
            row["positive_voxels"] = positive_voxels
            row["total_voxels"] = total_voxels
            row["has_label"] = positive_voxels > 0
            row["positive_ratio"] = positive_voxels / total_voxels if total_voxels else 0.0

        row["shape_match"] = bool(row["image_shape"] and row["image_shape"] == row["mask_shape"])
    except Exception as error:
        row["error"] = repr(error)
    return row


def summarize_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    labeled = sum(1 for row in rows if row["has_label"])
    missing_images = sum(1 for row in rows if not row["image_exists"])
    missing_masks = sum(1 for row in rows if not row["mask_exists"])
    shape_mismatch = sum(1 for row in rows if row["image_exists"] and row["mask_exists"] and not row["shape_match"])
    errors = sum(1 for row in rows if row["error"])
    positive_ratios = [float(row["positive_ratio"]) for row in rows if row["mask_exists"] and not row["error"]]
    return {
        "total_samples": total,
        "labeled_samples": labeled,
        "unlabeled_samples": total - labeled,
        "labeled_ratio": labeled / total if total else 0.0,
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "shape_mismatch": shape_mismatch,
        "errors": errors,
        "mean_positive_ratio": float(np.mean(positive_ratios)) if positive_ratios else 0.0,
        "median_positive_ratio": float(np.median(positive_ratios)) if positive_ratios else 0.0,
    }


def shape_summary_rows(sample_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counter = Counter((row["split"], row["image_shape"], row["mask_shape"]) for row in sample_rows)
    rows = []
    for (split, image_shape, mask_shape), count in sorted(counter.items()):
        rows.append({"split": split, "image_shape": image_shape, "mask_shape": mask_shape, "count": count})
    return rows


def source_summary_rows(sample_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    split_totals = Counter(str(row.get("split", "")) for row in sample_rows)
    for row in sample_rows:
        split = str(row.get("split", ""))
        source = str(row.get("source", "UNKNOWN") or "UNKNOWN")
        grouped.setdefault((split, source), []).append(row)

    rows: List[Dict[str, Any]] = []
    for (split, source), group_rows in sorted(grouped.items()):
        summary = summarize_split(group_rows)
        split_total = int(split_totals.get(split, 0))
        rows.append(
            {
                "split": split,
                "source": source,
                "source_ratio_in_split": len(group_rows) / split_total if split_total else 0.0,
                **summary,
            }
        )
    return rows


def source_split_matrix_rows(source_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = sorted({str(row["source"]) for row in source_rows})
    splits = sorted({str(row["split"]) for row in source_rows})
    counts = {
        (str(row["source"]), str(row["split"])): int(row["total_samples"])
        for row in source_rows
    }

    rows: List[Dict[str, Any]] = []
    for source in sources:
        row: Dict[str, Any] = {"source": source}
        total = 0
        for split in splits:
            count = counts.get((source, split), 0)
            row[split] = count
            total += count
        row["total"] = total
        rows.append(row)
    return rows


def _float_values(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return np.asarray(values, dtype=np.float64)


def _summary_stats(rows: Sequence[Dict[str, Any]], key: str, prefix: str) -> Dict[str, Any]:
    values = _float_values(rows, key)
    if values.size == 0:
        return {
            f"{prefix}_min": "",
            f"{prefix}_p25": "",
            f"{prefix}_median": "",
            f"{prefix}_mean": "",
            f"{prefix}_p75": "",
            f"{prefix}_max": "",
            f"{prefix}_std": "",
        }
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_std": float(np.std(values)),
    }


def source_size_distribution_rows(sample_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in sample_rows:
        split = str(row.get("split", ""))
        source = str(row.get("source", "UNKNOWN") or "UNKNOWN")
        grouped.setdefault((split, source), []).append(row)

    rows: List[Dict[str, Any]] = []
    for (split, source), group_rows in sorted(grouped.items()):
        output: Dict[str, Any] = {
            "split": split,
            "source": source,
            "total_samples": len(group_rows),
            "unique_image_shapes": len({str(row.get("image_shape", "")) for row in group_rows if row.get("image_shape")}),
            "unique_mask_shapes": len({str(row.get("mask_shape", "")) for row in group_rows if row.get("mask_shape")}),
        }
        output.update(_summary_stats(group_rows, "image_dim0", "image_dim0"))
        output.update(_summary_stats(group_rows, "image_dim1", "image_dim1"))
        output.update(_summary_stats(group_rows, "image_dim2", "image_dim2"))
        output.update(_summary_stats(group_rows, "image_voxels", "image_voxels"))
        output.update(_summary_stats(group_rows, "positive_voxels", "positive_voxels"))
        output.update(_summary_stats(group_rows, "positive_ratio", "positive_ratio"))
        rows.append(output)
    return rows


def source_shape_distribution_rows(sample_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], int] = Counter(
        (str(row.get("split", "")), str(row.get("source", "UNKNOWN") or "UNKNOWN"), str(row.get("image_shape", "")))
        for row in sample_rows
    )
    source_totals = Counter((str(row.get("split", "")), str(row.get("source", "UNKNOWN") or "UNKNOWN")) for row in sample_rows)
    rows: List[Dict[str, Any]] = []
    for (split, source, image_shape), count in sorted(grouped.items()):
        total = int(source_totals.get((split, source), 0))
        rows.append(
            {
                "split": split,
                "source": source,
                "image_shape": image_shape,
                "count": int(count),
                "ratio_in_source": int(count) / total if total else 0.0,
            }
        )
    return rows


def size_distribution_fieldnames() -> List[str]:
    fields = ["split", "source", "total_samples", "unique_image_shapes", "unique_mask_shapes"]
    for prefix in ("image_dim0", "image_dim1", "image_dim2", "image_voxels", "positive_voxels", "positive_ratio"):
        fields.extend(
            [
                f"{prefix}_min",
                f"{prefix}_p25",
                f"{prefix}_median",
                f"{prefix}_mean",
                f"{prefix}_p75",
                f"{prefix}_max",
                f"{prefix}_std",
            ]
        )
    return fields


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_split_size_distribution_outputs(
    output_dir: Path,
    splits: Sequence[str],
    size_rows: Sequence[Dict[str, Any]],
    shape_rows: Sequence[Dict[str, Any]],
) -> None:
    size_fields = size_distribution_fieldnames()
    shape_fields = ["split", "source", "image_shape", "count", "ratio_in_source"]
    for split in splits:
        split_dir = output_dir / "size_distribution" / str(split)
        split_size_rows = [row for row in size_rows if str(row.get("split", "")) == str(split)]
        split_shape_rows = [row for row in shape_rows if str(row.get("split", "")) == str(split)]
        write_csv(split_dir / "source_size_distribution.csv", split_size_rows, size_fields)
        write_csv(split_dir / "source_shape_distribution.csv", split_shape_rows, shape_fields)


def _plot_boxplot_by_source(axis_obj, split_rows: Sequence[Dict[str, Any]], sources: Sequence[str], key: str, title: str) -> None:
    plot_sources: List[str] = []
    values_by_source = []
    for source in sources:
        values = _float_values([row for row in split_rows if str(row.get("source", "UNKNOWN")) == source], key)
        if values.size > 0:
            plot_sources.append(source)
            values_by_source.append(values)
    if not values_by_source:
        axis_obj.set_title(title)
        axis_obj.axis("off")
        return
    boxplot_kwargs = {"showfliers": False}
    label_arg = "tick_labels" if "tick_labels" in inspect.signature(axis_obj.boxplot).parameters else "labels"
    boxplot_kwargs[label_arg] = plot_sources
    axis_obj.boxplot(values_by_source, **boxplot_kwargs)
    axis_obj.set_title(title)
    axis_obj.tick_params(axis="x", rotation=45, labelsize=8)
    axis_obj.grid(True, axis="y", alpha=0.25)


def save_size_distribution_plots(output_dir: Path, sample_rows: Sequence[Dict[str, Any]], splits: Sequence[str]) -> None:
    plot_specs = [
        ("image_dim0", "Dim 0"),
        ("image_dim1", "Dim 1"),
        ("image_dim2", "Dim 2 / Depth"),
        ("positive_voxels", "Lesion voxels"),
    ]
    for split in splits:
        split_rows = [row for row in sample_rows if str(row.get("split", "")) == str(split)]
        sources = sorted({str(row.get("source", "UNKNOWN") or "UNKNOWN") for row in split_rows})
        if not split_rows or not sources:
            continue
        fig, axes = plt.subplots(1, len(plot_specs), figsize=(4.2 * len(plot_specs), 4.2), dpi=150)
        if len(plot_specs) == 1:
            axes = [axes]
        for axis_obj, (key, title) in zip(axes, plot_specs):
            _plot_boxplot_by_source(axis_obj, split_rows, sources, key, title)
        fig.suptitle(f"{split}: size distribution by source")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        plot_path = output_dir / "size_distribution" / str(split) / "source_size_boxplot.pdf"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    return np.clip((image - low) / (high - low + 1e-8), 0.0, 1.0)


def choose_slice(mask: np.ndarray, axis: int, label_threshold: float) -> int:
    axis = int(np.clip(axis, 0, mask.ndim - 1))
    positive = mask > label_threshold
    if not np.any(positive):
        return mask.shape[axis] // 2
    reduce_axes = tuple(item for item in range(mask.ndim) if item != axis)
    slice_scores = positive.sum(axis=reduce_axes)
    return int(np.argmax(slice_scores))


def take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    axis = int(np.clip(axis, 0, volume.ndim - 1))
    return np.take(volume, int(index), axis=axis)


def save_preview(record: SampleRecord, output_path: Path, axis: int, label_threshold: float) -> Dict[str, Any]:
    image_img = nib.load(str(record.image_path))
    mask_img = nib.load(str(record.mask_path))
    image = load_image_array(image_img)
    mask = load_mask_array(mask_img)
    axis = int(np.clip(axis, 0, min(image.ndim, mask.ndim) - 1))
    slice_index = choose_slice(mask, axis, label_threshold)
    image_slice = normalize_for_display(take_slice(image, axis, slice_index))
    mask_slice = take_slice(mask, axis, slice_index) > label_threshold

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=150)
    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(mask_slice, cmap="gray")
    axes[1].set_title("GT")
    axes[2].imshow(image_slice, cmap="gray")
    axes[2].imshow(mask_slice, cmap="Reds", alpha=0.35)
    axes[2].set_title("Overlay")
    for axis_obj in axes:
        axis_obj.axis("off")
    fig.tight_layout(pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "split": record.split,
        "index": record.index,
        "case_id": record.case_id,
        "source": record.source,
        "slice_axis": axis,
        "slice_index": slice_index,
        "output_path": display_path(output_path),
    }


def select_visual_records(records: Sequence[SampleRecord], rows: Sequence[Dict[str, Any]], count: int, seed: int) -> List[SampleRecord]:
    if count <= 0:
        return []
    by_index = {record.index: record for record in records}
    valid_labeled = [
        row["index"]
        for row in rows
        if row["has_label"] and row["image_exists"] and row["mask_exists"] and not row["error"]
    ]
    valid_unlabeled = [
        row["index"]
        for row in rows
        if not row["has_label"] and row["image_exists"] and row["mask_exists"] and not row["error"]
    ]
    rng = np.random.default_rng(seed)

    selected: List[int] = []
    for pool in (valid_labeled, valid_unlabeled):
        if len(selected) >= count or not pool:
            continue
        take = min(count - len(selected), len(pool))
        selected.extend(int(item) for item in rng.choice(pool, take, replace=False))
    return [by_index[index] for index in sorted(selected)]


def select_source_visual_records(
    records: Sequence[SampleRecord],
    rows: Sequence[Dict[str, Any]],
    count_per_source: int,
    seed: int,
) -> List[SampleRecord]:
    if count_per_source <= 0:
        return []

    by_index = {record.index: record for record in records}
    valid_rows = [
        row
        for row in rows
        if row["image_exists"] and row["mask_exists"] and not row["error"]
    ]
    rows_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in valid_rows:
        rows_by_source.setdefault(str(row.get("source", "UNKNOWN") or "UNKNOWN"), []).append(row)

    rng = np.random.default_rng(seed)
    selected_indices: List[int] = []
    for source in sorted(rows_by_source):
        source_rows = rows_by_source[source]
        labeled = [row for row in source_rows if row["has_label"]]
        pool = labeled or source_rows
        take = min(int(count_per_source), len(pool))
        chosen_positions = rng.choice(len(pool), take, replace=False)
        selected_indices.extend(int(pool[int(position)]["index"]) for position in np.atleast_1d(chosen_positions))

    return [by_index[index] for index in sorted(selected_indices)]


def write_text_summary(
    path: Path,
    split_summaries: Dict[str, Dict[str, Any]],
    source_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:
    lines = [f"Output: {display_path(output_dir)}", ""]
    sources_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for row in source_rows:
        sources_by_split.setdefault(str(row["split"]), []).append(row)

    for split, summary in split_summaries.items():
        source_text = ", ".join(
            f"{row['source']}={row['total_samples']}"
            for row in sorted(sources_by_split.get(split, []), key=lambda item: str(item["source"]))
        )
        lines.append(
            "{split}: total={total}, labeled={labeled}/{total}, unlabeled={unlabeled}, shape_mismatch={mismatch}, errors={errors}, sources={sources}".format(
                split=split,
                total=summary["total_samples"],
                labeled=summary["labeled_samples"],
                unlabeled=summary["unlabeled_samples"],
                mismatch=summary["shape_mismatch"],
                errors=summary["errors"],
                sources=source_text or "none",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> Path:
    data_root = args.data_root
    run_name = args.run_name or f"cyst_data_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sample_rows: List[Dict[str, Any]] = []
    split_summaries: Dict[str, Dict[str, Any]] = {}
    visual_rows: List[Dict[str, Any]] = []

    for split in args.splits:
        records = read_split_file(data_root, split)
        split_rows = [analyze_record(record, args.label_threshold) for record in records]
        all_sample_rows.extend(split_rows)
        split_summaries[split] = summarize_split(split_rows)

        if not args.skip_visualization:
            if args.visual_selection == "per_source":
                selected = select_source_visual_records(records, split_rows, args.visuals_per_source, args.seed)
            else:
                selected = select_visual_records(records, split_rows, args.num_visuals, args.seed)
            for order, record in enumerate(selected):
                preview_path = output_dir / "visualizations" / split / f"{record.source}_{order:02d}_{record.case_id}_idx{record.index}.png"
                try:
                    visual_rows.append(save_preview(record, preview_path, args.slice_axis, args.label_threshold))
                except Exception as error:
                    visual_rows.append(
                        {
                            "split": split,
                            "index": record.index,
                            "case_id": record.case_id,
                            "source": record.source,
                            "slice_axis": args.slice_axis,
                            "slice_index": "",
                            "output_path": display_path(preview_path),
                            "error": repr(error),
                        }
                    )

    sample_fields = [
        "split",
        "index",
        "case_id",
        "source",
        "image_path",
        "mask_path",
        "image_exists",
        "mask_exists",
        "image_shape",
        "mask_shape",
        "image_dim0",
        "image_dim1",
        "image_dim2",
        "mask_dim0",
        "mask_dim1",
        "mask_dim2",
        "image_voxels",
        "mask_voxels",
        "image_spacing",
        "mask_spacing",
        "shape_match",
        "has_label",
        "positive_voxels",
        "total_voxels",
        "positive_ratio",
        "error",
    ]
    split_fields = [
        "split",
        "total_samples",
        "labeled_samples",
        "unlabeled_samples",
        "labeled_ratio",
        "missing_images",
        "missing_masks",
        "shape_mismatch",
        "errors",
        "mean_positive_ratio",
        "median_positive_ratio",
    ]
    split_rows = [{"split": split, **summary} for split, summary in split_summaries.items()]
    source_rows = source_summary_rows(all_sample_rows)
    source_fields = [
        "split",
        "source",
        "source_ratio_in_split",
        "total_samples",
        "labeled_samples",
        "unlabeled_samples",
        "labeled_ratio",
        "missing_images",
        "missing_masks",
        "shape_mismatch",
        "errors",
        "mean_positive_ratio",
        "median_positive_ratio",
    ]

    write_csv(output_dir / "sample_stats.csv", all_sample_rows, sample_fields)
    write_csv(output_dir / "split_summary.csv", split_rows, split_fields)
    write_csv(output_dir / "source_summary.csv", source_rows, source_fields)
    source_size_rows = source_size_distribution_rows(all_sample_rows)
    source_shape_rows = source_shape_distribution_rows(all_sample_rows)
    write_csv(output_dir / "source_size_distribution.csv", source_size_rows, size_distribution_fieldnames())
    write_csv(
        output_dir / "source_shape_distribution.csv",
        source_shape_rows,
        ["split", "source", "image_shape", "count", "ratio_in_source"],
    )
    write_split_size_distribution_outputs(output_dir, args.splits, source_size_rows, source_shape_rows)
    save_size_distribution_plots(output_dir, all_sample_rows, args.splits)
    source_matrix_rows = source_split_matrix_rows(source_rows)
    source_matrix_fields = ["source", *args.splits, "total"]
    write_csv(output_dir / "source_split_matrix.csv", source_matrix_rows, source_matrix_fields)
    write_csv(output_dir / "shape_summary.csv", shape_summary_rows(all_sample_rows), ["split", "image_shape", "mask_shape", "count"])
    write_csv(output_dir / "visualization_index.csv", visual_rows, ["split", "index", "case_id", "source", "slice_axis", "slice_index", "output_path", "error"])

    write_json(
        output_dir / "summary.json",
        {
            "data_root": display_path(data_root),
            "output_dir": display_path(output_dir),
            "splits": split_summaries,
            "sources": source_rows,
            "source_size_distribution": source_size_rows,
            "args": {
                "splits": args.splits,
                "num_visuals": args.num_visuals,
                "visual_selection": args.visual_selection,
                "visuals_per_source": args.visuals_per_source,
                "seed": args.seed,
                "slice_axis": args.slice_axis,
                "label_threshold": args.label_threshold,
                "skip_visualization": args.skip_visualization,
            },
        },
    )
    write_text_summary(output_dir / "summary.txt", split_summaries, source_rows, output_dir)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = run_analysis(args)
    print(f"Analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
