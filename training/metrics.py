from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_divide(numerator: float, denominator: float, empty_value: float = 0.0) -> float:
    if denominator == 0:
        return empty_value
    return float(numerator) / float(denominator)


def confusion_counts(prediction, target) -> Dict[str, int]:
    pred = _to_numpy(prediction) > 0
    gt = _to_numpy(target) > 0
    pred = pred.astype(bool).reshape(-1)
    gt = gt.astype(bool).reshape(-1)

    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    total = tp + tn + fp + fn
    pred_positive = tp + fp
    gt_positive = tp + fn
    dice = _safe_divide(2 * tp, 2 * tp + fp + fn, empty_value=1.0)
    iou = _safe_divide(tp, tp + fp + fn, empty_value=1.0)
    accuracy = _safe_divide(tp + tn, total, empty_value=1.0)
    precision = _safe_divide(tp, tp + fp, empty_value=1.0 if tp + fn == 0 else 0.0)
    recall = _safe_divide(tp, tp + fn, empty_value=1.0)
    specificity = _safe_divide(tn, tn + fp, empty_value=1.0)
    f1_empty_value = 1.0 if pred_positive == 0 and gt_positive == 0 else 0.0
    f1 = _safe_divide(2 * precision * recall, precision + recall, empty_value=f1_empty_value)
    return {
        "Dice": dice,
        "IoU": iou,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Pred Positives": pred_positive,
        "GT Positives": gt_positive,
        "Pred Positive Ratio": _safe_divide(pred_positive, total, empty_value=0.0),
        "GT Positive Ratio": _safe_divide(gt_positive, total, empty_value=0.0),
        "Specificity": specificity,
        "F1": f1,
    }


def surface_metrics(prediction, target) -> Dict[str, float]:
    try:
        from medpy import metric
    except ImportError:
        return {"HD95": math.nan, "ASD": math.nan}

    pred = _to_numpy(prediction) > 0
    gt = _to_numpy(target) > 0
    if not np.any(pred) and not np.any(gt):
        return {"HD95": 0.0, "ASD": 0.0}
    if not np.any(pred) or not np.any(gt):
        return {"HD95": math.nan, "ASD": math.nan}
    try:
        return {"HD95": float(metric.binary.hd95(pred, gt)), "ASD": float(metric.binary.asd(pred, gt))}
    except Exception:
        return {"HD95": math.nan, "ASD": math.nan}


def _nanmean_or_nan(values: list[float]) -> float:
    if not values:
        return math.nan
    array = np.asarray(values, dtype=np.float64)
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return math.nan
    return float(valid.mean())


class MetricAccumulator:
    def __init__(self, compute_surface: bool = False) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.loss_sum = 0.0
        self.loss_items = 0
        self.inference_time = 0.0
        self.inference_items = 0
        self.compute_surface = bool(compute_surface)
        self.hd95_values: list[float] = []
        self.asd_values: list[float] = []

    def update(self, prediction, target, loss: Optional[float] = None) -> None:
        pred_np = _to_numpy(prediction)
        target_np = _to_numpy(target)
        counts = confusion_counts(pred_np, target_np)
        self.tp += counts["tp"]
        self.tn += counts["tn"]
        self.fp += counts["fp"]
        self.fn += counts["fn"]

        batch_size = int(pred_np.shape[0]) if pred_np.ndim > 0 else 1
        if loss is not None:
            self.loss_sum += float(loss) * batch_size
            self.loss_items += batch_size

        if self.compute_surface:
            for pred_item, target_item in zip(pred_np, target_np):
                values = surface_metrics(pred_item, target_item)
                self.hd95_values.append(values["HD95"])
                self.asd_values.append(values["ASD"])

    def add_inference_time(self, seconds: float, items: int) -> None:
        self.inference_time += float(seconds)
        self.inference_items += int(items)

    def summary(self) -> Dict[str, float]:
        output = metrics_from_counts(self.tp, self.tn, self.fp, self.fn)
        output["Loss"] = _safe_divide(self.loss_sum, self.loss_items, empty_value=math.nan)
        output["Inference Time"] = _safe_divide(self.inference_time, self.inference_items, empty_value=math.nan)
        output["FPS"] = _safe_divide(self.inference_items, self.inference_time, empty_value=math.nan)
        output["TP"] = self.tp
        output["TN"] = self.tn
        output["FP"] = self.fp
        output["FN"] = self.fn
        if self.compute_surface:
            output["HD95"] = _nanmean_or_nan(self.hd95_values)
            output["ASD"] = _nanmean_or_nan(self.asd_values)
        else:
            output["HD95"] = math.nan
            output["ASD"] = math.nan
        return output
