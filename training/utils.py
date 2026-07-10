from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def get_nested(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def sanitize_name(value: Any) -> str:
    text = str(value or "default").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "default"


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class _SeedWorker:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def __call__(self, worker_id: int) -> None:
        worker_seed = self.seed + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)


def seed_worker(seed: int):
    return _SeedWorker(seed)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def format_large_number(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    for suffix in ("", "K", "M", "G", "T"):
        if abs(value) < 1000:
            return f"{value:.2f}{suffix}"
        value /= 1000
    return f"{value:.2f}P"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def display_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cyst")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def describe_device(device: torch.device, visible_ids: str) -> str:
    if device.type != "cuda":
        return "CPU"
    names = []
    for index in range(torch.cuda.device_count()):
        try:
            names.append(f"cuda:{index}={torch.cuda.get_device_name(index)}")
        except Exception:
            names.append(f"cuda:{index}")
    return f"visible GPU ids [{visible_ids}], " + ", ".join(names)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
