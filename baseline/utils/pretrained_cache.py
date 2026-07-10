from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import torch


def _project_pretrain_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "pretrain"


def pretrain_root() -> Path:
    configured = os.environ.get("CYST_PRETRAIN_DIR")
    root = Path(configured) if configured else _project_pretrain_dir()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(root)
    return root


def checkpoint_dir() -> Path:
    path = pretrain_root() / "hub" / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _candidate_dirs():
    root = pretrain_root()
    yield checkpoint_dir()
    yield root / "checkpoints"
    yield root
    yield Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    yield Path("pretrained")
    yield Path("checkpoints")


def ensure_pretrain_cache() -> Path:
    root = pretrain_root()
    for directory in _candidate_dirs():
        directory.mkdir(parents=True, exist_ok=True)
    return root


def find_cached_checkpoint(url: str, min_bytes: int = 0):
    filename = Path(urlparse(url).path).name
    if not filename:
        return None
    for directory in _candidate_dirs():
        path = directory / filename
        if path.exists() and path.stat().st_size >= int(min_bytes):
            return str(path)
    return None


def load_cached_state_dict(url: str, min_bytes: int = 0):
    cached_path = find_cached_checkpoint(url, min_bytes=min_bytes)
    if cached_path is not None:
        return torch.load(cached_path, map_location="cpu")
    return torch.hub.load_state_dict_from_url(
        url,
        model_dir=str(checkpoint_dir()),
        map_location="cpu",
        progress=True,
        check_hash=True,
    )


def ensure_cached_checkpoint(url: str, min_bytes: int = 0):
    cached_path = find_cached_checkpoint(url, min_bytes=min_bytes)
    if cached_path is not None:
        return cached_path
    torch.hub.load_state_dict_from_url(
        url,
        model_dir=str(checkpoint_dir()),
        map_location="cpu",
        progress=True,
        check_hash=True,
    )
    return find_cached_checkpoint(url, min_bytes=min_bytes)
