from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


_LARGEST_ORDER_TOKENS = {"largest", "high", "higher", "different", "most_different", "diverse", "farthest"}
_SMALLEST_ORDER_TOKENS = {"smallest", "low", "lower", "closest", "similar", "most_similar", "mean", "nearest_mean"}


def _is_group_based_mode(mode: str | None) -> bool:
    return str(mode or "").lower().replace("-", "_") == "group_based"


class SliceSelector:
    """Select 2D slices from a 3D volume [B, C, D, H, W]."""

    def __init__(
        self,
        mode: str = "uniform",
        num_slices: int | None = None,
        seed: int | None = 42,
        num_groups: int | None = None,
        samples_per_group: int = 1,
        similarity_metric: str = "mad",
        selection_order: str = "closest",
        **_: Any,
    ) -> None:
        self.mode = str(mode or "uniform").lower().replace("-", "_")
        self.seed = None if seed is None else int(seed)
        requested_slices = None if num_slices is None else max(1, int(num_slices))
        default_groups = requested_slices if _is_group_based_mode(self.mode) and requested_slices is not None else 5
        self.num_groups = max(1, int(default_groups if num_groups is None else num_groups))
        self.samples_per_group = max(1, int(samples_per_group))
        group_based_count = self.num_groups * self.samples_per_group
        if _is_group_based_mode(self.mode):
            self.num_slices = group_based_count
        else:
            self.num_slices = 5 if requested_slices is None else requested_slices
        self.similarity_metric = str(similarity_metric or "mad").lower()
        self.selection_order = str(selection_order or "closest").lower().replace("-", "_")
        if self.selection_order in _LARGEST_ORDER_TOKENS:
            self.select_largest = True
        elif self.selection_order in _SMALLEST_ORDER_TOKENS:
            self.select_largest = False
        else:
            raise ValueError(f"Unsupported group-based selection_order: {selection_order!r}")
        self._generator = torch.Generator(device="cpu")
        if self.seed is not None:
            self._generator.manual_seed(self.seed)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "SliceSelector":
        cfg = dict(cfg or {})
        group_cfg: dict[str, Any] = {}
        for group_key in ("group_based", "group-based"):
            value = cfg.pop(group_key, None)
            if isinstance(value, dict):
                group_cfg.update(value)
        cfg.update(group_cfg)
        return cls(**cfg)

    def __call__(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if volume.ndim != 5:
            raise ValueError(f"SliceSelector expects [B,C,D,H,W], got {tuple(volume.shape)}")
        batch_size, channels, depth, height, width = volume.shape
        if depth <= 0:
            raise ValueError("Cannot select slices from a volume with D=0.")

        indices = self._select_indices(volume)
        gather_index = indices[:, None, :, None, None].expand(batch_size, channels, indices.shape[1], height, width)
        selected = torch.gather(volume, dim=2, index=gather_index).permute(0, 2, 1, 3, 4).contiguous()
        return selected, indices

    def _select_indices(self, volume: torch.Tensor) -> torch.Tensor:
        mode = self.mode
        if mode == "uniform":
            return self._uniform_indices(volume)
        if mode == "random":
            return self._random_indices(volume)
        if mode in {"middle", "center", "centre", "mid", "single", "single_center", "center_slice", "middle_slice"}:
            return self._middle_indices(volume)
        if _is_group_based_mode(mode):
            return self._group_based_indices(volume)
        raise ValueError(f"Unsupported slice selection mode: {self.mode}")

    def _middle_indices(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, _, _ = volume.shape
        index = torch.tensor([depth // 2], device=volume.device, dtype=torch.long)
        return index.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _uniform_indices(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, _, _ = volume.shape
        if self.num_slices == 1:
            return self._middle_indices(volume)
        positions = torch.linspace(0, depth - 1, steps=self.num_slices, device=volume.device)
        indices = positions.round().long().clamp_(0, depth - 1)
        return indices.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _random_indices(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, _, _ = volume.shape
        rows = []
        for _ in range(batch_size):
            if self.num_slices <= depth:
                row = torch.randperm(depth, generator=self._generator, device="cpu")[: self.num_slices]
            else:
                row = torch.randint(depth, (self.num_slices,), generator=self._generator, device="cpu")
            rows.append(torch.sort(row).values)
        return torch.stack(rows, dim=0).to(device=volume.device, dtype=torch.long)

    def _group_based_indices(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, _, _ = volume.shape
        rows = []
        with torch.no_grad():
            for batch_index in range(batch_size):
                selected = self._group_based_indices_for_one(volume[batch_index], depth)
                rows.append(selected.to(device=volume.device, dtype=torch.long))
        return torch.stack(rows, dim=0)

    def _group_based_indices_for_one(self, sample: torch.Tensor, depth: int) -> torch.Tensor:
        device = sample.device
        boundaries = torch.linspace(0, depth, steps=min(self.num_groups, depth) + 1, device=device).round().long()
        chosen: list[int] = []

        for group_index in range(len(boundaries) - 1):
            start = int(boundaries[group_index].item())
            end = int(boundaries[group_index + 1].item())
            if end <= start:
                continue
            candidates = torch.arange(start, end, device=device)
            group = sample[:, start:end].permute(1, 0, 2, 3).contiguous()
            scores = self._group_based_scores(group)
            take = min(self.samples_per_group, len(candidates))
            local = torch.topk(scores, k=take, largest=self.select_largest, sorted=True).indices
            group_chosen = [int(item) for item in candidates[local].detach().cpu().tolist()]
            while len(group_chosen) < self.samples_per_group:
                group_chosen.append(group_chosen[-1] if group_chosen else int(candidates[len(candidates) // 2].item()))
            chosen.extend(group_chosen)

        target = self.num_slices
        if len(chosen) < target:
            fallback = torch.linspace(0, depth - 1, steps=target, device=device).round().long().detach().cpu().tolist()
            for item in fallback:
                index = int(item)
                if index not in chosen:
                    chosen.append(index)
                if len(chosen) >= target:
                    break
            for item in fallback:
                if len(chosen) >= target:
                    break
                chosen.append(int(item))

        if not chosen:
            chosen = [depth // 2]
        while len(chosen) < target:
            chosen.append(chosen[-1])

        return torch.tensor(sorted(chosen[:target]), device=device, dtype=torch.long).clamp_(0, depth - 1)

    def _group_based_scores(self, group: torch.Tensor) -> torch.Tensor:
        if group.shape[0] == 1:
            return torch.ones(1, device=group.device)

        if self.similarity_metric in {"cos", "cosine", "cosine_similarity"}:
            flattened = group.float().flatten(1)
            normalized = F.normalize(flattened, dim=1, eps=1e-6)
            similarity = normalized @ normalized.t()
            # Lower average similarity means less redundant; negate for topk.
            return -similarity.mean(dim=1)

        if self.similarity_metric not in {"mad", "mean_absolute_deviation"}:
            raise ValueError(f"Unsupported group-based similarity_metric: {self.similarity_metric}")

        center = group.float().mean(dim=0, keepdim=True)
        return (group.float() - center).abs().flatten(1).mean(dim=1)
