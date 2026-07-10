from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import nn


@dataclass
class SegmentationOutput:
    logits: Any
    features: Optional[Any] = None
    aux: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        yield self.logits
        yield self.features

    def __getitem__(self, index):
        if index == 0:
            return self.logits
        if index == 1:
            return self.features
        raise IndexError(index)


class BaseSegmentationModel(nn.Module):
    def set_architecture_config(self, **kwargs) -> None:
        self.architecture_config = dict(kwargs)

    def build_output(self, logits, features=None, aux=None) -> SegmentationOutput:
        return SegmentationOutput(logits=logits, features=features, aux=aux or {})
