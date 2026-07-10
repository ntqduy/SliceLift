from __future__ import annotations

from .hybrid_base import ExperimentHybridModel


class HybridUNet3DNNUNet2D(ExperimentHybridModel):
    """Hybrid architecture: UNet3D encoder + nnUNet2D encoder."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("encoder_3d_type", "unet3d")
        kwargs.setdefault("encoder_2d_type", "nnunet")
        super().__init__(*args, **kwargs)


__all__ = ["HybridUNet3DNNUNet2D"]
