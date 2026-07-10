from __future__ import annotations

from .Unet3D import UNet3DEncoder


class UNet3Plus3DEncoder(UNet3DEncoder):
    encoder_name = "unet3plus3d"


__all__ = ["UNet3Plus3DEncoder"]
