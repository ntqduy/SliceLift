from __future__ import annotations

from .Unet3D import UNet3DEncoder


class UNetPlusPlus3DEncoder(UNet3DEncoder):
    encoder_name = "unetpp3d"


__all__ = ["UNetPlusPlus3DEncoder"]
