from __future__ import annotations

from .Unet2D import UNet2DEncoder


class UNetPlusPlus2DEncoder(UNet2DEncoder):
    encoder_name = "unetpp2d"


__all__ = ["UNetPlusPlus2DEncoder"]
