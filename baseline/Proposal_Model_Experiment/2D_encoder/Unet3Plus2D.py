from __future__ import annotations

from .Unet2D import UNet2DEncoder


class UNet3Plus2DEncoder(UNet2DEncoder):
    encoder_name = "unet3plus2d"


__all__ = ["UNet3Plus2DEncoder"]
