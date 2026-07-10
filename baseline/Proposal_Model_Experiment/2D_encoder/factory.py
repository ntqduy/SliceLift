from __future__ import annotations

from typing import Any

from torch import nn

from .Unet2D import UNet2DEncoder
from .Unet3Plus2D import UNet3Plus2DEncoder
from .UnetPlusPlus2D import UNetPlusPlus2DEncoder
from .nnUnet2D import NNUNet2DEncoder


def normalise_2d_encoder_name(name: str | None) -> str:
    key = str(name or "unet").lower().replace("-", "_").replace("+", "plus")
    aliases = {
        "unet": "unet",
        "unet2d": "unet",
        "unet_2d": "unet",
        "unetplusplus": "unetpp",
        "unet_plus_plus": "unetpp",
        "unetpp": "unetpp",
        "unet2plus": "unetpp",
        "unet_2_plus": "unetpp",
        "unet3plus": "unet3plus",
        "unet_3_plus": "unet3plus",
        "nnunet": "nnunet",
        "nnunet2d": "nnunet",
        "nn_unet2d": "nnunet",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported 2D encoder: {name}")
    return aliases[key]


def build_2d_encoder(name: str | None, **kwargs: Any) -> nn.Module:
    key = normalise_2d_encoder_name(name)
    if key == "unet":
        return UNet2DEncoder(**kwargs)
    if key == "unetpp":
        return UNetPlusPlus2DEncoder(**kwargs)
    if key == "unet3plus":
        return UNet3Plus2DEncoder(**kwargs)
    if key == "nnunet":
        return NNUNet2DEncoder(**kwargs)
    raise AssertionError(key)


__all__ = [
    "UNet2DEncoder",
    "UNetPlusPlus2DEncoder",
    "UNet3Plus2DEncoder",
    "NNUNet2DEncoder",
    "build_2d_encoder",
    "normalise_2d_encoder_name",
]
