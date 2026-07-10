from __future__ import annotations

from typing import Any

from torch import nn

from .UNet3Plus3D import UNet3Plus3DEncoder
from .UNetPlusPlus3D import UNetPlusPlus3DEncoder
from .Unet3D import UNet3DEncoder
from .nnUnet3D import NNUNet3DEncoder


def normalise_3d_encoder_name(name: str | None) -> str:
    key = str(name or "unet3d").lower().replace("-", "_").replace("+", "plus")
    aliases = {
        "unet": "unet",
        "unet3d": "unet",
        "unet_3d": "unet",
        "unetpp": "unetpp",
        "unetplusplus": "unetpp",
        "unet_plus_plus": "unetpp",
        "unetplusplus3d": "unetpp",
        "unet_plus_plus3d": "unetpp",
        "unetpp3d": "unetpp",
        "unet3dpp": "unetpp",
        "unet3dplusplus": "unetpp",
        "unet3plus": "unet3plus",
        "unet_3_plus": "unet3plus",
        "unet3plus3d": "unet3plus",
        "unet_3_plus3d": "unet3plus",
        "unet3d_3plus": "unet3plus",
        "nnunet": "nnunet",
        "nnunet3d": "nnunet",
        "nn_unet3d": "nnunet",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported 3D encoder: {name}")
    return aliases[key]


def build_3d_encoder(name: str | None, **kwargs: Any) -> nn.Module:
    key = normalise_3d_encoder_name(name)
    if key == "unet":
        return UNet3DEncoder(**kwargs)
    if key == "unetpp":
        return UNetPlusPlus3DEncoder(**kwargs)
    if key == "unet3plus":
        return UNet3Plus3DEncoder(**kwargs)
    if key == "nnunet":
        return NNUNet3DEncoder(**kwargs)
    raise AssertionError(key)


__all__ = [
    "UNet3DEncoder",
    "UNetPlusPlus3DEncoder",
    "UNet3Plus3DEncoder",
    "NNUNet3DEncoder",
    "build_3d_encoder",
    "normalise_3d_encoder_name",
]
