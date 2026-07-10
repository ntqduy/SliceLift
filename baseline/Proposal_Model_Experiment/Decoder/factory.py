from __future__ import annotations

from typing import Any

from torch import nn

from .decoder_ablation_style import (
    FullEncoderSingleDecoder2DDecoder,
    FullEncoderSingleDecoder3DDecoder,
    SingleEncoderFullDecoder2DDecoder,
    SingleEncoderFullDecoder3DDecoder,
)
from .decoder_2_plus_style import UNetPlusPlus2DDecoder, UNetPlusPlus3DDecoder
from .decoder_3_plus_style import UNet3Plus2DDecoder, UNet3Plus3DDecoder
from .decoder_same_scale_style import SameScale2DDecoder, SameScale3DDecoder


def normalise_decoder_model(name: str | None) -> str:
    key = str(name or "unet3d").lower().replace("-", "_").replace("+", "plus")
    aliases = {
        "unet": "unet3d",
        "unet3d": "unet3d",
        "unet_3d": "unet3d",
        "unetpp": "unetpp3d",
        "unetplusplus": "unetpp3d",
        "unet_plus_plus": "unetpp3d",
        "unetpp3d": "unetpp3d",
        "unetplusplus3d": "unetpp3d",
        "unet_plus_plus3d": "unetpp3d",
        "unet_plus_plus_3d": "unetpp3d",
        "unet3plus": "unet3plus3d",
        "unet_3plus": "unet3plus3d",
        "unet_3_plus": "unet3plus3d",
        "unet3plus3d": "unet3plus3d",
        "unet_3plus3d": "unet3plus3d",
        "unet_3_plus3d": "unet3plus3d",
        "unet_3_plus_3d": "unet3plus3d",
        "nnunet": "nnunet3d",
        "nnunet3d": "nnunet3d",
        "nn_unet": "nnunet3d",
        "nn_unet3d": "nnunet3d",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported decoder model: {name}")
    return aliases[key]


def normalise_decoder_style(name: str | None) -> str:
    key = str(name or "same_scale").lower().replace("-", "_").replace("+", "plus")
    aliases = {
        "same": "same_scale",
        "same_scale": "same_scale",
        "skip": "same_scale",
        "unet": "same_scale",
        "unet_decoder": "same_scale",
        "nested": "nested_dense",
        "nested_dense": "nested_dense",
        "unetpp": "nested_dense",
        "unetplusplus": "nested_dense",
        "unet_plus_plus": "nested_dense",
        "full": "full_scale",
        "full_scale": "full_scale",
        "unet3plus": "full_scale",
        "unet_3_plus": "full_scale",
        "full_encoder_single_decoder": "full_encoder_single_decoder",
        "all_encoder_single_decoder": "full_encoder_single_decoder",
        "full_encoder_one_decoder": "full_encoder_single_decoder",
        "all_encoder_one_decoder": "full_encoder_single_decoder",
        "single_encoder_full_decoder": "single_encoder_full_decoder",
        "single_encoder_full_decode": "single_encoder_full_decoder",
        "same_encoder_full_decoder": "single_encoder_full_decoder",
        "same_encoder_full_decode": "single_encoder_full_decoder",
        "same_scale_full_decoder": "single_encoder_full_decoder",
        "same_scale_full_decode": "single_encoder_full_decoder",
        "single_skip_full_decoder": "single_encoder_full_decoder",
        "single_skip_full_decode": "single_encoder_full_decoder",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported decoder style: {name}")
    return aliases[key]


def build_3d_decoder(style: str | None, **kwargs: Any) -> nn.Module:
    key = normalise_decoder_style(style)
    decoder_model = normalise_decoder_model(kwargs.pop("decoder_model", None))
    if key == "same_scale":
        if decoder_model == "nnunet3d":
            kwargs.setdefault("activation_name", "leaky_relu")
            kwargs["residual"] = "none"
        if decoder_model == "unet3d":
            kwargs.setdefault("residual", "conv")
            kwargs.setdefault("conv_bias", True)
            kwargs.setdefault("up_kernel_size", 3)
        return SameScale3DDecoder(**kwargs)
    if key == "nested_dense":
        return UNetPlusPlus3DDecoder(**kwargs)
    if key == "full_scale":
        return UNet3Plus3DDecoder(**kwargs)
    if key == "full_encoder_single_decoder":
        return FullEncoderSingleDecoder3DDecoder(**kwargs)
    if key == "single_encoder_full_decoder":
        return SingleEncoderFullDecoder3DDecoder(**kwargs)
    raise AssertionError(key)


def build_2d_decoder(style: str | None, **kwargs: Any) -> nn.Module:
    key = normalise_decoder_style(style)
    decoder_model = str(kwargs.pop("decoder_model", "") or "").lower().replace("-", "_").replace("+", "plus")
    if key == "same_scale":
        if decoder_model in {"nnunet", "nnunet2d", "nn_unet", "nn_unet2d"}:
            kwargs.setdefault("activation_name", "leaky_relu")
        elif decoder_model in {"unet", "unet2d", "unet_2d", ""}:
            kwargs.setdefault("upsample_mode", "bilinear")
        return SameScale2DDecoder(**kwargs)
    if key == "nested_dense":
        return UNetPlusPlus2DDecoder(**kwargs)
    if key == "full_scale":
        return UNet3Plus2DDecoder(**kwargs)
    if key == "full_encoder_single_decoder":
        return FullEncoderSingleDecoder2DDecoder(**kwargs)
    if key == "single_encoder_full_decoder":
        return SingleEncoderFullDecoder2DDecoder(**kwargs)
    raise AssertionError(key)


__all__ = [
    "SameScale2DDecoder",
    "SameScale3DDecoder",
    "FullEncoderSingleDecoder2DDecoder",
    "FullEncoderSingleDecoder3DDecoder",
    "SingleEncoderFullDecoder2DDecoder",
    "SingleEncoderFullDecoder3DDecoder",
    "UNetPlusPlus2DDecoder",
    "UNetPlusPlus3DDecoder",
    "UNet3Plus2DDecoder",
    "UNet3Plus3DDecoder",
    "build_3d_decoder",
    "build_2d_decoder",
    "normalise_decoder_model",
    "normalise_decoder_style",
]
