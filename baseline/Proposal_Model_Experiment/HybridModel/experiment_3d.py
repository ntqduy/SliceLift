from __future__ import annotations

from typing import Any

import torch

from .common import (
    BaseSegmentationModel,
    build_3d_decoder,
    build_3d_encoder,
    channels_from_config,
    main_logits,
    normalise_3d_encoder_name,
    normalise_decoder_model,
    normalise_decoder_style,
)


class Experiment3DSegModel(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        encoder_type: str = "unet",
        encoder_channels: list[int] | tuple[int, ...] | None = None,
        channels: list[int] | tuple[int, ...] | None = None,
        base_channels: int = 16,
        decoder_model: str | None = None,
        decoder_style: str = "same_scale",
        normalization: str = "batchnorm",
        normalization_3d: str | None = None,
        decoder_normalization: str | None = None,
        deep_supervision: bool = False,
        dropout: float = 0.0,
        residual: str | None = None,
        conv_bias: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.encoder_type = normalise_3d_encoder_name(encoder_type)
        self.decoder_model = normalise_decoder_model(decoder_model or encoder_type)
        self.decoder_style = normalise_decoder_style(decoder_style)
        self.channels = channels_from_config(encoder_channels or channels, base_channels)
        encoder_normalization = str(normalization_3d or normalization)
        decoder_norm = str(decoder_normalization or encoder_normalization)
        self.encoder_3d = build_3d_encoder(
            self.encoder_type,
            in_channels=in_channels,
            channels=self.channels,
            base_channels=base_channels,
            normalization=encoder_normalization,
            dropout=float(dropout),
            residual=residual,
            conv_bias=bool(conv_bias),
        )
        self.decoder_3d = build_3d_decoder(
            self.decoder_style,
            decoder_model=self.decoder_model,
            channels=self.channels,
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            normalization=decoder_norm,
            dropout=float(dropout),
            residual=residual,
            conv_bias=bool(conv_bias),
        )
        self.model_name = f"proposal_exp_3d_{self.encoder_type}_{self.decoder_model}_{self.decoder_style}"
        self.backbone_name = f"{self.encoder_type}3d_encoder"
        self.deep_supervision = bool(deep_supervision)
        self.set_architecture_config(
            model_name=self.model_name,
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            encoder_type=self.encoder_type,
            decoder_model=self.decoder_model,
            decoder_style=self.decoder_style,
            channels=self.channels,
            deep_supervision=self.deep_supervision,
            normalization=encoder_normalization,
            decoder_normalization=decoder_norm,
            dropout=float(dropout),
            residual=residual,
            conv_bias=bool(conv_bias),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.encoder_3d(x)
        output = self.decoder_3d(features)
        logits = main_logits(output)
        if return_features:
            return self.build_output(logits, features={"encoder": features, "decoder_outputs": output})
        return output if isinstance(output, tuple) and self.deep_supervision else logits


FullExperiment3D = Experiment3DSegModel

__all__ = ["Experiment3DSegModel", "FullExperiment3D"]
