from __future__ import annotations

from typing import Any

import torch

from .common import (
    BaseSegmentationModel,
    SlicePositionEncoder,
    build_2d_decoder,
    build_2d_encoder,
    channels_from_config,
    normalise_2d_encoder_name,
    normalise_decoder_style,
)


class Experiment2DSegModel(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        encoder_type: str = "unet",
        encoder_channels: list[int] | tuple[int, ...] | None = None,
        channels: list[int] | tuple[int, ...] | None = None,
        base_channels: int = 16,
        decoder_style: str | None = None,
        normalization: str = "batchnorm",
        dropout: float = 0.0,
        use_position_encoder: bool = False,
        position_embedding_dim: int = 32,
        max_position_embeddings: int = 512,
        **_: Any,
    ) -> None:
        super().__init__()
        self.encoder_type = normalise_2d_encoder_name(encoder_type)
        self.channels = channels_from_config(encoder_channels or channels, base_channels)
        if decoder_style is None:
            decoder_style = {"unet": "same_scale", "unetpp": "nested_dense", "unet3plus": "full_scale", "nnunet": "same_scale"}[self.encoder_type]
        self.decoder_style = normalise_decoder_style(decoder_style)
        self.encoder_2d = build_2d_encoder(
            self.encoder_type,
            in_channels=in_channels,
            channels=self.channels,
            base_channels=base_channels,
            normalization=normalization,
            dropout=float(dropout),
        )
        self.decoder_2d = build_2d_decoder(
            self.decoder_style,
            decoder_model=self.encoder_type,
            channels=self.channels,
            num_classes=num_classes,
            normalization=normalization,
            dropout=float(dropout),
        )
        self.use_position_encoder = bool(use_position_encoder)
        self.expects_slice_indices = self.use_position_encoder
        self.position_encoder = (
            SlicePositionEncoder(self.channels, embedding_dim=position_embedding_dim, max_positions=max_position_embeddings)
            if self.use_position_encoder
            else None
        )
        self.model_name = f"proposal_exp_2d_{self.encoder_type}"
        self.backbone_name = f"{self.encoder_type}2d_encoder"
        self.set_architecture_config(
            model_name=self.model_name,
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            encoder_type=self.encoder_type,
            decoder_style=self.decoder_style,
            channels=self.channels,
            normalization=normalization,
            dropout=float(dropout),
            use_position_encoder=self.use_position_encoder,
        )

    @staticmethod
    def _normalise_slice_indices(x: torch.Tensor, slice_indices: Any | None) -> torch.Tensor:
        if slice_indices is None:
            return torch.zeros(int(x.shape[0]), device=x.device, dtype=torch.long)
        if not isinstance(slice_indices, torch.Tensor):
            slice_indices = torch.as_tensor(slice_indices, device=x.device)
        slice_indices = slice_indices.to(device=x.device, dtype=torch.long).reshape(-1)
        if slice_indices.numel() == 1 and int(x.shape[0]) > 1:
            slice_indices = slice_indices.expand(int(x.shape[0]))
        if slice_indices.numel() != int(x.shape[0]):
            raise ValueError(f"slice_indices must contain {int(x.shape[0])} values, got {int(slice_indices.numel())}.")
        return slice_indices

    def _apply_position_encoder(self, features: list[torch.Tensor], slice_indices: Any | None) -> list[torch.Tensor]:
        if self.position_encoder is None:
            return features
        indices = self._normalise_slice_indices(features[0], slice_indices)
        return self.position_encoder.forward_2d(features, indices)

    def forward(self, x: torch.Tensor, return_features: bool = False, slice_indices: Any | None = None, slice_index: Any | None = None):
        if slice_indices is None:
            slice_indices = slice_index
        features = self.encoder_2d(x)
        features = self._apply_position_encoder(features, slice_indices)
        logits = self.decoder_2d(features)
        if return_features:
            return self.build_output(logits, features={"encoder": features})
        return logits


FullExperiment2D = Experiment2DSegModel

__all__ = ["Experiment2DSegModel", "FullExperiment2D"]
