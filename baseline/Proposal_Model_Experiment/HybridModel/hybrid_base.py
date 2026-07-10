from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import (
    BaseSegmentationModel,
    SlicePositionEncoder,
    SliceSelector,
    build_2d_encoder,
    build_3d_decoder,
    build_3d_encoder,
    channels_from_config,
    checkpoint_state_dict,
    component_load_report,
    main_logits,
    normalise_2d_encoder_name,
    normalise_3d_encoder_name,
    normalise_decoder_model,
    normalise_decoder_style,
    normalise_encoder_fusion_mode,
    normalization_3d as make_normalization_3d,
)


class ExperimentHybridModel(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        encoder_2d_type: str = "unet",
        encoder_3d_type: str = "unet",
        encoder_2d_channels: list[int] | tuple[int, ...] | None = None,
        encoder_3d_channels: list[int] | tuple[int, ...] | None = None,
        channels: list[int] | tuple[int, ...] | None = None,
        base_channels: int = 16,
        decoder_model: str | None = None,
        decoder_style: str = "full_scale",
        slice_selection: Mapping[str, Any] | None = None,
        encoder_fusion_mode: str = "add",
        normalization: str = "batchnorm",
        normalization_2d: str | None = None,
        normalization_3d: str | None = None,
        decoder_normalization: str | None = None,
        deep_supervision: bool = False,
        dropout: float = 0.0,
        residual: str | None = None,
        conv_bias: bool = False,
        use_position_encoder: bool = True,
        position_embedding_dim: int = 32,
        max_position_embeddings: int = 512,
        freeze_2d_encoder: bool = False,
        freeze_3d_encoder: bool = False,
        freeze_3d_decoder: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.encoder_2d_type = normalise_2d_encoder_name(encoder_2d_type)
        self.encoder_3d_type = normalise_3d_encoder_name(encoder_3d_type)
        self.decoder_model = normalise_decoder_model(decoder_model or "unet3plus3d")
        self.decoder_style = normalise_decoder_style(decoder_style)
        self.encoder_2d_channels = channels_from_config(encoder_2d_channels or channels, base_channels)
        self.encoder_3d_channels = channels_from_config(encoder_3d_channels or channels, base_channels)
        if len(self.encoder_2d_channels) != len(self.encoder_3d_channels):
            raise ValueError("2D and 3D encoders must expose the same number of stages.")
        norm_2d = str(normalization_2d or normalization)
        norm_3d = str(normalization_3d or normalization)
        decoder_norm = str(decoder_normalization or norm_3d)

        self.num_classes = int(num_classes)
        self.encoder_fusion_mode = normalise_encoder_fusion_mode(encoder_fusion_mode)
        self.slice_selection_cfg = dict(slice_selection or {})
        self.slice_selector = SliceSelector.from_config(self.slice_selection_cfg)
        self.use_position_encoder = bool(use_position_encoder)
        self.deep_supervision = bool(deep_supervision)

        self.encoder_2d = build_2d_encoder(
            self.encoder_2d_type,
            in_channels=in_channels,
            channels=self.encoder_2d_channels,
            base_channels=base_channels,
            normalization=norm_2d,
            dropout=float(dropout),
        )
        self.encoder_3d = build_3d_encoder(
            self.encoder_3d_type,
            in_channels=in_channels,
            channels=self.encoder_3d_channels,
            base_channels=base_channels,
            normalization=norm_3d,
            dropout=float(dropout),
            residual=residual,
            conv_bias=bool(conv_bias),
        )
        self.decoder_3d = build_3d_decoder(
            self.decoder_style,
            decoder_model=self.decoder_model,
            channels=self.encoder_3d_channels,
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            normalization=decoder_norm,
            dropout=float(dropout),
            residual=residual,
            conv_bias=bool(conv_bias),
        )
        self.position_encoder = (
            SlicePositionEncoder(self.encoder_2d_channels, embedding_dim=position_embedding_dim, max_positions=max_position_embeddings)
            if self.use_position_encoder
            else None
        )
        self.fusion_projections = nn.ModuleList(
            [
                nn.Conv3d(ch2d, ch3d, kernel_size=1, bias=False)
                if self.encoder_fusion_mode == "add" and ch2d != ch3d
                else nn.Identity()
                for ch2d, ch3d in zip(self.encoder_2d_channels, self.encoder_3d_channels)
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(ch3d + ch2d if self.encoder_fusion_mode == "concat" else ch3d, ch3d, kernel_size=1, bias=False),
                    make_normalization_3d(ch3d, norm_3d),
                    nn.ReLU(inplace=True),
                )
                for ch2d, ch3d in zip(self.encoder_2d_channels, self.encoder_3d_channels)
            ]
        )
        self.model_name = f"proposal_exp_hybrid_{self.encoder_3d_type}3d_{self.encoder_2d_type}2d_{self.decoder_model}_{self.decoder_style}"
        self.backbone_name = f"hybrid_{self.encoder_3d_type}3d_{self.encoder_2d_type}2d_{self.decoder_model}_{self.decoder_style}_encoder"
        self.pretrain_loading_report: dict[str, Any] | None = None

        if freeze_2d_encoder:
            self._freeze(self.encoder_2d)
        if freeze_3d_encoder:
            self._freeze(self.encoder_3d)
        if freeze_3d_decoder:
            self._freeze(self.decoder_3d)

        self.set_architecture_config(
            model_name=self.model_name,
            in_channels=int(in_channels),
            num_classes=self.num_classes,
            encoder_2d_type=self.encoder_2d_type,
            encoder_3d_type=self.encoder_3d_type,
            decoder_model=self.decoder_model,
            decoder_style=self.decoder_style,
            encoder_2d_channels=self.encoder_2d_channels,
            encoder_3d_channels=self.encoder_3d_channels,
            slice_selection=self.slice_selection_cfg,
            deep_supervision=self.deep_supervision,
            use_position_encoder=self.use_position_encoder,
            normalization_2d=norm_2d,
            normalization_3d=norm_3d,
            decoder_normalization=decoder_norm,
            encoder_fusion_mode=self.encoder_fusion_mode,
            freeze_2d_encoder=bool(freeze_2d_encoder),
            freeze_3d_encoder=bool(freeze_3d_encoder),
            freeze_3d_decoder=bool(freeze_3d_decoder),
            dropout=float(dropout),
            residual=residual,
            conv_bias=bool(conv_bias),
        )

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _encode_2d(self, volume: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        selected_slices, slice_indices = self.slice_selector(volume)
        features_2d = self.encoder_2d(selected_slices)
        if self.position_encoder is not None:
            features_2d = self.position_encoder.forward_selected(features_2d, slice_indices)
        return features_2d, slice_indices

    def _scatter_2d_features_to_3d(self, feature_2d: torch.Tensor, feature_3d: torch.Tensor, slice_indices: torch.Tensor, stage_index: int) -> torch.Tensor:
        if feature_2d.ndim != 5:
            raise ValueError(f"Hybrid 2D features must be [B,K,C,H,W], got {tuple(feature_2d.shape)}")
        batch_size, num_slices, channels, height, width = feature_2d.shape
        target_hw = tuple(int(item) for item in feature_3d.shape[-2:])
        if (height, width) != target_hw:
            feature_2d = F.interpolate(
                feature_2d.reshape(batch_size * num_slices, channels, height, width),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, num_slices, channels, *target_hw)

        depth = int(feature_3d.shape[-3])
        divisor = 2 ** int(stage_index)
        mapped_indices = (slice_indices.to(device=feature_3d.device, dtype=torch.long) // divisor).clamp_(0, depth - 1)
        source = feature_2d.to(device=feature_3d.device, dtype=feature_3d.dtype).permute(0, 2, 1, 3, 4).contiguous()
        scatter_index = mapped_indices[:, None, :, None, None].expand(batch_size, channels, num_slices, *target_hw)
        lifted = feature_3d.new_zeros((batch_size, channels, depth, *target_hw))
        lifted.scatter_add_(2, scatter_index, source)
        counts = feature_3d.new_zeros((batch_size, 1, depth, 1, 1))
        counts.scatter_add_(2, mapped_indices[:, None, :, None, None], torch.ones((batch_size, 1, num_slices, 1, 1), device=feature_3d.device, dtype=feature_3d.dtype))
        return lifted / counts.clamp_min(1.0)

    def _fuse_stage_features(self, feature_3d: torch.Tensor, lifted_2d: torch.Tensor, stage_index: int) -> torch.Tensor:
        projection = self.fusion_projections[stage_index]
        block = self.fusion_blocks[stage_index]
        if self.encoder_fusion_mode == "concat":
            return block(torch.cat([feature_3d, lifted_2d], dim=1))
        return block(feature_3d + projection(lifted_2d))

    def _encode_3d_with_slice_fusion(self, features_2d: list[torch.Tensor], slice_indices: torch.Tensor, volume: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        lifted_2d_features: list[torch.Tensor] = []

        def fuse_fn(stage_index: int, feature_3d: torch.Tensor) -> torch.Tensor:
            lifted = self._scatter_2d_features_to_3d(features_2d[stage_index], feature_3d, slice_indices, stage_index)
            lifted_2d_features.append(lifted)
            return self._fuse_stage_features(feature_3d, lifted, stage_index)

        fused, raw = self.encoder_3d.forward_with_fusion(volume, fuse_fn)
        return fused, raw, lifted_2d_features

    def forward(self, volume: torch.Tensor, return_features: bool = False):
        features_2d, slice_indices = self._encode_2d(volume)
        fused, features_3d, lifted_2d = self._encode_3d_with_slice_fusion(features_2d, slice_indices, volume)
        output = self.decoder_3d(fused)
        logits = main_logits(output)
        if return_features:
            return self.build_output(
                logits,
                features={
                    "encoder": fused,
                    "encoder_2d": features_2d,
                    "encoder_3d": features_3d,
                    "lifted_2d": lifted_2d,
                    "slice_indices": slice_indices,
                    "deep_outputs": list(output[1:]) if isinstance(output, tuple) else [],
                },
            )
        return output if isinstance(output, tuple) and self.deep_supervision else logits

    def load_pretrained_components(
        self,
        ckpt_2d: str | Path,
        ckpt_3d: str | Path,
        load_2d_encoder: bool = True,
        load_3d_encoder: bool = True,
        load_3d_decoder: bool = True,
        strict: bool = False,
        freeze_2d_encoder: bool = False,
        freeze_3d_encoder: bool = False,
        freeze_3d_decoder: bool = False,
        log_path: str | Path | None = None,
        ckpt_3d_decoder: str | Path | None = None,
    ) -> dict[str, Any]:
        ckpt_2d = Path(ckpt_2d)
        ckpt_3d = Path(ckpt_3d)
        ckpt_3d_decoder = Path(ckpt_3d_decoder) if ckpt_3d_decoder is not None else ckpt_3d
        report: dict[str, Any] = {
            "stage1_2d_ckpt": str(ckpt_2d) if load_2d_encoder else None,
            "stage2_3d_ckpt": str(ckpt_3d) if load_3d_encoder else None,
            "stage2_3d_encoder_ckpt": str(ckpt_3d) if load_3d_encoder else None,
            "stage2_3d_decoder_ckpt": str(ckpt_3d_decoder) if load_3d_decoder else None,
            "loaded_2d_encoder": False,
            "loaded_3d_encoder": False,
            "loaded_3d_decoder": False,
            "load_2d_encoder_from_stage1": bool(load_2d_encoder),
            "load_3d_encoder_from_stage2": bool(load_3d_encoder),
            "load_3d_decoder_from_stage2": bool(load_3d_decoder),
            "components": {},
        }
        state_2d = torch.load(ckpt_2d, map_location="cpu") if load_2d_encoder else {}
        state_3d_encoder = torch.load(ckpt_3d, map_location="cpu") if load_3d_encoder else {}
        if load_3d_decoder:
            state_3d_decoder = (
                state_3d_encoder
                if load_3d_encoder and ckpt_3d_decoder == ckpt_3d
                else torch.load(ckpt_3d_decoder, map_location="cpu")
            )
        else:
            state_3d_decoder = {}

        if load_2d_encoder:
            component = component_load_report(self.encoder_2d, checkpoint_state_dict(state_2d), prefix="encoder_2d.", label="2D encoder", strict=strict)
            report["components"]["2d_encoder"] = component
            report["loaded_2d_encoder"] = component["num_loaded_keys"] > 0
            if self.position_encoder is not None:
                position_component = component_load_report(
                    self.position_encoder,
                    checkpoint_state_dict(state_2d),
                    prefix="position_encoder.",
                    label="2D position encoder",
                    strict=False,
                )
                report["components"]["2d_position_encoder"] = position_component
                report["loaded_2d_position_encoder"] = position_component["num_loaded_keys"] > 0
            if freeze_2d_encoder:
                self._freeze(self.encoder_2d)
                if self.position_encoder is not None:
                    self._freeze(self.position_encoder)
        if load_3d_encoder:
            component = component_load_report(self.encoder_3d, checkpoint_state_dict(state_3d_encoder), prefix="encoder_3d.", label="3D encoder", strict=strict)
            report["components"]["3d_encoder"] = component
            report["loaded_3d_encoder"] = component["num_loaded_keys"] > 0
            if freeze_3d_encoder:
                self._freeze(self.encoder_3d)
        if load_3d_decoder:
            component = component_load_report(self.decoder_3d, checkpoint_state_dict(state_3d_decoder), prefix="decoder_3d.", label="3D decoder", strict=strict)
            report["components"]["3d_decoder"] = component
            report["loaded_3d_decoder"] = component["num_loaded_keys"] > 0
            if freeze_3d_decoder:
                self._freeze(self.decoder_3d)

        report["num_loaded_2d_keys"] = report.get("components", {}).get("2d_encoder", {}).get("num_loaded_keys", 0)
        report["num_loaded_2d_position_keys"] = report.get("components", {}).get("2d_position_encoder", {}).get("num_loaded_keys", 0)
        report["num_loaded_3d_encoder_keys"] = report.get("components", {}).get("3d_encoder", {}).get("num_loaded_keys", 0)
        report["num_loaded_3d_decoder_keys"] = report.get("components", {}).get("3d_decoder", {}).get("num_loaded_keys", 0)
        report["num_skipped_keys"] = sum(item.get("num_skipped_keys", 0) for item in report["components"].values())
        self.pretrain_loading_report = report
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


HybridExperiment = ExperimentHybridModel

__all__ = ["ExperimentHybridModel", "HybridExperiment"]
