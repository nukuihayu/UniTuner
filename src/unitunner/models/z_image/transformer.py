# Copyright (c) 2025 Z-Image Team. Apache-2.0.
# Adapted from https://github.com/Tongyi-MAI/Z-Image via Musubi Tuner.
# Modified to separate native computation from training and execution policies.
"""Compute native Z-Image reverse-flow predictions with exact artifact key paths."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.config.routing import TreadConfig
from unitunner.execution.routing import TargetSpan, sample_token_route
from unitunner.models.components.features import FeatureRecorder
from unitunner.models.z_image.config.model import ZImageConfig
from unitunner.models.z_image.layers import FinalLayer, RMSNorm, RopeEmbedder, TimestepEmbedder, ZImageTransformerBlock


class ZImageTransformer(nn.Module):
    """Own native parameters; consume normalized reverse time without solver sign conversion.

    Input latents use [B,C,F,H,W], captions [B,L,caption_dim], mask [B,L]
    (True means real text). Image padding tokens are active attention keys.
    Caption padding is rounded to 32 per sample. Returned predictions have the
    latent shape and native sign clean-minus-noise. No trainer is imported.
    """

    def __init__(self, config: ZImageConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_recorder: FeatureRecorder | None = None
        self.token_routing: TreadConfig | None = None
        self.feature_width: int = config.dim
        self.feature_patch: tuple[int, int, int] = (1, 2, 2)
        self.all_x_embedder = nn.ModuleDict()
        self.all_final_layer = nn.ModuleDict()
        for patch, frame_patch in zip(config.patch_sizes, config.frame_patch_sizes, strict=True):
            key = f"{patch}-{frame_patch}"
            patch_dim = patch * patch * frame_patch * config.in_channels
            self.all_x_embedder[key] = nn.Linear(patch_dim, config.dim)
            self.all_final_layer[key] = FinalLayer(config.dim, patch_dim)
        self.noise_refiner = nn.ModuleList(
            [ZImageTransformerBlock(config, modulation=True) for _ in range(config.num_refiner_layers)]
        )
        self.context_refiner = nn.ModuleList(
            [ZImageTransformerBlock(config, modulation=False) for _ in range(config.num_refiner_layers)]
        )
        self.t_embedder = TimestepEmbedder(config)
        self.cap_embedder = nn.Sequential(
            RMSNorm(config.caption_dim, config.norm_eps), nn.Linear(config.caption_dim, config.dim)
        )
        # Deterministic finite initialization also supports genuine from-scratch use.
        self.x_pad_token = nn.Parameter(torch.zeros(1, config.dim))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, config.dim))
        self.layers = nn.ModuleList([ZImageTransformerBlock(config, modulation=True) for _ in range(config.num_layers)])
        self.rope_embedder = RopeEmbedder(config)

    def _validate_inputs(
        self, latents: Tensor, timesteps: Tensor, captions: Tensor, mask: Tensor | None, patch: int, frame_patch: int
    ) -> None:
        """Reject structural mismatches before projection without synchronizing devices."""
        if (patch, frame_patch) not in tuple(zip(self.config.patch_sizes, self.config.frame_patch_sizes, strict=True)):
            raise ValueError("Unsupported Z-Image spatial/temporal patch pair")
        if latents.ndim != 5 or latents.shape[1] != self.config.in_channels:
            raise ValueError("Z-Image latents require [B,C,F,H,W] with configured channels")
        batch, _, frames, height, width = latents.shape
        if min(batch, frames, height, width) <= 0 or frames % frame_patch or height % patch or width % patch:
            raise ValueError("Latent spatial/temporal sizes must be positive multiples of their patch sizes")
        if (
            timesteps.shape != (batch,)
            or captions.ndim != 3
            or captions.shape[0] != batch
            or captions.shape[2] != self.config.caption_dim
            or captions.shape[1] == 0
        ):
            raise ValueError("Expected timesteps [B] and nonempty captions [B,L,caption_dim]")
        if mask is not None and (
            mask.shape != captions.shape[:2] or mask.dtype != torch.bool or mask.device != latents.device
        ):
            raise ValueError("Caption mask must be boolean [B,L] on the latent device")
        if (
            captions.device != latents.device
            or timesteps.device != latents.device
            or not timesteps.is_floating_point()
            or not latents.is_floating_point()
            or captions.dtype != latents.dtype
        ):
            raise ValueError(
                "Floating latents, captions and timesteps must share device; caption and latent dtypes must match"
            )
        padded_caption = captions.shape[1] + (-captions.shape[1]) % 32
        if (
            padded_caption + frames // frame_patch >= self.config.axes_lengths[0]
            or height // patch > self.config.axes_lengths[1]
            or width // patch > self.config.axes_lengths[2]
        ):
            raise ValueError("Z-Image position coordinates exceed configured RoPE tables")

    def predict(
        self,
        latents: Tensor,
        timesteps: Tensor,
        captions: Tensor,
        caption_mask: Tensor | None = None,
        *,
        patch_size: int = 2,
        frame_patch_size: int = 1,
    ) -> Tensor:
        """Predict native reverse velocity at normalized reverse timesteps [B].

        Captions and latents share floating dtype/device. Values are not modified;
        gradients reach every used projection and input. Sparse masks retain
        learned interior padding up to the final valid text position; this avoids
        truncating late valid tokens by incorrectly using the number of True entries.
        Time is multiplied by the configured native scale exactly once.
        """
        self._validate_inputs(latents, timesteps, captions, caption_mask, patch_size, frame_patch_size)
        batch, channels, frames, height, width = latents.shape
        frame_tokens, height_tokens, width_tokens = (
            frames // frame_patch_size,
            height // patch_size,
            width // patch_size,
        )
        image_length = frame_tokens * height_tokens * width_tokens
        caption_length = captions.shape[1]
        padded_caption_length = caption_length + (-caption_length) % 32
        positions = torch.arange(padded_caption_length, device=latents.device)[None]
        if caption_mask is None:
            original_mask = (positions < caption_length).expand(batch, -1)
            caption_lengths = torch.full((batch,), padded_caption_length, device=latents.device, dtype=torch.int64)
            caption_attention_mask = None
        else:
            original_mask = F.pad(caption_mask, (0, padded_caption_length - caption_length))
            occupied_length = torch.where(original_mask, positions + 1, 0).amax(dim=1)
            caption_lengths = ((occupied_length + 31) // 32) * 32
            caption_attention_mask = positions < caption_lengths[:, None]
        captions = F.pad(captions, (0, 0, 0, padded_caption_length - caption_length))
        conditioning = self.t_embedder(timesteps * self.config.time_scale)
        tokens = (
            latents.reshape(
                batch, channels, frame_tokens, frame_patch_size, height_tokens, patch_size, width_tokens, patch_size
            )
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(batch, image_length, -1)
        )
        key = f"{patch_size}-{frame_patch_size}"
        tokens = self.all_x_embedder[key](tokens)
        conditioning = conditioning.to(tokens.dtype)
        grid = torch.meshgrid(
            torch.arange(1, frame_tokens + 1, device=latents.device),
            torch.arange(height_tokens, device=latents.device),
            torch.arange(width_tokens, device=latents.device),
            indexing="ij",
        )
        image_positions = torch.stack(grid, dim=-1).reshape(1, image_length, 3).expand(batch, -1, -1).clone()
        image_positions[:, :, 0] += caption_lengths[:, None]
        image_padding = (-image_length) % 32
        if image_padding:
            tokens = torch.cat(
                (tokens, self.x_pad_token[None].expand(batch, image_padding, -1).to(tokens.dtype)), dim=1
            )
            image_positions = F.pad(image_positions, (0, 0, 0, image_padding))
        image_frequencies = self.rope_embedder(image_positions)
        for block in self.noise_refiner:
            tokens = block(tokens, image_frequencies, conditioning, None)
        captions = self.cap_embedder(captions)
        captions = torch.where(original_mask[:, :, None], captions, self.cap_pad_token.to(captions.dtype))
        caption_positions = torch.zeros(batch, padded_caption_length, 3, dtype=torch.int64, device=latents.device)
        caption_positions[:, :, 0] = positions + 1
        caption_frequencies = self.rope_embedder(caption_positions)
        for block in self.context_refiner:
            captions = block(captions, caption_frequencies, None, caption_attention_mask)
        unified = torch.cat((tokens, captions), dim=1)
        frequencies = torch.cat((image_frequencies, caption_frequencies), dim=1)
        attention_mask = (
            None if caption_attention_mask is None else F.pad(caption_attention_mask, (tokens.shape[1], 0), value=True)
        )
        route = sample_token_route(
            self.token_routing,
            training=self.training,
            length=unified.shape[1],
            spans=((TargetSpan(0, image_length),),),
            device=unified.device,
        )
        for index, block in enumerate(self.layers):
            if route is not None and route.applies(index):
                updated = block(
                    route.select(unified),
                    route.select(frequencies),
                    conditioning,
                    None if attention_mask is None else route.select(attention_mask),
                )
                unified = route.restore(unified, updated)
            else:
                unified = block(unified, frequencies, conditioning, attention_mask)
            if self.feature_recorder is not None:
                self.feature_recorder.record(index, unified[:, :image_length])
        prediction = self.all_final_layer[key](unified, conditioning)[:, :image_length]
        return (
            prediction.reshape(
                batch, frame_tokens, height_tokens, width_tokens, frame_patch_size, patch_size, patch_size, channels
            )
            .permute(0, 7, 1, 4, 2, 5, 3, 6)
            .reshape(batch, channels, frames, height, width)
        )

    forward = predict
