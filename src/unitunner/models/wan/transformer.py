# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Compute native Wan video predictions with explicit host geometry and scheduler time."""

import math

import torch
from torch import Tensor, nn

from unitunner.config.routing import TreadConfig
from unitunner.execution.routing import TargetSpan, sample_token_route
from unitunner.models.components.features import FeatureRecorder
from unitunner.models.wan.config.model import WanConfig
from unitunner.models.wan.geometry import VideoGrid, rotary_embedding, timestep_embedding
from unitunner.models.wan.layers import Head, ImageProjection, WanBlock


class WanTransformer(nn.Module):
    """Own one native Wan expert; return FP32 CFHW predictions per clip.

    The initial validated path uses FP32 parameters with native BF16 SDPA.
    Text padding participates in biased projections and is never masked away.
    Video conditioning is concatenated before patch embedding. This component
    owns neither scheduler/expert switching nor VAE temporal history.
    """

    def __init__(self, config: WanConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_recorder: FeatureRecorder | None = None
        self.token_routing: TreadConfig | None = None
        self.feature_width: int = config.hidden_size
        self.feature_patch: tuple[int, int, int] = config.patch_size
        width = config.hidden_size
        self.patch_embedding = nn.Conv3d(config.in_channels, width, config.patch_size, stride=config.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, width), nn.GELU(approximate="tanh"), nn.Linear(width, width)
        )
        self.time_embedding = nn.Sequential(nn.Linear(config.timestep_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))
        self.blocks = nn.ModuleList([WanBlock(config) for _ in range(config.num_layers)])
        self.head = Head(config)
        if config.image_context:
            self.img_emb = ImageProjection(width, first_last_frame=config.variant == "flf2v")
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Apply native initialization, including the zero output projection."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for embedding in (self.text_embedding, self.time_embedding):
            for module in embedding.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)

    def predict(
        self,
        latents: tuple[Tensor, ...],
        timesteps: Tensor,
        captions: tuple[Tensor, ...],
        *,
        sequence_length: int,
        conditioning: tuple[Tensor, ...] | None = None,
        image_features: Tensor | None = None,
        frame_indices: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[Tensor, ...]:
        """Predict native velocity for variable CFHW clips at 0-1000-unit times.

        Wan 2.1 accepts B times; 2.2 accepts B or BL token times. Captions are
        unpadded SD tensors, with S at most text_length. Optional image features
        are B2571280 for 2.1 I2V or B22571280 for explicit2.1 FLF2V, with first
        then last image per row. Shapes/devices/dtypes are validated before
        patching. No RCM mask, additional noise or CFG is applied here.
        """
        grids = self.validate_inputs(
            latents, timesteps, captions, sequence_length, conditioning, image_features, frame_indices
        )
        config = self.config
        embedded = []
        rotations = []
        for index, (clip, grid) in enumerate(zip(latents, grids, strict=True)):
            if conditioning is not None:
                clip = torch.cat((clip, conditioning[index]), dim=0)
            tokens = self.patch_embedding(clip.unsqueeze(0)).flatten(2).transpose(1, 2)
            embedded.append(
                torch.cat((tokens, tokens.new_zeros(1, sequence_length - tokens.shape[1], config.hidden_size)), dim=1)
            )
            frames = None if frame_indices is None else frame_indices[index]
            rotations.append(rotary_embedding(grid, config.hidden_size // config.num_heads, clip.device, frames))
        values = torch.cat(embedded)
        # FP32 masters alone do not protect time projections from ambient AMP.
        with torch.autocast(device_type=values.device.type, enabled=False):
            if config.version == "2.1":
                time = self.time_embedding(timestep_embedding(timesteps, config.timestep_dim).float())
                modulation = self.time_projection(time).unflatten(1, (6, config.hidden_size))
            else:
                times = timesteps[:, None].expand(-1, sequence_length) if timesteps.ndim == 1 else timesteps
                features = (
                    timestep_embedding(times.flatten(), config.timestep_dim)
                    .unflatten(0, (len(latents), sequence_length))
                    .float()
                )
                time = self.time_embedding(features)
                modulation = self.time_projection(time).unflatten(2, (6, config.hidden_size))
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat((caption, caption.new_zeros(config.text_length - caption.shape[0], config.text_dim)))
                    for caption in captions
                ]
            )
        )
        if image_features is not None:
            context = torch.cat((self.img_emb(image_features), context), dim=1)
        frequencies = tuple(rotations)
        route = sample_token_route(
            self.token_routing,
            training=self.training,
            length=values.shape[1],
            spans=tuple((TargetSpan(0, math.prod(grid)),) for grid in grids),
            device=values.device,
        )
        for index, block in enumerate(self.blocks):
            if route is not None and route.applies(index):
                selected_frequencies = tuple(
                    rotation.index_select(0, route.indices[row, : route.targets[row][0].kept])
                    for row, rotation in enumerate(frequencies)
                )
                updated = block(
                    route.select(values),
                    route.select(modulation) if modulation.ndim == 4 else modulation,
                    context,
                    selected_frequencies,
                )
                values = route.restore(values, updated)
            else:
                values = block(values, modulation, context, frequencies)
            if self.feature_recorder is not None:
                for row, grid in zip(values, grids, strict=True):
                    self.feature_recorder.record(index, row[: math.prod(grid)].unsqueeze(0))
        predictions = self.head(values, time)
        outputs = []
        for row, grid in zip(predictions, grids, strict=True):
            patches = row[: math.prod(grid)].view(*grid, *config.patch_size, config.out_channels)
            clips = torch.einsum("fhwpqrc->cfphqwr", patches)
            outputs.append(
                clips.reshape(
                    config.out_channels, *(a * b for a, b in zip(grid, config.patch_size, strict=True))
                ).float()
            )
        return tuple(outputs)

    forward = predict

    def validate_inputs(
        self,
        latents: tuple[Tensor, ...],
        timesteps: Tensor,
        captions: tuple[Tensor, ...],
        sequence_length: int,
        conditioning: tuple[Tensor, ...] | None,
        image_features: Tensor | None,
        frame_indices: tuple[tuple[int, ...], ...] | None,
    ) -> tuple[VideoGrid, ...]:
        """Validate host topology without inspecting or synchronizing tensor values."""
        config = self.config
        batch = len(latents)
        if not batch or len(captions) != batch or sequence_length < 1:
            raise ValueError("Wan requires nonempty matching clips/captions and positive sequence_length")
        if conditioning is not None and len(conditioning) != batch:
            raise ValueError("Wan conditioning batch differs from latents")
        if frame_indices is not None and len(frame_indices) != batch:
            raise ValueError("Wan frame_indices batch differs from latents")
        allowed_times = ((batch,),) if config.version == "2.1" else ((batch,), (batch, sequence_length))
        if tuple(timesteps.shape) not in allowed_times:
            raise ValueError("Wan timesteps must be B for 2.1, B or BL for 2.2")
        reference = self.patch_embedding.weight
        if reference.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Wan transformer requires FP32/FP16/BF16 parameters")
        tensors = [*latents, *captions, timesteps]
        if conditioning is not None:
            tensors.extend(conditioning)
        if image_features is not None:
            tensors.append(image_features)
        if any(value.device != reference.device or value.dtype != torch.float32 for value in tensors):
            raise ValueError("Wan inputs must be FP32 on the model device")
        if config.image_context:
            image_shape = (batch, 2, 257, 1280) if config.variant == "flf2v" else (batch, 257, 1280)
            if image_features is None or image_features.shape != image_shape:
                raise ValueError(f"Wan2.1 {config.variant} requires image_features shape{image_shape}")
        elif image_features is not None:
            raise ValueError("Wan image_features only apply to2.1 I2V/FLF2V")
        grids = []
        for index, (clip, caption) in enumerate(zip(latents, captions, strict=True)):
            if (
                clip.ndim != 4
                or caption.ndim != 2
                or caption.shape[1] != config.text_dim
                or not 0 < caption.shape[0] <= config.text_length
            ):
                raise ValueError("Wan requires CFHW clips and nonempty SD captions within text_length")
            channels = clip.shape[0]
            if conditioning is not None:
                condition = conditioning[index]
                if condition.ndim != 4 or condition.shape[1:] != clip.shape[1:]:
                    raise ValueError("Wan condition geometry must match its latent clip")
                channels += condition.shape[0]
            if channels != config.in_channels:
                raise ValueError("Wan latent plus conditioning channels differ from in_channels")
            if any(size < patch or size % patch for size, patch in zip(clip.shape[1:], config.patch_size, strict=True)):
                raise ValueError("Wan clip dimensions must be positive multiples of patch_size")
            f, h, w = (size // patch for size, patch in zip(clip.shape[1:], config.patch_size, strict=True))
            grid = (f, h, w)
            if max(grid) > 1024 or math.prod(grid) > sequence_length:
                raise ValueError("Wan grid exceeds native rotary table or sequence_length")
            if frame_indices is not None:
                frames = frame_indices[index]
                if len(frames) != f or any(type(frame) is not int or not 0 <= frame < 1024 for frame in frames):
                    raise ValueError("Wan frame_indices require one integer in [0,1024) per patched frame")
            grids.append(grid)
        return tuple(grids)
