# Copyright 2025-2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Diffusers PR #14355 (abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc)
# via the pinned Musubi and old MixTuner implementations.
"""Compute native joint audio/video predictions without training or resource orchestration."""

from collections.abc import Callable, Sequence
from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.config.routing import TreadConfig
from unitunner.execution.routing import TargetSpan, sample_token_route
from unitunner.models.components.features import FeatureRecorder
from unitunner.models.minimax_h3.config.model import MiniMaxH3Config
from unitunner.models.minimax_h3.layers import DiTBlock, FinalLayer, RotationFrequencies, TimeEmbedder, TokenRefiner
from unitunner.models.minimax_h3.layout import TARGET_FPS, MiniMaxH3PackedLayout, MiniMaxH3RowSegment
from unitunner.models.minimax_h3.packing import MiniMaxH3Targets, pack_audio_rows, pack_video_rows, unpack_targets
from unitunner.models.minimax_h3.positions import build_position_grid
from unitunner.models.minimax_h3.timesteps import build_timestep_rows


class MiniMaxH3Transformer(nn.Module):
    """Own native packed AV residual blocks and explicit FP32 projection/time islands.

    Eager resident Torch attention only; input times use0=noise,1=clean.
    The first component acceptance uses FP32 compute and supplied artifacts.
    Caller owns adapters, execution strategy, frozen encoders and sampling.
    """

    adaln_t_table: Tensor
    adaln_basis: Tensor
    adaln_mean: Tensor

    def __init__(
        self,
        config: MiniMaxH3Config,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_recorder: FeatureRecorder | None = None
        self.token_routing: TreadConfig | None = None
        self.feature_width: int = config.hidden_size
        self.feature_patch: tuple[int, int, int] = config.patch_size
        self.video_patch_proj = nn.Linear(
            config.video_patch_dim, config.hidden_size, dtype=torch.float32, device=device
        )
        self.audio_patch_proj = nn.Linear(
            config.audio_in_channels, config.hidden_size, dtype=torch.float32, device=device
        )
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size, dtype=dtype, device=device)
        if config.adaln_curve_grid is not None:
            self.time_embedder = None
            self.register_buffer(
                "adaln_t_table",
                torch.empty(config.adaln_curve_grid, config.time_embed_dim, dtype=torch.float32, device=device),
            )
        else:
            self.time_embedder = TimeEmbedder(
                config.timestep_input_dim,
                config.time_embed_hidden_size,
                config.time_embed_dim,
                device=device,
            )
            if config.adaln_rank is not None:
                # load-time mean-centered SVD basis of the SiLU'd time-embedding curve
                self.register_buffer(
                    "adaln_basis",
                    torch.empty(config.time_embed_dim, config.adaln_rank, dtype=torch.float32, device=device),
                )
                self.register_buffer(
                    "adaln_mean",
                    torch.empty(config.time_embed_dim, dtype=torch.float32, device=device),
                )
        self.rope = RotationFrequencies(config.rope_inv_freq_len, device)
        self.token_refiner = TokenRefiner(
            config,
            dtype=dtype,
            device=device,
        )
        self.blocks = nn.ModuleList([DiTBlock(config, dtype=dtype, device=device) for _ in range(config.num_layers)])
        self.final_layer = FinalLayer(
            config.hidden_size,
            config.adaln_in_features,
            config.video_patch_dim,
            config.audio_in_channels,
            norm_eps=config.final_norm_eps,
            dtype=dtype,
            apply_silu=not config.is_pruned,
            device=device,
        )

    @property
    def device(self) -> torch.device:
        """Return the device owning the native FP32 visual projection."""
        return self.video_patch_proj.weight.device

    @property
    def dtype(self) -> torch.dtype:
        # norms are never quantized, so this sentinel stays the compute dtype even when a
        # checkpoint quantizes more Linears than the published ConvRot INT8 scope
        """Return the resident compute dtype independently of FP32 projection islands."""
        first = self.blocks[0]
        if not isinstance(first, DiTBlock):
            raise TypeError("MiniMax-H3 compute dtype requires a native DiT block")
        return first.norm1.weight.dtype

    def _timestep_embeddings(self, unique_timesteps: Tensor, execution_device: torch.device) -> Tensor:
        """Build native eager, curve-interpolated or basis-projected AdaLN embeddings."""
        if unique_timesteps.ndim != 1:
            raise ValueError("MiniMax-H3 unique timesteps must be a one-dimensional tensor")
        if not self.config.is_pruned:
            if self.time_embedder is None:
                raise RuntimeError("Full MiniMax-H3 requires its time embedder")
            return self.time_embedder(unique_timesteps.to(execution_device)).to(self.dtype)

        if self.config.adaln_rank is not None:
            # self-pruned AdaLN: the exact FP32 time-embedding curve projected onto the
            # load-time basis; the trailing constant channel applies the curve mean via
            # the per-block projections (their weights are W @ [basis | mean])
            if self.time_embedder is None:
                raise RuntimeError("Self-pruned MiniMax-H3 requires its time embedder")
            embedding = self.time_embedder(unique_timesteps.to(execution_device))
            centered = F.silu(embedding) - self.adaln_mean
            with torch.autocast(device_type=execution_device.type, enabled=False):
                coefficients = F.pad(centered @ self.adaln_basis, (0, 1), value=1.0)
            return coefficients.to(self.dtype)

        # pruned AdaLN: FP32 linear interpolation over the published [1025, 8] curve
        # table; the model time t = 1 - sigma lives in [0, 1] and maps onto the grid
        table = self.adaln_t_table
        timesteps_fp32 = unique_timesteps.to(device=table.device, dtype=torch.float32)
        position = timesteps_fp32.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        fraction = position - lower.to(position.dtype)
        embedding_fp32 = torch.lerp(table[lower], table[lower + 1], fraction[:, None])
        return embedding_fp32.to(device=execution_device, dtype=self.dtype)

    def _rotation_table(
        self,
        position_ids: Tensor,
        dtype: torch.dtype,
        *,
        fine_position_ids: Tensor | None = None,
        fine_bands: int = 0,
    ) -> Tensor:
        """Construct native FP32 rotations including optional unstretched fine bands."""
        positions = position_ids.to(device=self.rope.inv_freq.device, dtype=torch.float32)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("MiniMax-H3 position ids must be [B,S,3]")
        per_axis = positions.unsqueeze(-1) * self.rope.inv_freq.reshape(1, 1, 1, -1)
        temporal, height, width = per_axis.unbind(dim=2)
        if fine_position_ids is not None and fine_bands:
            # temporal-stretch experiment: the leading (sub-lattice) bands rotate by the
            # unstretched grid so the per-token lattice phase stays as trained, while the
            # remaining bands carry the stretched physical timeline
            if fine_bands >= self.rope.inv_freq.shape[0]:
                raise ValueError(
                    f"MiniMax-H3 temporal fine bands {fine_bands} must leave at least one of the"
                    f" {self.rope.inv_freq.shape[0]} rotary bands on the stretched clock"
                )
            fine_positions = fine_position_ids.to(device=self.rope.inv_freq.device, dtype=torch.float32)
            if fine_positions.shape != positions.shape:
                raise ValueError("MiniMax-H3 fine position ids must match the stretched grid shape")
            fine_temporal = fine_positions[..., 0].unsqueeze(-1) * self.rope.inv_freq.reshape(1, 1, -1)
            temporal = torch.cat((fine_temporal[..., :fine_bands], temporal[..., fine_bands:]), dim=-1)
        half = torch.cat((temporal, height, width), dim=-1)
        angles = torch.cat((half, half), dim=-1)
        pair_count = angles.shape[-1] // 2
        angles = angles[..., :pair_count]
        cosine, sine = torch.cos(angles), torch.sin(angles)
        return (
            torch.stack((cosine, -sine, sine, cosine), dim=-1)
            .reshape(
                angles.shape[0],
                angles.shape[1],
                1,
                pair_count,
                2,
                2,
            )
            .to(dtype)
        )

    @staticmethod
    def _validate_text_tags(tags: Tensor, batch_size: int, text_length: int) -> Tensor:
        """Check batch/text geometry before native token-tag validation."""
        tags = torch.as_tensor(tags)
        if tags.shape != (batch_size, text_length):
            raise ValueError(f"MiniMax-H3 text token tags must be [{batch_size},{text_length}]")
        return tags

    @staticmethod
    def _validate_condition_rows(
        latents: Sequence[Tensor],
        segments: Sequence[MiniMaxH3RowSegment],
        pack: Callable[[Tensor], Tensor],
        *,
        batch_size: int,
        label: str,
    ) -> list[Tensor]:
        """Pack explicitly aligned condition segments with matching batch ownership."""
        if len(latents) != len(segments):
            raise ValueError(f"MiniMax-H3 expected {len(segments)} {label} tensors, got {len(latents)}")
        rows = []
        for tensor, segment in zip(latents, segments, strict=True):
            if tensor.shape[0] != batch_size:
                raise ValueError(f"MiniMax-H3 {segment.role} batch size does not match the targets")
            packed = pack(tensor)
            if packed.shape[1] != segment.row_count:
                raise ValueError(
                    f"MiniMax-H3 {segment.role} has {packed.shape[1]} packed rows, expected {segment.row_count}"
                )
            rows.append(packed)
        return rows

    def predict(
        self,
        *,
        video_latents: Tensor,
        audio_latents: Tensor,
        text_hidden_states: Tensor,
        text_token_tags: Tensor,
        layout: MiniMaxH3PackedLayout,
        model_t_video: float | Tensor,
        model_t_audio: float | Tensor,
        visual_condition_latents: Sequence[Tensor] = (),
        audio_condition_latents: Sequence[Tensor] = (),
        visual_condition_clean: float = 0.999,
        audio_condition_clean: float = 1.0,
    ) -> MiniMaxH3Targets:
        """Return native video/audio predictions for one packed AV sample at clean-fraction times."""
        if video_latents.ndim != 5 or tuple(video_latents.shape[2:]) != (
            layout.target_video.frames,
            layout.target_video.height,
            layout.target_video.width,
        ):
            raise ValueError("MiniMax-H3 target video tensor does not match the packed layout")
        if audio_latents.ndim != 4 or tuple(audio_latents.shape[1:]) != (
            self.config.audio_in_channels,
            2,
            layout.target_audio_frames,
        ):
            raise ValueError("MiniMax-H3 target audio tensor does not match the packed layout")
        batch_size = video_latents.shape[0]
        if batch_size != 1:
            raise ValueError(f"MiniMax-H3 R1 requires batch_size=1, got {batch_size}; use gradient accumulation")
        if audio_latents.shape[0] != batch_size:
            raise ValueError("MiniMax-H3 target video and audio batch sizes differ")
        if text_hidden_states.shape != (batch_size, layout.text_length, self.config.text_dim):
            raise ValueError(
                f"MiniMax-H3 text hidden states must be [{batch_size},{layout.text_length},{self.config.text_dim}]"
            )
        text_token_tags = self._validate_text_tags(text_token_tags, batch_size, layout.text_length)

        execution_device = self.video_patch_proj.weight.device
        video_dtype = video_latents.dtype
        audio_dtype = audio_latents.dtype
        video_latents = video_latents.to(execution_device)
        audio_latents = audio_latents.to(execution_device)
        text_hidden_states = text_hidden_states.to(execution_device)
        visual_condition_latents = tuple(tensor.to(execution_device) for tensor in visual_condition_latents)
        audio_condition_latents = tuple(tensor.to(execution_device) for tensor in audio_condition_latents)

        visual_segments = tuple(segment for segment in layout.segments if segment.kind == "visual_condition")
        audio_segments = tuple(segment for segment in layout.segments if segment.kind == "audio_condition")
        if layout.task == "fl2va":
            expected_visual_geometries = layout.visual_conditions
        elif layout.task == "ref2va":
            expected_visual_geometries = tuple(
                reference.video for reference in layout.references if reference.kind in {"image", "video"}
            )
        else:
            expected_visual_geometries = ()
        expected_audio_frames = (
            tuple(reference.audio_frames for reference in layout.references if reference.audio_frames)
            if layout.task == "ref2va"
            else ()
        )
        for tensor, segment, geometry in zip(
            visual_condition_latents, visual_segments, expected_visual_geometries, strict=True
        ):
            actual = tuple(tensor.shape[2:]) if tensor.ndim == 5 else tuple(tensor.shape)
            if geometry is None:
                raise ValueError("Visual reference is missing its video geometry")
            expected = (geometry.frames, geometry.height, geometry.width)
            if actual != expected:
                raise ValueError(
                    f"MiniMax-H3 {segment.role} geometry expected {'x'.join(map(str, expected))}, got {'x'.join(map(str, actual))}"
                )
        for tensor, segment, frames in zip(audio_condition_latents, audio_segments, expected_audio_frames, strict=True):
            actual = tuple(tensor.shape[1:]) if tensor.ndim == 4 else tuple(tensor.shape)
            expected = (self.config.audio_in_channels, 2, frames)
            if actual != expected:
                raise ValueError(
                    f"MiniMax-H3 {segment.role} geometry expected {'x'.join(map(str, expected))}, got {'x'.join(map(str, actual))}"
                )
        visual_rows = self._validate_condition_rows(
            visual_condition_latents,
            visual_segments,
            pack_video_rows,
            batch_size=batch_size,
            label="visual condition",
        )
        audio_rows = self._validate_condition_rows(
            audio_condition_latents,
            audio_segments,
            pack_audio_rows,
            batch_size=batch_size,
            label="audio condition",
        )
        visual_rows.append(pack_video_rows(video_latents))
        audio_rows.append(pack_audio_rows(audio_latents))
        with torch.autocast(device_type=execution_device.type, enabled=False):
            projected_video = self.video_patch_proj(torch.cat(visual_rows, dim=1).to(torch.float32)).to(self.dtype)
            projected_audio = self.audio_patch_proj(torch.cat(audio_rows, dim=1).to(torch.float32)).to(self.dtype)

        text = self.condition_proj(text_hidden_states.to(self.dtype))
        text = self.token_refiner(text)
        parts = []
        visual_offset = 0
        audio_offset = 0
        for segment in layout.segments:
            if segment.kind == "text":
                parts.append(text)
            elif segment.kind in {"visual_condition", "target_video"}:
                parts.append(projected_video[:, visual_offset : visual_offset + segment.row_count])
                visual_offset += segment.row_count
            else:
                parts.append(projected_audio[:, audio_offset : audio_offset + segment.row_count])
                audio_offset += segment.row_count
        hidden_states = torch.cat(parts, dim=1)
        if hidden_states.shape[1] != layout.row_count:
            raise RuntimeError("MiniMax-H3 packed residual stream length drifted from its layout")

        timestep_rows = build_timestep_rows(
            layout,
            text_token_tags=text_token_tags,
            model_t_video=model_t_video,
            model_t_audio=model_t_audio,
            visual_condition_clean=visual_condition_clean,
            audio_condition_clean=audio_condition_clean,
        )
        timestep_embeddings = self._timestep_embeddings(timestep_rows.unique_timesteps, execution_device)
        position_ids = build_position_grid(layout, device=execution_device)
        fine_positions = (
            build_position_grid(replace(layout, output_fps=TARGET_FPS, temporal_fine_bands=0), device=execution_device)
            if layout.temporal_fine_bands and layout.output_fps != TARGET_FPS
            else None
        )
        rotation_table = self._rotation_table(
            position_ids, hidden_states.dtype, fine_position_ids=fine_positions, fine_bands=layout.temporal_fine_bands
        )

        target_segment = layout.target_video_segment
        route = sample_token_route(
            self.token_routing,
            training=self.training,
            length=hidden_states.shape[1],
            spans=((TargetSpan(target_segment.start, target_segment.stop),),),
            device=hidden_states.device,
        )
        for index, block in enumerate(self.blocks):
            if route is not None and route.applies(index):
                segments = tuple(
                    (route.boundary(start), route.boundary(stop), time)
                    for start, stop, time in timestep_rows.block_segments
                )
                updated = block(
                    route.select(hidden_states), timestep_embeddings, segments, route.select(rotation_table)
                )
                hidden_states = route.restore(hidden_states, updated)
            else:
                hidden_states = block(hidden_states, timestep_embeddings, timestep_rows.block_segments, rotation_table)
            if self.feature_recorder is not None:
                self.feature_recorder.record(index, hidden_states[:, layout.target_video_segment.row_slice])

        video_rows, audio_rows = self.final_layer(
            hidden_states,
            timestep_embeddings,
            video_slice=layout.target_video_segment.row_slice,
            audio_slice=layout.target_audio_segment.row_slice,
            video_timestep_index=timestep_rows.video_timestep_index,
            audio_timestep_index=timestep_rows.audio_timestep_index,
        )
        targets = unpack_targets(layout, video_rows, audio_rows)
        return MiniMaxH3Targets(video=targets.video.to(video_dtype), audio=targets.audio.to(audio_dtype))

    forward = predict
