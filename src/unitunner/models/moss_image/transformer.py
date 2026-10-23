# Adapted from OpenMOSS MOSS-Image and kohya-ss/sd-scripts, Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Compute native MOSS-Image velocities without trainer, adapter or execution-policy dependencies."""

import torch
from torch import Tensor, nn

from unitunner.config.routing import TreadConfig
from unitunner.execution.routing import TargetSpan, sample_token_route
from unitunner.models.components.features import FeatureRecorder
from unitunner.models.moss_image.config.model import MossImageConfig
from unitunner.models.moss_image.editing import reference_rope
from unitunner.models.moss_image.layers import MossImageBlock, RMSNorm, RotaryEmbedding, TextEmbedder


class MossImageTransformer(nn.Module):
    """Predict noise-minus-clean velocities on normalized packed BCHW latents.

    Preserve published parameter keys, text refiners, shared timestep modulation
    and four-axis half-split RoPE. Callers own mode, dtype/device and reversible
    adapter/recompute/compile installations. Base and SFT share this topology.
    """

    def __init__(self, config: MossImageConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_recorder: FeatureRecorder | None = None
        self.token_routing: TreadConfig | None = None
        self.feature_width: int = config.dim
        self.feature_patch: tuple[int, int, int] = (1, 1, 1)
        self.video_embedder = nn.Linear(config.latent_channels, config.dim)
        self.text_embedder = TextEmbedder(
            config.text_dim, config.dim, config.ffn_dim, config.num_heads, config.text_blocks
        )
        self.rope = RotaryEmbedding(config.dim // config.num_heads, config.rope_base)
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim)
        )
        self.shared_block_modulation = nn.Linear(config.dim, 6 * config.dim)
        self.layers = nn.ModuleList(
            MossImageBlock(config.dim, config.ffn_dim, config.num_heads) for _ in range(config.num_layers)
        )
        self.final_norm_video = RMSNorm(config.dim)
        self.final_linear_video = nn.Linear(config.dim, config.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(config.dim, 2 * config.dim)

    def time_modulation(self, timesteps: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
        """Embed scalar or B sigma*1000 using FP64 phases and native FP32 cosine/sine order."""
        parameter = self.video_embedder.weight
        times = timesteps.to(device=parameter.device, dtype=torch.float32).reshape(-1).expand(batch_size)
        half = self.config.freq_dim // 2
        frequencies = 10000 ** (-torch.arange(half, device=parameter.device, dtype=torch.float64) / half)
        phases = times.double()[:, None] * frequencies
        embedding = torch.cat((phases.cos(), phases.sin()), -1).float()
        vector = self.time_embedding(embedding.to(parameter.dtype)).float()
        modulation = self.shared_block_modulation(vector.to(parameter.dtype)).view(batch_size, 6, -1)
        head = self.video_head_modulation(vector.to(parameter.dtype)).view(batch_size, 2, -1)
        return modulation, head

    def reference_inputs(self, references: tuple[Tensor, ...], batch_size: int) -> tuple[Tensor, Tensor]:
        """Project normalized BCHW references; singleton references may broadcast over a batch."""
        parameter = self.video_embedder.weight
        if self.config.variant != "edit" or not references:
            raise ValueError("MOSS reference conditioning requires the edit variant and reference images")
        for value in references:
            if (
                value.ndim != 4
                or value.shape[0] not in (1, batch_size)
                or value.shape[1] != self.config.latent_channels
                or min(value.shape[-2:]) < 1
                or not value.is_floating_point()
                or value.device != parameter.device
            ):
                raise ValueError(
                    "MOSS references require floating BCHW on the model device with matching channels/batch"
                )
        tokens = torch.cat(
            tuple(
                self.video_embedder(value.expand(batch_size, -1, -1, -1).flatten(2).transpose(1, 2).to(parameter.dtype))
                for value in references
            ),
            1,
        )
        positions = reference_rope(
            tuple((value.shape[-2], value.shape[-1]) for value in references), self.rope.default_bands(parameter.device)
        ).to(tokens.dtype)
        return tokens, positions.expand(batch_size, -1, -1)

    def _prepare_reference_kv(
        self,
        references: tuple[Tensor, ...],
        batch_size: int,
        skip_layers: tuple[int, ...] = (),
    ) -> tuple[tuple[Tensor, Tensor], ...]:
        """Build sampler-local K/V; skipping a layer also changes downstream reference features."""
        if torch.is_grad_enabled() or any(module.training for module in self.modules()):
            raise ValueError("MOSS reference K/V prefill is inference-only")
        values, rope = self.reference_inputs(references, batch_size)
        modulation, _ = self.time_modulation(values.new_zeros(1), batch_size)
        cached = []
        for index, block in enumerate(self.layers):
            if index in skip_layers:
                # These entries are never read by this same skipped-layer route.
                cached.append((values, values))
            else:
                values = block(values, rope, modulation, (values.shape[1],) * batch_size, kv_output=cached)
        return tuple(cached)

    def predict(
        self,
        latents: Tensor,
        timesteps: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
        *,
        skip_layers: tuple[int, ...] = (),
        references: tuple[Tensor, ...] = (),
        _reference_kv: tuple[tuple[Tensor, Tensor], ...] | None = None,
    ) -> Tensor:
        """Return BCHW velocity with differentiable latent/context inputs and native times.

        Latents and context are floating BCHW/BLD on the model device. Text uses
        nonempty right-padded prefix masks (bool or binary numeric), never sparse
        masks. Time is a scalar or B vector in native sigma*1000 units. Boundary
        mask conversion uses one CPU read; attention excludes all padded keys.
        Skip layers are valid main-block indices, supported only in eval mode.
        """
        parameter = self.video_embedder.weight
        if (
            latents.ndim != 4
            or not latents.numel()
            or latents.shape[1] != self.config.latent_channels
            or not latents.is_floating_point()
            or latents.device != parameter.device
        ):
            raise ValueError("MOSS requires nonempty floating BCHW packed latents on the model device")
        batch_size, _, height, width = latents.shape
        if (
            context.ndim != 3
            or context.shape[0] != batch_size
            or context.shape[1] < 1
            or context.shape[2] != self.config.text_dim
            or not context.is_floating_point()
            or context.device != parameter.device
        ):
            raise ValueError("MOSS requires nonempty floating BLD text features on the model device")
        if timesteps.ndim > 1 or timesteps.numel() not in (1, batch_size) or not timesteps.is_floating_point():
            raise ValueError("MOSS native timesteps must be floating scalar or B values")
        if context_mask is None:
            lengths = (context.shape[1],) * batch_size
        else:
            host = context_mask.detach().cpu()
            if host.shape != context.shape[:2] or not ((host == 0) | (host == 1)).all():
                raise ValueError("MOSS context masks must be binary BL tensors")
            host = host.bool()
            counts = host.sum(1)
            if (counts < 1).any() or not torch.equal(host, torch.arange(host.shape[1])[None] < counts[:, None]):
                raise ValueError("MOSS context masks require nonempty right-padded prefixes")
            lengths = tuple(counts.tolist())
        skipped = frozenset(skip_layers)
        if any(type(index) is not int or not 0 <= index < len(self.layers) for index in skipped):
            raise ValueError("MOSS skip-layer index outside main blocks")
        if self.training and skipped:
            raise ValueError("MOSS layer skipping is inference-only")
        if self.config.variant == "edit" and not references and _reference_kv is None:
            raise ValueError("MOSS edit requires reference images")
        if self.config.variant != "edit" and (references or _reference_kv is not None):
            raise ValueError("Reference conditioning requires MOSS edit variant")
        if _reference_kv is not None and (
            torch.is_grad_enabled() or self.training or references or len(_reference_kv) != len(self.layers)
        ):
            raise ValueError("MOSS cached references require inference without live references")
        if self.config.variant == "edit" and self.token_routing is not None:
            raise ValueError("MOSS edit does not yet support token routing")
        image_length = height * width
        video = self.video_embedder(latents.flatten(2).transpose(1, 2).to(parameter.dtype))
        rope = self.rope(height, width, context.shape[1], torch.tensor(lengths, device=latents.device)).to(video.dtype)
        text = self.text_embedder.proj(context.to(parameter.dtype))
        for block in self.text_embedder.blocks:
            text = block(text, rope[:, image_length:], lengths)
        hidden = torch.cat((video, text), 1)
        modulation, head = self.time_modulation(timesteps, batch_size)
        joint_lengths = tuple(image_length + length for length in lengths)
        reference_count, reference_modulation = 0, None
        if references:
            reference_tokens, reference_positions = self.reference_inputs(references, batch_size)
            reference_count = reference_tokens.shape[1]
            hidden = torch.cat((hidden, reference_tokens), 1)
            rope = torch.cat((rope, reference_positions), 1)
            reference_modulation, _ = self.time_modulation(timesteps.new_zeros(1), batch_size)
        route = sample_token_route(
            self.token_routing,
            training=self.training,
            length=hidden.shape[1],
            spans=((TargetSpan(0, image_length),),),
            device=hidden.device,
        )
        for index, block in enumerate(self.layers):
            if index not in skipped:
                if route is not None and route.applies(index):
                    updated = block(
                        route.select(hidden),
                        route.select(rope),
                        modulation,
                        tuple(route.boundary(length) for length in joint_lengths),
                    )
                    hidden = route.restore(hidden, updated)
                elif reference_count or _reference_kv is not None:
                    hidden = block(
                        hidden,
                        rope,
                        modulation,
                        joint_lengths,
                        reference_count,
                        reference_modulation,
                        None if _reference_kv is None else _reference_kv[index],
                    )
                else:
                    hidden = block(hidden, rope, modulation, joint_lengths)
                if self.feature_recorder is not None:
                    self.feature_recorder.record(index, hidden[:, :image_length])
        hidden = self.final_norm_video(hidden[:, :image_length].to(parameter.dtype))
        hidden = hidden * (1 + head[:, 1, None]) + head[:, 0, None]
        return (
            self.final_linear_video(hidden.to(parameter.dtype)).transpose(1, 2).reshape(batch_size, -1, height, width)
        )

    forward = predict
