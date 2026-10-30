# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See licenses/Apache-2.0.txt.
"""Compute native Qwen Image predictions with explicit ordered image grids."""

from math import prod

import torch
from torch import Tensor, nn

from unituner.config.routing import TreadConfig
from unituner.execution.routing import TargetSpan, sample_token_route
from unituner.models.components.features import FeatureRecorder
from unituner.models.qwen_image.config.model import QwenImageConfig
from unituner.models.qwen_image.layers import AdaptiveOutputNorm, QwenImageBlock, QwenTimeEmbedding, RMSNorm
from unituner.models.qwen_image.positions import ImageGrid, build_rotary


class QwenImageTransformer(nn.Module):
    """Own native dual-stream weights; borrow caller tokens, masks and normalized times.

    No checkpoint, routing, residency or training owner lives in this network.
    Ordered grids are shared across samples; callers with different geometry
    must make independent predictions rather than attending synthetic padding.
    """

    def __init__(self, config: QwenImageConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_recorder: FeatureRecorder | None = None
        self.token_routing: TreadConfig | None = None
        self.feature_width: int = config.num_attention_heads * config.attention_head_dim
        self.feature_patch: tuple[int, int, int] = (1, 2, 2)
        width = config.hidden_size
        self.time_text_embed = QwenTimeEmbedding(width, layered=config.variant == "layered")
        self.txt_norm = RMSNorm(config.joint_attention_dim)
        self.img_in = nn.Linear(config.in_channels, width)
        self.txt_in = nn.Linear(config.joint_attention_dim, width)
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageBlock(
                    width,
                    config.num_attention_heads,
                    zero_time=config.zero_reference_time,
                    output_text=index < config.num_layers - 1,
                )
                for index in range(config.num_layers)
            ]
        )
        self.norm_out = AdaptiveOutputNorm(width)
        self.proj_out = nn.Linear(width, config.patch_size**2 * config.out_channels)

    def predict(
        self,
        image: Tensor,
        text: Tensor,
        timesteps: Tensor,
        grids: tuple[ImageGrid, ...],
        *,
        text_mask: Tensor | None = None,
        layer_condition: Tensor | None = None,
        target_count: int | None = None,
    ) -> Tensor:
        """Predict BNC velocity for all target/reference tokens in their supplied order.

        Image is BNC native packed patches, text is BLD with optional boolean
        prefix mask. Time is a B vector in normalized flow units, not0..1000.
        Layered requires a B int64 RGB/RGBA condition (0 or1). Text padding keys
        are excluded; callers must trim unused all-padding trailing positions.
        Edit-2511 applies zero-time modulation after the first target grid.
        The model does not select target tokens or invert prediction signs.
        """
        if image.ndim != 3 or min(image.shape) < 1 or image.shape[-1] != self.config.in_channels:
            raise ValueError("Qwen Image requires nonempty BNC image patches with native input channels")
        if (
            text.ndim != 3
            or text.shape[0] != image.shape[0]
            or text.shape[1] < 1
            or text.shape[-1] != self.config.joint_attention_dim
            or text.device != image.device
            or text.dtype != image.dtype
            or not image.is_floating_point()
        ):
            raise ValueError("Qwen Image text must be matching floating BLD context on the image device")
        if (
            timesteps.shape != (image.shape[0],)
            or timesteps.device != image.device
            or not timesteps.is_floating_point()
            or not torch.all(torch.isfinite(timesteps) & (timesteps >= 0) & (timesteps <= 1))
        ):
            raise ValueError("Qwen Image requires finite normalized B flow times in[0,1]")
        if not grids or sum(prod(grid) for grid in grids) != image.shape[1]:
            raise ValueError("Qwen Image grids must describe exactly every image token")
        if text_mask is not None:
            if text_mask.shape != text.shape[:2] or text_mask.dtype != torch.bool or text_mask.device != text.device:
                raise ValueError("Qwen Image text_mask must be a matching BL boolean tensor")
            if not text_mask[:, 0].all() or not text_mask[:, -1].any() or (text_mask[:, 1:] & ~text_mask[:, :-1]).any():
                raise ValueError("Qwen Image text masks require nonempty prefixes and trimmed batch padding")
        if self.config.variant == "layered":
            if (
                layer_condition is None
                or layer_condition.shape != (image.shape[0],)
                or layer_condition.device != image.device
                or layer_condition.dtype != torch.int64
                or not torch.all((layer_condition == 0) | (layer_condition == 1))
            ):
                raise ValueError("Qwen Image layered requires B int64 RGB/RGBA condition in{0,1}")
        elif layer_condition is not None:
            raise ValueError("Qwen Image RGB/RGBA condition applies only to layered")
        visual_count = prod(grids[0]) if target_count is None else target_count
        if not 0 < visual_count <= image.shape[1]:
            raise ValueError("Qwen Image target_count must describe nonempty leading target tokens")
        rotary = build_rotary(
            grids,
            text.shape[1],
            self.config.axes_dims_rope,
            layered=self.config.variant == "layered",
            device=image.device,
        )
        hidden = self.img_in(image)
        context = self.txt_in(self.txt_norm(text))
        if hidden.dtype in (torch.bfloat16, torch.float16):
            times = (timesteps * 1000).to(hidden.dtype) / 1000
        else:
            times = timesteps.to(hidden.dtype)
        zero_index = prod(grids[0]) if self.config.zero_reference_time else None
        if self.config.zero_reference_time:
            times = torch.cat((times, times * 0))
        time = self.time_text_embed(times, hidden.dtype, layer_condition)
        route = sample_token_route(
            self.token_routing,
            training=self.training,
            length=hidden.shape[1],
            spans=((TargetSpan(0, visual_count),),),
            device=hidden.device,
        )
        for index, block in enumerate(self.transformer_blocks):
            if route is not None and route.applies(index):
                image_rope, text_rope = rotary
                updated_context, updated = block(
                    route.select(hidden),
                    context,
                    text_mask,
                    time,
                    route.select(image_rope, dim=0, batched=False),
                    text_rope,
                    None if zero_index is None else route.boundary(zero_index),
                )
                context, hidden = updated_context, route.restore(hidden, updated)
            else:
                context, hidden = block(hidden, context, text_mask, time, *rotary, zero_index)
            if self.feature_recorder is not None:
                self.feature_recorder.record(index, hidden[:, :visual_count])
        if self.config.zero_reference_time:
            time = time.chunk(2)[0]
        return self.proj_out(self.norm_out(hidden, time))

    forward = predict
