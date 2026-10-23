# Derived from Black Forest Labs FLUX, Apache-2.0.
"""Compute native FLUX.2 tokens without training or execution-policy ownership."""

import torch
from torch import Tensor, nn

from unitunner.config.routing import TreadConfig
from unitunner.execution.routing import TargetSpan, sample_token_route
from unitunner.models.components.features import FeatureRecorder
from unitunner.models.flux2.config.model import Flux2Config
from unitunner.models.flux2.layers import (
    DoubleStreamBlock,
    LastLayer,
    MLPEmbedder,
    Modulation,
    SingleStreamBlock,
    rotary_embedding,
    timestep_embedding,
)


class Flux2Transformer(nn.Module):
    """Own published Dev/Klein weight keys and globally shared modulation.

    References are explicit image tokens with caller-defined four-axis IDs.
    This dense model consumes all supplied text/image tokens without masking.
    Execution owners install recompute/compile boundaries outside the network.
    """

    def __init__(self, config: Flux2Config) -> None:
        super().__init__()
        self.config = config
        self.feature_recorder: FeatureRecorder | None = None
        self.token_routing: TreadConfig | None = None
        self.feature_width: int = config.hidden_size
        self.feature_patch: tuple[int, int, int] = (1, 1, 1)
        dim = config.hidden_size
        self.img_in = nn.Linear(config.in_channels, dim, bias=False)
        self.time_in = MLPEmbedder(256, dim)
        self.txt_in = nn.Linear(config.context_in_dim, dim, bias=False)
        self.guidance_in = MLPEmbedder(256, dim) if config.use_guidance_embed else None
        self.double_blocks = nn.ModuleList(
            DoubleStreamBlock(dim, config.num_heads, config.mlp_ratio) for _ in range(config.depth)
        )
        self.single_blocks = nn.ModuleList(
            SingleStreamBlock(dim, config.num_heads, config.mlp_ratio) for _ in range(config.depth_single_blocks)
        )
        self.double_stream_modulation_img = Modulation(dim, double=True)
        self.double_stream_modulation_txt = Modulation(dim, double=True)
        self.single_stream_modulation = Modulation(dim, double=False)
        self.final_layer = LastLayer(dim, config.in_channels)

    def predict(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None,
        *,
        target_count: int | None = None,
    ) -> Tensor:
        """Return BxLxC noise-minus-clean velocity for packed image/reference tokens.

        X/context are nonempty BLD floating tensors with configured widths. IDs
        are matching BxLx4 integer or floating positions, and timesteps are B noise fractions.
        Dev requires B embedded guidance; Klein rejects unused guidance. All
        tensors share a device. Inputs, RNG and modes remain caller-owned. FP32
        resident execution is the initial policy; low precision needs acceptance.
        """
        if x.ndim != 3 or ctx.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or ctx.shape[1] == 0:
            raise ValueError("FLUX.2 image/context must be nonempty BLD tensors")
        batch = x.shape[0]
        if x.shape[2] != self.config.in_channels or ctx.shape != (batch, ctx.shape[1], self.config.context_in_dim):
            raise ValueError("FLUX.2 image/context widths or batches differ from topology")
        if x_ids.shape != (batch, x.shape[1], 4) or ctx_ids.shape != (batch, ctx.shape[1], 4):
            raise ValueError("FLUX.2 IDs must match image/context BxLx4 geometry")
        if timesteps.shape != (batch,):
            raise ValueError("FLUX.2 timesteps must be a B vector of noise fractions")
        if any(not value.is_floating_point() or value.device != x.device for value in (x, ctx, timesteps)):
            raise ValueError("FLUX.2 inputs must be floating tensors on the image device")
        if any(
            value.device != x.device
            or value.dtype
            not in (torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64)
            for value in (x_ids, ctx_ids)
        ):
            raise ValueError("FLUX.2 IDs must be real integer/floating positions on the image device")
        if self.guidance_in is not None:
            if (
                guidance is None
                or guidance.shape != (batch,)
                or guidance.device != x.device
                or not guidance.is_floating_point()
            ):
                raise ValueError("FLUX.2 Dev requires floating B embedded guidance on the image device")
        elif guidance is not None:
            raise ValueError("FLUX.2 Klein does not consume embedded guidance")
        vector = self.time_in(timestep_embedding(timesteps))
        if self.guidance_in is not None and guidance is not None:
            vector = vector + self.guidance_in(timestep_embedding(guidance))
        img_first, img_second = self.double_stream_modulation_img(vector)
        txt_first, txt_second = self.double_stream_modulation_txt(vector)
        single, _ = self.single_stream_modulation(vector)
        if img_second is None or txt_second is None:
            raise RuntimeError("FLUX.2 double-stream modulation requires two groups")
        img, txt = self.img_in(x), self.txt_in(ctx)
        positions = rotary_embedding(torch.cat((ctx_ids, x_ids), dim=1), self.config.axes_dim, self.config.theta)
        if (
            self.token_routing is not None
            and self.training
            and torch.is_grad_enabled()
            and (target_count is None or not 0 < target_count <= img.shape[1])
        ):
            raise ValueError("FLUX.2 TREAD requires explicit target_count within image tokens")
        target_length = img.shape[1] if target_count is None else target_count
        route = sample_token_route(
            self.token_routing,
            training=self.training,
            length=img.shape[1],
            spans=((TargetSpan(0, target_length),),),
            device=img.device,
        )
        joint_route = (
            None
            if route is None
            else route.relocate(img.shape[1] + txt.shape[1], (TargetSpan(txt.shape[1], txt.shape[1] + target_length),))
        )
        for index, block in enumerate(self.double_blocks):
            if route is not None and joint_route is not None and route.applies(index):
                updated, txt = block(
                    route.select(img),
                    txt,
                    joint_route.select(positions, dim=2),
                    (img_first, img_second),
                    (txt_first, txt_second),
                )
                img = route.restore(img, updated)
            else:
                img, txt = block(img, txt, positions, (img_first, img_second), (txt_first, txt_second))
            if self.feature_recorder is not None and self.feature_recorder.active:
                for row, ids in zip(img, x_ids, strict=True):
                    self.feature_recorder.record(index, row[ids[:, 0] == 0].unsqueeze(0))
        joint = torch.cat((txt, img), dim=1)
        for index, block in enumerate(self.single_blocks, start=len(self.double_blocks)):
            if joint_route is not None and joint_route.applies(index):
                updated = block(joint_route.select(joint), joint_route.select(positions, dim=2), single)
                joint = joint_route.restore(joint, updated)
            else:
                joint = block(joint, positions, single)
            if self.feature_recorder is not None and self.feature_recorder.active:
                for row, ids in zip(joint[:, txt.shape[1] :], x_ids, strict=True):
                    self.feature_recorder.record(index, row[ids[:, 0] == 0].unsqueeze(0))
        return self.final_layer(joint[:, txt.shape[1] :], vector)

    forward = predict
