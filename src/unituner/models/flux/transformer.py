# Derived from Black Forest Labs FLUX, Apache-2.0.
"""Compute native FLUX.1 tokens without trainer, routing or execution-policy state."""

import torch
from torch import Tensor, nn

from unituner.config.routing import TreadConfig
from unituner.execution.routing import TargetSpan, sample_token_route
from unituner.models.components.features import FeatureRecorder
from unituner.models.flux.config.model import FluxConfig
from unituner.models.flux.layers import (
    DoubleStreamBlock,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    rotary_embedding,
    timestep_embedding,
)


class FluxTransformer(nn.Module):
    """Own native Dev/Kontext/Fill weights and preserve published parameter keys.

    Inputs are already packed tokens, including any explicit reference or Fill
    channels. Native timesteps are noise fractions, not a 0-1000 clock. Execution
    owners may wrap blocks; this module does not install or discover resources.
    """

    def __init__(self, config: FluxConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_recorder: FeatureRecorder | None = None
        self.token_routing: TreadConfig | None = None
        self.feature_width: int = config.hidden_size
        self.feature_patch: tuple[int, int, int] = (1, 2, 2)
        dim = config.hidden_size
        self.img_in = nn.Linear(config.in_channels, dim)
        self.time_in = MLPEmbedder(256, dim)
        self.vector_in = MLPEmbedder(config.vec_in_dim, dim)
        self.guidance_in = MLPEmbedder(256, dim) if config.guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(config.context_in_dim, dim)
        self.double_blocks = nn.ModuleList(
            DoubleStreamBlock(dim, config.num_heads, config.mlp_ratio, config.qkv_bias) for _ in range(config.depth)
        )
        self.single_blocks = nn.ModuleList(
            SingleStreamBlock(dim, config.num_heads, config.mlp_ratio) for _ in range(config.depth_single_blocks)
        )
        self.final_layer = LastLayer(dim, config.out_channels)

    def predict(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        *,
        control_lengths: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Return native noise-minus-clean velocity in Bximage_tokensxout_channels.

        Image/text inputs use floating BLD tensors with configured channel widths;
        IDs use BxLx3 floating positions on the same device. Pooled text is BxV;
        time and embedded guidance are floating B vectors. Optional Kontext
        control lengths count valid trailing reference tokens per row, with zero
        padding at the end. Validate geometry before projecting large tensors.
        Inputs, RNG and module modes remain unchanged; all floating paths retain
        gradients. Low precision follows caller policy and needs separate parity.
        """
        if img.ndim != 3 or txt.ndim != 3 or img.shape[1] == 0 or txt.shape[1] == 0:
            raise ValueError("FLUX image/text must be nonempty BLD tensors")
        batch = img.shape[0]
        if img.shape[2] != self.config.in_channels or txt.shape != (batch, txt.shape[1], self.config.context_in_dim):
            raise ValueError("FLUX image/text widths or batch sizes differ from topology")
        if img_ids.shape != (batch, img.shape[1], 3) or txt_ids.shape != (batch, txt.shape[1], 3):
            raise ValueError("FLUX position IDs must match image/text BxLx3 geometry")
        if timesteps.shape != (batch,) or y.shape != (batch, self.config.vec_in_dim):
            raise ValueError("FLUX time/pooled text shapes differ from B and BxV")
        tensors = (img, txt, img_ids, txt_ids, timesteps, y)
        if any(not value.is_floating_point() or value.device != img.device for value in tensors):
            raise ValueError("FLUX inputs must be floating tensors on the image device")
        if self.config.guidance_embed:
            if (
                guidance is None
                or guidance.shape != (batch,)
                or guidance.device != img.device
                or not guidance.is_floating_point()
            ):
                raise ValueError("FLUX distilled guidance requires a floating B vector on the image device")
        elif guidance is not None:
            raise ValueError("FLUX topology without guidance embedding does not consume guidance")
        lengths = None
        if control_lengths is not None:
            if self.config.variant != "kontext":
                raise ValueError("FLUX control lengths require the Kontext variant")
            if len(control_lengths) != batch or any(
                type(length) is not int or length < 0 or length > img.shape[1] for length in control_lengths
            ):
                raise ValueError("FLUX control lengths must be B integers within the image token count")
            maximum = max(control_lengths)
            lengths = tuple(txt.shape[1] + img.shape[1] - maximum + length for length in control_lengths)
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps))
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        positions = rotary_embedding(torch.cat((txt_ids, img_ids), dim=1), self.config.axes_dim, self.config.theta)
        target_length = img.shape[1] - (max(control_lengths) if control_lengths else 0)
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
                    vec,
                    joint_route.select(positions, dim=2),
                    None if lengths is None else tuple(joint_route.boundary(length) for length in lengths),
                )
                img = route.restore(img, updated)
            else:
                img, txt = block(img, txt, vec, positions, lengths)
            if self.feature_recorder is not None:
                self.feature_recorder.record(index, img[:, :target_length])
        joint = torch.cat((txt, img), dim=1)
        for index, block in enumerate(self.single_blocks, start=len(self.double_blocks)):
            if joint_route is not None and joint_route.applies(index):
                updated = block(
                    joint_route.select(joint),
                    vec,
                    joint_route.select(positions, dim=2),
                    None if lengths is None else tuple(joint_route.boundary(length) for length in lengths),
                )
                joint = joint_route.restore(joint, updated)
            else:
                joint = block(joint, vec, positions, lengths)
            if self.feature_recorder is not None:
                self.feature_recorder.record(index, joint[:, txt.shape[1] : txt.shape[1] + target_length])
        return self.final_layer(joint[:, txt.shape[1] :], vec)

    forward = predict
