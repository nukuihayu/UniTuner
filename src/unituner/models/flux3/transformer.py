# Derived from Black Forest Labs FLUX3 Image Klein, Apache-2.0.
"""Compute native three-mode FLUX3 predictions with explicit packed sample geometry."""

import torch
from torch import Tensor, nn

from unituner.config.routing import TreadConfig
from unituner.execution.routing import TargetSpan, sample_token_route
from unituner.models.components.features import FeatureRecorder
from unituner.models.flux3.conditioning import MODES, Flux3Mode, Flux3Packed, validate_packed
from unituner.models.flux3.config.model import Flux3Config
from unituner.models.flux3.layers import (
    Flux3Block,
    Flux3Modulation,
    LastLayer,
    MLPEmbedder,
    ModeModulation,
    rope_embed,
    timestep_embedding,
)


class Flux3Transformer(nn.Module):
    """Own published FLUX3 image/reference/text branches and shared block keys.

    Base predicts instantaneous velocity. Distilled is a flow map and requires
    explicit destination time per sample. Empty reference mode uses one private
    phantom token, never included in attention or final joint packing. The model
    does not import trainers, execution owners, text providers or reference code.
    """

    def __init__(self, config: Flux3Config) -> None:
        super().__init__()
        self.config = config
        self.feature_recorder: FeatureRecorder | None = None
        self.token_routing: TreadConfig | None = None
        self.feature_width: int = config.hidden_size
        self.feature_patch: tuple[int, int, int] = (1, 1, 1)
        hidden = config.hidden_size
        self.emb_in = nn.ModuleDict(
            {mode: nn.Linear(config.image_channels, hidden, bias=False) for mode in ("image", "image_cond")}
        )
        self.txt_in = nn.Linear(config.text_channels, hidden, bias=False)
        self.time_in = MLPEmbedder(256, hidden)
        self.ref_time_in = MLPEmbedder(256, hidden) if config.variant == "distilled" else None
        self.early_stream_modulations = nn.ModuleDict(
            {mode: ModeModulation(hidden) for mode in ("image", "image_cond", "txt")}
        )
        self.single_stream_modulations = nn.ModuleDict(
            {mode: ModeModulation(hidden) for mode in ("image", "image_cond", "txt")}
        )
        self.content_mode_blocks = nn.ModuleDict(
            {
                mode: nn.ModuleList(Flux3Block(hidden, config.num_heads, config.mlp_ratio) for _ in range(config.depth))
                for mode in ("image", "image_cond")
            }
        )
        self.txt_mode_blocks = nn.ModuleList(
            Flux3Block(hidden, config.num_heads, config.mlp_ratio) for _ in range(config.depth)
        )
        self.single_blocks = nn.ModuleList(
            Flux3Block(hidden, config.num_heads, config.mlp_ratio) for _ in range(config.depth_single)
        )
        self.final_layer = nn.ModuleDict({"image": LastLayer(hidden, config.image_channels)})

    def early_blocks(self, mode: Flux3Mode) -> nn.ModuleList:
        """Borrow the native ordered block list for one early mode."""
        blocks = self.txt_mode_blocks if mode == "text" else self.content_mode_blocks[mode]
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError("FLUX3 early mode must own an ordered block list")
        return blocks

    def predict(self, packed: Flux3Packed, timesteps_ref: Tensor | None = None) -> Tensor:
        """Return sum(target lengths) x image_channels native velocity in image-mode order.

        Pack bounds isolate all samples in early and shared attention. Timesteps
        are floating noise fractions. For a distilled flow map, destination times
        have B rows; each destination is expanded over that sample's valid tokens
        in every mode. No implicit rounding, scheduler conversion or input mutation
        occurs. Autograd retains all active mode/time/parameter dependencies.
        """
        validate_packed(packed, image_channels=self.config.image_channels, text_channels=self.config.text_channels)
        if (timesteps_ref is not None) != (self.ref_time_in is not None):
            raise ValueError("FLUX3 destination timesteps are required exactly for distilled flow-map models")
        if timesteps_ref is not None and (
            timesteps_ref.shape != (packed.num_samples,)
            or not timesteps_ref.is_floating_point()
            or timesteps_ref.device != packed.data["image"].device
        ):
            raise ValueError("FLUX3 destination timesteps must be a floating B vector on the image device")
        data, ids, times = dict(packed.data), dict(packed.ids), dict(packed.timesteps)
        for mode in MODES:
            if data[mode].shape[0] == 0:
                data[mode] = data[mode].new_zeros(1, data[mode].shape[1])
                ids[mode] = ids[mode].new_zeros(1, 4)
                times[mode] = times[mode].new_zeros(1)
        hidden = {mode: (self.txt_in if mode == "text" else self.emb_in[mode])(data[mode]) for mode in MODES}
        dtype = hidden["image"].dtype
        vectors = {mode: self.time_in(timestep_embedding(times[mode].to(dtype))) for mode in MODES}
        if timesteps_ref is not None and self.ref_time_in is not None:
            reference = self.ref_time_in(timestep_embedding(timesteps_ref.to(dtype)))
            for mode in MODES:
                parts = [
                    reference[index].expand(end - start, -1)
                    for index, (start, end) in enumerate(packed.bounds[mode])
                    if end > start
                ]
                if parts:
                    vectors[mode] = vectors[mode] + torch.cat(parts)
        early: dict[Flux3Mode, Flux3Modulation] = {}
        single: dict[Flux3Mode, Flux3Modulation] = {}
        for mode in MODES:
            key = "txt" if mode == "text" else mode
            early[mode] = self.early_stream_modulations[key](vectors[mode])
            single[mode] = self.single_stream_modulations[key](vectors[mode])
        frequencies = {mode: rope_embed(ids[mode][None], self.config.axes_dim, self.config.theta) for mode in MODES}
        route = sample_token_route(
            self.token_routing,
            training=self.training,
            length=hidden["image"].shape[0],
            spans=(tuple(TargetSpan(first, last) for first, last in packed.bounds["image"]),),
            device=hidden["image"].device,
        )
        for depth in range(self.config.depth):
            for mode in MODES:
                blocks = self.early_blocks(mode)
                if mode == "image" and route is not None and route.applies(depth):
                    updated = blocks[depth](
                        route.select(hidden[mode], dim=0, batched=False),
                        route.select(frequencies[mode], dim=2),
                        tuple((route.boundary(first), route.boundary(last)) for first, last in packed.bounds[mode]),
                        tuple(route.select(group, dim=0, batched=False) for group in early[mode]),
                    )
                    hidden[mode] = route.restore(hidden[mode], updated, batched=False)
                else:
                    hidden[mode] = blocks[depth](hidden[mode], frequencies[mode], packed.bounds[mode], early[mode])
                if mode == "image" and self.feature_recorder is not None:
                    for first, last in packed.bounds["image"]:
                        self.feature_recorder.record(depth, hidden[mode][first:last].unsqueeze(0))
        chunks, positions, modulations = [], [], ([], [], [])
        bounds, placements, offset = [], [], 0
        for index in range(packed.num_samples):
            start = offset
            for mode in MODES:
                first, last = packed.bounds[mode][index]
                if last > first:
                    chunks.append(hidden[mode][first:last])
                    positions.append(ids[mode][first:last])
                    for group in range(3):
                        modulations[group].append(single[mode][group][first:last])
                    if mode == "image":
                        placements.append((first, last, offset))
                    offset += last - first
            bounds.append((start, offset))
        joint = torch.cat(chunks)
        joint_frequencies = rope_embed(torch.cat(positions)[None], self.config.axes_dim, self.config.theta)
        shift, scale, gate = (torch.cat(group) for group in modulations)
        joint_route = (
            None
            if route is None
            else route.relocate(
                joint.shape[0], tuple(TargetSpan(start, start + last - first) for first, last, start in placements)
            )
        )
        for index, block in enumerate(self.single_blocks, start=self.config.depth):
            if joint_route is not None and joint_route.applies(index):
                updated = block(
                    joint_route.select(joint, dim=0, batched=False),
                    joint_route.select(joint_frequencies, dim=2),
                    tuple((joint_route.boundary(first), joint_route.boundary(last)) for first, last in bounds),
                    tuple(joint_route.select(group, dim=0, batched=False) for group in (shift, scale, gate)),
                )
                joint = joint_route.restore(joint, updated, batched=False)
            else:
                joint = block(joint, joint_frequencies, tuple(bounds), (shift, scale, gate))
            if self.feature_recorder is not None:
                for first, last, start in placements:
                    self.feature_recorder.record(index, joint[start : start + last - first].unsqueeze(0))
        images = torch.empty_like(hidden["image"])
        for first, last, start in placements:
            images[first:last] = joint[start : start + last - first]
        return self.final_layer["image"](images, vectors["image"])

    forward = predict
