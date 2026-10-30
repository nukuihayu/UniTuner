# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Compute Qwen Image 2.1 target velocities from native interleaved conditioning."""

import torch
from torch import Tensor, nn

from unitunner.config.routing import TreadConfig
from unitunner.execution.routing import TargetSpan, sample_token_route
from unitunner.models.components.features import FeatureRecorder
from unitunner.models.qwen_image_2_1.config.model import QwenImage21Config
from unitunner.models.qwen_image_2_1.layers import FinalNorm, QwenImage21Block, TextProjection, TimeEmbedding
from unitunner.models.qwen_image_2_1.sequence import ImageGrid, JointSequence, interleave_sequence


class QwenImage21Transformer(nn.Module):
    """Own native single-stream parameters, with no training/execution resource state.

    Callers provide target/reference BNC tokens and BLD text without image tokens.
    Explicit CPU geometry excludes padding, inserts references among text and
    appends targets. Execution owners can install recompute/compile on blocks.
    """

    def __init__(self, config: QwenImage21Config) -> None:
        super().__init__()
        self.config = config
        self.feature_recorder: FeatureRecorder | None = None
        self.token_routing: TreadConfig | None = None
        self.feature_width: int = config.num_attention_heads * config.attention_head_dim
        self.feature_patch: tuple[int, int, int] = (1, 1, 1)
        width = config.hidden_size
        self.img_in = nn.Linear(config.in_channels, width, bias=False)
        self.txt_in = TextProjection(config.context_in_dim, width, config.eps)
        self.time_text_embed = TimeEmbedding(width)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 4 * width, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21Block(
                    width, config.num_attention_heads, config.attention_head_dim, config.mlp_ratio, config.eps
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm_out = FinalNorm(width, config.eps)
        self.proj_out = nn.Linear(width, config.out_channels, bias=False)

    def build_sequence(
        self,
        target: Tensor,
        text: Tensor,
        grids: tuple[tuple[ImageGrid, ...], ...],
        *,
        text_lengths: tuple[int, ...] | None = None,
        references: tuple[Tensor, ...] = (),
        image_slots: tuple[tuple[int, ...], ...] | None = None,
    ) -> JointSequence:
        """Validate native geometry and project/interleave caller-owned image/text tokens.

        Grids contain references first and target last, one layout per sample or
        one shared layout. Every grid is (1,H,W), H*W matching its BNC tensor.
        Lengths describe nonempty text prefixes. Slots are ordered text insertion
        indices (duplicates allowed), one per reference and sample. References
        require explicit slots; no multimodal positions are inferred. Invalid
        geometry fails before projections; no casting, RNG draws or input mutation.
        """
        if (
            target.ndim != 3
            or min(target.shape) < 1
            or target.shape[-1] != self.config.in_channels
            or not target.is_floating_point()
        ):
            raise ValueError("Qwen Image 2.1 target must be nonempty floating BNC with native input width")
        batch_size = target.shape[0]
        if (
            text.ndim != 3
            or text.shape[0] != batch_size
            or text.shape[1] < 1
            or text.shape[-1] != self.config.context_in_dim
            or text.dtype != target.dtype
            or text.device != target.device
        ):
            raise ValueError("Qwen Image 2.1 text must be matching floating BLD context")
        if any(
            reference.ndim != 3
            or reference.shape[0] != batch_size
            or reference.shape[1] < 1
            or reference.shape[-1] != self.config.in_channels
            or reference.dtype != target.dtype
            or reference.device != target.device
            for reference in references
        ):
            raise ValueError("Qwen Image 2.1 references must match target batch, channels, device and dtype")
        if len(grids) == 1:
            grids = grids * batch_size
        if len(grids) != batch_size:
            raise ValueError("Qwen Image 2.1 grids require one shared layout or one per sample")
        lengths = text_lengths if text_lengths is not None else (text.shape[1],) * batch_size
        slots = image_slots if image_slots is not None else ((),) * batch_size
        if len(lengths) != batch_size or len(slots) != batch_size:
            raise ValueError("Qwen Image 2.1 text lengths and image slots must match batch size")
        for row_grids, length, row_slots in zip(grids, lengths, slots, strict=True):
            if type(length) is not int or not 1 <= length <= text.shape[1]:
                raise ValueError("Qwen Image 2.1 text lengths must be nonempty integer prefixes")
            if len(row_grids) != len(references) + 1 or len(row_slots) != len(references):
                raise ValueError("Qwen Image 2.1 reference, grid and insertion slot counts must agree")
            if (
                any(type(slot) is not int or not 0 <= slot <= length for slot in row_slots)
                or tuple(sorted(row_slots)) != row_slots
            ):
                raise ValueError("Qwen Image 2.1 image slots must be ordered integer text positions")
            for grid, image in zip(row_grids, (*references, target), strict=True):
                if (
                    len(grid) != 3
                    or any(type(size) is not int or size < 1 for size in grid)
                    or grid[0] != 1
                    or grid[1] * grid[2] != image.shape[1]
                ):
                    raise ValueError("Qwen Image 2.1 image grids must describe exactly one frame of H*W tokens")
        return interleave_sequence(
            self.img_in(target),
            self.txt_in(text),
            tuple(self.img_in(reference) for reference in references),
            grids,
            lengths,
            slots,
            self.config.axes_dims_rope,
        )

    def predict(
        self,
        target: Tensor,
        text: Tensor,
        timesteps: Tensor,
        grids: tuple[tuple[ImageGrid, ...], ...],
        *,
        text_lengths: tuple[int, ...] | None = None,
        references: tuple[Tensor, ...] = (),
        image_slots: tuple[tuple[int, ...], ...] | None = None,
    ) -> Tensor:
        """Return target-only BNC velocities at normalized flow times in [0,1].

        Conditioning text/reference tokens use a separate shared zero-time row.
        Target tokens use one supplied B timestep per sample. Grids, lengths and
        slots obey build_sequence. Preserve all input/parameter gradients, caller
        modes, RNG and storage. The network does not invert prediction signs.
        """
        if (
            target.ndim != 3
            or timesteps.shape != (target.shape[0],)
            or not timesteps.is_floating_point()
            or timesteps.device != target.device
            or not torch.all(torch.isfinite(timesteps) & (timesteps >= 0) & (timesteps <= 1))
        ):
            raise ValueError("Qwen Image 2.1 requires one finite normalized flow timestep per sample")
        sequence = self.build_sequence(
            target, text, grids, text_lengths=text_lengths, references=references, image_slots=image_slots
        )
        values = sequence.tokens
        if values.dtype in (torch.bfloat16, torch.float16):
            times = (timesteps * 1000).to(values.dtype) / 1000
        else:
            times = timesteps.to(values.dtype)
        time_features = self.time_text_embed(torch.cat((times, times.new_zeros(1))), values.dtype)
        modulation = self.modulation(time_features)
        route = sample_token_route(
            self.token_routing,
            training=self.training,
            length=values.shape[1],
            spans=tuple((TargetSpan(offset, offset + target.shape[1]),) for offset in sequence.target_offsets),
            device=values.device,
        )
        for index, block in enumerate(self.transformer_blocks):
            if route is not None and route.applies(index):
                segments = tuple(
                    tuple(
                        (route.boundary(start, row), route.boundary(stop, row), causal) for start, stop, causal in runs
                    )
                    for row, runs in enumerate(sequence.segments)
                )
                updated = block(
                    route.select(values),
                    modulation,
                    route.select(sequence.rotary),
                    segments,
                    route.select(sequence.target_mask),
                )
                values = route.restore(values, updated)
            else:
                values = block(values, modulation, sequence.rotary, sequence.segments, sequence.target_mask)
            if self.feature_recorder is not None:
                self.feature_recorder.record(
                    index,
                    torch.stack(
                        [
                            values[row, offset : offset + target.shape[1]]
                            for row, offset in enumerate(sequence.target_offsets)
                        ]
                    ),
                )
        targets = torch.stack(
            [values[index, offset : offset + target.shape[1]] for index, offset in enumerate(sequence.target_offsets)]
        )
        return self.proj_out(self.norm_out(targets, time_features[:-1]))

    forward = predict
