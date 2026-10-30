# Derived from Krea2 and pinned Musubi Tuner; see THIRD_PARTY_NOTICES.md.
"""Compute Krea2 image-first single-stream velocity with native checkpoint geometry."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unituner.config.routing import TreadConfig
from unituner.execution.routing import TargetSpan, sample_token_route
from unituner.models.components.features import FeatureRecorder
from unituner.models.krea2.config.model import Krea2Config
from unituner.models.krea2.layers import (
    LastLayer,
    RMSNorm,
    SingleStreamBlock,
    TextFusionTransformer,
    rotary_embedding,
    timestep_embedding,
)


class Krea2Transformer(nn.Module):
    """Own RAW/Turbo shared topology without optimizer, recompute or residency policy.

    Input patches and positions are image-first. Selected text layers use BLND;
    lengths are validated host prefix sizes. Native parameters keep their exact
    artifact names, including zero-centered scales and raw modulation offsets.
    """

    def __init__(self, config: Krea2Config) -> None:
        super().__init__()
        self.config = config
        self.feature_recorder: FeatureRecorder | None = None
        self.token_routing: TreadConfig | None = None
        self.feature_width: int = config.hidden_size
        self.feature_patch: tuple[int, int, int] = (1, 2, 2)
        self.first = nn.Linear(config.in_channels * config.patch_size**2, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.hidden_size, config.num_heads, config.num_kv_heads, config.mlp_multiplier, config.bias
                )
                for _ in range(config.num_layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.timestep_dim, config.hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.txtfusion = TextFusionTransformer(
            config.text_layers,
            config.text_dim,
            config.text_heads,
            config.text_kv_heads,
            config.mlp_multiplier,
            config.bias,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.text_dim),
            nn.Linear(config.text_dim, config.hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.last = LastLayer(config.hidden_size, config.patch_size, config.in_channels)
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(config.hidden_size, config.hidden_size * 6))

    def predict(
        self,
        image: Tensor,
        text: Tensor,
        timesteps: Tensor,
        positions: Tensor,
        *,
        text_lengths: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Return BNC native noise-minus-data velocity from image patches and BLND text.

        Times are normalized B fractions; native time embedding multiplies by1000.
        Positions are FP32/FP64 B(N+L)3; text positions are supplied by the caller.
        Text lengths describe nonempty valid prefixes before right padding.
        Joint state and positions pad to256; uniform attention trims prefixes and
        heterogeneous attention masks keys. Keep all input/parameter gradients,
        caller mode/RNG/residency and input storage unchanged.
        """
        if (
            image.ndim != 3
            or min(image.shape) < 1
            or not image.is_floating_point()
            or image.shape[-1] != self.config.in_channels * self.config.patch_size**2
            or text.ndim != 4
            or text.shape[0] != image.shape[0]
            or text.shape[1] < 1
            or text.shape[2:] != (self.config.text_layers, self.config.text_dim)
            or text.device != image.device
            or text.dtype != image.dtype
        ):
            raise ValueError("Krea2 requires matching floating BNC patches and BLND selected text features")
        if (
            timesteps.shape != (image.shape[0],)
            or not timesteps.is_floating_point()
            or timesteps.device != image.device
            or not torch.all(torch.isfinite(timesteps) & (timesteps >= 0) & (timesteps <= 1))
        ):
            raise ValueError("Krea2 requires finite normalized B timesteps in[0,1] on the image device")
        if (
            positions.shape != (image.shape[0], image.shape[1] + text.shape[1], 3)
            or positions.dtype not in (torch.float32, torch.float64)
            or positions.device != image.device
        ):
            raise ValueError("Krea2 requires image-first floating B(N+L)3 positions on the image device")
        lengths = (text.shape[1],) * image.shape[0] if text_lengths is None else text_lengths
        if len(lengths) != image.shape[0] or any(
            type(length) is not int or not 1 <= length <= text.shape[1] for length in lengths
        ):
            raise ValueError("Krea2 text_lengths must be nonempty host integer prefixes, one per sample")
        image_count = image.shape[1]
        image = self.first(image)
        time = self.tmlp(timestep_embedding(timesteps, self.config.timestep_dim, dtype=image.dtype))
        modulation = self.tproj(time)
        text = self.txtmlp(self.txtfusion(text, lengths))
        values = torch.cat((image, text), dim=1)
        padding = -values.shape[1] % 256
        values, positions = F.pad(values, (0, 0, 0, padding)), F.pad(positions, (0, 0, 0, padding))
        rotary = rotary_embedding(positions, self.config.rotary_axes, self.config.theta)
        joint_lengths = tuple(image_count + length for length in lengths)
        route = sample_token_route(
            self.token_routing,
            training=self.training,
            length=values.shape[1],
            spans=((TargetSpan(0, image_count),),),
            device=values.device,
        )
        for index, block in enumerate(self.blocks):
            if route is not None and route.applies(index):
                updated = block(
                    route.select(values),
                    modulation,
                    route.select(rotary, dim=2),
                    tuple(route.boundary(length) for length in joint_lengths),
                )
                values = route.restore(values, updated)
            else:
                values = block(values, modulation, rotary, joint_lengths)
            if self.feature_recorder is not None:
                self.feature_recorder.record(index, values[:, :image_count])
        return self.last(values, time)[:, :image_count]

    forward = predict
