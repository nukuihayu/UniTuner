"""Compute and load native qwen image VAEs with explicit execution ownership."""

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn
from torch.nn import functional as F

from unituner.artifacts.formats.precision import FloatingPrecision
from unituner.models.components.posterior import DiagonalPosterior
from unituner.models.qwen_image.config.vae import QwenImageVAEConfig
from unituner.precision import compute_context

# Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.


@dataclass(slots=True)
class TemporalHistory:
    """Own differentiable convolution inputs for exactly one encode/decode operation."""

    tensors: dict[int, Tensor] = field(default_factory=dict)
    started_upsamples: set[int] = field(default_factory=set)


@dataclass(slots=True)
class HistoryCursor:
    """Borrow one history and walk the same convolution order for a temporal chunk."""

    history: TemporalHistory
    index: int = 0


class CausalConv3d(nn.Conv3d):
    """Preserve Conv3d weight keys with explicit left-only temporal zero padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        *,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        temporal, height, width = (padding, padding, padding) if isinstance(padding, int) else padding
        self.causal_padding = (width, width, height, height, 2 * temporal, 0)

    def convolve(self, input: Tensor, previous: Tensor | None = None) -> Tensor:
        """Convolve BCTHW, replacing left temporal zeros with previous input frames."""
        padding = list(self.causal_padding)
        if previous is not None and padding[4] > 0:
            input = torch.cat((previous, input), dim=2)
            padding[4] -= previous.shape[2]
        return super().forward(F.pad(input, padding))

    forward = convolve


def cached_convolution(convolution: CausalConv3d, values: Tensor, cursor: HistoryCursor) -> Tensor:
    """Retain the last two convolution inputs without detaching their gradient graph."""
    index = cursor.index
    previous = cursor.history.tensors.get(index)
    recent = values[:, :, -2:].clone()
    if recent.shape[2] < 2 and previous is not None:
        recent = torch.cat((previous[:, :, -1:], recent), dim=2)
    result = convolution(values, previous)
    cursor.history.tensors[index] = recent
    cursor.index += 1
    return result


class SpatialUpsample(nn.Module):
    """Upsample in FP32 nearest-exact arithmetic, restoring the supplied dtype."""

    def upsample(self, values: Tensor) -> Tensor:
        """Return BCHW spatially doubled features without temporal interpolation."""
        return F.interpolate(values.float(), scale_factor=(2.0, 2.0), mode="nearest-exact").to(values.dtype)

    forward = upsample


class VAEResample(nn.Module):
    """Preserve native spatial and first-frame temporal resampling semantics."""

    def __init__(self, width: int, mode: Literal["upsample2d", "upsample3d", "downsample2d", "downsample3d"]) -> None:
        super().__init__()
        self.mode = mode
        if mode.startswith("upsample"):
            self.resample = nn.Sequential(SpatialUpsample(), nn.Conv2d(width, width // 2, 3, padding=1))
        else:
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(width, width, 3, stride=2))
        if mode == "upsample3d":
            self.time_conv: CausalConv3d | None = CausalConv3d(width, 2 * width, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample3d":
            self.time_conv = CausalConv3d(width, width, (3, 1, 1), stride=(2, 1, 1))
        else:
            self.time_conv = None

    def resample_features(self, values: Tensor, cursor: HistoryCursor) -> Tensor:
        """Resample BCTHW while borrowing only this operation's temporal slots.

        The first frame bypasses temporal upsampling/downsampling. Subsequent
        chunks use their prior inputs; upsampling interleaves the two channel
        groups in native time order before independent spatial interpolation.
        """
        batch, channels, frames, height, width = values.shape
        if self.mode == "upsample3d":
            if self.time_conv is None:
                raise RuntimeError("Missing temporal upsampling convolution")
            index = cursor.index
            previous = cursor.history.tensors.get(index)
            if index not in cursor.history.started_upsamples:
                cursor.history.started_upsamples.add(index)
            else:
                recent = values[:, :, -2:].clone()
                if recent.shape[2] < 2:
                    prefix = torch.zeros_like(recent) if previous is None else previous[:, :, -1:]
                    recent = torch.cat((prefix, recent), dim=2)
                values = self.time_conv(values, previous)
                cursor.history.tensors[index] = recent
                values = values.reshape(batch, 2, channels, frames, height, width)
                values = torch.stack((values[:, 0], values[:, 1]), dim=3).reshape(
                    batch, channels, frames * 2, height, width
                )
            cursor.index += 1
        frames = values.shape[2]
        values = self.resample(values.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width))
        values = values.reshape(batch, frames, *values.shape[1:]).permute(0, 2, 1, 3, 4)
        if self.mode == "downsample3d":
            if self.time_conv is None:
                raise RuntimeError("Missing temporal downsampling convolution")
            index = cursor.index
            previous = cursor.history.tensors.get(index)
            if previous is None:
                cursor.history.tensors[index] = values.clone()
            else:
                recent = values[:, :, -1:].clone()
                values = self.time_conv(torch.cat((previous[:, :, -1:], values), dim=2))
                cursor.history.tensors[index] = recent
            cursor.index += 1
        return values

    forward = resample_features


# Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.


class QwenImageVAETiling(BaseModel):
    """Declare pixel-space tile extents and strides for both encode and decode.

    All values must be multiples of the VAE spatial factor, checked when
    used. Strides are distances between tile origins, not overlap widths.
    Tiling approximates whole-image attention/convolution at spatial seams.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    height: int = Field(default=256, gt=0)
    width: int = Field(default=256, gt=0)
    stride_height: int = Field(default=192, gt=0)
    stride_width: int = Field(default=192, gt=0)

    @model_validator(mode="after")
    def validate_strides(self) -> Self:
        """Reject gaps between adjacent spatial tiles."""
        if self.stride_height > self.height or self.stride_width > self.width:
            raise ValueError("Qwen Image vae tile strides cannot exceed tile extents")
        return self


class QwenImageVAEExecution(BaseModel):
    """Select immutable per-operation memory policies without changing model state.

    Batch slices are independent; every spatial tile owns a complete temporal
    history. Neither policy truncates gradients. Autograd still retains the
    intermediates needed by backward, so training memory is not bounded by
    a single tile. Execution choices are not weight artifact topology.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    batch_slice_size: int | None = Field(default=None, gt=0)
    tiling: QwenImageVAETiling | None = None


def _blend(previous: Tensor, current: Tensor, extent: int, *, axis: int) -> Tensor:
    """Blend an overlap with native host scalar weights without in-place writes.

    Build each weight in Python before FP32 conversion to retain the native
    per-row/per-column arithmetic. Updated tiles must feed later neighbors,
    including vertical blending before horizontal blending.
    """
    extent = min(extent, previous.shape[axis], current.shape[axis])
    if extent == 0:
        return current
    shape = [1] * current.ndim
    shape[axis] = extent
    old_weight = torch.tensor([1 - index / extent for index in range(extent)], dtype=current.dtype, device="cpu")
    new_weight = torch.tensor([index / extent for index in range(extent)], dtype=current.dtype, device="cpu")
    overlap = previous.narrow(axis, previous.shape[axis] - extent, extent) * old_weight.to(current.device).reshape(
        shape
    )
    overlap = overlap + current.narrow(axis, 0, extent) * new_weight.to(current.device).reshape(shape)
    return torch.cat((overlap, current.narrow(axis, extent, current.shape[axis] - extent)), dim=axis)


def execute_vae(
    values: Tensor,
    transform: Callable[[Tensor], Tensor],
    *,
    execution: QwenImageVAEExecution | None,
    spatial_factor: int,
    encoding: bool,
) -> Tensor:
    """Apply FP32 BCTHW transform using native slice, overlap and crop ordering.

    The caller disables autocast and validates input layout. Decode transform
    returns unclamped pixels; this operation preserves the native distinction
    between whole-image clamping and unclamped tiled output. Inputs/parameters
    remain unchanged and all output pieces retain their gradient graphs.
    """
    if execution is None:
        result = transform(values)
        return result if encoding else result.clamp(-1, 1)
    tiling = execution.tiling
    if tiling is not None and any(
        value % spatial_factor for value in (tiling.height, tiling.width, tiling.stride_height, tiling.stride_width)
    ):
        raise ValueError("Qwen Image vae tile extents/strides must be multiples of spatial_factor")
    if execution.batch_slice_size is not None and values.shape[0] > execution.batch_slice_size:
        return torch.cat(
            [
                execute_vae(
                    part,
                    transform,
                    execution=QwenImageVAEExecution(tiling=tiling),
                    spatial_factor=spatial_factor,
                    encoding=encoding,
                )
                for part in values.split(execution.batch_slice_size)
            ]
        )
    divisor = 1 if encoding else spatial_factor
    if tiling is None or (values.shape[-2] <= tiling.height // divisor and values.shape[-1] <= tiling.width // divisor):
        result = transform(values)
        return result if encoding else result.clamp(-1, 1)
    tile_height, tile_width = tiling.height // divisor, tiling.width // divisor
    stride_height, stride_width = tiling.stride_height // divisor, tiling.stride_width // divisor
    output_divisor = spatial_factor if encoding else 1
    output_stride_height = tiling.stride_height // output_divisor
    output_stride_width = tiling.stride_width // output_divisor
    overlap_height = (tiling.height - tiling.stride_height) // output_divisor
    overlap_width = (tiling.width - tiling.stride_width) // output_divisor
    previous_row: list[Tensor] = []
    result_rows = []
    for top in range(0, values.shape[-2], stride_height):
        row, pieces = [], []
        for column, left in enumerate(range(0, values.shape[-1], stride_width)):
            tile = transform(values[..., top : top + tile_height, left : left + tile_width])
            if previous_row:
                tile = _blend(previous_row[column], tile, overlap_height, axis=3)
            if row:
                tile = _blend(row[-1], tile, overlap_width, axis=4)
            row.append(tile)
            pieces.append(tile[..., :output_stride_height, :output_stride_width])
        result_rows.append(torch.cat(pieces, dim=-1))
        previous_row = row
    height, width = values.shape[-2:]
    output_height = height // spatial_factor if encoding else height * spatial_factor
    output_width = width // spatial_factor if encoding else width * spatial_factor
    return torch.cat(result_rows, dim=-2)[..., :output_height, :output_width]


# Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.


class VAERMSNorm(nn.Module):
    """Normalize channels with native L2 normalization, width scaling and learned gamma."""

    def __init__(self, width: int, *, temporal: bool) -> None:
        super().__init__()
        self.scale = width**0.5
        self.gamma = nn.Parameter(torch.ones((width, 1, 1, 1) if temporal else (width, 1, 1)))

    def normalize(self, values: Tensor) -> Tensor:
        """Apply BCTHW or BCHW channel normalization with native epsilon1e-12."""
        return F.normalize(values, dim=1) * self.scale * self.gamma + 0.0

    forward = normalize


class VAEResidual(nn.Module):
    """Compute two causal convolutions with channel norms, SiLU and an optional shortcut."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = VAERMSNorm(in_channels, temporal=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = VAERMSNorm(out_channels, temporal=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = (
            CausalConv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def transform(self, values: Tensor, cursor: HistoryCursor) -> Tensor:
        """Preserve the shortcut and operation-local histories through backward."""
        shortcut = self.conv_shortcut(values)
        values = cached_convolution(self.conv1, F.silu(self.norm1(values)), cursor)
        values = cached_convolution(self.conv2, self.dropout(F.silu(self.norm2(values))), cursor)
        return values + shortcut

    forward = transform


class VAEAttention(nn.Module):
    """Apply one-head spatial attention independently to every video frame."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = VAERMSNorm(width, temporal=False)
        self.to_qkv = nn.Conv2d(width, 3 * width, 1)
        self.proj = nn.Conv2d(width, width, 1)

    def attend(self, values: Tensor) -> Tensor:
        """Attend within H*W; no temporal keys or artificial causal mask are introduced."""
        batch, channels, frames, height, width = values.shape
        flat = values.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        qkv = self.to_qkv(self.norm(flat)).reshape(batch * frames, 1, channels * 3, -1)
        query, key, value = qkv.permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        result = F.scaled_dot_product_attention(query, key, value)
        result = result.squeeze(1).permute(0, 2, 1).reshape(batch * frames, channels, height, width)
        result = self.proj(result).reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return result + values

    forward = attend


class VAEMiddle(nn.Module):
    """Preserve native residual-attention-residual order and parameter paths."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([VAEResidual(width, width, dropout), VAEResidual(width, width, dropout)])
        self.attentions = nn.ModuleList([VAEAttention(width)])

    def transform(self, values: Tensor, cursor: HistoryCursor) -> Tensor:
        """Return differentiable features with two residual history pairs."""
        values = self.resnets[0](values, cursor)
        return self.resnets[1](self.attentions[0](values), cursor)

    forward = transform


class VAEEncoder(nn.Module):
    """Compute native causal downsampling stages and raw Gaussian moment channels."""

    def __init__(self, config: QwenImageVAEConfig) -> None:
        super().__init__()
        widths = [config.base_channels * value for value in (1, *config.channel_multipliers)]
        self.conv_in = CausalConv3d(config.input_channels, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        scale = 1.0
        for index, (in_channels, out_channels) in enumerate(pairwise(widths)):
            for _ in range(config.layers_per_block):
                self.down_blocks.append(VAEResidual(in_channels, out_channels, config.dropout))
                if scale in config.attention_scales:
                    self.down_blocks.append(VAEAttention(out_channels))
                in_channels = out_channels
            if index < len(config.channel_multipliers) - 1:
                mode = "downsample3d" if config.temporal_downsample[index] else "downsample2d"
                self.down_blocks.append(VAEResample(out_channels, mode))
                scale /= 2
        self.mid_block = VAEMiddle(widths[-1], config.dropout)
        self.norm_out = VAERMSNorm(widths[-1], temporal=True)
        self.conv_out = CausalConv3d(widths[-1], 2 * config.latent_channels, 3, padding=1)

    def encode(self, values: Tensor, cursor: HistoryCursor) -> Tensor:
        """Encode a temporal chunk, retaining history only through the supplied cursor."""
        values = cached_convolution(self.conv_in, values, cursor)
        for block in self.down_blocks:
            values = block(values) if isinstance(block, VAEAttention) else block(values, cursor)
        values = self.mid_block(values, cursor)
        return cached_convolution(self.conv_out, F.silu(self.norm_out(values)), cursor)

    forward = encode


class VAEUpBlock(nn.Module):
    """Compute the native extra decoder residual and optional spatial/temporal upsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: QwenImageVAEConfig,
        mode: Literal["upsample2d", "upsample3d"] | None,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList()
        for _ in range(config.layers_per_block + 1):
            self.resnets.append(VAEResidual(in_channels, out_channels, config.dropout))
            in_channels = out_channels
        self.upsamplers: nn.ModuleList | None = nn.ModuleList([VAEResample(out_channels, mode)]) if mode else None

    def transform(self, values: Tensor, cursor: HistoryCursor) -> Tensor:
        """Return upsampled chunk features while preserving first-frame behavior."""
        for block in self.resnets:
            values = block(values, cursor)
        if self.upsamplers is not None:
            values = self.upsamplers[0](values, cursor)
        return values

    forward = transform


class VAEDecoder(nn.Module):
    """Compute native inverse spatial/temporal stages without persistent feature state."""

    def __init__(self, config: QwenImageVAEConfig) -> None:
        super().__init__()
        widths = [
            config.base_channels * value
            for value in (config.channel_multipliers[-1], *config.channel_multipliers[::-1])
        ]
        self.conv_in = CausalConv3d(config.latent_channels, widths[0], 3, padding=1)
        self.mid_block = VAEMiddle(widths[0], config.dropout)
        self.up_blocks = nn.ModuleList()
        temporal = config.temporal_downsample[::-1]
        for index, (in_channels, out_channels) in enumerate(pairwise(widths)):
            mode = None
            if index < len(config.channel_multipliers) - 1:
                mode = "upsample3d" if temporal[index] else "upsample2d"
            self.up_blocks.append(VAEUpBlock(in_channels // 2 if index else in_channels, out_channels, config, mode))
        self.norm_out = VAERMSNorm(widths[-1], temporal=True)
        self.conv_out = CausalConv3d(widths[-1], config.input_channels, 3, padding=1)

    def decode(self, values: Tensor, cursor: HistoryCursor) -> Tensor:
        """Decode one latent frame with the operation's causal feature histories."""
        values = self.mid_block(cached_convolution(self.conv_in, values, cursor), cursor)
        for block in self.up_blocks:
            values = block(values, cursor)
        return cached_convolution(self.conv_out, F.silu(self.norm_out(values)), cursor)

    forward = decode


# Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.


class QwenImageVAE(nn.Module):
    """Own native RGB/RGBA FP32 VAE weights, never cross-call temporal history.

    Video uses BCTHW and 1+4k pixels; independent layers use BCLHW through the
    explicit layer methods. Default encoding uses posterior mean, as the native
    Musubi cache caller does. Optional sampling requires caller-supplied noise.
    No method changes RNG, train/eval mode or parameter residency. Gradients
    flow through all temporal chunks and raw latent normalization.
    """

    compute_precision: FloatingPrecision = "fp32"

    def __init__(self, config: QwenImageVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = VAEEncoder(config)
        self.quant_conv = CausalConv3d(2 * config.latent_channels, 2 * config.latent_channels, 1)
        self.post_quant_conv = CausalConv3d(config.latent_channels, config.latent_channels, 1)
        self.decoder = VAEDecoder(config)

    def _validate(self, values: Tensor, *, pixels: bool) -> None:
        """Reject ambiguous layout, precision, channels or device before convolution."""
        channels = self.config.input_channels if pixels else self.config.latent_channels
        parameter = next(self.parameters())
        if (
            values.ndim != 5
            or min(values.shape) < 1
            or values.shape[1] != channels
            or values.dtype != torch.float32
            or parameter.dtype != torch.float32
            or values.device != parameter.device
        ):
            raise ValueError("Qwen Image vae requires nonempty FP32 BCTHW/BCLHW with matching channels/device")
        if pixels and any(size % self.config.spatial_factor for size in values.shape[-2:]):
            raise ValueError("Qwen Image vae pixel H/W must be divisible by spatial_factor")

    def _normalization(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Build native host-FP32 mean/reciprocal-std before device transfer."""
        shape = (1, self.config.latent_channels, 1, 1, 1)
        mean = torch.tensor(self.config.latents_mean, dtype=torch.float32, device="cpu").reshape(shape)
        reciprocal = 1.0 / torch.tensor(self.config.latents_std, dtype=torch.float32, device="cpu").reshape(shape)
        return mean.to(values.device), reciprocal.to(values.device)

    def normalize(self, latents: Tensor) -> Tensor:
        """Convert raw BCTHW/BCLHW latents with (raw-mean)*reciprocal_std, without RNG."""
        self._validate(latents, pixels=False)
        mean, reciprocal = self._normalization(latents)
        return (latents - mean) * reciprocal

    def denormalize(self, latents: Tensor) -> Tensor:
        """Invert native normalized BCTHW/BCLHW using division by reciprocal_std."""
        self._validate(latents, pixels=False)
        mean, reciprocal = self._normalization(latents)
        return latents / reciprocal + mean

    def _encode_moments(self, pixels: Tensor) -> Tensor:
        """Encode one spatial tile/batch slice using a local causal history."""
        history = TemporalHistory()
        outputs = [self.encoder(pixels[:, :, :1], HistoryCursor(history))]
        for index in range(1, pixels.shape[2], 4):
            outputs.append(self.encoder(pixels[:, :, index : index + 4], HistoryCursor(history)))
        return self.quant_conv(torch.cat(outputs, dim=2))

    def posterior(self, pixels: Tensor, *, execution: QwenImageVAEExecution | None = None) -> DiagonalPosterior:
        """Return raw BCTHW Gaussian moments, clamping log variance to[-30,20].

        Pixels use [-1,1] and T=1+4k. Process the first frame then four-frame
        chunks with local differentiable histories. The moments are not latent
        normalized. The shared posterior's sample method requires explicit
        noise; this operation never samples or detaches input/parameter graphs.
        """
        self._validate(pixels, pixels=True)
        if (pixels.shape[2] - 1) % 4:
            raise ValueError("Qwen Image causal encoding requires1+4k frames; use encode_layers for independent images")
        with compute_context(pixels.device, self.compute_precision):
            moments = execute_vae(
                pixels,
                self._encode_moments,
                execution=execution,
                spatial_factor=self.config.spatial_factor,
                encoding=True,
            )
            mean, log_variance = moments.float().chunk(2, dim=1)
            return DiagonalPosterior(mean, log_variance.clamp(-30, 20))

    def encode(
        self,
        pixels: Tensor,
        *,
        posterior_noise: Tensor | None = None,
        execution: QwenImageVAEExecution | None = None,
    ) -> Tensor:
        """Encode FP32 video pixels into normalized latents, using mean unless noise is supplied."""
        posterior = self.posterior(pixels, execution=execution)
        return self.normalize(posterior.mean if posterior_noise is None else posterior.sample(posterior_noise))

    def _decode_pixels(self, latents: Tensor) -> Tensor:
        """Decode one spatial tile/batch slice without clipping native pixels."""
        history = TemporalHistory()
        values = self.post_quant_conv(latents)
        outputs = [
            self.decoder(values[:, :, index : index + 1], HistoryCursor(history)) for index in range(values.shape[2])
        ]
        return torch.cat(outputs, dim=2)

    def decode_raw(self, latents: Tensor, *, execution: QwenImageVAEExecution | None = None) -> Tensor:
        """Decode raw BCTHW latents with T=1+4*(latent_T-1).

        Each latent frame advances a local causal history. Clipping and all
        convolutions remain differentiable; no history is stored on the module.
        Native whole-image output clamps to[-1,1]. Active spatial tiling blends
        unclamped decoder output, as old/Musubi do; it can exceed that range.
        """
        self._validate(latents, pixels=False)
        with compute_context(latents.device, self.compute_precision):
            return execute_vae(
                latents,
                self._decode_pixels,
                execution=execution,
                spatial_factor=self.config.spatial_factor,
                encoding=False,
            )

    def decode(self, latents: Tensor, *, execution: QwenImageVAEExecution | None = None) -> Tensor:
        """Decode normalized BCTHW latents with native full/tiled clipping semantics."""
        return self.decode_raw(self.denormalize(latents), execution=execution)

    def encode_layers(
        self,
        pixels: Tensor,
        *,
        posterior_noise: Tensor | None = None,
        execution: QwenImageVAEExecution | None = None,
    ) -> Tensor:
        """Encode independent RGB/RGBA BCLHW layers, resetting causal history per layer."""
        self._validate(pixels, pixels=True)
        expected_shape = (
            pixels.shape[0],
            self.config.latent_channels,
            pixels.shape[2],
            pixels.shape[3] // self.config.spatial_factor,
            pixels.shape[4] // self.config.spatial_factor,
        )
        if posterior_noise is not None and (
            posterior_noise.shape != expected_shape
            or posterior_noise.dtype != pixels.dtype
            or posterior_noise.device != pixels.device
        ):
            raise ValueError("Qwen Image layer posterior noise must match encoded BCLHW shape/dtype/device")
        return torch.cat(
            [
                self.encode(
                    pixels[:, :, index : index + 1],
                    posterior_noise=None if posterior_noise is None else posterior_noise[:, :, index : index + 1],
                    execution=execution,
                )
                for index in range(pixels.shape[2])
            ],
            dim=2,
        )

    def decode_layers(self, latents: Tensor, *, execution: QwenImageVAEExecution | None = None) -> Tensor:
        """Decode independent normalized BCLHW layers; never expand them as causal video."""
        self._validate(latents, pixels=False)
        return torch.cat(
            [self.decode(latents[:, :, index : index + 1], execution=execution) for index in range(latents.shape[2])],
            dim=2,
        )
