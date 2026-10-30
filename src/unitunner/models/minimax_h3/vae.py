"""Compute and load native minimax h3 VAEs with explicit execution ownership."""

import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.artifacts.tensors import TensorReader, save_tensors
from unitunner.models.components.posterior import DiagonalPosterior
from unitunner.models.components.vae_loading import vae_tensors
from unitunner.models.minimax_h3.config.audio_vae import MiniMaxH3AudioVAEConfig
from unitunner.models.minimax_h3.config.video_vae import MiniMaxH3VideoVAEConfig

# Native H3 audio math follows the pinned Musubi and old MixTuner implementation.
# DAC: Copyright (c) 2023-present, Descript (MIT).
# BigVGAN: Copyright (c) 2024 NVIDIA CORPORATION (MIT).
# HiFi-GAN: Copyright (c) 2020 Jungil Kong (MIT).
# Julius: Copyright 2020 Alexandre Défossez (MIT).
# Antialiasing lineage: junjun3518/alias-free-torch (Apache-2.0).
# Full upstream notices and licenses are distributed under unitunner/licenses.


def audio_snake(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # x + 1/beta * sin^2(alpha * x)
    """Apply native Snake activation with reciprocal beta epsilon 1e-9."""
    t = torch.sin(alpha * x)
    # Keep the sine value intact for both branches of its square's backward.
    return (t * t).mul_((beta + 1e-9).reciprocal()).add_(x)


class AudioSnake(nn.Module):
    """Snake activation with per-channel alpha (encoder side)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return audio_snake(x, self.alpha, self.alpha)

    forward = transform


class AudioSnakeBeta(nn.Module):
    """AudioSnakeBeta := x + 1/beta * sin^2(alpha * x); alpha/beta stored in log scale."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(in_features))
        self.beta = nn.Parameter(torch.empty(in_features))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha).view(1, -1, 1)
        beta = torch.exp(self.beta).view(1, -1, 1)
        return audio_snake(x, alpha, beta)

    forward = transform


def kaiser_sinc_filter(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    # returns filter [1, 1, kernel_size]
    """Build a unit-sum Kaiser-windowed sinc low-pass kernel in native FP32 order."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # kaiser window design
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    # Normalize filter to have sum = 1, otherwise there is a small leakage of
    # the constant component in the input signal.
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class AudioUpsample(nn.Module):
    """Upsample with native replicate padding, grouped filtering and gain."""

    filter: torch.Tensor

    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size),
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = F.conv_transpose1d(x, self.filter.expand(C, -1, -1).to(x.dtype), stride=self.stride, groups=C).mul_(
            self.ratio
        )
        x = x[..., self.pad_left : -self.pad_right]
        return x

    forward = transform


class AudioLowPassFilter(nn.Module):
    """Apply native replicate-padded grouped low-pass convolution."""

    filter: torch.Tensor

    def __init__(self, cutoff: float = 0.5, half_width: float = 0.6, stride: int = 1, kernel_size: int = 12) -> None:
        super().__init__()
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter(cutoff, half_width, kernel_size))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        _, C, _ = x.shape
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(C, -1, -1).to(x.dtype), stride=self.stride, groups=C)

    forward = transform


class AudioDownsample(nn.Module):
    """Downsample through the native antialiasing low-pass filter."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.lowpass = AudioLowPassFilter(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)

    forward = transform


class AudioAntialiasedActivation(nn.Module):
    """upsample x2 -> pointwise activation -> downsample x2 (anti-aliased)."""

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.act = activation
        self.upsample = AudioUpsample(up_ratio, up_kernel_size)
        self.downsample = AudioDownsample(down_ratio, down_kernel_size)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x

    forward = transform


class AudioResidualUnit(nn.Module):
    """Apply the native dilated DAC residual convolution pair."""

    def __init__(self, dim: int = 16, dilation: int = 1) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            AudioSnake(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            AudioSnake(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return y.add_(x)

    forward = transform


class AudioEncoderBlock(nn.Module):
    """Apply three DAC residual units and one strided convolution."""

    def __init__(self, dim: int = 16, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            AudioResidualUnit(dim // 2, dilation=1),
            AudioResidualUnit(dim // 2, dilation=3),
            AudioResidualUnit(dim // 2, dilation=9),
            AudioSnake(dim // 2),
            nn.Conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    forward = transform


class AudioEncoder(nn.Module):
    """Encode mono waveforms through the native DAC stride cascade."""

    def __init__(self, d_model: int = 64, strides: tuple[int, ...] = (2, 4, 4, 5, 5), d_latent: int = 2048) -> None:
        super().__init__()
        block: list[nn.Module] = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block += [AudioEncoderBlock(d_model, stride=stride)]
        block += [
            AudioSnake(d_model),
            nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*block)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    forward = transform


class AudioGatedMlp(nn.Module):
    """Apply native LayerNorm and approximate-tanh GELU gating."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the native residual projection in checkpoint order."""
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)).mul_(self.w1(x)))

    forward = project


class AudioCausalAttention(nn.Module):
    """Apply native causal SDPA, head averaging and latent-width adaptive pooling."""

    zero_k_bias: torch.Tensor

    def __init__(self, in_dim: int, out_dim: int, num_heads: int) -> None:
        super().__init__()
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.empty(in_dim))
        self.v_bias = nn.Parameter(torch.empty(in_dim))
        self.register_buffer("zero_k_bias", torch.empty(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def attend(self, x: torch.Tensor) -> torch.Tensor:
        """Attend causally and pool the head-averaged feature axis."""
        B, N, _C = x.shape
        qkv = F.linear(x, weight=self.qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        # mean over heads then pool down to the latent width (in_dim >> out_dim)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = F.adaptive_avg_pool1d(torch.mean(x, dim=1), self.out_dim)
        return self.proj(x)

    forward = attend


class AudioAttentionProjection(nn.Module):
    """Project encoder features into the native posterior mean head."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = AudioCausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)

        self.norm2 = nn.LayerNorm(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = AudioGatedMlp(in_features=out_dim, hidden_features=hidden_dim)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # x -> torch.Tensor: [B, T, in_dim]
        """Compute the native residual projection in checkpoint order."""
        x = self.proj(self.norm3(x)).add_(self.attn(self.norm1(x)))
        # LayerNorm backward retains x, so the residual sum must allocate.
        return x + self.mlp(self.norm2(x))

    forward = project


def audio_convolution_padding(kernel_size: int, dilation: int = 1) -> int:
    """Compute native dilation-preserving symmetric convolution padding."""
    return int((kernel_size * dilation - dilation) / 2)


class AudioVocoderResidualBlock(nn.Module):
    """Apply native antialiased residual branches in convolution order."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=audio_convolution_padding(kernel_size, d),
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=audio_convolution_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList(
            [AudioAntialiasedActivation(activation=AudioSnakeBeta(channels)) for _ in range(self.num_layers)]
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2, strict=True):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt.add_(x)
        return x

    forward = transform


class AudioVocoder(nn.Module):
    """AudioVocoder vocoder (MiniMax H3 32 kHz configuration).

    use_bias_at_final=False, use_tanh_at_final=False (output clamped to [-1, 1]).
    """

    def __init__(
        self,
        num_mels: int = 2048,
        upsample_initial_channel: int = 1024,
        upsample_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ) -> None:
        super().__init__()
        if (
            not upsample_rates
            or not resblock_kernel_sizes
            or len(upsample_rates) != len(upsample_kernel_sizes)
            or len(resblock_kernel_sizes) != len(resblock_dilation_sizes)
        ):
            raise ValueError("AudioVocoder requires matching nonempty upsampling and residual presets")
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            self.ups.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose1d(
                            upsample_initial_channel // (2**i),
                            upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(AudioVocoderResidualBlock(ch, k, d))

        self.activation_post = AudioAntialiasedActivation(activation=AudioSnakeBeta(ch))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            stage = self.ups[i]
            if not isinstance(stage, nn.ModuleList):
                raise TypeError("AudioVocoder upsampling stages must be ModuleLists")
            for upsample in stage:
                x = upsample(x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            if xs is None:
                raise RuntimeError("AudioVocoder requires at least one residual kernel")
            x = xs.div_(self.num_kernels)

        x = self.activation_post(x)
        return self.conv_post(x).clamp_(-1.0, 1.0)

    forward = transform


# Native H3 DAC/BigVGAN vae; lineage and full notices are in audio_vae_blocks.py
# and the distributed THIRD_PARTY_NOTICES.md and licenses directory.


class MiniMaxH3AudioVAE(nn.Module):
    """Own native 32kHz stereo waveform encoding and optional BigVGAN decoding.

    Waveforms are B2L in [-1,1]; normalized latents are B32ST with S=2 and
    40 frames/second. Each channel is processed independently. Encoding uses
    mean_proj only: logs_proj remains an exact artifact tensor, with no posterior
    sampling or RNG. Checkpoint-owned Snake/bias/normalization values require
    strict artifact loading before use. Execution currently requires FP32.
    """

    latents_mean: Tensor
    latents_std: Tensor

    def __init__(self, config: MiniMaxH3AudioVAEConfig, *, include_decoder: bool = True) -> None:
        super().__init__()
        self.config = config
        self.sample_rate = 32000
        self.samples_per_latent = math.prod(config.encoder_strides)
        self.latents_per_second = self.sample_rate // self.samples_per_latent
        self.encoder = AudioEncoder(config.encoder_channels, config.encoder_strides, config.hidden_channels)
        self.pre_block = AudioAttentionProjection(config.hidden_channels, 32, num_heads=8)
        self.mean_proj = nn.Conv1d(32, 32, 1)
        self.logs_proj = nn.Conv1d(32, 32, 1)
        self.dec_in_proj: nn.Conv1d | None = None
        self.decoder: AudioVocoder | None = None
        if include_decoder:
            self.dec_in_proj = nn.Conv1d(32, config.hidden_channels, 1)
            self.decoder = AudioVocoder(
                num_mels=config.hidden_channels, upsample_initial_channel=config.decoder_channels
            )
        self.register_buffer("latents_mean", torch.empty(32))
        self.register_buffer("latents_std", torch.empty(32))

    def encode(self, waveform: Tensor) -> Tensor:
        """Return normalized means after zero right-padding L to a multiple of800.

        Input is nonempty FP32 B2L at32kHz. Output B32ST has T=ceil(L/800).
        No noise or log-variance is calculated. Preserve gradients to inputs and
        active encoder parameters, without changing waveform storage or RNG.
        """
        if waveform.ndim != 3 or min(waveform.shape) < 1 or waveform.shape[1] != 2 or waveform.dtype != torch.float32:
            raise ValueError("H3 audio encoding requires nonempty FP32 B2L stereo waveforms at32kHz")
        batch_size, stereo_channels, length = waveform.shape
        right_pad = math.ceil(length / self.samples_per_latent) * self.samples_per_latent - length
        waveform = F.pad(waveform, (0, right_pad))
        values = waveform.reshape(batch_size * stereo_channels, 1, -1)
        values = self.encoder(values)
        values = self.pre_block(values.transpose(1, 2)).transpose(1, 2)
        latents = self.mean_proj(values)
        mean = self.latents_mean.view(1, -1, 1).to(device=latents.device, dtype=latents.dtype)
        std = self.latents_std.view(1, -1, 1).to(device=latents.device, dtype=latents.dtype)
        latents = (latents - mean) / std
        return latents.reshape(batch_size, stereo_channels, latents.shape[1], latents.shape[2]).permute(0, 2, 1, 3)

    def decode(self, latents: Tensor) -> Tensor:
        """Decode normalized FP32 B32ST into B2(T*800) clamped stereo waveforms.

        Native channel deinterleaving and multiply-then-add normalization remain
        differentiable. No output trimming or resampling occurs here.
        """
        if self.decoder is None or self.dec_in_proj is None:
            raise RuntimeError("H3 audio vae was loaded without its decoder; reload with include_decoder=True")
        if (
            latents.ndim != 4
            or min(latents.shape) < 1
            or latents.shape[1:3] != (32, 2)
            or latents.dtype != torch.float32
        ):
            raise ValueError("H3 audio decoding requires nonempty FP32 B32ST latents with S=2")
        batch_size, channels, stereo_channels, frames = latents.shape
        latents = latents.permute(0, 2, 1, 3).reshape(batch_size * stereo_channels, channels, frames)
        mean = self.latents_mean.view(1, -1, 1).to(device=latents.device, dtype=latents.dtype)
        std = self.latents_std.view(1, -1, 1).to(device=latents.device, dtype=latents.dtype)
        latents = latents * std + mean
        values = self.decoder(self.dec_in_proj(latents))
        return values.reshape(batch_size, stereo_channels, -1)


_AUDIO_CONFIG_KEY = "unitunner.minimax_h3.audio_vae.config"
_AUDIO_DECODER_KEY = "unitunner.minimax_h3.audio_vae.decoder"
_AUDIO_DECODER_PREFIXES = ("decoder.", "dec_in_proj.")
_CPU = torch.device("cpu")


def save_audio_vae(model: MiniMaxH3AudioVAE, path: Path, *, metadata: Mapping[str, str] | None = None) -> None:
    """Atomically publish native VAE tensors and authoritative topology/inventory."""
    save_tensors(
        model.state_dict(),
        path,
        metadata={
            **({} if metadata is None else metadata),
            _AUDIO_CONFIG_KEY: model.config.model_dump_json(),
            _AUDIO_DECODER_KEY: "omitted" if model.decoder is None else "present",
        },
    )


def load_audio_vae(
    path: Path,
    *,
    config: MiniMaxH3AudioVAEConfig | None = None,
    include_decoder: bool = True,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.float32,
) -> MiniMaxH3AudioVAE:
    """Load exact single/sharded native weights through meta without consuming RNG.

    Published topology is used without metadata or explicit config. Known Musubi
    component prefixes are removed. FP32 encoder-only loading accepts an
    entire full artifact or a complete encoder artifact; partial decoder keys
    are rejected. All supplied tensors are checked for shape, floating type and
    finite values. Omitted decoder tensors are validated then released one at a
    time. Returned CPU tensors may share mmap storage; sources stay immutable.
    """
    if dtype != torch.float32:
        raise ValueError("H3 audio vae currently requires FP32 loading")
    with TensorReader(path) as reader:
        metadata = reader.metadata()
    serialized = metadata.get(_AUDIO_CONFIG_KEY)
    embedded = None if serialized is None else MiniMaxH3AudioVAEConfig.model_validate_json(serialized)
    if config is None:
        config = MiniMaxH3AudioVAEConfig() if embedded is None else embedded
    elif embedded is not None and embedded != config:
        raise ValueError("H3 audio vae explicit topology differs from artifact metadata")
    with torch.device("meta"):
        full_model = MiniMaxH3AudioVAE(config)
        model = full_model if include_decoder else MiniMaxH3AudioVAE(config, include_decoder=False)
    full_expected, expected = full_model.state_dict(), model.state_dict()
    supplied: set[str] = set()
    tensors = {}
    for name, tensor in vae_tensors(path, prefixes=("audio_vae.", "vae.")):
        supplied.add(name)
        if name not in full_expected:
            raise ValueError(f"H3 audio vae unexpected tensor: {name}")
        if (
            tensor.shape != full_expected[name].shape
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"H3 audio vae invalid tensor shape/type/values: {name}")
        if not torch.isfinite(tensor.to(dtype=dtype)).all():
            raise ValueError(f"H3 audio vae tensor exceeds requested dtype range: {name}")
        if name == "latents_std" and not torch.all(tensor > 0):
            raise ValueError("H3 audio vae latents_std must be positive")
        if name in expected:
            tensors[name] = tensor.to(device=device, dtype=dtype)
    encoder_keys = {name for name in full_expected if not name.startswith(_AUDIO_DECODER_PREFIXES)}
    if supplied not in (set(full_expected), encoder_keys) or tensors.keys() != expected.keys():
        raise ValueError("H3 audio vae requires a complete native encoder/decoder tensor inventory")
    decoder_state = metadata.get(_AUDIO_DECODER_KEY)
    actual_decoder_state = "present" if supplied == set(full_expected) else "omitted"
    if decoder_state is not None and decoder_state != actual_decoder_state:
        raise ValueError("H3 audio vae decoder metadata differs from tensor inventory")
    model.load_state_dict(tensors, strict=True, assign=True)
    return model


# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Diffusers PR #14355 at abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
# through the pinned Musubi and old MixTuner implementations.


class VideoCausalConv3d(nn.Conv3d):
    """Spatially reflected, temporally causal 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def transform(self, input: torch.Tensor) -> torch.Tensor:
        temporal, height, width = self.causal_padding
        if height or width:
            input = F.pad(input, (width, width, height, height, 0, 0), mode="reflect")
        if temporal:
            input = F.pad(input, (0, 0, 0, 0, temporal * 2, 0), mode="constant")
        return F.conv3d(
            input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )

    forward = transform


class VideoFrameGroupNorm(nn.GroupNorm):
    """Compute group-normalization statistics independently for each frame."""

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        """Normalize BCHW or frame-isolated BCFHW tensors."""
        if input.ndim != 5:
            return super().forward(input)
        batch_size, channels, frames, height, width = input.shape
        input = input.permute(0, 2, 1, 3, 4).contiguous()
        input = input.view(batch_size * frames, channels, 1, height, width)
        input = super().forward(input)
        input = input.view(batch_size, frames, channels, height, width)
        return input.permute(0, 2, 1, 3, 4).contiguous()

    forward = normalize


def video_group_norm(num_channels: int) -> VideoFrameGroupNorm:
    """Construct native 32-group frame-local normalization with epsilon 1e-6."""
    return VideoFrameGroupNorm(32, num_channels, eps=1e-6, affine=True)


class VideoDownsample(nn.Module):
    """Retain native asymmetric spatial reflection and causal temporal stride."""

    def __init__(self, in_channels: int, out_channels: int, time_stride: int = 1, space_stride: int = 2) -> None:
        super().__init__()
        self.space_stride = space_stride
        self.conv = VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(time_stride, space_stride, space_stride),
            padding=(1, 0, 0),
        )

    def transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.space_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(hidden_states)

    forward = transform


class VideoResnetBlock(nn.Module):
    """Apply two frame-normalized causal convolutions and a native residual sum."""

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = video_group_norm(in_channels)
        self.conv1 = VideoCausalConv3d(in_channels, self.out_channels, kernel_size=3, padding=1)
        self.norm2 = video_group_norm(self.out_channels)
        self.conv2 = VideoCausalConv3d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.nin_shortcut = (
            VideoCausalConv3d(in_channels, self.out_channels, kernel_size=1)
            if in_channels != self.out_channels
            else None
        )

    def transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(F.silu(self.norm2(hidden_states)))
        if self.nin_shortcut is not None:
            residual = self.nin_shortcut(residual)
        return residual + hidden_states

    forward = transform


class VideoEncoderStage(nn.Module):
    """One native convolutional encoder level, retaining block/downsample checkpoint keys."""

    def __init__(
        self, input_channels: int, output_channels: int, layers: int, time_stride: int, space_stride: int
    ) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            [
                VideoResnetBlock(input_channels if index == 0 else output_channels, output_channels)
                for index in range(layers)
            ]
        )
        self.downsample = (
            VideoDownsample(output_channels, output_channels, time_stride=time_stride, space_stride=space_stride)
            if space_stride * time_stride > 1
            else None
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Apply each native encoder level in checkpoint order."""
        for block in self.block:
            values = block(values)
        return self.downsample(values) if self.downsample is not None else values

    forward = encode


class VideoEncoder(nn.Module):
    """Compute unnormalized Gaussian moments with the native convolutional encoder."""

    def __init__(
        self,
        ch: int,
        ch_mult: tuple[int, ...],
        space_down: tuple[int, ...],
        time_down: tuple[int, ...],
        num_res_blocks: int | tuple[int, ...],
        in_channels: int,
        z_channels: int,
        double_z: bool = True,
    ) -> None:
        super().__init__()
        num_levels = len(ch_mult)
        layers_per_level = [num_res_blocks] * num_levels if isinstance(num_res_blocks, int) else list(num_res_blocks)
        block_channels = [ch * multiplier for multiplier in ch_mult]
        input_channels = [block_channels[0], *block_channels[:-1]]

        self.num_levels = num_levels
        self.num_res_blocks = layers_per_level
        self.conv_in = VideoCausalConv3d(in_channels, input_channels[0], kernel_size=3, padding=1)
        self.down = nn.ModuleList()
        for level in range(num_levels):
            down = VideoEncoderStage(
                input_channels[level],
                block_channels[level],
                layers_per_level[level],
                time_down[level],
                space_down[level],
            )
            self.down.append(down)

        self.norm_out = video_group_norm(block_channels[-1])
        output_channels = 2 * z_channels if double_z else z_channels
        self.conv_out = VideoCausalConv3d(block_channels[-1], output_channels, kernel_size=3, padding=1)

    def transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down in self.down:
            hidden_states = down(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)

    forward = transform


def create_video_token_ids(
    patch_dims: tuple[int, ...], device: torch.device | str, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build normalized patch-center coordinates in temporal, height, width order."""
    grids = [2.0 * (torch.arange(0.5, size, dtype=dtype, device=device) / size) - 1.0 for size in patch_dims]
    return torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, len(patch_dims) - 1).unsqueeze(0)


class VideoRotaryEmbedding(nn.Module):
    """Build axis-major three-dimensional decoder RoPE with a half-split rotation."""

    inv_freq: torch.Tensor

    def __init__(self, dim: int, rotary_base: float = 100.0, n_dim: int = 3) -> None:
        super().__init__()
        if dim % (2 * n_dim):
            raise ValueError(f"Rotary dimension {dim} must be divisible by {2 * n_dim}")
        inv_freq = 1.0 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def transform(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None].float() * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()

    forward = transform


def _apply_video_rotary(hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Rotate the leading head coordinates and retain the unrotated tail."""
    cosine, sine = (value.to(hidden_states.dtype) for value in rotary_emb)
    rotary_dim = cosine.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cosine + rotated * sine, passthrough), dim=-1)


class VideoDecoderFeedForward(nn.Module):
    """Apply the native SiLU-gated two-projection decoder feed-forward branch."""

    def __init__(self, dim: int, mult: int = 4, bias: bool = True) -> None:
        super().__init__()
        inner_dim = dim * mult
        self.w1 = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.w2 = nn.Linear(inner_dim, dim, bias=bias)

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the native gated feed-forward projection."""
        gate, value = self.w1(hidden_states).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * value)

    forward = project


class VideoDecoderAttention(nn.Module):
    """Apply per-head interleaved QKV, FP32 RMS QK normalization and Torch SDPA."""

    def __init__(self, heads: int, dim_head: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_qkv = nn.Linear(inner_dim, inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(inner_dim, inner_dim, bias=bias)

    def attend(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Attend over all decoder rows, retaining native QKV grouping."""
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.to_qkv(hidden_states).view(batch_size, sequence_length, self.heads, 3 * self.dim_head)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self.norm_q(query.float()).to(query.dtype)
        key = self.norm_k(key.float()).to(key.dtype)
        if rotary_emb is not None:
            query = _apply_video_rotary(query, rotary_emb)
            key = _apply_video_rotary(key, rotary_emb)
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.to_out(hidden_states)

    forward = attend


class VideoDecoderBlock(nn.Module):
    """Apply native FP32 RMS normalization and learned residual scale vectors."""

    def __init__(self, heads: int, dim_head: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        dim = heads * dim_head
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.attn = VideoDecoderAttention(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.ff = VideoDecoderFeedForward(dim=dim, bias=bias)
        self.scale2 = nn.Parameter(torch.empty(dim))

    def transform(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        normalized = F.rms_norm(
            hidden_states.float(),
            (hidden_states.shape[-1],),
            self.norm1.weight.float(),
            self.norm1.eps,
        ).to(hidden_states.dtype)
        hidden_states = hidden_states + self.attn(normalized, rotary_emb) * self.scale1
        normalized = F.rms_norm(
            hidden_states.float(),
            (hidden_states.shape[-1],),
            self.norm2.weight.float(),
            self.norm2.eps,
        ).to(hidden_states.dtype)
        return hidden_states + self.ff(normalized) * self.scale2

    forward = transform


class VideoDecoder(nn.Module):
    """Decode latent tokens with native register rows and spatiotemporal unpatching."""

    mask_token: torch.Tensor

    def __init__(
        self,
        patch_size: int = 16,
        patch_size_t: int = 4,
        in_channels: int = 24,
        out_channels: int = 3,
        num_layers: int = 36,
        heads: int = 32,
        dim_head: int = 64,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        bias: bool = True,
        eps: float = 1e-5,
        num_register_tokens: int = 4,
    ) -> None:
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.pos_embed = VideoRotaryEmbedding(int(dim_head * rope_dim_ratio), rope_theta, n_dim=3)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))
        self.register_buffer("mask_token", torch.empty(1, 1, dim))
        self.transformer_blocks = nn.ModuleList(
            [VideoDecoderBlock(heads=heads, dim_head=dim_head, bias=bias, eps=eps) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channels, frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch_size, frames * height * width, channels)
        hidden_states = self.x_embedder(hidden_states)
        num_patches = hidden_states.shape[1]
        hidden_states = torch.cat(
            (
                hidden_states,
                self.register_tokens.expand(batch_size, -1, -1),
                torch.zeros_like(hidden_states[:, :1]),
            ),
            dim=1,
        )

        position_ids = create_video_token_ids((frames, height, width), hidden_states.device)
        position_ids = position_ids.expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        rotary_emb = self.pos_embed(torch.cat((position_ids, suffix_ids), dim=1))
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.proj_out(self.norm_out(hidden_states))[:, :num_patches]
        hidden_states = hidden_states.view(
            batch_size,
            frames,
            height,
            width,
            self.out_channels,
            self.patch_size_t,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            frames * self.patch_size_t,
            height * self.patch_size,
            width * self.patch_size,
        )

    forward = transform


# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Diffusers PR #14355 at abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
# through the pinned Musubi and old MixTuner implementations.


IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
]

LATENTS_STD = [
    1.2223774194717407,
    1.2767263650894165,
    1.68317747116088865,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.96531379222869875,
    1.05698859691619875,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049925,
    0.7197399735450745,
    0.69362932443618775,
    2.961095094680786,
    2.7694199085235595,
    3.0496184825897215,
    2.1088054180145265,
    3.276226282119751,
    3.1627357006073,
    2.28168129920959475,
    2.6127843856811525,
]


class MiniMaxH3VideoVAE(nn.Module):
    """Own native video Gaussian moments and a separately optional ViT decoder.

    Pixel inputs are [-1,1]; latents use stored per-channel affine normalization.
    Temporal chunks and spatial tile blends retain the native arithmetic. No
    posterior RNG, media decoding, residency or cache identity is owned here.
    """

    latents_mean: Tensor
    latents_std: Tensor
    pixel_mean: Tensor
    pixel_std: Tensor

    def __init__(self, config: MiniMaxH3VideoVAEConfig, *, include_decoder: bool = True) -> None:
        super().__init__()
        self.config = config
        self.vae_ratio = math.prod(config.spatial_strides)
        self.vae_ratio_t = math.prod(config.temporal_strides)
        self.clip_length = config.clip_length
        self.token_drop = config.token_drop
        self.frame_pre_padding = (-self.clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(self.clip_length / self.vae_ratio_t)
        self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)
        self.use_tiling = config.tiling
        self.tile_size = config.tile_size
        self.tile_overlap_min = config.tile_overlap
        self.encoder = VideoEncoder(
            ch=config.channels,
            ch_mult=config.channel_multipliers,
            space_down=config.spatial_strides,
            time_down=config.temporal_strides,
            num_res_blocks=config.residual_blocks,
            in_channels=3,
            z_channels=24,
            double_z=True,
        )
        self.quant_conv = nn.Conv3d(48, 48, kernel_size=1)
        self.post_quant_conv: nn.Conv3d | None = None
        self.decoder: VideoDecoder | None = None
        if include_decoder:
            self.post_quant_conv = nn.Conv3d(24, 24, kernel_size=1)
            decoder = config.decoder
            self.decoder = VideoDecoder(
                patch_size=self.vae_ratio,
                patch_size_t=self.vae_ratio_t,
                in_channels=24,
                out_channels=3,
                num_layers=decoder.num_layers,
                heads=decoder.num_heads,
                dim_head=decoder.head_dim,
                rope_theta=decoder.rope_theta,
                rope_dim_ratio=decoder.rope_dim_ratio,
                bias=decoder.bias,
                eps=decoder.eps,
                num_register_tokens=decoder.num_register_tokens,
            )
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    def initialize_nonpersistent_buffers(self, device: torch.device) -> None:
        """Materialize fixed pixel/rotary buffers after meta artifact construction."""
        self.pixel_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1, 1)
        self.pixel_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1, 1)
        if self.decoder is not None:
            config = self.config.decoder
            self.decoder.pos_embed = VideoRotaryEmbedding(
                int(config.head_dim * config.rope_dim_ratio), config.rope_theta
            ).to(device)

    def _validate_values(self, values: Tensor, *, pixels: bool) -> None:
        """Validate native FP32 layout before padding, tiling or normalization."""
        channels = 3 if pixels else 24
        if (
            values.ndim not in ((4, 5) if pixels else (5,))
            or min(values.shape) < 1
            or values.shape[1] != channels
            or values.dtype != torch.float32
        ):
            raise ValueError(
                f"H3 video {'pixels' if pixels else 'latents'} require nonempty FP32 B{channels}FHW tensors"
            )
        if pixels and (values.shape[-2] % self.vae_ratio or values.shape[-1] % self.vae_ratio):
            raise ValueError("H3 video pixel dimensions must align to spatial compression")

    def normalize(self, latents: Tensor) -> Tensor:
        """Normalize unscaled B24FHW latents with native subtract-then-divide order."""
        self._validate_values(latents, pixels=False)
        return (latents - self.latents_mean.view(1, -1, 1, 1, 1).to(latents)) / self.latents_std.view(
            1, -1, 1, 1, 1
        ).to(latents)

    def posterior(self, pixels: Tensor) -> DiagonalPosterior:
        """Return raw FP32 moments with native log-variance clamp [-30,20]."""
        mean, log_variance = self.encode_moments(pixels).chunk(2, dim=1)
        return DiagonalPosterior(mean, log_variance.clamp(-30.0, 20.0))

    def encode(self, pixels: Tensor, *, posterior_noise: Tensor | None = None) -> Tensor:
        """Return normalized posterior mean, or sample using explicit caller noise."""
        posterior = self.posterior(pixels)
        return self.normalize(posterior.mean if posterior_noise is None else posterior.sample(posterior_noise))

    def _split_tiles(self, length: int) -> tuple[list[int], list[int], list[int]]:
        """Split a compression-aligned axis with native distributed overlap."""
        if self.tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / self.tile_size)
        while self.tile_size * num_tiles - self.tile_overlap_min * (num_tiles - 1) < length:
            num_tiles += 1
        overlaps = [self.tile_overlap_min] * (num_tiles - 1)
        remaining = self.tile_size * num_tiles - sum(overlaps) - length
        for index in range(remaining // self.vae_ratio):
            overlaps[index % (num_tiles - 1)] += self.vae_ratio
        starts = [0]
        for index in range(num_tiles - 1):
            starts.append(starts[-1] + self.tile_size - overlaps[index])
        return starts, [self.tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(first: torch.Tensor, second: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
        """Blend the first tail into the second prefix with native linear weights."""
        extent = min(first.shape[dim], second.shape[dim], extent)
        positions = torch.arange(extent, device=second.device, dtype=second.dtype)
        shape = [1] * first.ndim
        shape[dim] = extent
        first_weight = (1 - positions / extent).view(shape)
        second_weight = (positions / extent).view(shape)
        first_slice = [slice(None)] * first.ndim
        first_slice[dim] = slice(-extent, None)
        second_slice = [slice(None)] * second.ndim
        second_slice[dim] = slice(0, extent)
        blended = first[tuple(first_slice)] * first_weight + second[tuple(second_slice)] * second_weight
        if extent == second.shape[dim]:
            return blended
        rest = [slice(None)] * second.ndim
        rest[dim] = slice(extent, None)
        return torch.cat((blended, second[tuple(rest)]), dim=dim)

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        """Blend neighboring original tiles and concatenate cropped rows."""
        stitched_rows = []
        for row_index, row in enumerate(tiles):
            stitched_row = []
            for column_index, tile in enumerate(row):
                if row_index:
                    tile = self._blend(tiles[row_index - 1][column_index], tile, height_overlaps[row_index - 1], -2)
                if column_index:
                    tile = self._blend(row[column_index - 1], tile, width_overlaps[column_index - 1], -1)
                if row_index < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[column_index]]
                stitched_row.append(tile)
            stitched_rows.append(torch.cat(stitched_row, dim=-1))
        return torch.cat(stitched_rows, dim=-2)

    def _encode_clip(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode one temporal clip, optionally blending native spatial tiles."""
        if not self.use_tiling:
            return self.quant_conv(self.encoder(pixels))
        height_starts, height_lengths, height_overlaps = self._split_tiles(pixels.shape[-2])
        width_starts, width_lengths, width_overlaps = self._split_tiles(pixels.shape[-1])
        tiles = [
            [
                self.quant_conv(
                    self.encoder(
                        pixels[
                            ...,
                            top : top + tile_height,
                            left : left + tile_width,
                        ]
                    )
                )
                for left, tile_width in zip(width_starts, width_lengths, strict=True)
            ]
            for top, tile_height in zip(height_starts, height_lengths, strict=True)
        ]
        latent_height_overlaps = [value // self.vae_ratio for value in height_overlaps]
        latent_width_overlaps = [value // self.vae_ratio for value in width_overlaps]
        return self._stitch_tiles(tiles, latent_height_overlaps, latent_width_overlaps)

    def _decode_clip(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode one temporal clip, optionally blending native spatial tiles."""
        if self.decoder is None or self.post_quant_conv is None:
            raise RuntimeError("MiniMax-H3 VAE was loaded without its decoder")
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(latents))
        height_starts, height_lengths, height_overlaps = self._split_tiles(latents.shape[-2] * self.vae_ratio)
        width_starts, width_lengths, width_overlaps = self._split_tiles(latents.shape[-1] * self.vae_ratio)
        tiles = [
            [
                self.decoder(
                    self.post_quant_conv(
                        latents[
                            ...,
                            top // self.vae_ratio : (top + tile_height) // self.vae_ratio,
                            left // self.vae_ratio : (left + tile_width) // self.vae_ratio,
                        ]
                    )
                )
                for left, tile_width in zip(width_starts, width_lengths, strict=True)
            ]
            for top, tile_height in zip(height_starts, height_lengths, strict=True)
        ]
        return self._stitch_tiles(tiles, height_overlaps, width_overlaps)

    def _encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
        """Right-pad raw frames to clips, concatenate moments and drop native tail tokens."""
        frame_count = pixels.shape[2]
        if frame_count % self.clip_length:
            padding = pixels[:, :, -1:].repeat(1, 1, (-frame_count) % self.clip_length, 1, 1)
            pixels = torch.cat((pixels, padding), dim=2)
        moments = torch.cat(
            [
                self._encode_clip(pixels[:, :, start : start + self.clip_length])
                for start in range(0, pixels.shape[2], self.clip_length)
            ],
            dim=2,
        )
        return moments[:, :, : -self.token_drop] if self.token_drop else moments

    def _decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode overlapping native token chunks and trim only synthesized tail frames."""
        if self.decoder is None:
            raise RuntimeError("MiniMaxH3 video vae was loaded without its decoder; reload with include_decoder=True")
        chunk_tokens = self.tokens_chunk_size
        chunk_frames = chunk_tokens * self.vae_ratio_t
        padded_token_count = latents.shape[2] + self.token_drop
        pad_tokens = (-padded_token_count) % chunk_tokens
        num_chunks = (padded_token_count + pad_tokens) // chunk_tokens - int(self.token_drop > 0)
        if num_chunks < 1:
            pad_tokens += chunk_tokens
            num_chunks = 1
        if pad_tokens:
            latents = torch.cat((latents, latents[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)), dim=2)

        decoded = []
        overlap = None
        for index in range(num_chunks):
            start = index * chunk_tokens
            clip = self._decode_clip(latents[:, :, start : start + chunk_tokens + self.token_overlap])
            for part_index in range(int(self.token_drop > 0) + 1):
                frame_start = part_index * chunk_frames
                part = clip[:, :, frame_start : frame_start + chunk_frames]
                part = part[:, :, self.frame_pre_padding :]
                if part_index == 0:
                    if overlap is not None:
                        part = self._blend(overlap, part, self.frame_overlap, dim=-3)
                    decoded.append(part)
                else:
                    overlap = part
        if overlap is not None:
            decoded.append(overlap)
        pixels = torch.cat(decoded, dim=2)

        if pad_tokens:
            intra_tail = self.clip_length % self.vae_ratio_t
            tokens_before_padding = latents.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (tokens_before_padding + offset) % chunk_tokens == 0 else self.vae_ratio_t
                for offset in range(pad_tokens)
            )
            pixels = pixels[:, :, :-pad_frames]
        return pixels

    def encode_moments(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode BCHW or BCFHW pixels in [-1,1] into raw FP32 Gaussian moments."""
        self._validate_values(pixels, pixels=True)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(2)
        pixels = (pixels + 1.0) * 0.5
        pixels = (pixels - self.pixel_mean.to(pixels)) / self.pixel_std.to(pixels)
        if pixels.shape[2] == 1:
            moments = self._encode_clip(pixels)[:, :, -1:]
        else:
            moments = self._encode_video(pixels)
        return moments.float()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized B24FHW latents, repeating one-frame context before cropping."""
        self._validate_values(latents, pixels=False)
        latent_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        latent_std = self.latents_std.view(1, -1, 1, 1, 1).to(latents)
        latents = latents * latent_std + latent_mean
        if latents.shape[2] == 1:
            # Native single-frame decoding repeats the token for temporal context,
            # then retains the first pixel frame.
            pixels = self._decode_video(latents.repeat(1, 1, 2, 1, 1))[:, :, :1]
        else:
            pixels = self._decode_video(latents)
        pixels = pixels.float() * self.pixel_std.to(pixels) + self.pixel_mean.to(pixels)
        return pixels.clamp(0.0, 1.0) * 2.0 - 1.0


_VIDEO_CONFIG_KEY = "unitunner.minimax_h3.video_vae.config"
_VIDEO_DECODER_KEY = "unitunner.minimax_h3.video_vae.decoder"
_VIDEO_DECODER_PREFIXES = ("decoder.", "post_quant_conv.")


def save_video_vae(model: MiniMaxH3VideoVAE, path: Path, *, metadata: Mapping[str, str] | None = None) -> None:
    """Atomically publish native VAE tensors and authoritative topology/inventory."""
    save_tensors(
        model.state_dict(),
        path,
        metadata={
            **({} if metadata is None else metadata),
            _VIDEO_CONFIG_KEY: model.config.model_dump_json(),
            _VIDEO_DECODER_KEY: "omitted" if model.decoder is None else "present",
        },
    )


def load_video_vae(
    path: Path,
    *,
    config: MiniMaxH3VideoVAEConfig | None = None,
    include_decoder: bool = True,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.float32,
) -> MiniMaxH3VideoVAE:
    """Load exact single/sharded native weights through meta without consuming RNG.

    Published topology is used without metadata or explicit config. Known Musubi
    component prefixes are removed. FP32 encoder-only loading accepts an
    entire full artifact or a complete encoder artifact; partial decoder keys
    are rejected. All supplied tensors are checked for shape, floating type and
    finite values. Omitted decoder tensors are validated then released one at a
    time. Returned CPU tensors may share mmap storage; sources stay immutable.
    """
    if dtype != torch.float32:
        raise ValueError("H3 video vae currently requires FP32 loading")
    with TensorReader(path) as reader:
        metadata = reader.metadata()
    serialized = metadata.get(_VIDEO_CONFIG_KEY)
    embedded = None if serialized is None else MiniMaxH3VideoVAEConfig.model_validate_json(serialized)
    if config is None:
        config = MiniMaxH3VideoVAEConfig() if embedded is None else embedded
    elif embedded is not None and embedded != config:
        raise ValueError("H3 video vae explicit topology differs from artifact metadata")
    with torch.device("meta"):
        full_model = MiniMaxH3VideoVAE(config)
        model = full_model if include_decoder else MiniMaxH3VideoVAE(config, include_decoder=False)
    full_expected, expected = full_model.state_dict(), model.state_dict()
    supplied: set[str] = set()
    tensors = {}
    for name, tensor in vae_tensors(path, prefixes=("first_stage_model.", "video_vae.", "vae.")):
        supplied.add(name)
        if name not in full_expected:
            raise ValueError(f"H3 video vae unexpected tensor: {name}")
        if (
            tensor.shape != full_expected[name].shape
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"H3 video vae invalid tensor shape/type/values: {name}")
        if not torch.isfinite(tensor.to(dtype=dtype)).all():
            raise ValueError(f"H3 video vae tensor exceeds requested dtype range: {name}")
        if name == "latents_std" and not torch.all(tensor > 0):
            raise ValueError("H3 video vae latents_std must be positive")
        if name in expected:
            tensors[name] = tensor.to(device=device, dtype=dtype)
    encoder_keys = {name for name in full_expected if not name.startswith(_VIDEO_DECODER_PREFIXES)}
    if supplied not in (set(full_expected), encoder_keys) or tensors.keys() != expected.keys():
        raise ValueError("H3 video vae requires a complete native encoder/decoder tensor inventory")
    decoder_state = metadata.get(_VIDEO_DECODER_KEY)
    actual_decoder_state = "present" if supplied == set(full_expected) else "omitted"
    if decoder_state is not None and decoder_state != actual_decoder_state:
        raise ValueError("H3 video vae decoder metadata differs from tensor inventory")
    model.load_state_dict(tensors, strict=True, assign=True)
    model.initialize_nonpersistent_buffers(device)
    return model
