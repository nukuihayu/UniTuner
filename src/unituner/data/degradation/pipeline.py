"""OMGSR-F-1024 two-stage degradation with phase-stable interpolation."""

import math

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from unituner.data.degradation import color as image_ops
from unituner.data.degradation.config import BlurConfig, CompressionConfig, DegradationConfig, NoiseConfig, StageConfig
from unituner.data.degradation.diffjpeg import diffjpeg

type Pixels = NDArray[np.float32]

_RESIZE_MODES = ("area", "bilinear", "bicubic")
_KERNEL_SIZES = tuple(range(11, 42, 2))
_KERNEL_MAX = 41
_KERNEL_MID = 25


def _kernel_profile() -> tuple[tuple[int, ...], int, int]:
    """Return OMGSR-F-1024's fixed kernel support."""
    return _KERNEL_SIZES, _KERNEL_MAX, _KERNEL_MID


def _pad_kernel(kernel: Pixels) -> Pixels:
    padding = (_KERNEL_MAX - kernel.shape[0]) // 2
    return np.pad(kernel, ((padding, padding), (padding, padding))).astype(np.float32)


def sinc_kernel(size: int, cutoff: float) -> Pixels:
    """Return OMGSR's unit-sum circular low-pass kernel."""
    if size < 3 or size % 2 != 1 or not 0 < cutoff <= math.pi:
        raise ValueError("Sinc kernels require an odd size >=3 and cutoff in (0, pi]")
    axis = torch.arange(size, dtype=torch.float64) - size // 2
    radius = (axis[:, None].square() + axis[None, :].square()).sqrt()
    kernel = cutoff * torch.special.bessel_j1(cutoff * radius) / (2 * math.pi * radius.clamp_min(1e-12))
    kernel[size // 2, size // 2] = cutoff**2 / (4 * math.pi)
    return (kernel / kernel.sum()).float().numpy()


def _sinc_kernel(rng: np.random.Generator, *, stage: bool = False) -> Pixels:
    size = int(rng.choice(_KERNEL_SIZES))
    minimum = math.pi / (5 if stage and size >= _KERNEL_MID else 3)
    return _pad_kernel(sinc_kernel(size, float(rng.uniform(minimum, math.pi))))


def _sample(bounds: tuple[float, float], rng: np.random.Generator) -> float:
    return float(rng.uniform(*bounds))


def mixed_blur_kernel(rng: np.random.Generator, settings: BlurConfig) -> Pixels:
    """Sample and pad an OMGSR-F mixed blur kernel."""
    size = int(rng.choice(_KERNEL_SIZES))
    kind = int(rng.choice(6, p=settings.probabilities))
    sigma_x = _sample(settings.sigma, rng)
    sigma_y = sigma_x if kind % 2 == 0 else _sample(settings.sigma, rng)
    angle = rng.uniform(-math.pi, math.pi)
    axis = np.arange(size, dtype=np.float64) - size // 2
    xx, yy = np.meshgrid(axis, axis)
    u = (xx * np.cos(angle) + yy * np.sin(angle)) / sigma_x
    v = (-xx * np.sin(angle) + yy * np.cos(angle)) / sigma_y
    radius_squared = u**2 + v**2
    if kind < 2:
        kernel = np.exp(-0.5 * radius_squared)
    elif kind < 4:
        beta = (
            rng.uniform(settings.generalized_beta[0], 1)
            if rng.random() < 0.5
            else rng.uniform(1, settings.generalized_beta[1])
        )
        kernel = np.exp(-0.5 * radius_squared**beta)
    else:
        beta = (
            rng.uniform(settings.plateau_beta[0], 1)
            if rng.random() < 0.5
            else rng.uniform(1, settings.plateau_beta[1])
        )
        kernel = 1 / (1 + radius_squared**beta)
    return _pad_kernel((kernel / kernel.sum()).astype(np.float32))


def _filter(image: Pixels, kernel: Pixels) -> Pixels:
    """Apply OMGSR reflection-padded depthwise filtering, including tiny grids."""
    radius = kernel.shape[0] // 2
    if radius < min(image.shape[:2]):
        value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
        value = F.pad(value, (radius, radius, radius, radius), mode="reflect")
    else:
        padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
        value = torch.from_numpy(np.ascontiguousarray(padded)).permute(2, 0, 1)[None]
    weight = torch.from_numpy(np.ascontiguousarray(kernel))[None, None].repeat(3, 1, 1, 1)
    return F.conv2d(value, weight, groups=3)[0].permute(1, 2, 0).contiguous().numpy()


def _resize(
    image: Pixels,
    height: int,
    width: int,
    rng: np.random.Generator,
    *,
    mode: str | None = None,
) -> Pixels:
    """Resize on the integer grid so repeated half-pixel mappings compose."""
    selected = str(rng.choice(_RESIZE_MODES)) if mode is None else mode
    if selected not in _RESIZE_MODES:
        raise ValueError(f"Unsupported OMGSR resize mode: {selected}")
    value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    return (
        F.interpolate(
            value,
            size=(max(1, height), max(1, width)),
            mode=selected,
            align_corners=False if selected in ("bilinear", "bicubic") else None,
            antialias=False,
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )


def _resize_scale(stage: StageConfig, rng: np.random.Generator) -> float:
    direction = int(rng.choice(3, p=stage.resize_probabilities))
    if direction == 0:
        return float(rng.uniform(1, stage.resize_range[1]))
    if direction == 1:
        return float(rng.uniform(stage.resize_range[0], 1))
    return 1.0


def add_noise(
    image: Pixels,
    settings: NoiseConfig,
    rng: np.random.Generator,
    *,
    trace: list[dict] | None = None,
) -> Pixels:
    """Apply OMGSR Gaussian or Poisson noise and clip without rounding."""
    gaussian = rng.random() < settings.gaussian_probability
    gray = rng.random() < settings.gray_probability
    if gaussian:
        parameter = _sample(settings.gaussian_sigma, rng)
        shape = (*image.shape[:2], 1 if gray else 3)
        noise = rng.normal(0, parameter / 255, shape)
        kind = "gaussian"
    else:
        signal = (image * np.array([0.2989, 0.5870, 0.1140])).sum(2, keepdims=True) if gray else image
        signal = (signal * 255).round().clip(0, 255) / 255
        levels = 2 ** math.ceil(math.log2(max(1, np.unique(signal).size)))
        parameter = _sample(settings.poisson_scale, rng)
        noise = (rng.poisson(signal * levels) / levels - signal) * parameter
        kind = "poisson"
    if trace is not None:
        trace.append(
            {
                "op": "noise",
                "kind": kind,
                "parameter": parameter,
                "gray": gray,
                "noise_rms_255": float(np.sqrt(np.mean(noise**2)) * 255),
                "shape": list(image.shape[:2]),
            }
        )
    return (image + noise).clip(0, 1).astype(np.float32)


def _compress(
    image: Pixels,
    settings: CompressionConfig,
    rng: np.random.Generator,
    trace: list[dict] | None,
) -> Pixels:
    quality = _sample(settings.jpeg_quality, rng)
    result = diffjpeg(image.clip(0, 1).astype(np.float32), quality)
    if trace is not None:
        trace.append({"op": "compression", "codec": "diffjpeg", "quality": quality, "shape": list(image.shape[:2])})
    return result


@torch.autocast("cpu", enabled=False)
def degrade_pair(
    high: torch.Tensor,
    settings: DegradationConfig,
    seed: int,
    *,
    trace: list[dict] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned OMGSR-F LQ and the unchanged HQ supervision target."""
    if (
        high.device.type != "cpu"
        or high.dtype != torch.float32
        or high.ndim != 3
        or high.shape[0] != 3
        or min(high.shape[-2:]) < 32
        or not torch.isfinite(high).all()
        or high.min() < 0
        or high.max() > 1
    ):
        raise ValueError("Degradation requires CPU float32 CHW RGB [0,1], dimensions >=32")
    rng = np.random.default_rng(seed)
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(12)]
    image = high.detach().permute(1, 2, 0).numpy().copy()
    height, width = image.shape[:2]
    target = (max(1, int(height / settings.scale)), max(1, int(width / settings.scale)))

    def record(op: str, value: Pixels, **params) -> None:
        if trace is not None:
            trace.append({"op": op, "shape": list(value.shape[:2]), **params})

    record("source", image, scale=settings.scale, target=list(target))

    def blur(value: Pixels, stage: StageConfig, index: int) -> Pixels:
        blur_rng = streams[(index - 1) * 4]
        is_sinc = blur_rng.random() < stage.sinc_probability
        kernel = _sinc_kernel(blur_rng, stage=True) if is_sinc else mixed_blur_kernel(blur_rng, stage.blur)
        record("stage_sinc" if is_sinc else "blur", value, stage=index, kernel=kernel.tolist())
        return _filter(value, kernel)

    image = blur(image, settings.first, 1)
    resize_rng, noise_rng, codec_rng = streams[1:4]
    factor = _resize_scale(settings.first, resize_rng)
    requested = (max(1, int(height * factor)), max(1, int(width * factor)))
    image = _resize(image, *requested, resize_rng)
    record("resize", image, stage=1, factor=factor, requested_shape=list(requested))
    image = add_noise(image, settings.first.noise, noise_rng, trace=trace)
    image = _compress(image, settings.first.compression, codec_rng, trace)

    second_blur_rng = streams[4]
    if second_blur_rng.random() < settings.second.blur_probability:
        is_sinc = second_blur_rng.random() < settings.second.sinc_probability
        kernel = (
            _sinc_kernel(second_blur_rng, stage=True)
            if is_sinc
            else mixed_blur_kernel(second_blur_rng, settings.second.blur)
        )
        record("stage_sinc" if is_sinc else "blur", image, stage=2, kernel=kernel.tolist())
        image = _filter(image, kernel)
    resize_rng, noise_rng, codec_rng = streams[5:8]
    factor = _resize_scale(settings.second, resize_rng)
    requested = (max(1, int(target[0] * factor)), max(1, int(target[1] * factor)))
    image = _resize(image, *requested, resize_rng)
    record("resize", image, stage=2, factor=factor, requested_shape=list(requested))
    image = add_noise(image, settings.second.noise, noise_rng, trace=trace)

    sinc_rng = streams[8]
    final_kernel = _sinc_kernel(sinc_rng) if sinc_rng.random() < settings.final_sinc_probability else None

    def delivery(value: Pixels) -> Pixels:
        value = _resize(value, *target, streams[9])
        record("delivery_resize", value)
        if final_kernel is not None:
            record("final_sinc", value, kernel=final_kernel.tolist())
            value = _filter(value, final_kernel)
        return value

    if streams[10].random() < 0.5:
        image = delivery(image)
        image = _compress(image, settings.second.compression, codec_rng, trace)
    else:
        image = _compress(image, settings.second.compression, codec_rng, trace)
        image = delivery(image)
    record("native_lq", image)

    value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    if rng.random() < settings.gray_probability:
        value = image_ops.rgb_to_grayscale(value, num_output_channels=3)
    if rng.random() < settings.color_jitter_probability:
        for operation in rng.permutation(4):
            match int(operation):
                case 0:
                    value = image_ops.adjust_brightness(value, _sample(settings.brightness, rng))
                case 1:
                    value = image_ops.adjust_contrast(value, _sample(settings.contrast, rng))
                case 2:
                    value = image_ops.adjust_saturation(value, _sample(settings.saturation, rng))
                case 3:
                    value = image_ops.adjust_hue(value, float(rng.uniform(*settings.hue)))
    image = value.permute(1, 2, 0).contiguous().numpy()
    image = _resize(image, height, width, streams[11])
    record("alignment", image)
    value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    value = (value * 255).round().clamp(0, 255) / 255
    record("quantize", value.permute(1, 2, 0).numpy())
    return value.clamp(0, 1).contiguous(), high.clone()


def degrade_image(
    high: torch.Tensor,
    settings: DegradationConfig,
    seed: int,
    *,
    trace: list[dict] | None = None,
) -> torch.Tensor:
    """Return only the aligned degraded input."""
    return degrade_pair(high, settings, seed, trace=trace)[0]
