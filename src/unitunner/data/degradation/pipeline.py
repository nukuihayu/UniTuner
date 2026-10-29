"""Real-ESRGAN two-order degradation, with explicit optional extensions.

All stochastic state is local. Native LQ is completed before the separate
alignment resize; HYPIR-style final quantization follows that resize.
"""

import math

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from unitunner.data.degradation import color as image_ops
from unitunner.data.degradation.config import (
    BlurConfig,
    CompressionConfig,
    DegradationConfig,
    GrainConfig,
    NoiseConfig,
    StageConfig,
)

type Pixels = NDArray[np.float32]


def _resolution_scale(resolution: int, reference: int = 512, exponent: float = 0.5) -> float:
    """One HQ-derived spatial multiplier; never multiply again after resize."""
    if resolution < 1 or reference < 1:
        raise ValueError("Resolutions must be positive")
    return float(np.clip((resolution / reference) ** exponent, 0.5, 2.0))


def _scaled_odd_size(size: int, scale: float) -> int:
    """Scale an odd kernel support while retaining a centered odd grid."""
    return 2 * max(1, round((size // 2) * scale)) + 1


def sinc_kernel(size: int, cutoff: float) -> Pixels:
    """Return a unit-sum circular low-pass kernel; cutoff is in radians/pixel.

    The origin uses the analytic limit of the Bessel expression, avoiding a
    division by zero. Negative lobes preserve the intended ringing distribution.
    """
    if size < 3 or size % 2 != 1 or not 0 < cutoff <= math.pi:
        raise ValueError("Sinc kernels require an odd size >=3 and cutoff in (0, pi]")
    axis = torch.arange(size, dtype=torch.float64) - size // 2
    radius = (axis[:, None].square() + axis[None, :].square()).sqrt()
    kernel = cutoff * torch.special.bessel_j1(cutoff * radius) / (2 * math.pi * radius.clamp_min(1e-12))
    kernel[size // 2, size // 2] = cutoff**2 / (4 * math.pi)
    return (kernel / kernel.sum()).float().numpy()


def _resolution_sinc_kernel(rng: np.random.Generator, resolution: float, *, stage: bool = False) -> Pixels:
    """Sample a resolution-aware stage or final sinc kernel from official bounds."""
    spatial_scale = resolution
    base_size = int(rng.choice(np.arange(7, 22, 2)))
    size = _scaled_odd_size(base_size, spatial_scale)
    minimum = math.pi / (5 if stage and base_size >= 13 else 3)
    cutoff = min(math.pi, float(rng.uniform(minimum, math.pi)) / spatial_scale)
    return sinc_kernel(size, cutoff)


def _sample(bounds: tuple[float, float], rng: np.random.Generator) -> float:
    """Draw an independent uniform parameter, as in the official distribution."""
    return float(rng.uniform(*bounds))


def mixed_blur_kernel(
    rng: np.random.Generator,
    settings: BlurConfig,
    resolution: float = 1.0,
) -> Pixels:
    """Sample an official mixed PSF using the single HQ spatial multiplier."""
    size = int(rng.choice(np.arange(7, 22, 2)))
    spatial_scale = resolution
    size = _scaled_odd_size(size, spatial_scale)
    kind = int(rng.choice(6, p=settings.probabilities))
    sigma_x = _sample(settings.sigma, rng) * spatial_scale
    sigma_y = sigma_x if kind % 2 == 0 else _sample(settings.sigma, rng) * spatial_scale
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
            rng.uniform(settings.plateau_beta[0], 1) if rng.random() < 0.5 else rng.uniform(1, settings.plateau_beta[1])
        )
        kernel = 1 / (1 + radius_squared**beta)
    return (kernel / kernel.sum()).astype(np.float32)


def _filter(image: Pixels, kernel: Pixels) -> Pixels:
    """Convolve each channel using reflect-101 borders, preserving float32 output."""
    return np.asarray(cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101), dtype=np.float32)


def _resize(
    image: Pixels,
    height: int,
    width: int,
    rng: np.random.Generator,
    settings: DegradationConfig,
    *,
    factor: float | None = None,
) -> Pixels:
    """Resize on the actual integer grid, antialiasing continuous downsampling."""
    shrinking = height < image.shape[0] or width < image.shape[1]
    mode = str(rng.choice(settings.resize_modes))
    value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    return (
        F.interpolate(
            value,
            size=(max(1, height), max(1, width)) if factor is None else None,
            scale_factor=factor,
            recompute_scale_factor=False if factor is not None else None,
            mode=mode,
            align_corners=False if mode in ("bilinear", "bicubic") else None,
            antialias=settings.resize_antialias and shrinking and mode in ("bilinear", "bicubic"),
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )


def _resize_scale(stage: StageConfig, rng: np.random.Generator) -> float:
    """Sample the official unrestricted stage resize range."""
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
    """HYPIR Gaussian/Poisson sampling, followed by clipping without rounding."""
    source = image
    gaussian = rng.random() < settings.gaussian_probability
    gray = rng.random() < settings.gray_probability
    if gaussian:
        sigma = _sample(settings.gaussian_sigma, rng)
        shape = (*image.shape[:2], 1 if gray else 3)
        noise = rng.normal(0, sigma / 255, shape)
        kind, parameter = "gaussian", sigma
    else:
        signal = (source * np.array([0.2989, 0.5870, 0.1140])).sum(2, keepdims=True) if gray else source
        signal = (signal * 255).round().clip(0, 255) / 255
        levels = 2 ** math.ceil(math.log2(max(1, np.unique(signal).size)))
        amplitude = _sample(settings.poisson_scale, rng)
        noise = (rng.poisson(signal * levels) / levels - signal) * amplitude
        kind, parameter = "poisson", amplitude
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
    return (source + noise).clip(0, 1).astype(np.float32)


def add_grain(
    image: Pixels, settings: GrainConfig, rng: np.random.Generator, trace: list[dict] | None = None
) -> Pixels:
    """Draw a single optional post-alignment grain pass, independent of SR factor."""
    if rng.random() >= settings.probability:
        return image
    sigma = _sample(settings.sigma, rng)
    radius = _sample(settings.correlation, rng)
    noise = rng.standard_normal(image.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), max(0.05, radius))
    noise -= noise.mean()
    noise /= max(float(noise.std()), 1e-8)
    if trace is not None:
        trace.append({"op": "grain", "sigma": sigma, "correlation": radius, "shape": list(image.shape[:2])})
    return image + noise[..., None] * (sigma / 255)


def compress_image(image: Pixels, codec: str, quality: int) -> Pixels:
    """Perform a real RGB 8-bit JPEG/WebP/JPEG2000 round trip without silent fallback.

    JPEG/WebP quality uses 0..100, JPEG2000 uses OpenCV's 0..1000 compression
    scale. Input/output are CPU HWC float32 [0,1]. Unavailable codecs and failed
    encodes raise rather than replacing the requested degradation with identity.
    JPEG2000 inputs smaller than 32 pixels are reflect-padded for OpenJPEG and
    cropped back after decoding; other codecs keep their original dimensions.
    """
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
        or image.dtype != np.float32
        or not np.isfinite(image).all()
    ):
        raise ValueError("Compression requires nonempty finite HWC RGB float32 pixels")
    formats = {
        "jpeg": (".jpg", cv2.IMWRITE_JPEG_QUALITY, 100),
        "webp": (".webp", cv2.IMWRITE_WEBP_QUALITY, 100),
        "jpeg2000": (".jp2", cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 1000),
    }
    if codec not in formats:
        raise ValueError(f"Unsupported degradation codec: {codec}")
    extension, option, maximum = formats[codec]
    if not 0 <= quality <= maximum:
        raise ValueError(f"Invalid {codec} quality: {quality}")
    rgb = (image * 255).round().clip(0, 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    if codec == "jpeg2000" and min(height, width) < 32:
        rgb = np.pad(rgb, ((0, max(0, 32 - height)), (0, max(0, 32 - width)), (0, 0)), mode="reflect")
    try:
        options = [option, quality]
        if codec == "jpeg":
            # Make HYPIR's 4:2:0 chroma layout explicit for the real codec too.
            options += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420]
        success, encoded = cv2.imencode(extension, rgb[:, :, ::-1], options)
    except cv2.error as error:
        raise RuntimeError(f"{codec} degradation encoder is unavailable or failed") from error
    if not success:
        raise RuntimeError(f"{codec} degradation encode failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError(f"{codec} degradation decode failed")
    return decoded[:height, :width, ::-1].astype(np.float32) / np.float32(255)


def _compress(
    image: Pixels, settings: CompressionConfig, rng: np.random.Generator, trace: list[dict] | None = None
) -> Pixels:
    index = int(rng.choice(3, p=settings.probabilities))
    bounds = (settings.jpeg_quality, settings.webp_quality, settings.jpeg2000_quality)[index]
    codec = ("jpeg", "webp", "jpeg2000")[index]
    quality = round(_sample(bounds, rng))
    result = compress_image(image.clip(0, 1), codec, quality)
    if trace is not None:
        trace.append({"op": "compression", "codec": codec, "quality": quality, "shape": list(image.shape[:2])})
    if settings.recompression_probability > 0 and rng.random() < settings.recompression_probability:
        quality = round(_sample(bounds, rng))
        result = compress_image(result, codec, quality)
        if trace is not None:
            trace.append({"op": "recompression", "codec": codec, "quality": quality, "shape": list(image.shape[:2])})
    return result


def _usm(image: Pixels, spatial_scale: float) -> Pixels:
    """BasicSR-style thresholded USM on the degradation source only."""
    radius = _scaled_odd_size(51, spatial_scale)
    blur = cv2.GaussianBlur(image, (radius, radius), 0)
    residual = image - blur
    mask = (np.abs(residual) * 255 > 10).astype(np.float32)
    soft = cv2.GaussianBlur(mask, (radius, radius), 0)
    sharp = (image + 0.5 * residual).clip(0, 1)
    return (soft * sharp + (1 - soft) * image).astype(np.float32)


def _sample_scale(settings: DegradationConfig, rng: np.random.Generator) -> float:
    if settings.scales is not None:
        return float(rng.choice(settings.scales))
    return math.exp(rng.uniform(*np.log(settings.scale_range)))


@torch.autocast("cpu", enabled=False)
def degrade_pair(
    high: torch.Tensor, settings: DegradationConfig, seed: int, *, trace: list[dict] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return LQ and its HYPIR-style USM supervision target.

    Trace contains actual grids, kernels and sampled noise/codec values. It does
    not consume randomness. The original HQ and global random states are intact.
    """
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
    # Stable per-operator streams prevent image-dependent Poisson consumption
    # or optional noise categories from changing subsequent geometry or codecs.
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(12)]
    image = high.detach().permute(1, 2, 0).numpy().copy()
    height, width = image.shape[:2]
    scale = _sample_scale(settings, rng)
    spatial = _resolution_scale(min(height, width), settings.reference_resolution, settings.resolution_exponent)
    target = (max(1, int(height / scale)), max(1, int(width / scale)))
    # Official presets target x4. Avoid multiplying their intermediate loss of
    # detail by another x1.5/x2 at x6/x8. Keep x1..x4 exactly unchanged.
    relief = min(1.0, 4.0 / scale)

    def record(op: str, value: Pixels, **params) -> None:
        if trace is not None:
            trace.append({"op": op, "shape": list(value.shape[:2]), **params})

    record("source", image, scale=scale, spatial_scale=spatial, target=list(target))
    if settings.use_usm:
        image = _usm(image, spatial)
        record("usm", image)
    target_high = (
        torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()
        if settings.use_usm and settings.sharpen_target
        else high.clone()
    )

    def blur(value: Pixels, stage: StageConfig, index: int) -> Pixels:
        blur_rng = streams[(index - 1) * 4]
        if blur_rng.random() >= stage.blur_probability:
            return value
        is_sinc = blur_rng.random() < stage.sinc_probability
        kernel = (
            _resolution_sinc_kernel(blur_rng, spatial, stage=True)
            if is_sinc
            else mixed_blur_kernel(blur_rng, stage.blur, spatial)
        )
        record("stage_sinc" if is_sinc else "blur", value, stage=index, kernel=kernel.tolist())
        return _filter(value, kernel)

    for index, stage in enumerate((settings.first, settings.second), 1):
        # Optional ResShift-style single/two-order mixture; official default is 1.
        if index == 2 and settings.second_order_probability < 1 and rng.random() >= settings.second_order_probability:
            break
        image = blur(image, stage, index)
        resize_rng, noise_rng, codec_rng = streams[(index - 1) * 4 + 1 : index * 4]
        resize = _resize_scale(stage, resize_rng)
        if index == 2 and resize < 1:
            resize = 1.0 - (1.0 - resize) * relief
        base_h, base_w = (height, width) if index == 1 else target
        image = _resize(
            image,
            max(1, int(base_h * resize)),
            max(1, int(base_w * resize)),
            resize_rng,
            settings,
            factor=resize if index == 1 else None,
        )
        record("resize", image, stage=index, factor=resize)
        if noise_rng.random() < stage.noise_probability:
            image = add_noise(image, stage.noise, noise_rng, trace=trace)
        if index == 1 and codec_rng.random() < stage.compression_probability:
            image = _compress(image, stage.compression, codec_rng, trace)

    sinc = streams[8].random() < settings.final_sinc_probability
    kernel = _resolution_sinc_kernel(streams[8], max(0.5, spatial * relief)) if sinc else None

    def delivery(value: Pixels) -> Pixels:
        value = _resize(value, *target, streams[9], settings)
        record("delivery_resize", value)
        if kernel is not None:
            record("final_sinc", value, kernel=kernel.tolist())
            value = _filter(value, kernel)
        return value

    compression = streams[7].random() < settings.second.compression_probability
    if streams[10].random() < 0.5:
        image = delivery(image)
        if compression:
            image = _compress(image, settings.second.compression, streams[7], trace)
    else:
        if compression:
            image = _compress(image, settings.second.compression, streams[7], trace)
        image = delivery(image)
    record("native_lq", image)
    value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    # Same-size restoration contract; interpolation is not another degradation.
    if settings.output_mode == "aligned":
        value = F.interpolate(value[None], size=(height, width), mode="bicubic", align_corners=False)[0]
        record("alignment", value.permute(1, 2, 0).numpy())
        pixels = add_grain(value.permute(1, 2, 0).numpy(), settings.final_grain, streams[11], trace)
        value = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1)
    value = (value * 255).round().clamp(0, 255) / 255
    record("quantize", value.permute(1, 2, 0).numpy())
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
    return value.clamp(0, 1).contiguous(), target_high


def degrade_image(
    high: torch.Tensor, settings: DegradationConfig, seed: int, *, trace: list[dict] | None = None
) -> torch.Tensor:
    """LQ-only diagnostic wrapper; training/calibration must use degrade_pair."""
    return degrade_pair(high, settings, seed, trace=trace)[0]
