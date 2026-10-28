"""CPU super-resolution degradation with correlated acquisition and delivery scenes.

Photo, web-repost and low-light samples share blur/noise/codec operators.
These are synthetic processed-RGB approximations, not a calibrated RAW camera ISP.
"""

import math
from functools import lru_cache
from itertools import permutations

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
    NoiseConfig,
    StageConfig,
)

type Pixels = NDArray[np.float32]


def _resolution_scale(resolution: int) -> float:
    """Scale spatial kernels from a 1024-pixel short-edge reference."""
    if resolution < 1:
        raise ValueError("Kernel resolution must be positive")
    ratio = resolution / 1024
    exponent = 0.25 if ratio < 1 else 0.5
    return float(np.clip(ratio**exponent, 0.5, 2.0))


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
    # Taper the finite support: abrupt truncation can amplify edge overshoot
    # after DC normalization, especially for short kernels on coarse grids.
    window = torch.kaiser_window(size, periodic=False, beta=4.0, dtype=torch.float64)
    kernel *= window[:, None] * window[None, :]
    return (kernel / kernel.sum()).float().numpy()


def _resolution_sinc_kernel(rng: np.random.Generator, resolution: int) -> Pixels:
    """Sample an independent sinc layer at the current short-edge scale."""
    spatial_scale = _resolution_scale(resolution)
    size = _scaled_odd_size(int(rng.choice(np.arange(7, 22, 2))), spatial_scale)
    cutoff = min(math.pi, float(rng.uniform(math.pi / 2, math.pi)) / spatial_scale)
    return sinc_kernel(size, cutoff)


def _sample(bounds: tuple[float, float], rng: np.random.Generator, level: float | None = None) -> float:
    """Keep scene correlation while reserving 20% of draws for the full range."""
    position = (
        rng.random() if level is None or rng.random() < 0.2 else float(np.clip(level + rng.uniform(-0.15, 0.15), 0, 1))
    )
    return bounds[0] + (bounds[1] - bounds[0]) * position


def _defocus_kernel(size: int, radius: float) -> Pixels:
    """Integrate a circular aperture over pixel areas without truncating its rim."""
    size = max(size, 2 * math.ceil(radius) + 1)
    if radius == 0:
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, size // 2] = 1
        return kernel
    axis = (np.arange(size * 4, dtype=np.float64) - (size * 4 - 1) / 2) / 4
    xx, yy = np.meshgrid(axis, axis)
    aperture = xx**2 + yy**2 <= max(radius, 0.25) ** 2
    kernel = aperture.reshape(size, 4, size, 4).mean((1, 3))
    return (kernel / kernel.sum()).astype(np.float32)


def _motion_kernel(size: int, length: float, angle: float, acceleration: float, focus_radius: float = 0.0) -> Pixels:
    """Integrate a straight, nonuniform-speed handheld exposure, centered for aligned pairs.

    Equal time samples and bilinear splats preserve exposure energy and the
    subpixel centroid. Optional aperture convolution adds simultaneous defocus.
    """
    time = np.linspace(-0.5, 0.5, max(32, math.ceil(length * 8)), dtype=np.float64)
    along = length * (time + acceleration * (time**2 - 0.25))
    x = along * math.cos(angle)
    y = along * math.sin(angle)
    x -= x.mean()
    y -= y.mean()
    half = max(size // 2, math.ceil(max(float(np.abs(x).max()), float(np.abs(y).max()))) + 1)
    x += half
    y += half
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    kernel = np.zeros((2 * half + 1, 2 * half + 1), dtype=np.float64)
    for dx, wx in ((0, 1 - fx), (1, fx)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            np.add.at(kernel, (iy + dy, ix + dx), wx * wy / time.size)
    if focus_radius > 0:
        aperture = _defocus_kernel(2 * math.ceil(focus_radius) + 1, focus_radius)
        kernel = np.pad(kernel, aperture.shape[0] // 2)
        kernel = cv2.filter2D(kernel, -1, aperture, borderType=cv2.BORDER_CONSTANT).clip(0)
    return (kernel / kernel.sum()).astype(np.float32)


def mixed_blur_kernel(
    rng: np.random.Generator, settings: BlurConfig, resolution: int, *, severity: float | None = None
) -> Pixels:
    """Sample a PSF scaled to the layer input's current short edge."""
    size = int(rng.choice(np.arange(11, 42, 2)))
    choice = rng.random()
    spatial_scale = _resolution_scale(resolution)
    size = _scaled_odd_size(size, spatial_scale)
    if choice < settings.motion_probability:
        # Camera shake and imperfect focus can coexist. A continuous radius
        # starting at zero also includes essentially sharp-focus motion blur.
        return _motion_kernel(
            size,
            _sample(settings.motion_length, rng, severity) * spatial_scale,
            float(rng.uniform(-math.pi, math.pi)),
            float(rng.uniform(-0.6, 0.6)),
            _sample(settings.defocus_radius, rng, severity) * spatial_scale * float(rng.random()),
        )
    if choice < settings.motion_probability + settings.defocus_probability:
        return _defocus_kernel(size, _sample(settings.defocus_radius, rng, severity) * spatial_scale)
    kind = int(rng.choice(6, p=settings.probabilities))
    sigma_x = _sample(settings.sigma, rng, severity) * spatial_scale
    sigma_y = sigma_x if kind % 2 == 0 else _sample(settings.sigma, rng, severity) * spatial_scale
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
    shrinking = factor < 1 if factor is not None else height < image.shape[0] or width < image.shape[1]
    modes = settings.resize_modes
    if not shrinking:
        modes = tuple(mode for mode in modes if mode != "area") or modes
    mode = str(rng.choice(modes))
    value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    return (
        F.interpolate(
            value,
            size=(max(1, height), max(1, width)) if factor is None else None,
            scale_factor=factor,
            # Match later size-based resizes after fractional dimensions round
            # down, rather than accumulating a position-dependent grid offset.
            recompute_scale_factor=True if factor is not None else None,
            mode=mode,
            align_corners=False if mode in ("bilinear", "bicubic") else None,
            antialias=shrinking and mode in ("bilinear", "bicubic"),
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )


def _resize_scale(stage: StageConfig, rng: np.random.Generator, scale: float) -> float:
    """Sample up/down/identity scale, limiting downscale by the selected degradation factor."""
    direction = int(rng.choice(3, p=stage.resize_probabilities))
    if direction == 0:
        return float(rng.uniform(1, stage.resize_range[1]))
    if direction == 1:
        return float(rng.uniform(max(stage.resize_range[0], 1 / scale), 1))
    return 1.0


def add_noise(
    image: Pixels,
    settings: NoiseConfig,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
    camera: bool = False,
    severity: float | None = None,
    chroma_scale: float = 1.0,
) -> Pixels:
    """Gaussian/Poisson residuals in sRGB, or shot/read noise in linear light.

    Low-light acquisition selects the camera branch. Strength scales standard
    deviation, not variance. Poisson uses a fixed 255-level reference so image
    size and the number of unique colors cannot change its noise amplitude.
    """
    source = image.clip(0, 1)
    kind = 2 if camera else int(rng.choice(3, p=settings.probabilities))
    gray = rng.random() < settings.gray_probability
    shape = (*image.shape[:2], 1) if gray else image.shape
    exposure = 1.0
    if kind == 0:
        noise = rng.normal(0, _sample(settings.gaussian_sigma, rng, severity) / 255, size=shape)
    elif kind == 1:
        signal = (
            (source * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(2, keepdims=True) if gray else source
        )
        amplitude = _sample(settings.poisson_scale, rng, severity)
        noise = (rng.poisson(signal * 255) / 255 - signal) * amplitude
    else:
        source = np.where(source <= 0.04045, source / 12.92, ((source + 0.055) / 1.055) ** 2.4)
        # Capture fewer photons, then recover exposure after sensor noise/clipping.
        exposure = _sample(settings.exposure, rng, None if severity is None else 1 - severity)
        source = source * exposure
        shot = _sample(settings.iso_shot, rng, severity)
        read = _sample(settings.iso_read, rng, severity)
        gains = np.array([rng.uniform(*settings.iso_color_gain), 1, rng.uniform(*settings.iso_color_gain)])
        variance = source * gains * shot + read
        if gray:
            variance = variance.mean(2, keepdims=True)
        noise = rng.standard_normal(shape) * np.sqrt(variance.clip(0))
    if noise.shape[-1] == 3 and chroma_scale < 1:
        luminance = (noise * np.array([0.2126, 0.7152, 0.0722])).sum(2, keepdims=True)
        adjusted = luminance + chroma_scale * (noise - luminance)
        original_energy = float(np.square(noise).mean())
        adjusted_energy = float(np.square(adjusted).mean())
        if adjusted_energy > 0:
            adjusted *= math.sqrt(original_energy / adjusted_energy)
        noise = adjusted
    result = (source + noise * strength).clip(0, 1)
    if kind == 2:
        result = (result / exposure).clip(0, 1)
        result = np.where(result <= 0.0031308, result * 12.92, 1.055 * result ** (1 / 2.4) - 0.055)
    return result.clip(0, 1).astype(np.float32)


def _camera_processing(image: Pixels, severity: float) -> Pixels:
    """Approximate edge-preserving ISP denoising and stronger chroma smoothing."""
    spatial_scale = _resolution_scale(min(image.shape[:2]))
    image = np.asarray(
        cv2.bilateralFilter(
            image,
            _scaled_odd_size(5, spatial_scale),
            0.02 + 0.06 * severity,
            (0.8 + severity) * spatial_scale,
        ),
        dtype=np.float32,
    )
    luminance = (image * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(2, keepdims=True)
    chroma = image - luminance
    chroma = cv2.GaussianBlur(chroma, (0, 0), (0.6 + 0.6 * severity) * spatial_scale)
    return (luminance + chroma).clip(0, 1).astype(np.float32)


def _smooth_texture(
    image: Pixels,
    severity: float,
    *,
    strength: float,
    detail_protection: float = 0.6,
    downsample_factor: float = 1.0,
) -> Pixels:
    """Continuously suppress microtexture, limiting extra blur on coarse grids."""
    if strength == 0:
        return image.copy()
    # Limit extra texture suppression on coarse grids: downsampling already
    # removes detail, and every delivery pixel covers a larger HQ footprint.
    strength *= min(1.0, 2.0 / max(1.0, downsample_factor))
    # A single moderate pass removes microtexture without repeatedly flattening
    # mid-frequency facial and hair detail on the already downsampled grid.
    spatial_scale = _resolution_scale(min(image.shape[:2]))
    smooth = np.asarray(
        cv2.bilateralFilter(
            image,
            _scaled_odd_size(9, spatial_scale),
            0.05 + 0.10 * severity,
            (1.5 + 1.5 * severity) * spatial_scale,
        ),
        dtype=np.float32,
    )
    smooth = _camera_processing(smooth, severity)
    luminance = (image * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(2)
    local_sigma = 1.2 * spatial_scale
    mean = cv2.GaussianBlur(luminance, (0, 0), local_sigma)
    variance = (cv2.GaussianBlur(luminance * luminance, (0, 0), local_sigma) - mean * mean).clip(0)
    # Smooth weak microtexture more than eyelashes/hair contours, with a continuous
    # local-contrast map: no face masks, hard patch boundaries or geometric warp.
    blend = strength * (1 - detail_protection * variance / (variance + 0.0009))
    return (image + blend[..., None] * (smooth - image)).clip(0, 1).astype(np.float32)


def _residual_grain(
    image: Pixels,
    settings: NoiseConfig,
    rng: np.random.Generator,
    severity: float,
    *,
    strength: float,
    spatial_scale: float = 1.0,
) -> Pixels:
    """Add continuously scaled residual grain without resampling image structure.

    Only the noise residual is enlarged from its sampled grid. This avoids
    tying additional image blur to whether a tiny grain gain is nonzero.
    """
    if strength == 0:
        return image.copy()
    height, width = image.shape[:2]
    shape = (max(1, int(width / spatial_scale)), max(1, int(height / spatial_scale)))
    source = np.asarray(cv2.resize(image, shape, interpolation=cv2.INTER_AREA), dtype=np.float32)
    noisy = add_noise(source, settings, rng, camera=True, severity=severity, strength=0.8 + 0.4 * severity)
    residual = noisy - source
    luminance = (residual * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(2, keepdims=True)
    fine = cv2.GaussianBlur(luminance[..., 0], (0, 0), 0.65)[..., None]
    chroma = cv2.GaussianBlur(residual - luminance, (0, 0), 0.8 + severity)
    residual = 0.65 * residual + 0.7 * fine + (0.8 + severity) * chroma
    # Partial denoising acts on residual noise, never on the underlying picture.
    residual = 0.85 * residual + 0.15 * cv2.GaussianBlur(residual, (0, 0), 0.6)
    residual = cv2.resize(residual, (width, height), interpolation=cv2.INTER_LINEAR)
    return (image + strength * residual).clip(0, 1).astype(np.float32)


def _processing_gain(bounds: tuple[float, float], level: float, deadzone: float) -> float:
    """A zero atom joined continuously to positive gains, without mode selection.

    A positive lower bound explicitly requests an always-active operation;
    fixed bounds remain exact. Deadzone=1 disables zero-based ranges entirely.
    """
    if bounds[0] == 0:
        level = max(0.0, (level - deadzone) / (1 - deadzone)) if deadzone < 1 else 0.0
    return bounds[0] + (bounds[1] - bounds[0]) * level


def _sample_processing(
    settings: DegradationConfig, rng: np.random.Generator, acquisition_level: float = 0.5
) -> tuple[float, float, float, float, float]:
    """Sample processing with shared noise gain and acquisition-coupled grain."""
    denoising = float(rng.beta(1.5, 2.0))
    retention = float(rng.beta(1.5, 2.0)) * (1 - 0.5 * denoising)
    noise_level = float(rng.beta(1.5, 2.0))
    smooth = _processing_gain(settings.smear_strength, denoising, settings.processing_deadzone)
    noise_gain = _processing_gain(settings.noise_strength, noise_level, settings.processing_deadzone)
    # Strong acquisition noise both suppresses and often disables extra grain,
    # preventing a third noise pass from appearing in every sample.
    grain_budget = max(0.15, 1 - 0.85 * acquisition_level)
    grain = _processing_gain(settings.grain_strength, retention, 0.0) * math.sqrt(noise_gain) * grain_budget
    if rng.random() >= grain_budget:
        grain = 0.0
    footprint = math.exp(rng.uniform(math.log(settings.grain_scale_range[0]), math.log(settings.grain_scale_range[1])))
    sharpening = _processing_gain(settings.sharpen_strength, float(rng.random()), settings.processing_deadzone)
    return smooth, grain, footprint, sharpening, noise_gain


def _sample_severity(rng: np.random.Generator) -> float:
    """Invert a smooth CDF with 20/40/40 mass across diagnostic thirds.

    F(s)=0.1s+1.8s**2-0.9s**3 has strictly positive density on [0,1].
    No class is selected: every sample follows the same continuous pipeline.
    """
    quantile = float(rng.random())
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = (low + high) * 0.5
        cdf = middle * (0.1 + middle * (1.8 - 0.9 * middle))
        if cdf < quantile:
            low = middle
        else:
            high = middle
    return (low + high) * 0.5


def _sample_scale(settings: DegradationConfig, rng: np.random.Generator) -> float:
    """Sample log-uniform factors independently of damage severity."""
    if settings.scales is not None:
        return float(rng.choice(settings.scales))
    level = float(rng.random())
    low, high = (math.log(value) for value in settings.scale_range)
    return math.exp(low + (high - low) * level)


@lru_cache(maxsize=64)
def _layer_orders(layers: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Return every ordering of independent active degradation layers."""
    return tuple(permutations(layers))


@lru_cache(maxsize=64)
def _stage_orders(layers: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Order processing freely while keeping intermediate encoding last."""
    if "compression" not in layers:
        return _layer_orders(layers)
    processing = tuple(layer for layer in layers if layer != "compression")
    return tuple((*order, "compression") for order in _layer_orders(processing))


def _compression_after_resize_probability(acquisition_level: float) -> float:
    """Avoid enlarging codec blocks when acquisition noise is already strong."""
    return 0.55 + 0.30 * float(np.clip(acquisition_level, 0, 1))


def _sharpen(image: Pixels, amount: float) -> Pixels:
    """Model mild delivery sharpening in LQ only; leave the supervision target intact."""
    smooth = cv2.GaussianBlur(image, (0, 0), 0.7 * _resolution_scale(min(image.shape[:2])))
    return (image + amount * (image - smooth)).clip(0, 1).astype(np.float32)


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
        success, encoded = cv2.imencode(extension, rgb[:, :, ::-1], [option, quality])
    except cv2.error as error:
        raise RuntimeError(f"{codec} degradation encoder is unavailable or failed") from error
    if not success:
        raise RuntimeError(f"{codec} degradation encode failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError(f"{codec} degradation decode failed")
    return decoded[:height, :width, ::-1].astype(np.float32) / np.float32(255)


def _compress(image: Pixels, settings: CompressionConfig, rng: np.random.Generator, severity: float) -> Pixels:
    """Sample a real codec and rounded codec-native quality from its distribution."""
    index = int(rng.choice(3, p=settings.probabilities))
    quality_range = (settings.jpeg_quality, settings.webp_quality, settings.jpeg2000_quality)[index]
    codec = ("jpeg", "webp", "jpeg2000")[index]
    result = compress_image(image, codec, round(_sample(quality_range, rng, 1 - severity)))
    if rng.random() < settings.recompression_probability:
        # A cropped/reposted copy has a different codec block origin, but remains aligned to HQ.
        top, left = (int(value) for value in rng.integers(1, 8, size=2))
        shifted = np.pad(result, ((top, 0), (left, 0), (0, 0)), mode="reflect")
        result = compress_image(shifted, codec, round(_sample(quality_range, rng, 1 - severity)))
        result = result[top:, left:]
    return result


@torch.autocast("cpu", enabled=False)
def degrade_image(high: torch.Tensor, settings: DegradationConfig, seed: int) -> torch.Tensor:
    """Apply a correlated photo/web/low-light SR distribution with local RNG.

    Args:
        high: CPU float32 CHW RGB [0,1], each dimension >=32; never modified.
        settings: Dataset-owned two-stage settings; enabled is handled by the dataset.
        seed: Local seed, normally derived from dataset seed/epoch/item.

    Returns:
        Detached CPU float32 CHW RGB [0,1] in the original shape, quantized to
        8-bit after resize-back. NumPy/Python/Torch global RNGs are unchanged.

    Raises:
        ValueError: Input device, dtype, layout or intensity range is invalid.
        RuntimeError: A requested image codec fails.
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
    image = high.detach().permute(1, 2, 0).numpy()
    height, width = image.shape[:2]
    scene = str(rng.choice(("photo", "web", "low-light"), p=settings.scenario_probabilities))
    # Moderate and strong samples coexist; the same draw links blur, noise and codec damage.
    severity = _sample_severity(rng)
    noise_strength = {"photo": 1.0, "web": 1.0, "low-light": 1.15}[scene]
    codec_severity = severity if scene == "web" else severity * 0.5
    scale = _sample_scale(settings, rng)
    smear_strength, grain_strength, grain_scale, sharpening, noise_gain = _sample_processing(settings, rng, severity)
    noise_strength *= noise_gain
    selected_noise = tuple(
        noise_gain > 0 and rng.random() < stage.noise_probability for stage in (settings.first, settings.second)
    )
    late_noise = selected_noise[1] and rng.random() < 0.5
    if all(selected_noise):
        first_share = settings.first_noise_variance_share
        noise_gains = (math.sqrt(first_share), math.sqrt(1 - first_share))
    elif selected_noise[0]:
        noise_gains = (1.0, 0.0)
    elif selected_noise[1]:
        noise_gains = (0.0, 1.0)
    else:
        noise_gains = (0.0, 0.0)

    for index, stage in enumerate((settings.first, settings.second)):
        resize = _resize_scale(stage, rng, scale)
        blur = rng.random() < stage.blur_probability
        noise = selected_noise[index] and not (index == 1 and late_noise)
        compression = index == 0 and scene == "web" and rng.random() < stage.compression_probability
        active = tuple(
            name
            for name, enabled in (("resize", True), ("blur", blur), ("noise", noise), ("compression", compression))
            if enabled
        )
        orders = _stage_orders(active)
        for layer in orders[int(rng.integers(len(orders)))]:
            match layer:
                case "blur":
                    reference = min(image.shape[:2])
                    kernel = mixed_blur_kernel(rng, stage.blur, reference, severity=severity)
                    image = _filter(image, kernel).clip(0, 1)
                case "resize":
                    h, w = image.shape[:2] if index == 0 else (height / scale, width / scale)
                    image = _resize(
                        image, int(h * resize), int(w * resize), rng, settings, factor=resize if index == 0 else None
                    ).clip(0, 1)
                case "noise":
                    image = add_noise(
                        image,
                        stage.noise,
                        rng,
                        strength=noise_strength * noise_gains[index],
                        camera=scene == "low-light" and index == 0,
                        severity=severity,
                        chroma_scale=settings.second_noise_chroma_scale if index == 1 else 1.0,
                    )
                case "compression":
                    image = _compress(image, stage.compression, rng, codec_severity)
    sinc = rng.random() < settings.final_sinc_probability
    compression = rng.random() < settings.second.compression_probability
    processing = tuple(
        name
        for name, enabled in (
            ("smear", smear_strength > 0),
            ("sharpen", sharpening > 0),
            ("noise", grain_strength > 0),
        )
        if enabled
    )
    orders = _layer_orders(processing) if processing else ((),)
    order = orders[int(rng.integers(len(orders)))]

    def apply_processing(layer: str, value: Pixels) -> Pixels:
        match layer:
            case "smear":
                return _smooth_texture(
                    value,
                    severity,
                    strength=smear_strength,
                    detail_protection=settings.smear_detail_protection,
                    downsample_factor=max(height / value.shape[0], width / value.shape[1]),
                )
            case "sharpen":
                return _sharpen(value, sharpening)
            case "noise":
                if grain_strength > 0:
                    reduction = max(height / value.shape[0], width / value.shape[1])
                    value = _residual_grain(
                        value,
                        settings.first.noise,
                        rng,
                        severity,
                        strength=grain_strength,
                        spatial_scale=max(1.0, grain_scale / reduction),
                    )
                return value
        raise ValueError(f"Unknown processing layer: {layer}")

    # Keep the Real-ESRGAN delivery constraint: resize-back and sinc are one
    # atomic operation, exchanged only as a unit with the final codec pass.
    for layer in order:
        image = apply_processing(layer, image)

    def resize_and_sinc(value: Pixels) -> Pixels:
        if value.shape[:2] != (height, width):
            value = _resize(value, height, width, rng, settings).clip(0, 1)
        if sinc:
            value = _filter(value, _resolution_sinc_kernel(rng, min(value.shape[:2]))).clip(0, 1)
        return value

    if compression and rng.random() < _compression_after_resize_probability(severity):
        image = resize_and_sinc(image)
        image = _compress(image, settings.second.compression, rng, codec_severity)
    else:
        if compression:
            image = _compress(image, settings.second.compression, rng, codec_severity)
        image = resize_and_sinc(image)

    if late_noise:
        image = add_noise(
            image,
            settings.second.noise,
            rng,
            strength=noise_strength * noise_gains[1],
            severity=severity,
            chroma_scale=settings.second_noise_chroma_scale,
        )

    value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    if rng.random() < settings.gray_probability:
        value = image_ops.rgb_to_grayscale(value, num_output_channels=3)
    if rng.random() < settings.color_jitter_probability:
        for operation in rng.permutation(4):
            match int(operation):
                case 0:
                    value = image_ops.adjust_brightness(value, float(rng.uniform(*settings.brightness)))
                case 1:
                    value = image_ops.adjust_contrast(value, float(rng.uniform(*settings.contrast)))
                case 2:
                    value = image_ops.adjust_saturation(value, float(rng.uniform(*settings.saturation)))
                case 3:
                    value = image_ops.adjust_hue(value, float(rng.uniform(*settings.hue)))
    image = value.permute(1, 2, 0).numpy()
    return torch.from_numpy((image * 255).round().clip(0, 255).copy()).permute(2, 0, 1) / 255
