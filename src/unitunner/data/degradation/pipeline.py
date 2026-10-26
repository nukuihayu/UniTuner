"""CPU super-resolution degradation with correlated acquisition and delivery scenes.

Photo, web-repost and low-light samples share blur/noise/codec operators.
These are synthetic processed-RGB approximations, not a calibrated RAW camera ISP.
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
    NoiseConfig,
    StageConfig,
)

type Pixels = NDArray[np.float32]


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


def _sample(bounds: tuple[float, float], rng: np.random.Generator, level: float | None = None) -> float:
    """Sample uniformly alone, or with small variation around a shared scene severity."""
    position = rng.random() if level is None else float(np.clip(level + rng.uniform(-0.15, 0.15), 0, 1))
    return bounds[0] + (bounds[1] - bounds[0]) * position


def _optical_kernel(size: int, sigma: float, angle: float, *, motion: bool) -> Pixels:
    """Integrate a centered line exposure or circular aperture over pixel areas."""
    axis = (np.arange(size * 4, dtype=np.float64) - (size * 4 - 1) / 2) / 4
    xx, yy = np.meshgrid(axis, axis)
    sigma = max(sigma, 0.25)
    if motion:
        u = xx * np.cos(angle) + yy * np.sin(angle)
        v = -xx * np.sin(angle) + yy * np.cos(angle)
        aperture = (np.abs(u) <= math.sqrt(3) * sigma) & (np.abs(v) <= 0.5)
    else:
        aperture = xx**2 + yy**2 <= (2 * sigma) ** 2
    kernel = aperture.reshape(size, 4, size, 4).mean((1, 3))
    return (kernel / kernel.sum()).astype(np.float32)


def mixed_blur_kernel(
    rng: np.random.Generator, settings: BlurConfig, resolution: int, *, severity: float | None = None
) -> Pixels:
    """Sample an optical or reconstruction PSF; pass 1024 for local-pixel blur."""
    size = int(rng.choice(np.arange(11, 42, 2)))
    choice = rng.random()
    if choice < settings.sinc_probability:
        return sinc_kernel(size, float(rng.uniform(math.pi / (3 if size < 25 else 5), math.pi)))
    spatial_scale = float(np.clip(math.sqrt(resolution / 1024), 0.5, 1.5))
    size = 2 * max(1, round((size // 2) * spatial_scale)) + 1
    if choice < settings.sinc_probability + settings.motion_probability + settings.defocus_probability:
        return _optical_kernel(
            size,
            _sample(settings.sigma, rng, severity) * spatial_scale,
            rng.uniform(-math.pi, math.pi),
            motion=choice < settings.sinc_probability + settings.motion_probability,
        )
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
            antialias=shrinking and mode in ("bilinear", "bicubic"),
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )


def _resize_scale(stage: StageConfig, rng: np.random.Generator, scale: int) -> float:
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
        shot = _sample(settings.iso_shot, rng, severity)
        read = _sample(settings.iso_read, rng, severity)
        gains = np.array([rng.uniform(*settings.iso_color_gain), 1, rng.uniform(*settings.iso_color_gain)])
        variance = source * gains * shot + read
        if gray:
            variance = variance.mean(2, keepdims=True)
        noise = rng.standard_normal(shape) * np.sqrt(variance.clip(0))
    result = (source + noise * strength).clip(0, 1)
    if kind == 2:
        result = np.where(result <= 0.0031308, result * 12.92, 1.055 * result ** (1 / 2.4) - 0.055)
    return result.clip(0, 1).astype(np.float32)


def _camera_processing(image: Pixels, severity: float) -> Pixels:
    """Approximate edge-preserving ISP denoising and stronger chroma smoothing."""
    image = np.asarray(cv2.bilateralFilter(image, 5, 0.02 + 0.06 * severity, 0.8 + severity), dtype=np.float32)
    luminance = (image * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(2, keepdims=True)
    chroma = image - luminance
    chroma = cv2.GaussianBlur(chroma, (0, 0), 0.6 + 0.6 * severity)
    return (luminance + chroma).clip(0, 1).astype(np.float32)


def _processed_photo(
    image: Pixels, settings: NoiseConfig, rng: np.random.Generator, severity: float, *, grainy: bool
) -> Pixels:
    """Residual camera grain or texture suppression in a processed phone photo.

    Operates on the delivery grid, with no semantic masks or geometry changes.
    Grain is only partially denoised; texture smoothing preserves larger edges.
    """
    if grainy:
        noisy = add_noise(image, settings, rng, camera=True, severity=severity, strength=0.8 + 0.4 * severity)
        return (0.75 * noisy + 0.25 * _camera_processing(noisy, severity)).astype(np.float32)
    smooth = np.asarray(cv2.bilateralFilter(image, 9, 0.06 + 0.10 * severity, 1.5 + severity), dtype=np.float32)
    return _camera_processing(smooth, severity)


def _sharpen(image: Pixels, amount: float) -> Pixels:
    """Model mild delivery sharpening in LQ only; leave the supervision target intact."""
    smooth = cv2.GaussianBlur(image, (0, 0), 0.7)
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
    return compress_image(image, ("jpeg", "webp", "jpeg2000")[index], round(_sample(quality_range, rng, 1 - severity)))


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
    # Most samples are mild/moderate; the same draw links blur, noise and codec damage.
    severity = float(rng.beta(2, 3))
    noise_strength = {"photo": 0.35, "web": 0.5, "low-light": 1.0}[scene]
    codec_severity = severity if scene == "web" else severity * 0.5
    scale = int(rng.choice(settings.scales))
    for index, stage in enumerate((settings.first, settings.second)):
        resize = _resize_scale(stage, rng, scale)
        # Acquisition blur precedes resampling. Delivery blur acts on the LQ grid.
        operations = ("blur", "resize", "noise") if index == 0 else ("resize", "blur", "noise")
        for operation in operations:
            match operation:
                case "blur":
                    if rng.random() < stage.blur_probability:
                        reference = min(height, width) if index == 0 else 1024
                        kernel = mixed_blur_kernel(rng, stage.blur, reference, severity=severity)
                        image = _filter(image, kernel).clip(0, 1)
                case "resize":
                    h, w = image.shape[:2] if index == 0 else (height / scale, width / scale)
                    image = _resize(
                        image, int(h * resize), int(w * resize), rng, settings, factor=resize if index == 0 else None
                    ).clip(0, 1)
                case "noise":
                    if rng.random() < stage.noise_probability:
                        image = add_noise(
                            image,
                            stage.noise,
                            rng,
                            strength=noise_strength,
                            camera=scene == "low-light" and index == 0,
                            severity=severity,
                        )
                        if scene == "low-light" and index == 0:
                            image = _camera_processing(image, severity)
        if index == 0 and scene == "web" and rng.random() < stage.compression_probability:
            image = _compress(image, stage.compression, rng, codec_severity)
    image = _resize(image, height // scale, width // scale, rng, settings).clip(0, 1)
    if rng.random() < settings.final_sinc_probability:
        size = int(rng.choice(np.arange(7, 22, 2)))
        image = _filter(image, sinc_kernel(size, float(rng.uniform(math.pi / 2, math.pi)))).clip(0, 1)
    processing = str(rng.choice(("standard", "grainy", "smoothed"), p=settings.processing_probabilities))
    if processing == "grainy":
        # Previously enlarged phone photos retain finer grain than noise added
        # before the SR downsample. Encode this delivery-size grain as well.
        image = _resize(image, max(image.shape[0], height // 2), max(image.shape[1], width // 2), rng, settings).clip(
            0, 1
        )
    if processing != "standard":
        image = _processed_photo(image, settings.first.noise, rng, severity, grainy=processing == "grainy")
    if processing == "smoothed":
        image = _sharpen(image, float(rng.uniform(0.4, 0.8)))
    elif rng.random() < (0.6 if scene == "web" else 0.3):
        image = _sharpen(image, float(rng.uniform(0.1, 0.4)))
    if rng.random() < settings.second.compression_probability:
        image = _compress(image, settings.second.compression, rng, codec_severity)
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
    image = _resize(value.permute(1, 2, 0).numpy(), height, width, rng, settings)
    return torch.from_numpy((image * 255).round().clip(0, 255).copy()).permute(2, 0, 1) / 255
