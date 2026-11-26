"""OMGSR-F-1024 two-stage degradation with phase-stable interpolation."""

import math

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from unituner.data.degradation import color as image_ops
from unituner.data.degradation.config import (
    BlurConfig,
    CompressionConfig,
    DegradationConfig,
    ExtremeDegradationConfig,
    NoiseConfig,
    Scale,
    SecondStageConfig,
    StageConfig,
)
from unituner.data.degradation.diffjpeg import diffjpeg
from unituner.data.degradation.resize import resize_centered, scaled_shape

type Pixels = NDArray[np.float32]

_RESIZE_MODES = ("bilinear", "bicubic")
_SECOND_STAGE_RESIZE_MODES = ("area", "bilinear", "bicubic", "nearest")
_KERNEL_SIZES = tuple(range(11, 42, 2))
_KERNEL_MAX = 41
_KERNEL_MID = 25
_REAL_ESRGAN_RESIZE_MODES = ("area", "bilinear", "bicubic")
_WIDE_KERNEL = (_KERNEL_SIZES, _KERNEL_MAX, _KERNEL_MID)
_SMALL_KERNEL = (tuple(range(7, 22, 2)), 21, 13)


_UPSTREAM_RECIPE = "adabridge_dnf_omgsr"


def _upstream_stages() -> tuple[StageConfig, StageConfig]:
    first = StageConfig()
    second = SecondStageConfig(resize_range=(0.6, 1.2))
    updates = {"gaussian_probability": 0.5, "iso_probability": 0.0}
    return (
        first.model_copy(
            update={
                "noise": first.noise.model_copy(update=updates),
                "compression": first.compression.model_copy(update={"jpeg_probability": 1.0}),
            }
        ),
        second.model_copy(
            update={
                "noise": second.noise.model_copy(update=updates),
                "compression": second.compression.model_copy(update={"jpeg_probability": 1.0}),
            }
        ),
    )


def _kernel_profile() -> tuple[tuple[int, ...], int, int]:
    """Return OMGSR-F-1024's fixed kernel support."""
    return _KERNEL_SIZES, _KERNEL_MAX, _KERNEL_MID


def _pad_kernel(kernel: Pixels, support: int = _KERNEL_MAX) -> Pixels:
    padding = (support - kernel.shape[0]) // 2
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


def _sinc_kernel(rng: np.random.Generator, *, stage: bool = False, profile=_WIDE_KERNEL) -> Pixels:
    sizes, support, threshold = profile
    size = int(rng.choice(sizes))
    minimum = math.pi / (5 if stage and size >= threshold else 3)
    return _pad_kernel(sinc_kernel(size, float(rng.uniform(minimum, math.pi))), support)


def _sample(bounds: tuple[float, float], rng: np.random.Generator) -> float:
    return float(rng.uniform(*bounds))


def _sample_scale(scale: Scale, rng: np.random.Generator) -> float:
    return _sample(scale, rng) if isinstance(scale, tuple) else scale


def mixed_blur_kernel(rng: np.random.Generator, settings: BlurConfig, *, profile=_WIDE_KERNEL) -> Pixels:
    """Sample and pad an OMGSR-F mixed blur kernel."""
    sizes, support, _ = profile
    size = int(rng.choice(sizes))
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
            rng.uniform(settings.plateau_beta[0], 1) if rng.random() < 0.5 else rng.uniform(1, settings.plateau_beta[1])
        )
        kernel = 1 / (1 + radius_squared**beta)
    return _pad_kernel((kernel / kernel.sum()).astype(np.float32), support)


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


def _resize_modes_for_shape(image: Pixels, height: int, width: int, *, second_stage: bool = False) -> tuple[str, ...]:
    """Keep block-preserving interpolation inside the small second-stage resize only."""
    return _SECOND_STAGE_RESIZE_MODES if second_stage else _RESIZE_MODES


def _resize(
    image: Pixels,
    height: int,
    width: int,
    rng: np.random.Generator,
    *,
    mode: str | None = None,
    second_stage: bool = False,
    upstream: bool = False,
) -> Pixels:
    """Resize on the integer grid so repeated half-pixel mappings compose."""
    available = (
        _REAL_ESRGAN_RESIZE_MODES
        if upstream
        else _resize_modes_for_shape(image, height, width, second_stage=second_stage)
    )
    selected = str(rng.choice(available)) if mode is None else mode
    if selected not in _SECOND_STAGE_RESIZE_MODES:
        raise ValueError(f"Unsupported OMGSR resize mode: {selected}")
    if selected not in available:
        raise ValueError("Area and nearest resize are restricted to the second stage")
    return resize_centered(image, (max(1, height), max(1, width)), selected)


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
    """Apply one Gaussian, Poisson or camera-ISO noise draw and clip."""
    poisson_probability = 1 - settings.gaussian_probability - settings.iso_probability
    kind_index = int(rng.choice(3, p=(settings.gaussian_probability, poisson_probability, settings.iso_probability)))
    gray = rng.random() < settings.gray_probability
    if kind_index == 0:
        parameter = _sample(settings.gaussian_sigma, rng)
        shape = (*image.shape[:2], 1 if gray else 3)
        noise = rng.normal(0, parameter / 255, shape)
        kind = "gaussian"
    elif kind_index == 1:
        signal = (image * np.array([0.2989, 0.5870, 0.1140])).sum(2, keepdims=True) if gray else image
        signal = (signal * 255).round().clip(0, 255) / 255
        levels = 2 ** math.ceil(math.log2(max(1, np.unique(signal).size)))
        parameter = _sample(settings.poisson_scale, rng)
        noise = (rng.poisson(signal * levels) / levels - signal) * parameter
        kind = "poisson"
    else:
        shot = _sample(settings.iso_shot_noise, rng)
        read = _sample(settings.iso_read_noise, rng)
        gains = np.array(
            [_sample(settings.iso_color_gain, rng), 1.0, _sample(settings.iso_color_gain, rng)],
            dtype=np.float32,
        )
        variance = image.clip(0, 1) * gains * shot + read
        standard = rng.standard_normal(image.shape, dtype=np.float32)
        noise = standard * np.sqrt(variance.clip(1e-10), dtype=np.float32)
        if gray:
            noise = np.repeat(noise.mean(2, keepdims=True), 3, axis=2)
        parameter = {"shot": shot, "read": read, "gains": gains.tolist()}
        kind = "iso"
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


def _webp(image: Pixels, quality: float) -> Pixels:
    bgr = np.ascontiguousarray((image.clip(0, 1)[..., ::-1] * 255).round().astype(np.uint8))
    success, encoded = cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), round(quality)])
    if not success:
        raise RuntimeError("OpenCV failed to encode WebP degradation")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("OpenCV failed to decode WebP degradation")
    return (decoded[..., ::-1].astype(np.float32) / 255).copy()


def _compress(
    image: Pixels,
    settings: CompressionConfig,
    rng: np.random.Generator,
    trace: list[dict] | None,
) -> Pixels:
    jpeg = rng.random() < settings.jpeg_probability
    bounds = settings.jpeg_quality if jpeg else settings.webp_quality
    quality = _sample(bounds, rng)
    if jpeg and np.float32(quality) >= 100:
        quality = float(np.nextafter(np.float32(100), np.float32(0)))
    result = diffjpeg(image.clip(0, 1).astype(np.float32), quality) if jpeg else _webp(image, quality)
    if trace is not None:
        trace.append(
            {
                "op": "compression",
                "codec": "diffjpeg" if jpeg else "webp",
                "quality": quality,
                "shape": list(image.shape[:2]),
            }
        )
    return result


def _jpeg(image: Pixels, quality: int) -> Pixels:
    """Round-trip RGB through an actual JPEG bitstream."""
    bgr = np.ascontiguousarray((image[..., ::-1] * 255).round().astype(np.uint8))
    success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("OpenCV failed to encode JPEG degradation")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("OpenCV failed to decode JPEG degradation")
    return (decoded[..., ::-1].astype(np.float32) / 255).copy()


def _degrade_extreme(
    image: Pixels, settings: ExtremeDegradationConfig, scale: int, seed: int, trace: list[dict] | None
) -> Pixels:
    """Shuffle corruptions before the final downsample to the native LQ grid."""
    shape = image.shape[:2]
    intermediate = tuple(max(1, size // settings.intermediate_scale) for size in shape)
    native = tuple(max(1, size // scale) for size in shape)
    order_rng, blur_rng, noise_rng, jpeg_rng = [
        np.random.default_rng(child) for child in np.random.SeedSequence([seed, 0x4558]).spawn(4)
    ]
    operations = ["blur", "intermediate_resize", "noise", "compression"]
    if order_rng.random() < settings.shuffle_probability:
        order_rng.shuffle(operations)
    operations.append("delivery_resize")

    def record(op: str, **parameters) -> None:
        if trace is not None:
            trace.append({"op": op, "shape": list(image.shape[:2]), **parameters})

    record("source", recipe="extreme_bsrgan", scale=scale, target=list(native))
    for operation in operations:
        if operation in ("intermediate_resize", "delivery_resize"):
            grid = intermediate if operation == "intermediate_resize" else native
            image = resize_centered(image, grid, settings.resize_mode).clip(0, 1)
            record(operation, mode=settings.resize_mode)
        elif operation == "blur":
            sigma = _sample(settings.blur_sigma, blur_rng)
            side = 2 * math.ceil(3 * sigma) + 1
            axis = cv2.getGaussianKernel(side, sigma)
            image = _filter(image, (axis @ axis.T).astype(np.float32)).clip(0, 1)
            record(operation, sigma=sigma)
        elif operation == "noise":
            sigma = _sample(settings.gaussian_sigma, noise_rng)
            correlation = _sample(settings.rgb_correlation, noise_rng)
            common = noise_rng.standard_normal((*image.shape[:2], 1), dtype=np.float32)
            independent = noise_rng.standard_normal(image.shape, dtype=np.float32)
            noise = math.sqrt(correlation) * common + math.sqrt(1 - correlation) * independent
            image = (image + noise * (sigma / 255)).clip(0, 1)
            record(operation, kind="correlated_gaussian", sigma=sigma, correlation=correlation)
        else:
            quality = round(_sample(settings.jpeg_quality, jpeg_rng))
            image = _jpeg(image, quality)
            record(operation, codec="jpeg", quality=quality)
    record("native_lq")
    image = resize_centered(image, shape, "bicubic").clip(0, 1)
    record("alignment", mode="bicubic")
    return image


def _linear_camera_noise(image: Pixels, rng: np.random.Generator) -> Pixels:
    """Apply shot and read noise in a linear sensor approximation."""
    linear = np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)
    gain = np.array([rng.uniform(1, 1.8), 1.0, rng.uniform(1, 1.8)], dtype=np.float32)
    sensor = linear / gain
    shot = float(rng.uniform(0.0005, 0.006))
    read = float(rng.uniform(0.00001, 0.0002))
    noise = rng.standard_normal(image.shape, dtype=np.float32) * np.sqrt(sensor * shot + read)
    rendered = ((sensor + noise) * gain).clip(0, 1)
    return np.where(rendered <= 0.0031308, 12.92 * rendered, 1.055 * rendered ** (1 / 2.4) - 0.055)


def _degrade_bsrgan_plus(
    image: Pixels, settings: ExtremeDegradationConfig, scale: int, seed: int, trace: list[dict] | None
) -> Pixels:
    """Run two KAIR-inspired stages followed by delivery resize and JPEG."""
    plus = settings.plus
    assert plus is not None
    shape = image.shape[:2]
    native = tuple(max(1, size // scale) for size in shape)
    minimum_grid = tuple(min(size, math.ceil(target * 1.25)) for size, target in zip(shape, native, strict=True))
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence([seed, 0x504C]).spawn(15)]
    order_rng, delivery_rng, *operation_rngs = streams
    noise_kinds = ("gaussian", "poisson", "speckle", "isp")
    stage1 = [(1, op) for op in ("blur", "resize", *noise_kinds, "jpeg")]
    stage2 = [(2, op) for op in ("blur", "resize", *noise_kinds)]
    global_shuffle = order_rng.random() < plus.global_shuffle_probability
    if global_shuffle:
        operations = stage1 + stage2
        order_rng.shuffle(operations)
    else:
        operations = stage1[:2] + [stage1[2 + i] for i in order_rng.permutation(4)] + stage1[6:] + stage2[:2]
        operations += [stage2[2 + i] for i in order_rng.permutation(4)]
    rngs = dict(zip(stage1 + stage2, operation_rngs, strict=True))

    def record(op: str, **parameters) -> None:
        if trace is not None:
            trace.append({"op": op, "shape": list(image.shape[:2]), **parameters})

    record(
        "source",
        recipe="bsrgan_plus",
        scale=scale,
        target=list(native),
        shuffle="global" if global_shuffle else "local",
    )
    optional = {"poisson": plus.poisson_probability, "speckle": plus.speckle_probability, "isp": plus.isp_probability}
    for stage, op in operations:
        rng = rngs[stage, op]
        if op in optional and rng.random() >= optional[op]:
            continue
        if op == "blur":
            sigma = _sample(settings.blur_sigma, rng)
            axis = cv2.getGaussianKernel(2 * math.ceil(3 * sigma) + 1, sigma)
            image = cv2.sepFilter2D(image, -1, axis, axis, borderType=cv2.BORDER_REFLECT_101).clip(0, 1)
            record(op, stage=stage, sigma=sigma)
        elif op == "resize":
            factor = float(rng.uniform(0.7, 1.3)) / math.sqrt(settings.intermediate_scale)
            grid = tuple(
                max(size, minimum)
                for size, minimum in zip(scaled_shape(image.shape[:2], factor), minimum_grid, strict=True)
            )
            mode = str(rng.choice(_REAL_ESRGAN_RESIZE_MODES))
            image = resize_centered(image, grid, mode).clip(0, 1)
            record(op, stage=stage, mode=mode)
        elif op == "jpeg":
            quality = round(_sample(settings.jpeg_quality, rng))
            image = _jpeg(image, quality)
            record("compression", stage=stage, codec="jpeg", quality=quality)
        else:
            if op == "gaussian":
                sigma = _sample(settings.gaussian_sigma, rng) / 255
                rho = _sample(settings.rgb_correlation, rng)
                common = rng.standard_normal((*image.shape[:2], 1), dtype=np.float32)
                independent = rng.standard_normal(image.shape, dtype=np.float32)
                image = (image + sigma * (math.sqrt(rho) * common + math.sqrt(1 - rho) * independent)).clip(0, 1)
            elif op == "poisson":
                levels = float(rng.uniform(128, 1024))
                image = (rng.poisson(image * levels).astype(np.float32) / levels).clip(0, 1)
            elif op == "speckle":
                sigma = float(rng.uniform(0.01, 0.06))
                image = (image * (1 + rng.standard_normal(image.shape, dtype=np.float32) * sigma)).clip(0, 1)
            else:
                image = _linear_camera_noise(image, rng)
            record("noise", stage=stage, kind=op)
    image = resize_centered(image, native, settings.resize_mode).clip(0, 1)
    record("delivery_resize", mode=settings.resize_mode)
    quality = round(_sample(settings.jpeg_quality, delivery_rng))
    image = _jpeg(image, quality)
    record("compression", codec="jpeg", quality=quality, final=True)
    record("native_lq")
    image = resize_centered(image, shape, "bicubic").clip(0, 1)
    record("alignment", mode="bicubic")
    return image


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
    extreme = settings.extreme
    if extreme is not None:
        scale_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4550]))
        extreme_scale = extreme.scale if scale_rng.random() < extreme.probability else extreme.secondary_scale
        if extreme_scale is not None:
            image = high.detach().permute(1, 2, 0).numpy()
            degrade = _degrade_bsrgan_plus if extreme.plus is not None else _degrade_extreme
            low = degrade(image, extreme, extreme_scale, seed, trace)
            low = (low * 255).round() / 255
            return torch.from_numpy(low).permute(2, 0, 1).contiguous(), high.clone()
    rng = np.random.default_rng(seed)
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(13)]
    image = high.detach().permute(1, 2, 0).numpy().copy()
    height, width = image.shape[:2]
    upstream = settings.upstream_rotation
    if upstream:
        first, second = _upstream_stages()
        small_grid = (height, width) == (512, 512)
        scale = 4.0 if small_grid else 8.0
        kernel_profile = _SMALL_KERNEL if small_grid else _WIDE_KERNEL
    else:
        first, second = settings.first, settings.second
        scale = _sample_scale(settings.scale, streams[12])
        kernel_profile = _WIDE_KERNEL
    if upstream:
        target = (max(1, height // int(scale)), max(1, width // int(scale)))
    else:
        target = (max(1, int(height / scale)), max(1, int(width / scale)))

    def record(op: str, value: Pixels, **params) -> None:
        if trace is not None:
            trace.append({"op": op, "shape": list(value.shape[:2]), **params})

    if upstream:
        record("source", image, scale=scale, target=list(target), recipe=_UPSTREAM_RECIPE)
    elif isinstance(settings.scale, tuple):
        record("source", image, scale=scale, scale_range=list(settings.scale), target=list(target))
    else:
        record("source", image, scale=scale, target=list(target))

    def blur(value: Pixels, stage: StageConfig, index: int, profile) -> Pixels:
        blur_rng = streams[(index - 1) * 4]
        is_sinc = blur_rng.random() < stage.sinc_probability
        kernel = (
            _sinc_kernel(blur_rng, stage=True, profile=profile)
            if is_sinc
            else mixed_blur_kernel(blur_rng, stage.blur, profile=profile)
        )
        record("stage_sinc" if is_sinc else "blur", value, stage=index, kernel=kernel.tolist())
        return _filter(value, kernel)

    image = blur(image, first, 1, kernel_profile)
    resize_rng, noise_rng, codec_rng = streams[1:4]
    factor = _resize_scale(first, resize_rng)
    requested = scaled_shape((height, width), factor)
    image = _resize(image, *requested, resize_rng, upstream=upstream)
    record("resize", image, stage=1, factor=factor, requested_shape=list(requested))
    image = add_noise(image, first.noise, noise_rng, trace=trace)
    image = _compress(image, first.compression, codec_rng, trace)

    resize_rng, noise_rng, codec_rng = streams[5:8]
    if streams[4].random() < second.blur_probability:
        image = blur(image, second, 2, kernel_profile)
    factor = _resize_scale(second, resize_rng)
    requested = scaled_shape(target, factor)
    image = _resize(image, *requested, resize_rng, second_stage=True, upstream=upstream)
    record("resize", image, stage=2, factor=factor, requested_shape=list(requested))
    image = add_noise(image, second.noise, noise_rng, trace=trace)

    sinc_rng = streams[8]
    final_sinc_probability = 0.8 if upstream else settings.final_sinc_probability
    final_kernel = (
        _sinc_kernel(sinc_rng, profile=kernel_profile) if sinc_rng.random() < final_sinc_probability else None
    )

    def delivery(value: Pixels) -> Pixels:
        value = _resize(value, *target, streams[9], upstream=upstream)
        record("delivery_resize", value)
        if final_kernel is not None:
            record("final_sinc", value, kernel=final_kernel.tolist())
            value = _filter(value, final_kernel)
        return value

    if streams[10].random() < 0.5:
        image = delivery(image)
        image = _compress(image, second.compression, codec_rng, trace)
    else:
        image = _compress(image, second.compression, codec_rng, trace)
        image = delivery(image)
    record("native_lq", image)

    value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    if not upstream and rng.random() < settings.gray_probability:
        value = image_ops.rgb_to_grayscale(value, num_output_channels=3)
    if not upstream and rng.random() < settings.color_jitter_probability:
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
    alignment_mode = str(streams[11].choice(_RESIZE_MODES))
    image = _resize(image, height, width, streams[11], mode=alignment_mode, upstream=upstream)
    record("alignment", image, mode=alignment_mode)
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
