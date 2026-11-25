"""Single-frame spatial adaptation of FastVR's public RealVSR recipe."""

import math
from fractions import Fraction

import av
import cv2
import numpy as np

from unituner.data.degradation.config import BlurConfig
from unituner.data.degradation.pipeline import _SMALL_KERNEL, Pixels, _sinc_kernel, mixed_blur_kernel

_RESIZE = (cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC)
_RESIZE_PROBABILITIES = (0.3333, 0.3333, 0.3334)
_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow")


def _mpeg(image: Pixels, codec: str, crf: float, preset: str, sigma: float) -> Pixels:
    height, width = image.shape[:2]
    pixels = image.clip(0, 255).astype(np.uint8)
    encoded_height = max(16, height + height % 2)
    encoded_width = max(16, width + width % 2)
    pixels = np.pad(pixels, ((0, encoded_height - height), (0, encoded_width - width), (0, 0)), mode="reflect")
    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24").reformat(format="yuv420p")
    frame.pts = 0
    frame.time_base = Fraction(1, 25)
    graph = av.filter.Graph()
    graph.link_nodes(graph.add_buffer(template=frame), graph.add("gblur", f"sigma={sigma}"), graph.add("buffersink"))
    graph.configure()
    graph.push(frame)
    frame = graph.pull()

    encoder = av.CodecContext.create(codec, "w")
    encoder.width = frame.width
    encoder.height = frame.height
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, 25)
    encoder.framerate = 25
    encoder.thread_count = 1
    encoder.options = {"crf": f"{crf:.1f}", "preset": preset}
    if codec == "libx265":
        encoder.options["x265-params"] = "pools=none:frame-threads=1:log-level=none"
    packets = encoder.encode(frame) + encoder.encode(None)
    decoder = av.CodecContext.create(encoder.codec.canonical_name, "r")
    decoder.thread_count = 1
    frames = [decoded for packet in packets for decoded in decoder.decode(packet)] + decoder.decode(None)
    if len(frames) != 1:
        raise RuntimeError(f"FastVR {codec} round trip produced {len(frames)} frames, expected one")
    return frames[0].to_ndarray(format="rgb24")[:height, :width].astype(np.float32)


def degrade_fastvr(image: Pixels, rng: np.random.Generator, trace: list[dict] | None) -> Pixels:
    """Apply FastVR's spatial stages to one aligned RGB frame in [0, 1]."""
    shape = image.shape[:2]
    value = (image * 255).clip(0, 255).astype(np.uint8)

    def record(op: str, result: Pixels, stage: int, **params) -> Pixels:
        if trace is not None:
            trace.append({"op": op, "shape": list(result.shape[:2]), "stage": stage, **params})
        return result

    def mpeg(pixels: Pixels, stage: int) -> Pixels:
        probability, bounds, shift, blur_bounds = (
            (0.5, (12, 20), 2, (0.01, 0.1)),
            (0.5, (18, 26), 3, (0.01, 0.1)),
            (0.5, (18, 26), 3, (0.01, 0.1)),
            (0.9, (20, 35), 5, (0.05, 0.4)),
        )[stage]
        if rng.random() >= probability:
            return pixels
        codec = "libx264" if rng.random() < 0.4 else "libx265"
        preset = str(rng.choice(_PRESETS))
        if codec == "libx265" and preset == "slow":
            preset = "medium"
        crf = float(rng.uniform(*bounds)) - (shift if codec == "libx264" else 0)
        result = _mpeg(pixels, codec, crf, preset, float(rng.uniform(*blur_bounds)))
        return record("compression", result, stage, codec=codec, crf=crf)

    def blur(pixels: Pixels, stage: int) -> Pixels:
        if stage != 1 and rng.random() >= 0.8:
            return pixels
        is_sinc = stage == 3 or rng.random() < 0.1
        blur = BlurConfig(sigma=(0.2, 3.0 if stage == 1 else 1.5))
        kernel = (
            _sinc_kernel(rng, stage=stage != 3, profile=_SMALL_KERNEL)
            if is_sinc
            else mixed_blur_kernel(rng, blur, profile=_SMALL_KERNEL)
        )
        result = cv2.filter2D(pixels, -1, kernel)
        return record("final_sinc" if stage == 3 else "blur", result, stage)

    def resize(pixels: Pixels, stage: int) -> Pixels:
        if stage == 3:
            size = shape
        else:
            up, down = (0.2, 0.7) if stage == 1 else (0.3, 0.4)
            draw = rng.random()
            if draw < up:
                factor = rng.uniform(1, 1.5 if stage == 1 else 1.2)
            elif draw < up + down:
                factor = rng.uniform(0.5 if stage == 1 else 0.3, 1)
            else:
                factor = 1
            size = tuple(max(2, int(dimension * factor) // 2 * 2) for dimension in pixels.shape[:2])
        mode = int(rng.choice(_RESIZE, p=_RESIZE_PROBABILITIES))
        result = cv2.resize(pixels, size[::-1], interpolation=mode)
        return record("delivery_resize" if stage == 3 else "resize", result, stage)

    def noise(pixels: Pixels, stage: int) -> Pixels:
        if rng.random() >= (0.8 if stage == 1 else 0.5):
            return pixels
        gray = rng.random() < 0.4
        if rng.random() < 0.5:
            sigma = rng.uniform(*((0.1, 3) if stage == 1 else (0.05, 5)))
            dimensions = (*pixels.shape[:2], 1 if gray else 3)
            perturbation = rng.standard_normal(dimensions, dtype=np.float32) * sigma
            kind = "gaussian"
        else:
            signal = pixels.astype(np.float32)
            if gray:
                signal = cv2.cvtColor(signal, cv2.COLOR_RGB2GRAY)[..., None]
            signal = signal.round().clip(0, 255)
            levels = 2 ** math.ceil(math.log2(np.unique(signal).size))
            scale = rng.uniform(*((0.05, 0.5) if stage == 1 else (0.02, 1)))
            perturbation = (rng.poisson(signal * levels).astype(np.float32) / levels - signal) * scale
            kind = "poisson"
        return record("noise", pixels + perturbation, stage, kind=kind, gray=gray)

    def jpeg(pixels: Pixels, stage: int) -> Pixels:
        if rng.random() >= 0.8:
            return pixels
        quality = round(rng.uniform(50, 95))
        bgr = pixels[..., ::-1].clip(0, 255).round().astype(np.uint8)
        success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("FastVR JPEG encoding failed")
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
        return record("compression", result, stage, codec="jpeg", quality=quality)

    value = mpeg(value, 0)
    for stage in (1, 2, 3):
        operations = [resize, blur, mpeg] if stage == 3 else [blur, resize, noise, jpeg, mpeg]
        if rng.random() < (0.5 if stage == 3 else 0.2):
            rng.shuffle(operations)
        for operation in operations:
            value = operation(value, stage)
    return value.clip(0, 255).astype(np.float32) / 255
