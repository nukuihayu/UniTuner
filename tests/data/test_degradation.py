"""Check official stage semantics, spatial adaptation and optional extensions."""

import json
import math
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.nn import functional as F

pytest.importorskip("cv2")
from unitunner.data.degradation.config import (
    BlurConfig,
    CompressionConfig,
    DegradationConfig,
    GrainConfig,
    NoiseConfig,
    SecondCompressionConfig,
    SecondNoiseConfig,
    SecondStageConfig,
    StageConfig,
)
from unitunner.data.degradation.pipeline import (
    _resize,
    _resize_scale,
    _resolution_scale,
    add_grain,
    add_noise,
    compress_image,
    degrade_image,
    degrade_pair,
    mixed_blur_kernel,
    sinc_kernel,
)


@pytest.mark.parametrize("scale", [1, 2, 4, 6, 8, 16])
@pytest.mark.parametrize("shape", [(32, 48), (128, 192)])
def test_finite_deterministic_output_immutable_hq_and_local_rng(scale, shape):
    high = torch.from_numpy(np.random.default_rng(17).random((3, *shape), dtype=np.float32))
    saved = high.clone()
    states = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    cfg = DegradationConfig(scales=(float(scale),))
    trace = []
    result = degrade_image(high, cfg, 13, trace=trace)
    torch.testing.assert_close(result, degrade_image(high, cfg, 13), rtol=0, atol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(result, degrade_image(high, cfg, 13), rtol=0, atol=0)
    assert result.shape == high.shape
    assert torch.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    torch.testing.assert_close(saved, high, rtol=0, atol=0)
    assert torch.equal(states[0], torch.get_rng_state())
    np.testing.assert_array_equal(states[1][1], np.random.get_state()[1])
    assert states[1][2:] == np.random.get_state()[2:]
    assert states[2] == random.getstate()
    assert trace[-1]["op"] == "quantize"


@pytest.mark.parametrize("scale", [2, 4, 6, 8, 16])
def test_native_and_aligned_outputs_are_quantized(scale):
    high = torch.from_numpy(np.random.default_rng(4).random((3, 129, 193), dtype=np.float32))
    cfg = DegradationConfig(scales=(float(scale),), output_mode="native")
    low = degrade_image(high, cfg, 29)
    assert low.shape == (3, 129 // scale, 193 // scale)
    torch.testing.assert_close(low * 255, (low * 255).round(), atol=2e-5, rtol=0)
    aligned = degrade_image(high, cfg.model_copy(update={"output_mode": "aligned"}), 29)
    torch.testing.assert_close(aligned * 255, (aligned * 255).round(), atol=2e-5, rtol=0)


def test_fixed_two_stage_order_and_native_delivery_both_branches():
    cfg = DegradationConfig(
        scales=(4.0,),
        use_usm=False,
        final_grain=GrainConfig(probability=0),
        final_sinc_probability=1,
        first=StageConfig(blur_probability=1, sinc_probability=0),
        second=SecondStageConfig(blur_probability=1, sinc_probability=0),
    )
    tails = set()
    for seed in range(64):
        trace = []
        degrade_image(torch.full((3, 128, 192), 0.5), cfg, seed, trace=trace)
        ops = [e["op"] for e in trace]
        assert ops[:8] == ["source", "blur", "resize", "noise", "compression", "blur", "resize", "noise"]
        tails.add(tuple(ops[8:]))
        final = next(e for e in trace if e["op"] == "final_sinc")
        assert final["shape"] == [32, 48]
        assert len([e for e in trace if e["op"] == "noise"]) == 2
        assert all(e["shape"] != [128, 192] for e in trace[8:] if e["op"] == "compression")
    assert tails == {
        ("delivery_resize", "final_sinc", "compression", "native_lq", "alignment", "quantize"),
        ("compression", "delivery_resize", "final_sinc", "native_lq", "alignment", "quantize"),
    }


@pytest.mark.parametrize("output_mode", ["native", "aligned"])
def test_native_resize_matches_independent_torch_reference_without_damage(output_mode):
    cfg = DegradationConfig(
        scales=(4.0,),
        use_usm=False,
        final_grain=GrainConfig(probability=0),
        output_mode=output_mode,
        final_sinc_probability=0,
        resize_modes=("bilinear",),
        first=StageConfig(blur_probability=0, noise_probability=0, compression_probability=0),
        second=SecondStageConfig(blur_probability=0, noise_probability=0, compression_probability=0),
    )
    high = torch.from_numpy(np.random.default_rng(3).random((3, 129, 193), dtype=np.float32))
    trace = []
    actual = degrade_image(high, cfg, 8, trace=trace)
    reference = high[None]
    for event in trace:
        if event["op"] in ("resize", "delivery_resize"):
            if event.get("stage") == 1:
                reference = F.interpolate(reference, scale_factor=event["factor"], mode="bilinear", align_corners=False)
            else:
                reference = F.interpolate(reference, size=event["shape"], mode="bilinear", align_corners=False)
    if output_mode == "aligned":
        reference = F.interpolate(reference, size=high.shape[-2:], mode="bicubic", align_corners=False)
    reference = (reference[0] * 255).round().clamp(0, 255) / 255
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "resolution,expected", [(128, 0.5), (512, 1), (1024, math.sqrt(2)), (1280, math.sqrt(2.5)), (4096, 2)]
)
def test_resolution_mapping_uses_hq_reference(resolution, expected):
    assert _resolution_scale(resolution) == pytest.approx(expected)
    assert _resolution_scale(resolution, exponent=0) == 1


def test_blur_kernel_matches_analytic_gaussian_and_official_support():
    cfg = BlurConfig(sigma=(2.0, 2.0), probabilities=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    sizes = set()
    for seed in range(100):
        kernel = mixed_blur_kernel(np.random.default_rng(seed), cfg)
        sizes.add(len(kernel))
        a = np.arange(len(kernel)) - len(kernel) // 2
        reference = np.exp(-(a[:, None] ** 2 + a[None, :] ** 2) / 8)
        np.testing.assert_allclose(kernel, reference / reference.sum(), rtol=1e-6, atol=1e-8)
    assert sizes == set(range(7, 22, 2))


def test_sinc_is_untapered_official_circular_filter():
    kernel = sinc_kernel(21, math.pi / 3)
    a = torch.arange(-10, 11, dtype=torch.float64)
    r = (a[:, None] ** 2 + a[None, :] ** 2).sqrt()
    w = math.pi / 3
    expected = w * torch.special.bessel_j1(w * r) / (2 * math.pi * r)
    expected[10, 10] = w**2 / (4 * math.pi)
    expected /= expected.sum()
    np.testing.assert_allclose(kernel, expected.numpy(), atol=1e-8, rtol=1e-6)
    assert kernel.min() < 0
    assert kernel.sum() == pytest.approx(1, abs=1e-6)


def test_stage_sinc_replaces_blur_and_uses_same_hq_multiplier():
    cfg = DegradationConfig(
        scales=(8.0,),
        resolution_exponent=0,
        use_usm=False,
        final_grain=GrainConfig(probability=0),
        first=StageConfig(sinc_probability=1),
        second=SecondStageConfig(blur_probability=1, sinc_probability=1),
        final_sinc_probability=1,
    )
    trace = []
    degrade_image(torch.full((3, 400, 400), 0.5), cfg, 8, trace=trace)
    assert len([e for e in trace if e["op"] == "stage_sinc"]) == 2
    assert not any(e["op"] == "blur" for e in trace)
    for event in trace:
        if "kernel" in event:
            assert len(event["kernel"]) in range(7, 22, 2)


def test_resize_ranges_not_clipped_to_selected_sr_factor():
    rng = np.random.default_rng(0)
    stage = StageConfig(resize_probabilities=(0.0, 1.0, 0.0))
    draws = [_resize_scale(stage, rng) for _ in range(1000)]
    assert min(draws) < 0.16 and max(draws) > 0.99


@pytest.mark.parametrize("codec,quality", [("diffjpeg", 95), ("jpeg", 95), ("webp", 95), ("jpeg2000", 900)])
def test_codecs_preserve_rgb_and_tiny_grids(codec, quality):
    source = np.full((16, 24, 3), [0.8, 0.3, 0.1], dtype=np.float32)
    result = compress_image(source, codec, quality)
    assert result.shape == source.shape and result.dtype == np.float32
    np.testing.assert_allclose(result.mean((0, 1)), source.mean((0, 1)), atol=0.025)


def test_diffjpeg_matches_pinned_hypir_reference_including_odd_borders():
    root = Path(__file__).parent / "fixtures"
    metadata = json.loads((root / "hypir_diffjpeg.json").read_text(encoding="utf8"))
    with np.load(root / "hypir_diffjpeg.npz") as reference:
        for case in metadata["cases"]:
            source = reference[case["input"]]
            saved = source.copy()
            actual = compress_image(source, "diffjpeg", case["quality"])
            np.testing.assert_allclose(actual, reference[case["output"]], atol=2e-6, rtol=0)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                np.testing.assert_array_equal(actual, compress_image(source, "diffjpeg", case["quality"]))
            np.testing.assert_array_equal(source, saved)


@pytest.mark.parametrize("quality", [0.0, 100.0, float("nan")])
def test_singular_diffjpeg_quality_is_rejected(quality):
    with pytest.raises(ValueError, match="strictly between"):
        compress_image(np.full((16, 16, 3), 0.5, dtype=np.float32), "diffjpeg", quality)
    with pytest.raises(ValidationError):
        CompressionConfig(jpeg_quality=(quality, quality))


def test_hypir_jpeg_and_real_codec_overrides_use_distinct_backends():
    high = torch.full((3, 64, 96), 0.5)
    backends: tuple[tuple[Literal["diffjpeg", "opencv"], str], ...] = (("diffjpeg", "diffjpeg"), ("opencv", "jpeg"))
    for backend, expected in backends:
        compression = CompressionConfig(probabilities=(1.0, 0.0, 0.0), jpeg_backend=backend)
        cfg = DegradationConfig(
            scales=(4.0,),
            first=StageConfig(compression=compression),
            second=SecondStageConfig(compression=SecondCompressionConfig(**compression.model_dump())),
        )
        trace = []
        degrade_image(high, cfg, 13, trace=trace)
        assert [e["codec"] for e in trace if e["op"] == "compression"] == [expected, expected]


def test_final_grain_is_gray_correlated_and_bounded():
    source = np.full((128, 128, 3), 0.5, dtype=np.float32)
    cfg = GrainConfig(probability=1, sigma=(4.0, 4.0), correlation=(0.8, 0.8))
    trace = []
    result = add_grain(source, cfg, np.random.default_rng(0), trace)
    residual = result - source
    assert trace[0]["op"] == "grain" and len(trace) == 1
    np.testing.assert_array_equal(residual[..., 0], residual[..., 1])
    assert residual.std() * 255 == pytest.approx(4, rel=0.01)
    assert abs(residual.mean()) < 1e-6
    assert np.corrcoef(residual[1:, :, 0].ravel(), residual[:-1, :, 0].ravel())[0, 1] > 0.05


def test_noise_mixture_frequency():
    counts = {"gaussian": 0, "poisson": 0}
    rng = np.random.default_rng(1)
    grain_count = 0
    for _ in range(2000):
        source = np.full((8, 8, 3), 0.5, dtype=np.float32)
        trace = []
        add_noise(source, NoiseConfig(), rng, trace=trace)
        counts[trace[0]["kind"]] += 1
        trace = []
        add_grain(source, GrainConfig(), rng, trace)
        grain_count += len(trace)
    assert counts["gaussian"] / 2000 == pytest.approx(0.5, abs=0.035)
    assert grain_count / 2000 == pytest.approx(0.1, abs=0.025)


@pytest.mark.parametrize("scale", [1.0, 4.0, 6.0, 8.0])
def test_grain_only_runs_once_after_alignment_without_changing_other_draws(scale):
    cfg = DegradationConfig(scales=(scale,), final_grain=GrainConfig(probability=0))
    grain_cfg = cfg.model_copy(update={"final_grain": GrainConfig(probability=1)})
    source = torch.from_numpy(np.random.default_rng(4).random((3, 96, 144), dtype=np.float32))
    original, grain = [], []
    degrade_image(source, cfg, 13, trace=original)
    degrade_image(source, grain_cfg, 13, trace=grain)
    assert original == [e for e in grain if e["op"] != "grain"]
    assert [e["op"] for e in grain][-3:] == ["alignment", "grain", "quantize"]
    assert next(e for e in grain if e["op"] == "grain")["shape"] == [96, 144]
    assert len([e for e in grain if e["op"] == "noise"]) == 2
    native = []
    degrade_image(source, grain_cfg.model_copy(update={"output_mode": "native"}), 13, trace=native)
    assert not any(e["op"] == "grain" for e in native)


@pytest.mark.parametrize("use_usm,sharpen_target", [(True, True), (True, False), (False, True)])
def test_hypir_usm_supervision_and_input_immutability(use_usm, sharpen_target):
    source = torch.from_numpy(np.random.default_rng(1).random((3, 64, 96), dtype=np.float32))
    saved = source.clone()
    cfg = DegradationConfig(use_usm=use_usm, sharpen_target=sharpen_target, resolution_exponent=0)
    low, target = degrade_pair(source, cfg, 17)
    # Independent HYPIR reference: fixed 51-wide Gaussian,
    # reflect-padded Torch convolution, thresholded soft mask, weight 0.5.
    import cv2

    axis = cv2.getGaussianKernel(51, 0)
    kernel = torch.from_numpy((axis @ axis.T).astype(np.float32))[None, None].repeat(3, 1, 1, 1)

    def filtered(value):
        return F.conv2d(F.pad(value[None], (25, 25, 25, 25), mode="reflect"), kernel, groups=3)[0]

    residual = source - filtered(source)
    mask = filtered((residual.abs() * 255 > 10).float())
    expected = mask * (source + 0.5 * residual).clamp(0, 1) + (1 - mask) * source
    if use_usm and sharpen_target:
        torch.testing.assert_close(target, expected, rtol=0, atol=2e-6)
        assert not torch.equal(target, source)
    else:
        torch.testing.assert_close(target, source, rtol=0, atol=0)
    torch.testing.assert_close(source, saved, rtol=0, atol=0)
    torch.testing.assert_close(low, degrade_image(source, cfg, 17), rtol=0, atol=0)


def test_reference_kernel_does_not_depend_on_sr_factor():
    traces = []
    for scale in (2.0, 4.0, 8.0):
        trace = []
        cfg = DegradationConfig(scales=(scale,), resolution_exponent=0, first=StageConfig(sinc_probability=0))
        degrade_image(torch.full((3, 128, 192), 0.5), cfg, 17, trace=trace)
        traces.append(next(e["kernel"] for e in trace if e["op"] == "blur"))
    assert traces[0] == traces[1] == traces[2]


@pytest.mark.parametrize("scale", [2.0, 4.0, 6.0, 8.0, 16.0])
def test_secondary_resize_uses_unmodified_hypir_draw_at_every_sr_factor(scale):
    source = torch.from_numpy(np.random.default_rng(1).random((3, 128, 192), dtype=np.float32))
    cfg = DegradationConfig(scales=(scale,))
    trace = []
    degrade_image(source, cfg, 13, trace=trace)
    resize = next(e for e in trace if e["op"] == "resize" and e.get("stage") == 2)
    raw_rng = np.random.default_rng(np.random.SeedSequence(13).spawn(12)[5])
    raw_resize = _resize_scale(cfg.second, raw_rng)
    assert resize["factor"] == pytest.approx(raw_resize, abs=1e-15)
    assert resize["shape"] == [max(1, int((128 // scale) * raw_resize)), max(1, int((192 // scale) * raw_resize))]


@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("gray", [False, True])
def test_noise_matches_hypir_add_then_clip_semantics(gaussian, gray):
    # Include bicubic/sinc overshoot: pre-clipping would change the residual.
    source = np.random.default_rng(2).uniform(-0.15, 1.15, (48, 64, 3)).astype(np.float32)
    # Explicit official ranges keep this a semantics check independent of defaults.
    cfg = NoiseConfig(
        gaussian_probability=float(gaussian),
        gray_probability=float(gray),
        gaussian_sigma=(1.0, 30.0),
        poisson_scale=(0.05, 3.0),
    )
    actual = add_noise(source, cfg, np.random.default_rng(17))
    rng = np.random.default_rng(17)
    rng.random(2)  # Gaussian and gray choices
    if gaussian:
        sigma = rng.uniform(1, 30)
        residual = rng.normal(0, sigma / 255, (48, 64, 1 if gray else 3))
    else:
        signal = (source * np.array([0.2989, 0.5870, 0.1140])).sum(2, keepdims=True) if gray else source
        signal = np.clip(np.round(signal * 255), 0, 255) / 255
        levels = 2 ** math.ceil(math.log2(np.unique(signal).size))
        amplitude = rng.uniform(0.05, 3)
        residual = (rng.poisson(signal * levels) / levels - signal) * amplitude
    expected = np.clip(source + residual, 0, 1).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, np.clip(source.clip(0, 1) + residual, 0, 1).astype(np.float32))


def test_gaussian_noise_uses_full_official_sigma():
    cfg = NoiseConfig(gaussian_probability=1, gaussian_sigma=(10.0, 10.0), gray_probability=1)
    result = add_noise(np.full((256, 256, 3), 0.5, dtype=np.float32), cfg, np.random.default_rng(1))
    assert result.std() * 255 == pytest.approx(10, rel=0.02)
    np.testing.assert_array_equal(result[..., 0], result[..., 1])


@pytest.mark.parametrize("scale", [1, 2, 4, 6, 8, 16])
def test_pipeline_keeps_hypir_noise_amplitude_at_every_scale(scale):
    stage = StageConfig(
        blur_probability=0, noise_probability=0, compression_probability=0, resize_probabilities=(0.0, 0.0, 1.0)
    )
    second = SecondStageConfig(
        blur_probability=0,
        compression_probability=0,
        resize_probabilities=(0.0, 0.0, 1.0),
        noise=SecondNoiseConfig(gaussian_probability=1.0, gaussian_sigma=(25.0, 25.0)),
    )
    cfg = DegradationConfig(
        scales=(float(scale),),
        first=stage,
        second=second,
        final_sinc_probability=0,
        final_grain=GrainConfig(probability=0),
        output_mode="native",
    )
    source = torch.full((3, 512, 768), 0.5)
    trace = []
    result = degrade_image(source, cfg, 18, trace=trace)
    noise = next(e for e in trace if e["op"] == "noise")
    assert noise["shape"] == [512 // scale, 768 // scale]
    assert noise["parameter"] == 25
    assert result.std() * 255 == pytest.approx(25, rel=0.06)


@pytest.mark.parametrize("removed", ["high_scale_protection", "noise_grid_adaptation"])
def test_removed_adaptation_switches_are_rejected(removed):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DegradationConfig.model_validate({removed: True})


def test_optional_second_order_skip_keeps_final_lq_delivery():
    trace = []
    cfg = DegradationConfig(scales=(4.0,), second_order_probability=0)
    degrade_image(torch.full((3, 64, 96), 0.5), cfg, 7, trace=trace)
    assert len([e for e in trace if e["op"] == "noise"]) == 1
    assert next(e for e in trace if e["op"] == "native_lq")["shape"] == [16, 24]


@pytest.mark.parametrize("legacy", ["smear_strength", "grain_strength", "noise_strength", "late_noise_probability"])
def test_removed_heuristics_fail_instead_of_silently_applying(legacy):
    with pytest.raises(ValidationError):
        DegradationConfig.model_validate({legacy: 0.5})


def test_optional_antialias_suppresses_checkerboard_alias():
    source = np.broadcast_to((np.indices((128, 128)).sum(0) % 2)[..., None], (128, 128, 3)).astype(np.float32)
    cfg = DegradationConfig(resize_modes=("bilinear",), resize_antialias=True)
    actual = _resize(source, 31, 31, np.random.default_rng(0), cfg)
    assert actual[2:-2, 2:-2].std() < 0.005


@pytest.mark.parametrize("shape", [(45, 70), (9, 13)])
@pytest.mark.parametrize("synthetic", [False, True])
def test_calibration_tiles_cover_edges_and_degrade_after_crop(tmp_path, shape, synthetic):
    """Check every tile against an independent pixel slice and degradation call."""
    from PIL import Image

    from unitunner.data.media.restoration import iter_aligned_images

    array = np.random.default_rng(7).integers(0, 256, (*shape, 3), dtype=np.uint8)
    path = tmp_path / "high.png"
    Image.fromarray(array).save(path)
    settings = DegradationConfig(enabled=True) if synthetic else None
    before = torch.get_rng_state().clone()
    tiles = list(
        iter_aligned_images(
            None if synthetic else path,
            path,
            resolution=(32, 32),
            seed=19,
            degradation=settings,
            sampling="tiles",
        )
    )
    starts = [(0, 0)] if shape == (9, 13) else [(y, x) for y in (0, 13) for x in (0, 32, 38)]
    padded = np.pad(array, ((0, max(0, 32 - shape[0])), (0, max(0, 32 - shape[1])), (0, 0)), mode="reflect")
    assert len(tiles) == len(starts)
    for index, ((low, high), (top, left)) in enumerate(zip(tiles, starts, strict=True)):
        expected = torch.from_numpy(padded[top : top + 32, left : left + 32].copy()).permute(2, 0, 1).float() / 255
        expected_low, expected_high = (
            degrade_pair(expected, settings, 19 + index * 1_000_003) if settings else (expected, expected)
        )
        torch.testing.assert_close(high, expected_high * 2 - 1, atol=0, rtol=0)
        torch.testing.assert_close(low, expected_low * 2 - 1, atol=0, rtol=0)
    assert torch.equal(before, torch.get_rng_state())
