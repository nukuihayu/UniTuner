"""Check OMGSR-F-1024 degradation semantics and phase-stable resizing."""

import math
import random

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.nn import functional as F

pytest.importorskip("cv2")
from unituner.data.degradation.config import (
    BlurConfig,
    CompressionConfig,
    DegradationConfig,
    NoiseConfig,
    SecondCompressionConfig,
    SecondStageConfig,
    StageConfig,
)
from unituner.data.degradation.pipeline import (
    _kernel_profile,
    _resize,
    _resize_scale,
    add_noise,
    degrade_image,
    degrade_pair,
    mixed_blur_kernel,
    sinc_kernel,
)


@pytest.mark.parametrize("scale", [1.0, 4.0, 8.0, 16.0])
@pytest.mark.parametrize("shape", [(32, 48), (128, 192)])
def test_finite_deterministic_output_immutable_hq_and_local_rng(scale, shape):
    high = torch.from_numpy(np.random.default_rng(17).random((3, *shape), dtype=np.float32))
    saved = high.clone()
    states = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    cfg = DegradationConfig(scale=scale)
    trace = []
    result = degrade_image(high, cfg, 13, trace=trace)
    torch.testing.assert_close(result, degrade_image(high, cfg, 13), rtol=0, atol=0)
    assert result.shape == high.shape
    assert torch.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    torch.testing.assert_close(saved, high, rtol=0, atol=0)
    assert torch.equal(states[0], torch.get_rng_state())
    np.testing.assert_array_equal(states[1][1], np.random.get_state()[1])
    assert states[1][2:] == np.random.get_state()[2:]
    assert states[2] == random.getstate()
    assert trace[-1]["op"] == "quantize"


def test_defaults_match_omgsr_flux_1024():
    cfg = DegradationConfig()
    assert cfg.scale == 8
    assert cfg.first.resize_probabilities == (0.2, 0.7, 0.1)
    assert cfg.first.resize_range == (0.3, 1.5)
    assert cfg.first.noise.gaussian_sigma == (1.0, 15.0)
    assert cfg.first.noise.poisson_scale == (0.05, 2.0)
    assert cfg.first.compression.jpeg_quality == (60.0, 95.0)
    assert cfg.second.blur_probability == 0.5
    assert cfg.second.resize_probabilities == (0.3, 0.4, 0.3)
    assert cfg.second.resize_range == (0.6, 1.2)
    assert cfg.second.noise.gaussian_sigma == (1.0, 12.0)
    assert cfg.second.noise.poisson_scale == (0.05, 1.0)
    assert cfg.second.compression.jpeg_quality == (60.0, 100.0)
    assert cfg.final_sinc_probability == 0.8
    assert cfg.gray_probability == cfg.color_jitter_probability == 0
    assert _kernel_profile() == (tuple(range(11, 42, 2)), 41, 25)


@pytest.mark.parametrize(
    "removed",
    [
        "scales",
        "scale_range",
        "second_order_probability",
        "use_usm",
        "sharpen_target",
        "final_grain",
        "resize_antialias",
        "output_mode",
        "resize_modes",
    ],
)
def test_non_omgsr_degradation_parameters_are_rejected(removed):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DegradationConfig.model_validate({removed: False})


@pytest.mark.parametrize("scale", [0.5, 17.0, float("nan")])
def test_scale_parameter_is_strictly_bounded(scale):
    with pytest.raises(ValidationError):
        DegradationConfig(scale=scale)


@pytest.mark.parametrize("mode", ["area", "bilinear", "bicubic"])
def test_resize_uses_phase_stable_integer_grid(mode):
    source = np.random.default_rng(3).random((101, 99, 3), dtype=np.float32)
    target = (61, 60)
    actual = _resize(source, *target, np.random.default_rng(0), mode=mode)
    reference = F.interpolate(
        torch.from_numpy(source).permute(2, 0, 1)[None],
        size=target,
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
        antialias=False,
    )[0].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(actual, reference)


def test_phase_fix_differs_from_float_scale_factor_on_odd_grids():
    source = np.random.default_rng(4).random((101, 99, 3), dtype=np.float32)
    value = torch.from_numpy(source).permute(2, 0, 1)[None]
    fixed = _resize(source, 61, 60, np.random.default_rng(0), mode="bicubic")
    old = F.interpolate(value, scale_factor=0.613, mode="bicubic", align_corners=False)[0].permute(1, 2, 0).numpy()
    assert fixed.shape == old.shape
    assert np.max(np.abs(fixed - old)) > 1e-3


def test_fixed_two_stage_order_and_both_final_orders():
    cfg = DegradationConfig(
        scale=8,
        final_sinc_probability=1,
        first=StageConfig(sinc_probability=0),
        second=SecondStageConfig(blur_probability=1, sinc_probability=0),
    )
    tails = set()
    for seed in range(64):
        trace = []
        degrade_image(torch.full((3, 128, 192), 0.5), cfg, seed, trace=trace)
        ops = [event["op"] for event in trace]
        assert ops[:8] == ["source", "blur", "resize", "noise", "compression", "blur", "resize", "noise"]
        tails.add(tuple(ops[8:]))
        assert next(event for event in trace if event["op"] == "native_lq")["shape"] == [16, 24]
        assert len([event for event in trace if event["op"] == "noise"]) == 2
    assert tails == {
        ("delivery_resize", "final_sinc", "compression", "native_lq", "alignment", "quantize"),
        ("compression", "delivery_resize", "final_sinc", "native_lq", "alignment", "quantize"),
    }


def test_blur_kernel_matches_analytic_gaussian_and_flux_support():
    cfg = BlurConfig(sigma=(2.0, 2.0), probabilities=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    sizes = set()
    for seed in range(100):
        kernel = mixed_blur_kernel(np.random.default_rng(seed), cfg)
        support = np.flatnonzero(kernel.max(0) > 0)
        size = int(support[-1] - support[0] + 1)
        sizes.add(size)
        actual = kernel[support[0] : support[-1] + 1, support[0] : support[-1] + 1]
        axis = np.arange(size) - size // 2
        expected = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / 8)
        np.testing.assert_allclose(actual, expected / expected.sum(), rtol=1e-6, atol=1e-8)
        assert kernel.shape == (41, 41)
    assert sizes == set(range(11, 42, 2))


def test_sinc_is_untapered_official_circular_filter():
    kernel = sinc_kernel(21, math.pi / 3)
    axis = torch.arange(-10, 11, dtype=torch.float64)
    radius = (axis[:, None] ** 2 + axis[None, :] ** 2).sqrt()
    cutoff = math.pi / 3
    expected = cutoff * torch.special.bessel_j1(cutoff * radius) / (2 * math.pi * radius)
    expected[10, 10] = cutoff**2 / (4 * math.pi)
    expected /= expected.sum()
    np.testing.assert_allclose(kernel, expected.numpy(), atol=1e-8, rtol=1e-6)
    assert kernel.min() < 0


def test_resize_ranges_match_omgsr_and_second_stage_can_undershoot():
    cfg = DegradationConfig()
    assert cfg.first.resize_range == (0.3, 1.5)
    assert cfg.second.resize_range == (0.6, 1.2)
    draws = [_resize_scale(cfg.second, np.random.default_rng(seed)) for seed in range(1000)]
    assert min(draws) >= 0.6 and min(draws) < 0.61
    undershot = False
    for seed in range(64):
        trace = []
        degrade_image(torch.full((3, 128, 192), 0.5), cfg, seed, trace=trace)
        stage = next(event for event in trace if event["op"] == "resize" and event["stage"] == 2)
        assert stage["shape"] == stage["requested_shape"]
        undershot |= stage["shape"][0] < 16
    assert undershot


@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("gray", [False, True])
def test_noise_matches_omgsr_add_then_clip_semantics(gaussian, gray):
    source = np.random.default_rng(2).uniform(-0.15, 1.15, (48, 64, 3)).astype(np.float32)
    cfg = NoiseConfig(
        gaussian_probability=float(gaussian),
        gray_probability=float(gray),
        gaussian_sigma=(1.0, 30.0),
        poisson_scale=(0.05, 3.0),
    )
    actual = add_noise(source, cfg, np.random.default_rng(17))
    rng = np.random.default_rng(17)
    rng.random(2)
    if gaussian:
        sigma = rng.uniform(1, 30)
        residual = rng.normal(0, sigma / 255, (48, 64, 1 if gray else 3))
    else:
        signal = (source * np.array([0.2989, 0.5870, 0.1140])).sum(2, keepdims=True) if gray else source
        signal = np.clip(np.round(signal * 255), 0, 255) / 255
        levels = 2 ** math.ceil(math.log2(np.unique(signal).size))
        amplitude = rng.uniform(0.05, 3)
        residual = (rng.poisson(signal * levels) / levels - signal) * amplitude
    np.testing.assert_array_equal(actual, np.clip(source + residual, 0, 1).astype(np.float32))


def test_scale_eight_is_explicit_and_controls_native_grid():
    cfg = DegradationConfig(scale=8.0)
    trace = []
    result, target = degrade_pair(torch.full((3, 129, 193), 0.5), cfg, 7, trace=trace)
    assert trace[0]["scale"] == 8
    assert trace[0]["target"] == [16, 24]
    assert next(event for event in trace if event["op"] == "native_lq")["shape"] == [16, 24]
    assert result.shape == target.shape == (3, 129, 193)
    torch.testing.assert_close(target, torch.full_like(target, 0.5), rtol=0, atol=0)
    torch.testing.assert_close(result * 255, (result * 255).round(), atol=2e-5, rtol=0)


def test_diffjpeg_quality_schema_matches_omgsr_exclusive_upper_sampling():
    assert CompressionConfig().jpeg_quality == (60.0, 95.0)
    assert SecondCompressionConfig().jpeg_quality == (60.0, 100.0)
    with pytest.raises(ValidationError):
        CompressionConfig(jpeg_quality=(100.0, 100.0))
    trace = []
    degrade_image(torch.full((3, 128, 128), 0.5), DegradationConfig(), 9, trace=trace)
    qualities = [event["quality"] for event in trace if event["op"] == "compression"]
    assert 60 <= qualities[0] < 95
    assert 60 <= qualities[1] < 100


@pytest.mark.parametrize("shape", [(45, 70), (9, 13)])
@pytest.mark.parametrize("synthetic", [False, True])
def test_calibration_tiles_cover_edges_and_degrade_after_crop(tmp_path, shape, synthetic):
    from PIL import Image

    from unituner.data.media.restoration import iter_aligned_images

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
