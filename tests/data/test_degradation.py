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
    _compress,
    _kernel_profile,
    _resize,
    _resize_modes_for_shape,
    _resize_scale,
    _sample_scale,
    add_noise,
    degrade_image,
    degrade_pair,
    mixed_blur_kernel,
    sinc_kernel,
)


@pytest.mark.parametrize("quality", [50, 90])
def test_jpeg_round_trip_matches_legacy_quantization_and_rgb_order(quality):
    import cv2

    from unituner.data.degradation.pipeline import _jpeg

    ties = (np.arange(255, dtype=np.float32) + 0.5) / 255
    values = np.concatenate((np.nextafter(ties, 0), ties, np.nextafter(ties, 1)))
    image = values.reshape(15, 17, 3)[:, ::-1]
    bgr = np.ascontiguousarray((image[..., ::-1] * 255).round().astype(np.uint8))
    success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert success
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    expected = decoded[..., ::-1].astype(np.float32) / 255
    output = _jpeg(image, quality)
    assert output.flags.c_contiguous
    np.testing.assert_array_equal(output, expected)


def test_bsrgan_plus_camera_noise_preserves_srgb_values_and_input():
    from unituner.data.degradation.pipeline import _linear_camera_noise

    values = np.array([0, 0.003, 0.039, 0.04045, 0.041, 0.5, 0.99, 1], dtype=np.float32)
    pixels = np.resize(values, (8, 9, 3))[:, ::-1]
    original = pixels.copy()
    reference_rng = np.random.default_rng(45)
    actual_rng = np.random.default_rng(45)
    linear = np.where(pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4)
    gain = np.array([reference_rng.uniform(1, 1.8), 1.0, reference_rng.uniform(1, 1.8)], dtype=np.float32)
    sensor = linear / gain
    shot = float(reference_rng.uniform(0.0005, 0.006))
    read = float(reference_rng.uniform(0.00001, 0.0002))
    noise = reference_rng.standard_normal(pixels.shape, dtype=np.float32) * np.sqrt(sensor * shot + read)
    rendered = ((sensor + noise) * gain).clip(0, 1)
    expected = np.where(rendered <= 0.0031308, 12.92 * rendered, 1.055 * rendered ** (1 / 2.4) - 0.055)
    np.testing.assert_array_equal(_linear_camera_noise(pixels, actual_rng), expected)
    np.testing.assert_array_equal(pixels, original)
    assert actual_rng.random() == reference_rng.random()


@pytest.mark.parametrize("scale", [1.0, 4.0, 8.0, 16.0, (4.0, 8.0)])
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
    assert cfg.scale == 4
    assert cfg.first.resize_probabilities == (0.2, 0.7, 0.1)
    assert cfg.first.resize_range == (0.3, 1.5)
    assert cfg.first.noise.gaussian_probability == pytest.approx(1 / 3)
    assert cfg.first.noise.iso_probability == pytest.approx(1 / 3)
    assert cfg.first.noise.gaussian_sigma == (1.0, 15.0)
    assert cfg.first.noise.poisson_scale == (0.05, 2.0)
    assert cfg.first.noise.iso_shot_noise == (0.0008, 0.015)
    assert cfg.first.noise.iso_read_noise == (0.0001, 0.006)
    assert cfg.first.noise.iso_color_gain == (0.92, 1.15)
    assert cfg.first.compression.jpeg_probability == 0.5
    assert cfg.first.compression.jpeg_quality == (60.0, 95.0)
    assert cfg.first.compression.webp_quality == (60.0, 95.0)
    assert cfg.second.blur_probability == 0.5
    assert cfg.second.resize_probabilities == (0.3, 0.4, 0.3)
    assert cfg.second.resize_range == (0.8, 1.2)
    assert cfg.second.noise.gaussian_sigma == (1.0, 12.0)
    assert cfg.second.noise.poisson_scale == (0.05, 1.0)
    assert cfg.second.noise.iso_shot_noise == (0.001, 0.02)
    assert cfg.second.noise.iso_read_noise == (0.0002, 0.008)
    assert cfg.second.noise.iso_color_gain == (0.9, 1.2)
    assert cfg.second.compression.jpeg_quality == (60.0, 100.0)
    assert cfg.second.compression.webp_quality == (60.0, 100.0)
    assert cfg.final_sinc_probability == 0.8
    assert cfg.gray_probability == cfg.color_jitter_probability == 0
    assert _kernel_profile() == (tuple(range(11, 42, 2)), 41, 25)


@pytest.mark.parametrize(
    "scale",
    [0.5, 17.0, float("nan"), (0.5, 8.0), (4.0, 17.0), (8.0, 4.0), (4.0, float("nan"))],
)
def test_scale_parameter_is_strictly_bounded(scale):
    with pytest.raises(ValidationError):
        DegradationConfig(scale=scale)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_resize_matches_half_pixel_reference(mode):
    source = np.random.default_rng(3).random((101, 99, 3), dtype=np.float32)
    target = (61, 60)
    actual = _resize(source, *target, np.random.default_rng(0), mode=mode, second_stage=True)
    reference = (
        F.interpolate(torch.from_numpy(source).permute(2, 0, 1)[None], size=target, mode=mode, align_corners=False)[0]
        .permute(1, 2, 0)
        .numpy()
    )
    np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-5)


def test_nearest_resize_uses_pixel_centers_and_averages_exact_ties():
    source = np.array([[[0.0], [1.0], [3.0], [6.0]]], dtype=np.float32)
    actual = _resize(source, 1, 3, np.random.default_rng(0), mode="nearest", second_stage=True)
    np.testing.assert_array_equal(actual, np.array([[[0.0], [2.0], [6.0]]], dtype=np.float32))


def test_phase_fix_differs_from_float_scale_factor_on_odd_grids():
    source = np.random.default_rng(4).random((101, 99, 3), dtype=np.float32)
    value = torch.from_numpy(source).permute(2, 0, 1)[None]
    fixed = _resize(source, 61, 60, np.random.default_rng(0), mode="bicubic")
    old = F.interpolate(value, scale_factor=0.613, mode="bicubic", align_corners=False)[0].permute(1, 2, 0).numpy()
    assert fixed.shape == old.shape
    assert np.max(np.abs(fixed - old)) > 1e-3


def test_blocky_modes_are_only_available_for_second_stage_micro_resizing():
    source = np.random.default_rng(5).random((101, 99, 3), dtype=np.float32)
    for shape in ((61, 60), (101, 60), (101, 99), (102, 100)):
        assert _resize_modes_for_shape(source, *shape) == ("bilinear", "bicubic")
        assert _resize_modes_for_shape(source, *shape, second_stage=True) == ("area", "bilinear", "bicubic", "nearest")
    with pytest.raises(ValueError, match="restricted to the second stage"):
        _resize(source, 202, 198, np.random.default_rng(0), mode="nearest")
    with pytest.raises(ValueError, match="restricted to the second stage"):
        _resize(source, 202, 198, np.random.default_rng(0), mode="area")
    assert _resize(source, 102, 100, np.random.default_rng(0), mode="nearest", second_stage=True).shape == (102, 100, 3)


def test_pipeline_never_uses_blocky_modes_for_delivery_or_final_alignment(monkeypatch):
    from unituner.data.degradation import pipeline

    calls = []
    original = pipeline._resize

    def track(image, height, width, rng, **options):
        calls.append(((height, width), options.get("second_stage", False)))
        return original(image, height, width, rng, **options)

    monkeypatch.setattr(pipeline, "_resize", track)
    degrade_pair(torch.rand(3, 64, 64), DegradationConfig(scale=4), 17)
    assert [second_stage for _, second_stage in calls] == [False, True, False, False]
    assert calls[-1][0] == (64, 64)


def test_scale_range_is_continuous_deterministic_and_uses_integer_grid():
    cfg = DegradationConfig.model_validate_json('{"scale":[4,8]}', strict=True)
    assert cfg.scale == (4.0, 8.0)
    draws = [_sample_scale(cfg.scale, np.random.default_rng(seed)) for seed in range(100)]
    assert min(draws) >= 4 and max(draws) < 8
    assert len({round(value, 6) for value in draws}) == len(draws)
    assert _sample_scale(cfg.scale, np.random.default_rng(17)) == _sample_scale(cfg.scale, np.random.default_rng(17))

    trace = []
    result = degrade_image(torch.full((3, 129, 193), 0.5), cfg, 7, trace=trace)
    source = trace[0]
    assert source["scale_range"] == [4.0, 8.0]
    assert source["target"] == [int(129 / source["scale"]), int(193 / source["scale"])]
    assert next(event for event in trace if event["op"] == "native_lq")["shape"] == source["target"]
    assert result.shape == (3, 129, 193)


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


def test_second_stage_can_undershoot_native_lq_with_integer_phase_stable_size():
    cfg = DegradationConfig()
    assert cfg.first.resize_range == (0.3, 1.5)
    assert cfg.second.resize_range == (0.8, 1.2)
    draws = [_resize_scale(cfg.second, np.random.default_rng(seed)) for seed in range(1000)]
    assert min(draws) >= 0.8 and min(draws) < 0.81
    assert 1.19 < max(draws) <= 1.2
    undershot = False
    for seed in range(64):
        trace = []
        degrade_image(torch.full((3, 128, 192), 0.5), cfg, seed, trace=trace)
        native_target = trace[0]["target"]
        stage = next(event for event in trace if event["op"] == "resize" and event["stage"] == 2)
        assert stage["shape"] == stage["requested_shape"]
        undershot |= stage["shape"][0] < native_target[0]
    assert undershot


def test_1024_four_x_point_eight_case_rounds_to_nearest_integer_before_delivery(monkeypatch):
    monkeypatch.setattr("unituner.data.degradation.pipeline._resize_scale", lambda *_: 0.8)
    trace = []
    degrade_image(torch.full((3, 1024, 1024), 0.5), DegradationConfig(scale=4), 1, trace=trace)
    assert [event["shape"] for event in trace if event["op"] == "resize"] == [[819, 819], [205, 205]]
    assert next(event for event in trace if event["op"] == "native_lq")["shape"] == [256, 256]


@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("gray", [False, True])
def test_noise_matches_omgsr_add_then_clip_semantics(gaussian, gray):
    source = np.random.default_rng(2).uniform(-0.15, 1.15, (48, 64, 3)).astype(np.float32)
    cfg = NoiseConfig(
        gaussian_probability=float(gaussian),
        iso_probability=0,
        gray_probability=float(gray),
        gaussian_sigma=(1.0, 30.0),
        poisson_scale=(0.05, 3.0),
    )
    actual = add_noise(source, cfg, np.random.default_rng(17))
    rng = np.random.default_rng(17)
    rng.choice(3, p=(cfg.gaussian_probability, 1 - cfg.gaussian_probability, 0))
    rng.random()
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


@pytest.mark.parametrize("gray", [False, True])
def test_iso_noise_matches_signal_dependent_camera_model(gray):
    source = np.random.default_rng(2).uniform(0, 1, (24, 32, 3)).astype(np.float32)
    cfg = NoiseConfig(
        gaussian_probability=0,
        iso_probability=1,
        gray_probability=float(gray),
        iso_shot_noise=(0.002, 0.002),
        iso_read_noise=(0.0005, 0.0005),
        iso_color_gain=(0.95, 1.05),
    )
    trace = []
    actual = add_noise(source, cfg, np.random.default_rng(19), trace=trace)
    rng = np.random.default_rng(19)
    rng.choice(3, p=(0, 0, 1))
    rng.random()
    shot = rng.uniform(0.002, 0.002)
    read = rng.uniform(0.0005, 0.0005)
    gains = np.array([rng.uniform(0.95, 1.05), 1, rng.uniform(0.95, 1.05)], dtype=np.float32)
    variance = source * gains * shot + read
    standard = rng.standard_normal(source.shape, dtype=np.float32)
    residual = standard * np.sqrt(variance.clip(1e-10), dtype=np.float32)
    if gray:
        residual = np.repeat(residual.mean(2, keepdims=True), 3, axis=2)
    np.testing.assert_array_equal(actual, np.clip(source + residual, 0, 1).astype(np.float32))
    assert trace[0]["kind"] == "iso"
    assert trace[0]["parameter"]["shot"] == 0.002


def test_noise_probabilities_reject_an_overfull_distribution():
    with pytest.raises(ValidationError, match="sum to at most one"):
        NoiseConfig(gaussian_probability=0.6, iso_probability=0.5)


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


def test_jpeg_and_webp_quality_schema_and_sampling():
    assert CompressionConfig().jpeg_quality == (60.0, 95.0)
    assert CompressionConfig().webp_quality == (60.0, 95.0)
    assert SecondCompressionConfig().jpeg_quality == (60.0, 100.0)
    assert SecondCompressionConfig().webp_quality == (60.0, 100.0)
    with pytest.raises(ValidationError):
        CompressionConfig(jpeg_quality=(100.0, 100.0))
    with pytest.raises(ValidationError):
        CompressionConfig(webp_quality=(0.0, 95.0))

    source = np.random.default_rng(3).random((32, 40, 3), dtype=np.float32)
    for jpeg, codec in ((True, "diffjpeg"), (False, "webp")):
        trace = []
        settings = CompressionConfig(
            jpeg_probability=float(jpeg),
            jpeg_quality=(80.0, 80.0),
            webp_quality=(80.0, 80.0),
        )
        result = _compress(source, settings, np.random.default_rng(5), trace)
        assert result.shape == source.shape and result.dtype == np.float32
        assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
        assert trace == [{"op": "compression", "codec": codec, "quality": 80.0, "shape": [32, 40]}]

    codecs = set()
    for seed in range(32):
        trace = []
        degrade_image(torch.full((3, 128, 128), 0.5), DegradationConfig(), seed, trace=trace)
        compressions = [event for event in trace if event["op"] == "compression"]
        assert 60 <= compressions[0]["quality"] < 95
        assert 60 <= compressions[1]["quality"] < 100
        codecs.update(event["codec"] for event in compressions)
    assert codecs == {"diffjpeg", "webp"}


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


@pytest.mark.parametrize("size, scale", [(512, 4), (1024, 8)])
def test_upstream_rotation_uses_only_merged_omgsr_recipe(monkeypatch, size, scale):
    def fail_if_legacy_scale_is_sampled(*args):
        raise AssertionError("Upstream recipes must use their own scale")

    monkeypatch.setattr("unituner.data.degradation.pipeline._sample_scale", fail_if_legacy_scale_is_sampled)
    assert "upstream_rotation" not in DegradationConfig().model_dump()
    config = DegradationConfig(enabled=True, upstream_rotation=True, scale=(11.0, 12.0))
    high = torch.linspace(0, 1, 3 * size * size).reshape(3, size, size)
    for seed in (0, 1, 5):
        trace = []
        low, target = degrade_pair(high, config, seed, trace=trace)
        repeat, _ = degrade_pair(high, config, seed)
        assert torch.equal(low, repeat) and torch.equal(target, high)
        assert low.shape == high.shape and torch.isfinite(low).all()
        assert trace[0]["recipe"] == "adabridge_dnf_omgsr"
        assert trace[0]["scale"] == scale
        assert next(item for item in trace if item["op"] == "native_lq")["shape"] == [128, 128]
        assert sum(item["op"] == "noise" for item in trace) == 2
        assert next(item["mode"] for item in trace if item["op"] == "alignment") in ("bilinear", "bicubic")
        assert [item["codec"] for item in trace if item["op"] == "compression"] == ["diffjpeg", "diffjpeg"]


@pytest.mark.parametrize("mode", ["area", "bilinear", "bicubic", "nearest"])
@pytest.mark.parametrize(
    "source_shape,shape", [((101, 100), (60, 61)), ((1024, 1001), (131, 412)), ((31, 32), (62, 65))]
)
def test_resize_has_no_directional_grid_bias(mode, source_shape, shape):
    from unituner.data.degradation.resize import resize_centered

    image = np.random.default_rng(27).random((*source_shape, 3), dtype=np.float32)
    result = resize_centered(image, shape, mode)
    assert result.shape == (*shape, 3)
    np.testing.assert_allclose(result, resize_centered(image[::-1], shape, mode)[::-1], rtol=0, atol=1e-7)
    np.testing.assert_allclose(result, resize_centered(image[:, ::-1], shape, mode)[:, ::-1], rtol=0, atol=1e-7)
    legacy = _resize(image, *shape, np.random.default_rng(0), mode=mode, second_stage=True)
    np.testing.assert_array_equal(legacy, result)
    if mode != "nearest":
        upstream = _resize(image, *shape, np.random.default_rng(0), mode=mode, upstream=True)
        np.testing.assert_array_equal(upstream, result)


@pytest.mark.parametrize("mode", ["area", "bilinear", "bicubic", "nearest"])
def test_resize_chain_preserves_center_across_odd_and_even_grids(mode):
    from unituner.data.degradation.resize import resize_centered

    impulse = np.zeros((35, 34, 1), dtype=np.float32)
    impulse[16:19, 15:19] = 1
    value = impulse
    for shape in ((22, 21), (31, 37), (35, 34)):
        value = resize_centered(value, shape, mode)
        assert value.max() > 0
        np.testing.assert_allclose(value, value[::-1], rtol=0, atol=1e-7)
        np.testing.assert_allclose(value, value[:, ::-1], rtol=0, atol=1e-7)


@pytest.mark.parametrize("shape", [(64, 80), (65, 81)])
def test_compression_preserves_near_center_color_landmark(shape):
    from unituner.data.degradation.diffjpeg import diffjpeg

    y, x = np.indices(shape)
    cy, cx = (np.array(shape) - 1) / 2
    spot = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 50).astype(np.float32)
    image = spot[..., None] * np.array([1, 0, 0], dtype=np.float32)
    result = diffjpeg(image, 95)
    assert result.shape == image.shape
    weights = np.maximum(result[..., 0] - 0.15, 0)
    center = [(weights * coordinate).sum() / weights.sum() for coordinate in (y, x)]
    np.testing.assert_allclose(center, (cy, cx), rtol=0, atol=0.25)


def test_area_resize_uses_fractional_pixel_overlap():
    from unituner.data.degradation.resize import resize_centered, scaled_shape

    gradient = np.array([[[0.0], [0.0], [1.0], [0.0], [1.0]]], dtype=np.float32)
    expected = np.array([[[0.2], [0.6]]], dtype=np.float32)
    np.testing.assert_allclose(resize_centered(gradient, (1, 2), "area"), expected, atol=1e-7)
    assert scaled_shape((101, 99), 0.613) == (62, 61)
    assert scaled_shape((101, 99), 0.613, multiple=2) == (62, 60)


@pytest.mark.parametrize("scale", [12, 16])
def test_extreme_degradation_preserves_integer_native_grid_and_alignment(scale):
    from unituner.data.degradation.config import ExtremeDegradationConfig

    high = torch.rand((3, 1024, 1024), generator=torch.Generator().manual_seed(42))
    settings = DegradationConfig(upstream_rotation=True, extreme=ExtremeDegradationConfig(probability=1.0, scale=scale))
    orders = set()
    for seed in range(5):
        trace = []
        low, target = degrade_pair(high, settings, seed, trace=trace)
        assert low.shape == high.shape and torch.equal(target, high)
        assert torch.equal(low, degrade_pair(high, settings, seed)[0])
        operations = tuple(event["op"] for event in trace)
        orders.add(operations)
        assert set(operations[1:5]) == {"blur", "intermediate_resize", "noise", "compression"}
        assert operations[5:] == ("delivery_resize", "native_lq", "alignment")
        assert all(event["shape"] in ([1024, 1024], [256, 256]) for event in trace[1:5])
        assert next(event for event in trace if event["op"] == "intermediate_resize")["shape"] == [256, 256]
        assert trace[0]["recipe"] == "extreme_bsrgan"
        assert trace[-2:] == [
            {"op": "native_lq", "shape": [1024 // scale, 1024 // scale]},
            {"op": "alignment", "shape": [1024, 1024], "mode": "bicubic"},
        ]
    assert len(orders) > 1


def test_extreme_probability_zero_preserves_existing_rotation():
    from unituner.data.degradation.config import ExtremeDegradationConfig

    high = torch.rand((3, 33, 35), generator=torch.Generator().manual_seed(42))
    regular = DegradationConfig(upstream_rotation=True)
    disabled = DegradationConfig(upstream_rotation=True, extreme=ExtremeDegradationConfig(probability=0.0))
    assert "extreme" not in regular.model_dump()
    for seed in range(10):
        assert torch.equal(degrade_pair(high, regular, seed)[0], degrade_pair(high, disabled, seed)[0])


@pytest.mark.parametrize("probability, scale, native", [(0.0, 8, 128), (1.0, 12, 85)])
def test_both_configured_scales_use_shuffled_bsrgan(probability, scale, native):
    from unituner.data.degradation.config import ExtremeDegradationConfig

    high = torch.zeros(3, 1024, 1024)
    settings = DegradationConfig(
        upstream_rotation=True,
        extreme=ExtremeDegradationConfig(probability=probability, scale=12, secondary_scale=8, shuffle_probability=1.0),
    )
    orders = set()
    for seed in (0, 1, 2):
        trace = []
        low, target = degrade_pair(high, settings, seed, trace=trace)
        assert low.shape == high.shape and torch.equal(target, high)
        assert trace[0]["recipe"] == "extreme_bsrgan" and trace[0]["scale"] == scale
        assert [(item["op"], item["shape"]) for item in trace[-3:]] == [
            ("delivery_resize", [native, native]),
            ("native_lq", [native, native]),
            ("alignment", [1024, 1024]),
        ]
        assert trace[-1]["mode"] == "bicubic"
        orders.add(tuple(item["op"] for item in trace[1:5]))
    assert len(orders) > 1


@pytest.mark.parametrize("global_probability, mode", [(0.0, "local"), (1.0, "global")])
@pytest.mark.parametrize("size, scale", [((1024, 1024), 8), ((103, 99), 12)])
def test_bsrgan_plus_order_final_jpeg_and_unchanged_target(global_probability, mode, size, scale):
    from unituner.data.degradation.config import BSRGANPlusConfig, ExtremeDegradationConfig

    options = BSRGANPlusConfig(
        global_shuffle_probability=global_probability,
        poisson_probability=1.0,
        speckle_probability=1.0,
        isp_probability=1.0,
    )
    settings = DegradationConfig(extreme=ExtremeDegradationConfig(probability=1.0, scale=scale, plus=options))
    high = torch.rand((3, *size), generator=torch.Generator().manual_seed(42))
    trace = []
    low, target = degrade_pair(high, settings, 13, trace=trace)
    assert low.shape == high.shape and torch.isfinite(low).all() and torch.equal(target, high)
    assert torch.equal(low, degrade_pair(high, settings, 13)[0])
    assert trace[0]["recipe"] == "bsrgan_plus" and trace[0]["shuffle"] == mode
    assert [event["op"] for event in trace[-4:]] == ["compression", "delivery_resize", "native_lq", "alignment"]
    jpeg = trace[-4]
    assert jpeg["final"] and jpeg["codec"] == "jpeg" and jpeg["shape"] == trace[-5]["shape"]
    native = [axis // scale for axis in size]
    assert all(actual > target for actual, target in zip(jpeg["shape"], native, strict=True))
    assert all(event["shape"] == native for event in trace[-3:-1])
    assert trace[-1]["mode"] == "bicubic"
    for stage in (1, 2):
        events = [event for event in trace[1:-4] if event["stage"] == stage]
        assert {event["kind"] for event in events if event["op"] == "noise"} == {
            "gaussian",
            "poisson",
            "speckle",
            "isp",
        }
        assert {event["op"] for event in events} == (
            {"blur", "resize", "noise", "compression"} if stage == 1 else {"blur", "resize", "noise"}
        )
    if mode == "local":
        assert [event["op"] for event in trace[1:-4]] == [
            "blur",
            "resize",
            *["noise"] * 4,
            "compression",
            "blur",
            "resize",
            *["noise"] * 4,
        ]
    else:
        assert [event["stage"] for event in trace[1:-4]] != [1] * 7 + [2] * 6


def test_bsrgan_plus_delivery_modes_and_quality_bounds():
    from unituner.data.degradation.config import BSRGANPlusConfig, ExtremeDegradationConfig

    settings = DegradationConfig(extreme=ExtremeDegradationConfig(probability=1.0, scale=8, plus=BSRGANPlusConfig()))
    high = torch.full((3, 64, 64), 0.5)
    modes = set()
    alignment_modes = set()
    for seed in range(24):
        trace = []
        degrade_pair(high, settings, seed, trace=trace)
        modes.add(trace[-3]["mode"])
        alignment_modes.add(trace[-1]["mode"])
        assert all(50 <= event["quality"] <= 90 for event in trace if event["op"] == "compression")
        levels = [event["sigma"] for event in trace if event.get("kind") == "gaussian"]
        assert len(levels) == 2 and all(2 <= level <= 25 for level in levels)
    assert modes == {"bilinear", "bicubic", "area"}
    assert alignment_modes == {"bilinear", "bicubic"}


@pytest.mark.parametrize("scale", [4, 8])
def test_bsrgan_plus_intermediate_resize_uses_current_integer_grid(scale):
    from unituner.data.degradation.config import BSRGANPlusConfig, ExtremeDegradationConfig
    from unituner.data.degradation.resize import scaled_shape

    settings = DegradationConfig(
        extreme=ExtremeDegradationConfig(probability=1.0, scale=scale, intermediate_scale=2, plus=BSRGANPlusConfig())
    )
    high = torch.full((3, 96, 112), 0.5)
    factors_by_stage = {1: set(), 2: set()}
    saw_clamped_resize = False
    native = [96 // scale, 112 // scale]
    minimum = [(5 * value + 3) // 4 for value in native]
    for seed in range(48):
        trace = []
        low, target = degrade_pair(high, settings, seed, trace=trace)
        assert low.shape == high.shape and torch.equal(target, high)
        assert trace[-4]["shape"] == trace[-5]["shape"] and trace[-4]["final"]
        assert all(actual >= floor for actual, floor in zip(trace[-4]["shape"], minimum, strict=True))
        assert trace[-3]["shape"] == native
        assert trace[-1]["shape"] == [96, 112] and trace[-1]["mode"] in {"bilinear", "bicubic"}
        for i, event in enumerate(trace[:-4]):
            if event["op"] == "blur":
                pixels_per_native = min(size / target for size, target in zip(event["shape"], native, strict=True))
                assert event["sigma"] <= 0.55 * pixels_per_native
            if event["op"] != "resize":
                continue
            factor = event["factor"]
            assert 0.5 / scale <= factor <= 2.0
            assert event["source_shape"] == trace[i - 1]["shape"]
            requested = scaled_shape(tuple(event["source_shape"]), factor)
            assert event["shape"] == [max(size, floor) for size, floor in zip(requested, minimum, strict=True)]
            saw_clamped_resize |= list(requested) != event["shape"]
            factors_by_stage[event["stage"]].add("down" if factor < 1 else "up" if factor > 1 else "keep")
    assert all(directions == {"down", "up", "keep"} for directions in factors_by_stage.values())
    assert saw_clamped_resize


@pytest.mark.parametrize("flip_probability", [0.0, 1.0])
@pytest.mark.parametrize("seed, mode", [(22, "bicubic"), (23, "bilinear")])
def test_random_plus_patch_crops_after_full_image_jpeg(tmp_path, flip_probability, seed, mode):
    from PIL import Image

    from unituner.data.degradation.config import BSRGANPlusConfig, ExtremeDegradationConfig
    from unituner.data.degradation.pipeline import degrade_bsrgan_plus_native
    from unituner.data.degradation.resize import resize_centered
    from unituner.data.media.restoration import read_aligned_images

    pixels = np.random.default_rng(12).integers(0, 256, (104, 152, 3), dtype=np.uint8)
    path = tmp_path / "hq.png"
    Image.fromarray(pixels).save(path)
    extreme = ExtremeDegradationConfig(probability=1.0, scale=8, plus=BSRGANPlusConfig())
    settings = DegradationConfig(enabled=True, extreme=extreme)
    low, high = read_aligned_images(
        None, path, resolution=(64, 48), seed=seed, flip_probability=flip_probability, degradation=settings
    )
    trace = []
    native = degrade_bsrgan_plus_native(pixels.astype(np.float32) / 255, extreme, 8, seed, trace)
    assert trace[-3]["final"] and trace[-3]["shape"] == trace[-4]["shape"]
    assert trace[-2]["op"] == "delivery_resize" and trace[-2]["shape"] == [13, 19]
    generator = torch.Generator().manual_seed(seed)
    top = int(torch.randint(8, (), generator=generator))
    left = int(torch.randint(12, (), generator=generator))
    assert (top, left) != (0, 0)
    expected_low = resize_centered(native[top : top + 6, left : left + 8], (48, 64), mode).clip(0, 1)
    expected_low = torch.from_numpy((expected_low * 255).round() / 255).permute(2, 0, 1)
    high_crop = pixels[top * 8 : top * 8 + 48, left * 8 : left * 8 + 64]
    expected_high = torch.from_numpy(high_crop).permute(2, 0, 1).float() / 255
    if flip_probability == 1.0:
        expected_low, expected_high = expected_low.flip(-1), expected_high.flip(-1)
    assert torch.equal(low, expected_low * 2 - 1)
    assert torch.equal(high, expected_high * 2 - 1)
