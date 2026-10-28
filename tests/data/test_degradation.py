"""Check real degradation, interpolation coordinates, codecs and global-state isolation."""

import colorsys
import math
import random

import numpy as np
import pytest
import torch

pytest.importorskip("cv2")
from unitunner.data.degradation.color import adjust_hue
from unitunner.data.degradation.config import (
    BlurConfig,
    DegradationConfig,
    NoiseConfig,
    SecondNoiseConfig,
    SecondStageConfig,
    StageConfig,
)
from unitunner.data.degradation.pipeline import (
    _defocus_kernel,
    _motion_kernel,
    _resize,
    add_noise,
    compress_image,
    degrade_image,
    mixed_blur_kernel,
    sinc_kernel,
)


def test_kernels_preserve_dc_centroid_and_sinc_negative_lobes():
    rng = np.random.default_rng(42)
    for _ in range(32):
        kernel = mixed_blur_kernel(rng, BlurConfig(), 512)
        assert np.isfinite(kernel).all()
        assert kernel.sum(dtype=np.float64) == pytest.approx(1, abs=2e-7)
        axis = np.arange(kernel.shape[0]) - kernel.shape[0] // 2
        assert abs((kernel * axis[None, :]).sum()) < 1e-6
        assert abs((kernel * axis[:, None]).sum()) < 1e-6
    kernel = sinc_kernel(21, math.pi / 2)
    assert kernel.min() < 0
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-7)
    assert kernel.sum(dtype=np.float64) == pytest.approx(1, abs=2e-7)


@pytest.mark.parametrize("scale", [1, 4, 16])
@pytest.mark.parametrize("scene", [0, 1, 2])
@pytest.mark.parametrize("strengths", [(0.0, 0.0), (0.2, 0.7), (0.7, 0.2)])
def test_degradation_local_rng_finite_range_and_unchanged_hq(scale, scene, strengths):
    high = torch.from_numpy(np.random.default_rng(42).random((3, 64, 80), dtype=np.float32))
    original = high.clone()
    before_torch, before_numpy, before_python = torch.get_rng_state(), np.random.get_state(), random.getstate()
    settings = DegradationConfig(
        scales=(scale,),
        smear_strength=(strengths[0], strengths[0]),
        grain_strength=(strengths[1], strengths[1]),
        color_jitter_probability=1.0,
        gray_probability=0.5,
        scenario_probabilities=(float(scene == 0), float(scene == 1), float(scene == 2)),
    )
    low = degrade_image(high, settings, 7)
    torch.testing.assert_close(low, degrade_image(high, settings, 7), rtol=0, atol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(low, degrade_image(high, settings, 7), rtol=0, atol=0)
    assert not torch.equal(low, degrade_image(high, settings, 8))
    assert low.shape == high.shape and low.dtype == torch.float32
    assert torch.isfinite(low).all() and 0 <= low.min() <= low.max() <= 1
    torch.testing.assert_close(high, original, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), before_torch)
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0] and after_numpy[2:] == before_numpy[2:]
    np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
    assert random.getstate() == before_python


@pytest.mark.parametrize("kind", [0, 1, 2])
def test_noise_families_keep_gray_correlation_and_signal_range(kind):
    image = np.full((32, 32, 3), 0.5, dtype=np.float32)
    settings = NoiseConfig(probabilities=(float(kind == 0), float(kind == 1), float(kind == 2)), gray_probability=1.0)
    noisy = add_noise(image, settings, np.random.default_rng(3))
    assert noisy.dtype == np.float32 and 0 <= noisy.min() < noisy.max() <= 1
    np.testing.assert_array_equal(noisy[..., 0], noisy[..., 1])
    np.testing.assert_array_equal(noisy[..., 1], noisy[..., 2])


@pytest.mark.parametrize("codec,quality", [("jpeg", 95), ("webp", 95), ("jpeg2000", 900)])
def test_real_codecs_preserve_rgb_order_and_small_image_geometry(codec, quality):
    image = np.full((16, 24, 3), [0.8, 0.3, 0.1], dtype=np.float32)
    decoded = compress_image(image, codec, quality)
    assert decoded.shape == image.shape and decoded.dtype == np.float32
    np.testing.assert_allclose(decoded.mean((0, 1)), image.mean((0, 1)), atol=0.025)


def test_fractional_resize_uses_integer_grid_coordinates():
    ramp = np.broadcast_to(np.arange(7, dtype=np.float32)[None, :, None], (5, 7, 3)).copy()
    actual = _resize(ramp, 3, 4, np.random.default_rng(0), DegradationConfig(resize_modes=("bilinear",)), factor=0.6)
    # Triangle weights use the actual 7->4 grid rather than the requested 0.6.
    expected = np.array([11 / 21, 52 / 25, 98 / 25, 115 / 21], dtype=np.float32)
    assert actual.shape == (3, 4, 3)
    np.testing.assert_allclose(actual[1, :, 0], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("scale", [4, 8])
@pytest.mark.parametrize("factor", [0.83, 0.713])
def test_fractional_resize_roundtrip_preserves_interior_coordinates(vertical, scale, factor):
    # A coordinate ramp detects geometric drift independently of texture blur.
    ramp = np.broadcast_to(np.arange(1537, dtype=np.float32)[None, :, None], (32, 1537, 3)).copy()
    if vertical:
        ramp = ramp.transpose(1, 0, 2).copy()
    settings = DegradationConfig(resize_modes=("bilinear",))
    rng = np.random.default_rng(0)
    height, width = ramp.shape[:2]
    first = _resize(ramp, int(height * factor), int(width * factor), rng, settings, factor=factor)
    low = _resize(first, height // scale, width // scale, rng, settings)
    restored = _resize(low, height, width, rng, settings)
    interior = (slice(8, -8), slice(64, -64))
    if vertical:
        interior = interior[::-1]
    # Discrete antialias weights introduce small local ripples, not cumulative
    # geometric drift. Bound both the local error and its systematic bias.
    np.testing.assert_allclose(restored[interior], ramp[interior], rtol=0, atol=0.075)
    assert abs(float((restored[interior] - ramp[interior]).mean())) < 0.01


def test_hue_matches_scalar_colorsys():
    colors = [(0, 0, 0), (0.5, 0.5, 0.5), (1, 0, 0), (0, 1, 0), (0.2, 0.7, 0.4), (0.9, 0.2, 0.8)]
    image = torch.tensor(colors).T[:, None, :]
    for shift in (-0.5, -0.1, 0, 0.5):
        expected_colors = []
        for color in colors:
            hue, saturation, value = colorsys.rgb_to_hsv(*color)
            expected_colors.append(colorsys.hsv_to_rgb((hue + shift) % 1, saturation, value))
        torch.testing.assert_close(
            adjust_hue(image, shift), torch.tensor(expected_colors).T[:, None, :], atol=1e-6, rtol=1e-6
        )


@pytest.mark.parametrize("resolution", [512, 768, 1024, 1280, 1536, 2048, 4096])
def test_spatial_blur_tracks_image_scale(resolution):
    gaussian = BlurConfig(
        sigma=(1.0, 1.0),
        probabilities=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        motion_probability=0.0,
        defocus_probability=0.0,
    )
    kernel = mixed_blur_kernel(np.random.default_rng(23), gaussian, resolution)
    axis = np.arange(kernel.shape[0]) - kernel.shape[0] // 2
    measured_sigma = np.sqrt((kernel * axis[None, :] ** 2).sum())
    ratio = resolution / 1024
    expected = ratio ** (0.25 if ratio < 1 else 0.5)
    assert measured_sigma == pytest.approx(expected, rel=0.08)
    assert kernel.shape[0] % 2 == 1
    assert kernel.sum() == pytest.approx(1, abs=1e-6)


def test_each_blur_layer_uses_its_actual_input_short_edge(monkeypatch):
    from unitunner.data.degradation import pipeline

    resolutions = []
    original = pipeline.mixed_blur_kernel

    def capture(rng, settings, resolution, **kwargs):
        resolutions.append(resolution)
        return original(rng, settings, resolution, **kwargs)

    monkeypatch.setattr(pipeline, "mixed_blur_kernel", capture)
    high = torch.full((3, 64, 96), 0.5)
    for image in (high, high.transpose(1, 2)):
        result = degrade_image(
            image,
            DegradationConfig(
                scales=(1.0,),
                smear_strength=(0.0, 0.0),
                grain_strength=(0.0, 0.0),
                sharpen_strength=(0.0, 0.0),
                final_sinc_probability=0.0,
                first=StageConfig(
                    blur_probability=1.0,
                    resize_probabilities=(0.0, 0.0, 1.0),
                    noise_probability=0.0,
                    compression_probability=0.0,
                ),
                second=SecondStageConfig(
                    blur_probability=1.0,
                    resize_probabilities=(0.0, 0.0, 1.0),
                    noise_probability=0.0,
                    compression_probability=0.0,
                ),
            ),
            7,
        )
        assert result.shape == image.shape
    assert resolutions == [64, 64, 64, 64]


def test_resolution_scale_and_sinc_support_expand_through_4k():
    from unitunner.data.degradation.pipeline import _resolution_scale, _resolution_sinc_kernel

    assert _resolution_scale(64) == pytest.approx(0.5)
    assert _resolution_scale(256) == pytest.approx(math.sqrt(0.5))
    assert _resolution_scale(512) == pytest.approx(0.5**0.25)
    assert _resolution_scale(768) == pytest.approx(0.75**0.25)
    assert _resolution_scale(1024) == 1
    assert _resolution_scale(1280) == pytest.approx(math.sqrt(1.25))
    assert _resolution_scale(2048) == pytest.approx(math.sqrt(2))
    assert _resolution_scale(4096) == 2
    supports = [
        _resolution_sinc_kernel(np.random.default_rng(9), resolution).shape[0]
        for resolution in (1024, 1280, 2048, 4096)
    ]
    assert supports == sorted(supports)
    assert supports[0] < supports[-1]


@pytest.mark.parametrize("motion", [False, True])
def test_optical_exposure_kernels_are_centered_and_preserve_energy(motion):
    kernel = _motion_kernel(31, 8, 0, 0) if motion else _defocus_kernel(31, 4)
    assert np.isfinite(kernel).all() and kernel.min() >= 0
    assert kernel.sum() == pytest.approx(1, abs=1e-7)
    np.testing.assert_array_equal(kernel, kernel[::-1, ::-1])
    axis = np.arange(31) - 15
    x_variance = (kernel * axis[None, :] ** 2).sum()
    y_variance = (kernel * axis[:, None] ** 2).sum()
    if motion:
        assert x_variance > 10 * y_variance
    else:
        assert x_variance == pytest.approx(y_variance)
        assert kernel[15, 20] == 0  # Outside the circular aperture.


def test_poisson_noise_does_not_depend_on_unrelated_image_colors():
    flat = np.full((64, 64, 3), 0.5, dtype=np.float32)
    textured = flat.copy()
    textured[32:] = np.random.default_rng(0).random(textured[32:].shape, dtype=np.float32)
    settings = NoiseConfig(probabilities=(0.0, 1.0, 0.0), gray_probability=0.0)
    a = add_noise(flat, settings, np.random.default_rng(7))
    b = add_noise(textured, settings, np.random.default_rng(7))
    np.testing.assert_array_equal(a[:32], b[:32])


@pytest.mark.parametrize("exposure", [1.0, 0.65, 0.4])
def test_camera_noise_matches_shot_read_variance_in_linear_light(exposure):
    settings = NoiseConfig(
        probabilities=(0.0, 0.0, 1.0),
        exposure=(exposure, exposure),
        iso_shot=(0.001, 0.001),
        iso_read=(0.00001, 0.00001),
        iso_color_gain=(1.0, 1.0),
        gray_probability=0.0,
    )
    for linear in (0.1, 0.5):
        srgb = 1.055 * linear ** (1 / 2.4) - 0.055
        image = np.full((256, 256, 3), srgb, dtype=np.float32)
        noisy = add_noise(image, settings, np.random.default_rng(7))
        decoded = np.where(noisy <= 0.04045, noisy / 12.92, ((noisy + 0.055) / 1.055) ** 2.4)
        assert float(decoded.mean()) == pytest.approx(linear, abs=0.0002)
        assert float(decoded.var()) == pytest.approx(linear * 0.001 / exposure + 0.00001 / exposure**2, rel=0.02)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_downsampling_suppresses_high_frequency_aliasing(mode):
    stripe = np.broadcast_to((np.arange(128) % 2)[None, :, None], (128, 128, 3)).astype(np.float32)
    resized = _resize(stripe, 39, 39, np.random.default_rng(0), DegradationConfig(resize_modes=(mode,)))
    unfiltered = (
        torch.nn.functional.interpolate(torch.from_numpy(stripe).permute(2, 0, 1)[None], size=(39, 39), mode=mode)[0]
        .permute(1, 2, 0)
        .numpy()
    )
    assert resized[2:-2, 2:-2].std() < unfiltered[2:-2, 2:-2].std() * 0.15
    assert resized.mean() == pytest.approx(0.5, abs=0.002)


@pytest.mark.parametrize("scene", [0, 1, 2])
def test_only_web_reposts_encode_twice_and_low_light_processes_camera_noise(monkeypatch, scene):
    from unitunner.data.degradation import pipeline

    codecs, cameras = [], []
    original_compress, original_noise = pipeline._compress, pipeline.add_noise

    def encode(image, settings, rng, severity):
        codecs.append(image.shape[:2])
        return original_compress(image, settings, rng, severity)

    def noise(image, settings, rng, **kwargs):
        cameras.append(kwargs.get("camera", False))
        return original_noise(image, settings, rng, **kwargs)

    monkeypatch.setattr(pipeline, "_compress", encode)
    monkeypatch.setattr(pipeline, "add_noise", noise)
    settings = DegradationConfig(
        scales=(4,),
        scenario_probabilities=(float(scene == 0), float(scene == 1), float(scene == 2)),
        smear_strength=(0.0, 0.0),
        noise_strength=(1.0, 1.0),
        grain_strength=(0.0, 0.0),
        sharpen_strength=(0.0, 0.0),
        first=StageConfig(noise_probability=1.0, compression_probability=1.0),
        second=SecondStageConfig(noise_probability=1.0, compression_probability=1.0),
    )
    degrade_image(torch.full((3, 128, 192), 0.5), settings, 7)
    assert len(codecs) == (2 if scene == 1 else 1)
    assert codecs[-1] == (128, 192) or (codecs[-1][0] < 40 and codecs[-1][1] < 60)
    assert cameras == [scene == 2, False]


def test_degradation_rejects_invalid_scene_and_blur_probabilities():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="sum to one"):
        DegradationConfig(scenario_probabilities=(0.5, 0.4, 0.4))
    with pytest.raises(ValidationError, match="at most one"):
        BlurConfig(motion_probability=0.8, defocus_probability=0.8)
    with pytest.raises(ValidationError, match="sinc_probability"):
        BlurConfig(sinc_probability=0.1)


def test_scale_sampling_is_uniform_without_favoring_the_largest_factor(monkeypatch):
    from collections import Counter

    from unitunner.data.degradation import pipeline

    counts = Counter()
    scales = (2, 3, 4, 8)
    original = pipeline._resize_scale

    def capture(stage, rng, scale):
        if stage is settings.first:
            counts[scale] += 1
        return original(stage, rng, scale)

    monkeypatch.setattr(pipeline, "_resize_scale", capture)
    settings = DegradationConfig(
        scales=scales,
        scenario_probabilities=(1.0, 0.0, 0.0),
        smear_strength=(0.0, 0.0),
        noise_strength=(1.0, 1.0),
        grain_strength=(0.0, 0.0),
        sharpen_strength=(0.0, 0.0),
        first=StageConfig(blur_probability=0.0, noise_probability=0.0),
        second=SecondStageConfig(blur_probability=0.0, noise_probability=0.0, compression_probability=1.0),
    )
    high = torch.full((3, 64, 96), 0.5)
    for seed in range(1024):
        low = degrade_image(high, settings, seed)
        assert low.shape == high.shape
    assert set(counts) == set(scales)
    for count in counts.values():
        assert count == pytest.approx(256, rel=0.2)


def test_texture_suppression_preserves_edges_and_grain_preserves_mean():
    from unitunner.data.degradation.pipeline import _residual_grain, _smooth_texture

    flat = np.full((128, 128, 3), 0.4, dtype=np.float32)
    grainy = _residual_grain(flat, NoiseConfig(), np.random.default_rng(7), 0.8, strength=1)
    assert grainy.std() > 0.005
    assert abs(grainy.mean() - flat.mean()) < 0.01
    texture = flat.copy()
    texture[:, 64:] = 0.7
    texture += np.random.default_rng(8).normal(0, 0.015, texture.shape).astype(np.float32)
    smoothed = _smooth_texture(texture, 0.8, strength=1)
    assert smoothed[8:-8, 8:56].std() < texture[8:-8, 8:56].std() * 0.4
    assert smoothed[:, 68:].mean() - smoothed[:, :60].mean() > 0.25


def test_continuous_processing_supports_delivery_and_output_grid_encoding(monkeypatch):
    from unitunner.data.degradation import pipeline

    encoded = []
    original = pipeline._compress

    def capture(image, settings, rng, severity):
        encoded.append(image.shape[:2])
        return original(image, settings, rng, severity)

    monkeypatch.setattr(pipeline, "_compress", capture)
    cfg = DegradationConfig(
        scales=(4,),
        scenario_probabilities=(1.0, 0.0, 0.0),
        second=SecondStageConfig(compression_probability=1.0),
    )
    for seed in range(32):
        low = degrade_image(torch.full((3, 128, 192), 0.4), cfg, seed)
        assert low.shape == (3, 128, 192)
    assert (128, 192) in encoded
    assert any(height < 40 and width < 60 for height, width in encoded)
    assert all((height, width) == (128, 192) or (height < 40 and width < 60) for height, width in encoded)


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
        torch.testing.assert_close(high, expected * 2 - 1, atol=0, rtol=0)
        expected_low = degrade_image(expected, settings, 19 + index * 1_000_003) if settings else expected
        torch.testing.assert_close(low, expected_low * 2 - 1, atol=0, rtol=0)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("scale", [4, 8])
def test_deferred_noise_stays_before_constrained_delivery_group(monkeypatch, scale):
    from unitunner.data.degradation import pipeline

    shapes = []
    original = pipeline.add_noise

    def capture(image, settings, rng, **kwargs):
        shapes.append(image.shape[:2])
        return original(image, settings, rng, **kwargs)

    monkeypatch.setattr(pipeline, "add_noise", capture)
    cfg = DegradationConfig(
        scales=(scale,),
        scenario_probabilities=(1.0, 0.0, 0.0),
        smear_strength=(0.0, 0.0),
        noise_strength=(1.0, 1.0),
        grain_strength=(0.0, 0.0),
        sharpen_strength=(0.0, 0.0),
        resize_modes=("bilinear",),
        final_sinc_probability=0.0,
        first=StageConfig(blur_probability=0.0, noise_probability=0.0),
        second=SecondStageConfig(
            blur_probability=0.0,
            noise_probability=1.0,
            compression_probability=0.0,
            noise=SecondNoiseConfig(probabilities=(1.0, 0.0, 0.0), gaussian_sigma=(8.0, 8.0)),
        ),
    )
    for seed in range(32):
        shapes.clear()
        low = degrade_image(torch.full((3, 128, 192), 0.5), cfg, seed)
        assert len(shapes) == 1  # Deferring noise must not add it twice.
        assert (low[:, :, 1:] - low[:, :, :-1]).std().item() > 0


def test_disabling_noise_and_extra_processing_does_not_inject_noise(monkeypatch):
    from unitunner.data.degradation import pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled noise must not run on either grid")

    monkeypatch.setattr(pipeline, "add_noise", forbidden)
    cfg = DegradationConfig(
        scales=(4, 8),
        smear_strength=(0.0, 0.0),
        noise_strength=(1.0, 1.0),
        grain_strength=(0.0, 0.0),
        sharpen_strength=(0.0, 0.0),
        first=StageConfig(noise_probability=0.0),
        second=SecondStageConfig(noise_probability=0.0),
    )
    for seed in range(16):
        low = degrade_image(torch.full((3, 128, 192), 0.5), cfg, seed)
        assert low.std().item() < 1 / 255


def test_random_order_covers_noise_before_and_after_blur_in_both_stages(monkeypatch):
    from itertools import permutations

    from unitunner.data.degradation import pipeline

    cfg = DegradationConfig(
        scales=(4,),
        scenario_probabilities=(1.0, 0.0, 0.0),
        smear_strength=(0.0, 0.0),
        noise_strength=(1.0, 1.0),
        grain_strength=(0.0, 0.0),
        sharpen_strength=(0.0, 0.0),
        final_sinc_probability=0.0,
        first=StageConfig(blur_probability=1.0, noise_probability=1.0),
        second=SecondStageConfig(blur_probability=1.0, noise_probability=1.0, compression_probability=0.0),
    )
    active = 0
    traces = [[], []]
    original_scale, original_resize = pipeline._resize_scale, pipeline._resize
    original_filter, original_noise = pipeline._filter, pipeline.add_noise

    def scale(stage, rng, factor):
        nonlocal active
        active = 0 if stage is cfg.first else 1
        return original_scale(stage, rng, factor)

    def resize(*args, **kwargs):
        if "resize" not in traces[active]:
            traces[active].append("resize")
        return original_resize(*args, **kwargs)

    def blur(*args, **kwargs):
        traces[active].append("blur")
        return original_filter(*args, **kwargs)

    def noise(*args, **kwargs):
        traces[active].append("noise")
        return original_noise(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_resize_scale", scale)
    monkeypatch.setattr(pipeline, "_resize", resize)
    monkeypatch.setattr(pipeline, "_filter", blur)
    monkeypatch.setattr(pipeline, "add_noise", noise)
    observed = [set(), set()]
    for seed in range(128):
        traces = [[], []]
        degrade_image(torch.full((3, 64, 96), 0.4), cfg, seed)
        for index in range(2):
            observed[index].add(tuple(traces[index]))
    assert observed[0] == set(permutations(("blur", "resize", "noise")))
    assert observed[1] == set(permutations(("blur", "resize", "noise")))


def test_processing_gains_are_continuous_and_zero_preserves_image():
    from unitunner.data.degradation.pipeline import _residual_grain, _smooth_texture

    image = np.full((64, 64, 3), 0.4, dtype=np.float32)
    image += np.random.default_rng(4).normal(0, 0.015, image.shape).astype(np.float32)
    for operation in [
        lambda gain: _smooth_texture(image, 0.8, strength=gain),
        lambda gain: _residual_grain(
            image, NoiseConfig(), np.random.default_rng(13), 0.8, strength=gain, spatial_scale=2.7
        ),
    ]:
        np.testing.assert_array_equal(operation(0), image)
        half, full = operation(0.5), operation(1.0)
        np.testing.assert_allclose(half - image, 0.5 * (full - image), atol=1e-7)
        assert np.square(operation(1e-5) - image).mean() < 1e-10


def test_recompression_shifts_block_origin_without_shifting_image(monkeypatch):
    from unitunner.data.degradation import pipeline
    from unitunner.data.degradation.config import CompressionConfig

    image = np.random.default_rng(3).random((43, 51, 3), dtype=np.float32)
    calls = []

    def identity_codec(value, codec, quality):
        calls.append(value.shape)
        return value.copy()

    monkeypatch.setattr(pipeline, "compress_image", identity_codec)
    result = pipeline._compress(image, CompressionConfig(recompression_probability=1.0), np.random.default_rng(42), 0.8)
    assert len(calls) == 2 and calls[1][0] > calls[0][0] and calls[1][1] > calls[0][1]
    np.testing.assert_array_equal(result, image)


def test_delivery_history_covers_noise_before_and_after_smoothing(monkeypatch):
    from unitunner.data.degradation import pipeline

    calls = []
    smooth, grain = pipeline._smooth_texture, pipeline._residual_grain

    def record_smooth(*args, **kwargs):
        calls.append("smooth")
        return smooth(*args, **kwargs)

    def record_grain(*args, **kwargs):
        calls.append("grain")
        return grain(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_smooth_texture", record_smooth)
    monkeypatch.setattr(pipeline, "_residual_grain", record_grain)
    high = torch.rand(3, 64, 96)
    before = high.clone()
    orders = set()
    for seed in range(32):
        calls.clear()
        result = degrade_image(
            high,
            DegradationConfig(smear_strength=(0.3, 0.3), grain_strength=(0.3, 0.3), noise_strength=(1.0, 1.0)),
            seed,
        )
        orders.add(tuple(calls))
        torch.testing.assert_close(high, before, atol=0, rtol=0)
        assert result.shape == high.shape and torch.isfinite(result).all()
    assert orders == {("smooth", "grain"), ("grain", "smooth")}


def test_adaptive_smear_preserves_contrast_without_disabling_flat_texture_removal():
    from unitunner.data.degradation.pipeline import _smooth_texture

    image = np.full((96, 128, 3), 0.4, dtype=np.float32)
    image += np.random.default_rng(4).normal(0, 0.01, image.shape).astype(np.float32)
    image[:, 64:] += np.tile(np.array([0.0, 0.0, 0.15, 0.15], dtype=np.float32), 16)[None, :, None]
    uniform = _smooth_texture(image, 0.8, strength=1, detail_protection=0.0)
    adaptive = _smooth_texture(image, 0.8, strength=1, detail_protection=0.6)
    assert np.square(adaptive[:, 72:] - image[:, 72:]).mean() < np.square(uniform[:, 72:] - image[:, 72:]).mean()
    assert adaptive[8:-8, 8:56].std() < image[8:-8, 8:56].std() * 0.4
    flat = np.full((32, 32, 3), 0.4, dtype=np.float32)
    np.testing.assert_allclose(_smooth_texture(flat, 0.8, strength=1), flat, atol=1e-6)


@pytest.mark.parametrize("exposure", [(0.0, 1.0), (0.5, 1.1), (0.8, 0.5)])
def test_invalid_sensor_exposure_is_rejected(exposure):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        NoiseConfig(exposure=exposure)


def test_intermediate_compression_stays_after_blur(monkeypatch):
    from unitunner.data.degradation import pipeline

    trace = []
    original_compress, original_filter = pipeline._compress, pipeline._filter

    def encode(*args, **kwargs):
        trace.append("encode")
        return original_compress(*args, **kwargs)

    def blur(*args, **kwargs):
        trace.append("blur")
        return original_filter(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_compress", encode)
    monkeypatch.setattr(pipeline, "_filter", blur)
    cfg = DegradationConfig(
        scenario_probabilities=(0.0, 1.0, 0.0),
        smear_strength=(0.0, 0.0),
        noise_strength=(1.0, 1.0),
        grain_strength=(0.0, 0.0),
        sharpen_strength=(0.0, 0.0),
        first=StageConfig(blur_probability=1.0, noise_probability=0.0, compression_probability=1.0),
        second=SecondStageConfig(blur_probability=0.0, compression_probability=0.0, noise_probability=0.0),
        final_sinc_probability=0.0,
    )
    image = torch.rand(3, 96, 128)
    original = image.clone()
    observed = set()
    for seed in range(64):
        trace.clear()
        out = degrade_image(image, cfg, seed)
        observed.add(tuple(trace))
        assert out.shape == image.shape and torch.isfinite(out).all()
    assert observed == {("blur", "encode")}
    torch.testing.assert_close(image, original, atol=0, rtol=0)


@pytest.mark.parametrize("factor", [2.5, 4, 6, 8])
def test_coarse_grid_smoothing_preserves_more_texture(factor):
    from unitunner.data.degradation.pipeline import _smooth_texture

    image = np.full((64, 64, 3), 0.4, dtype=np.float32)
    image += np.random.default_rng(4).normal(0, 0.015, image.shape).astype(np.float32)
    baseline = _smooth_texture(image, 0.8, strength=0.7)
    limited = _smooth_texture(image, 0.8, strength=0.7, downsample_factor=factor)
    assert baseline.std() < limited.std() < image.std()
    assert np.square(limited - image).mean() < np.square(baseline - image).mean()


def test_continuous_scale_and_processing_sampling_cover_intermediate_values():
    from unitunner.data.degradation.pipeline import _sample_processing, _sample_scale

    cfg = DegradationConfig()
    rng = np.random.default_rng(37)
    scales = np.array([_sample_scale(cfg, rng) for _ in range(2048)])
    assert np.all((scales >= 1) & (scales <= 8))
    assert np.median(np.log(scales) / math.log(8)) == pytest.approx(0.5, abs=0.03)
    assert len(np.unique(scales.round(2))) > 400
    draws = np.array([_sample_processing(cfg, rng) for _ in range(2048)])
    assert np.all((draws >= 0) & (draws <= [0.8, 1.5, 4, 0.4, 1.0]))
    assert np.all(draws[:, 2] >= 1)
    assert np.corrcoef(draws[:, 0], draws[:, 1])[0, 1] < -0.1
    assert np.any((draws[:, 0] > 0.4) & (draws[:, 1] > 0.2))
    assert np.any((draws[:, 0] < 0.1) & (draws[:, 1] < 0.1))


@pytest.mark.parametrize("scale", [1.0, 2.5, 4.0, 6.0, 8.0, 11.7, 16.0])
def test_smear_layer_stays_before_final_resize_and_preserves_hq(monkeypatch, scale):
    from unitunner.data.degradation import pipeline

    grids = []
    original = pipeline._smooth_texture

    def capture(image, *args, **kwargs):
        grids.append(image.shape[:2])
        return original(image, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_smooth_texture", capture)
    high = torch.from_numpy(np.random.default_rng(15).random((3, 65, 97), dtype=np.float32))
    before = high.clone()
    cfg = DegradationConfig(
        scales=(scale,),
        grain_strength=(0.0, 0.0),
        sharpen_strength=(0.0, 0.0),
        final_sinc_probability=0.0,
        first=StageConfig(
            blur_probability=0.0,
            resize_probabilities=(0.0, 0.0, 1.0),
            noise_probability=0.0,
            compression_probability=0.0,
        ),
        second=SecondStageConfig(
            blur_probability=0.0,
            resize_probabilities=(0.0, 0.0, 1.0),
            noise_probability=0.0,
            compression_probability=0.0,
        ),
    )
    observed = set()
    for seed in range(32):
        grids.clear()
        low = degrade_image(high, cfg, seed)
        observed.update(grids)
        assert low.shape == high.shape and low.dtype == torch.float32
        assert torch.isfinite(low).all() and 0 <= low.min() <= low.max() <= 1
    expected = {(65, 97)} if scale == 1 else {(int(65 / scale), int(97 / scale))}
    assert observed == expected
    torch.testing.assert_close(high, before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale_range": (0.5, 8.0)},
        {"scale_range": (8.0, 2.0)},
        {"scale_range": (1.0, float("nan"))},
        {"scales": ()},
        {"scales": (17.0,)},
        {"grain_scale_range": (0.0, 2.0)},
        {"sharpen_strength": (0.0, 2.0)},
        {"compression_blur_probability": 0.25},
        {"processing_probabilities": (0.3, 0.45, 0.25)},
        {"grain_smear_probability": 0.45},
    ],
)
def test_invalid_or_removed_processing_settings_fail_explicitly(kwargs):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DegradationConfig(**kwargs)


def test_explicit_scale_candidates_override_continuous_range_and_roundtrip():
    from unitunner.data.degradation.pipeline import _sample_scale

    cfg = DegradationConfig(scales=(1.5, 2.75, 6.0), scale_range=(8.0, 16.0))
    restored = DegradationConfig.model_validate_json(cfg.model_dump_json())
    rng = np.random.default_rng(0)
    assert {_sample_scale(restored, rng) for _ in range(100)} == {1.5, 2.75, 6.0}


def test_defaults_merge_realesrgan_strength_with_unitunner_extensions():
    cfg = DegradationConfig()
    assert cfg.first.blur_probability == 1
    assert cfg.first.blur.sigma == (0.2, 3.0)
    assert cfg.first.resize_probabilities == (0.2, 0.7, 0.1)
    assert cfg.first.resize_range == (0.15, 1.5)
    assert cfg.first.noise_probability == 1
    assert cfg.first.noise.probabilities == (0.5, 0.5, 0.0)
    assert cfg.first.noise.gaussian_sigma == (1.0, 30.0)
    assert cfg.first.noise.poisson_scale == (0.05, 3.0)
    assert cfg.second.blur_probability == 0.8
    assert cfg.second.blur.sigma == (0.2, 1.5)
    assert cfg.second.resize_probabilities == (0.3, 0.4, 0.3)
    assert cfg.second.resize_range == (0.3, 1.2)
    assert cfg.second.noise_probability == 1
    assert cfg.second.noise.probabilities == (0.5, 0.5, 0.0)
    assert cfg.second.noise.gaussian_sigma == (1.0, 25.0)
    assert cfg.second.noise.poisson_scale == (0.05, 2.5)
    assert cfg.final_sinc_probability == 0.8
    assert cfg.noise_strength == (1.0, 1.0)
    assert cfg.first.blur.motion_probability == 0.2
    assert cfg.first.blur.defocus_probability == 0.15
    assert cfg.grain_strength == (0.0, 1.5)
    assert cfg.smear_strength == (0.0, 0.8)
    assert cfg.sharpen_strength == (0.0, 0.4)


def test_rectified_gains_cover_exact_zero_without_discontinuity_or_violating_positive_bounds():
    from unitunner.data.degradation.pipeline import _processing_gain, _sample_processing

    assert _processing_gain((0.0, 0.8), 0.2, 0.2) == 0
    assert _processing_gain((0.0, 0.8), 0.2 + 1e-8, 0.2) == pytest.approx(1e-8)
    assert _processing_gain((0.3, 0.3), 0.0, 1.0) == 0.3
    assert _processing_gain((0.0, 0.8), 1.0, 1.0) == 0
    rng = np.random.default_rng(20)
    draws = np.array([_sample_processing(DegradationConfig(), rng) for _ in range(10000)])
    no_smooth = draws[:, 0] == 0
    # Beta(1.5,2) CDF at the configured deadzone.
    threshold = DegradationConfig().processing_deadzone
    zero_mass = 2.5 * threshold**1.5 - 1.5 * threshold**2.5
    assert no_smooth.mean() == pytest.approx(zero_mass, abs=0.015)
    assert np.all(draws[:, 4] == 1)
    assert np.all(draws[:, 1] > 0)
    assert np.median(draws[:, 1]) > 0.2

    low_rng, high_rng = np.random.default_rng(91), np.random.default_rng(91)
    low_acquisition = _sample_processing(DegradationConfig(), low_rng, acquisition_level=0.0)
    high_acquisition = _sample_processing(DegradationConfig(), high_rng, acquisition_level=1.0)
    assert low_acquisition[4] == high_acquisition[4] == 1
    assert high_acquisition[1] == pytest.approx(low_acquisition[1] * 0.15)


def test_default_second_stage_preserves_late_noise_coverage(monkeypatch):
    from unitunner.data.degradation import pipeline

    grids = []

    def noise(image, *args, **kwargs):
        grids.append(image.shape[:2])
        return image

    monkeypatch.setattr(pipeline, "add_noise", noise)
    cfg = DegradationConfig(
        scales=(6.0,),
        scenario_probabilities=(1.0, 0.0, 0.0),
        noise_strength=(1.0, 1.0),
        grain_strength=(0.0, 0.0),
        smear_strength=(0.0, 0.0),
        sharpen_strength=(0.0, 0.0),
        first=StageConfig(noise_probability=0.0),
        second=SecondStageConfig(blur_probability=0.0, compression_probability=0.0),
    )
    high = torch.full((3, 192, 192), 0.5)
    for seed in range(512):
        degrade_image(high, cfg, seed)
    assert len(grids) == 512
    assert (32, 32) in grids
    assert any(shape != (32, 32) for shape in grids)


def test_final_resize_sinc_group_only_exchanges_with_compression(monkeypatch):
    from unitunner.data.degradation import pipeline

    events = []
    original_resize, original_filter = pipeline._resize, pipeline._filter

    def resize(image, height, width, *args, **kwargs):
        if image.shape[:2] != (height, width) and (height, width) == (64, 96):
            events.append("resize")
        return original_resize(image, height, width, *args, **kwargs)

    def filter_image(image, kernel):
        events.append("sinc")
        return original_filter(image, kernel)

    def compress(image, *args, **kwargs):
        events.append("compression")
        return image

    def noise(image, *args, **kwargs):
        events.append("noise")
        return image

    monkeypatch.setattr(pipeline, "_resize", resize)
    monkeypatch.setattr(pipeline, "_filter", filter_image)
    monkeypatch.setattr(pipeline, "_compress", compress)
    monkeypatch.setattr(pipeline, "add_noise", noise)
    cfg = DegradationConfig(
        scales=(4.0,),
        noise_strength=(1.0, 1.0),
        grain_strength=(0.0, 0.0),
        smear_strength=(0.0, 0.0),
        sharpen_strength=(0.0, 0.0),
        final_sinc_probability=1.0,
        first=StageConfig(
            blur_probability=0.0,
            resize_probabilities=(0.0, 0.0, 1.0),
            noise_probability=1.0,
            compression_probability=0.0,
        ),
        second=SecondStageConfig(
            blur_probability=0.0,
            resize_probabilities=(0.0, 0.0, 1.0),
            noise_probability=0.0,
            compression_probability=1.0,
        ),
    )
    observed = set()
    for seed in range(32):
        events.clear()
        degrade_image(torch.full((3, 64, 96), 0.5), cfg, seed)
        observed.add(tuple(events))
    assert observed == {
        ("noise", "resize", "sinc", "compression"),
        ("noise", "compression", "resize", "sinc"),
    }


def test_shared_zero_noise_gain_suppresses_every_noise_stage(monkeypatch):
    from unitunner.data.degradation import pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("An all-stage zero noise gain must bypass noise generation")

    monkeypatch.setattr(pipeline, "add_noise", forbidden)
    cfg = DegradationConfig(
        noise_strength=(0.0, 0.0),
        grain_strength=(1.0, 1.0),
        first=StageConfig(noise_probability=1.0),
        second=SecondStageConfig(noise_probability=1.0),
    )
    for seed in range(16):
        out = degrade_image(torch.full((3, 64, 96), 0.4), cfg, seed)
        assert torch.isfinite(out).all()


def test_zero_smoothing_also_bypasses_camera_denoising(monkeypatch):
    from unitunner.data.degradation import pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("Zero smoothing must bypass the camera denoiser as well")

    monkeypatch.setattr(pipeline, "_camera_processing", forbidden)
    cfg = DegradationConfig(
        smear_strength=(0.0, 0.0),
        noise_strength=(1.0, 1.0),
        scenario_probabilities=(0.0, 0.0, 1.0),
        first=StageConfig(noise_probability=1.0),
    )
    for seed in range(16):
        out = degrade_image(torch.full((3, 64, 96), 0.4), cfg, seed)
        assert torch.isfinite(out).all()


def test_processing_precedes_constrained_codec_group_without_extra_encode(monkeypatch):
    from unitunner.data.degradation import pipeline

    events = []
    compress, smooth = pipeline._compress, pipeline._smooth_texture

    def encode(image, *args, **kwargs):
        events.append(("encode", image.shape[:2]))
        return compress(image, *args, **kwargs)

    def denoise(image, *args, **kwargs):
        events.append(("smooth", image.shape[:2]))
        return smooth(image, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_compress", encode)
    monkeypatch.setattr(pipeline, "_smooth_texture", denoise)
    cfg = DegradationConfig(
        scales=(4.0,),
        smear_strength=(0.3, 0.3),
        scenario_probabilities=(1.0, 0.0, 0.0),
        second=SecondStageConfig(compression_probability=1.0),
    )
    observed = set()
    for seed in range(64):
        events.clear()
        degrade_image(torch.full((3, 64, 96), 0.4), cfg, seed)
        observed.add(tuple(events))
    assert all(events[0][0] == "smooth" for events in observed)
    grids = {shape for events in observed for _, shape in events}
    assert (64, 96) in grids
    assert any(height < 20 and width < 30 for height, width in grids)


def test_windowed_sinc_limits_short_kernel_step_ringing_without_removing_negative_lobes():
    # Previously this short, abruptly truncated kernel produced about 24% step
    # overshoot after normalization, exaggerated further by coarse-grid resizing.
    kernel = sinc_kernel(7, 1.8889296497771006)
    edge = kernel.sum(axis=0).cumsum()
    assert kernel.min() < 0
    assert edge.min() > -0.05 and edge.max() < 1.05
    np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-6)


@pytest.mark.parametrize("angle", [0, 0.7, 2.1])
@pytest.mark.parametrize("length,focus", [(0, 0), (10, 2), (96, 24)])
def test_handheld_psf_preserves_alignment_and_complete_support(angle, length, focus):
    kernel = _motion_kernel(7, length, angle, 0.6, focus)
    axis = np.arange(kernel.shape[0]) - kernel.shape[0] // 2
    assert np.isfinite(kernel).all() and kernel.min() >= 0
    assert kernel.sum(dtype=np.float64) == pytest.approx(1, abs=2e-7)
    assert abs((kernel * axis[None, :]).sum()) < 2e-6
    assert abs((kernel * axis[:, None]).sum()) < 2e-6
    assert np.abs(kernel[[0, -1]]).max() < 1e-12
    assert np.abs(kernel[:, [0, -1]]).max() < 1e-12


def test_defocus_broadens_handheld_psf_without_changing_centroid():
    def moment(kernel):
        axis = np.arange(kernel.shape[0]) - kernel.shape[0] // 2
        return (kernel * (axis[:, None] ** 2 + axis[None, :] ** 2)).sum()

    assert moment(_defocus_kernel(7, 0)) == 0
    assert moment(_defocus_kernel(7, 4)) > moment(_defocus_kernel(7, 1))
    assert moment(_motion_kernel(7, 10, 0.7, 0.6, 3)) > moment(_motion_kernel(7, 10, 0.7, 0.6))


@pytest.mark.parametrize(
    "active,count",
    [(("smear", "sharpen", "noise", "compression", "sinc", "resize"), 720), (("noise", "resize"), 2)],
)
def test_independent_layer_order_space_and_sampling(active, count):
    from unitunner.data.degradation.pipeline import _layer_orders

    orders = _layer_orders(active)
    assert len(orders) == len(set(orders)) == count
    for order in orders:
        assert set(order) == set(active)
    hits = np.bincount(np.random.default_rng(42).integers(count, size=count * 1000), minlength=count)
    assert hits.min() > 850 and hits.max() < 1150
    assert _layer_orders(("resize",)) == (("resize",),)


def test_correlated_parameter_draws_retain_full_range_exploration():
    from unitunner.data.degradation.pipeline import _sample

    rng = np.random.default_rng(42)
    values = np.array([_sample((2, 10), rng, 0.5) for _ in range(10000)])
    assert values.min() >= 2 and values.max() <= 10
    assert (values < 2.8).sum() > 130 and (values > 9.2).sum() > 130
    assert _sample((3, 3), rng, 0.9) == 3


@pytest.mark.parametrize(
    "field,value", [("motion_length", (0.0, 65.0)), ("defocus_radius", (0.0, 17.0)), ("motion_length", (-1.0, 4.0))]
)
def test_invalid_handheld_ranges_fail(field, value):
    with pytest.raises(ValueError):
        BlurConfig.model_validate({field: value})


def test_continuous_severity_matches_smooth_cdf_and_covers_tails():
    from unitunner.data.degradation.pipeline import _sample_severity

    rng = np.random.default_rng(42)
    values = np.array([_sample_severity(rng) for _ in range(20000)])
    assert 0 < values.min() < 0.01 and 0.99 < values.max() < 1
    np.testing.assert_allclose(
        np.histogram(values, bins=[0, 1 / 3, 2 / 3, 1])[0] / values.size, [0.2, 0.4, 0.4], atol=0.015, rtol=0
    )
    # Replaying uniforms checks inversion throughout the interval, not just bins.
    uniforms = np.random.default_rng(42).random(values.size)
    np.testing.assert_allclose(values * (0.1 + values * (1.8 - 0.9 * values)), uniforms, atol=1e-12, rtol=0)


def test_independent_scale_covers_all_severities_and_respects_explicit_choices():
    from unitunner.data.degradation.pipeline import _sample_scale, _sample_severity

    rng = np.random.default_rng(42)
    cfg = DegradationConfig()
    pairs = np.array([(_sample_severity(rng), _sample_scale(cfg, rng)) for _ in range(20000)])
    severity, log_scale = pairs[:, 0], np.log(pairs[:, 1]) / math.log(8)
    assert abs(np.corrcoef(severity, log_scale)[0, 1]) < 0.03
    for low, high in [(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1)]:
        selected = severity[(log_scale >= low) & (log_scale < high)]
        assert len(selected) > 6000
        np.testing.assert_allclose(
            np.histogram(selected, [0, 1 / 3, 2 / 3, 1])[0] / len(selected), [0.2, 0.4, 0.4], atol=0.025, rtol=0
        )
    for _ in range(20):
        assert _sample_scale(DegradationConfig(scale_range=(6.0, 6.0)), rng) == pytest.approx(6)
        assert _sample_scale(DegradationConfig(scales=(4.0, 6.0)), rng) in (4.0, 6.0)
