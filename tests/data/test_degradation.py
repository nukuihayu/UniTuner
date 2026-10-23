"""Check real degradation, interpolation coordinates, codecs and global-state isolation."""

import colorsys
import math
import random

import numpy as np
import pytest
import torch

pytest.importorskip("cv2")
from unitunner.data.degradation.color import adjust_hue
from unitunner.data.degradation.config import BlurConfig, DegradationConfig, NoiseConfig, SecondStageConfig, StageConfig
from unitunner.data.degradation.pipeline import (
    _optical_kernel,
    _resize,
    add_noise,
    compress_image,
    degrade_image,
    mixed_blur_kernel,
    sinc_kernel,
)


def test_kernels_preserve_dc_symmetry_and_sinc_negative_lobes():
    rng = np.random.default_rng(42)
    for _ in range(32):
        kernel = mixed_blur_kernel(rng, BlurConfig(), 512)
        assert np.isfinite(kernel).all()
        assert kernel.sum() == pytest.approx(1, abs=2e-7)
        np.testing.assert_allclose(kernel, kernel[::-1, ::-1], atol=1e-7)
    kernel = sinc_kernel(21, math.pi / 2)
    assert kernel.min() < 0
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-7)
    assert kernel.sum() == pytest.approx(1, abs=2e-7)


@pytest.mark.parametrize("scale", [1, 4, 16])
@pytest.mark.parametrize("scene", [0, 1, 2])
@pytest.mark.parametrize("processing", [0, 1, 2])
def test_degradation_local_rng_finite_range_and_unchanged_hq(scale, scene, processing):
    high = torch.from_numpy(np.random.default_rng(42).random((3, 64, 80), dtype=np.float32))
    original = high.clone()
    before_torch, before_numpy, before_python = torch.get_rng_state(), np.random.get_state(), random.getstate()
    settings = DegradationConfig(
        scales=(scale,),
        processing_probabilities=tuple(float(i == processing) for i in range(3)),
        color_jitter_probability=1.0,
        gray_probability=0.5,
        scenario_probabilities=tuple(float(i == scene) for i in range(3)),
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


def test_fractional_resize_uses_scale_coordinates_and_hue_matches_scalar_colorsys():
    ramp = np.broadcast_to(np.arange(7, dtype=np.float32)[None, :, None], (5, 7, 3)).copy()
    actual = _resize(ramp, 3, 4, np.random.default_rng(0), DegradationConfig(resize_modes=("bilinear",)), factor=0.6)
    # Integrated triangle weights at fractional source coordinates, including boundaries.
    expected = np.array([3 / 7, 2, 3.75, 5.25], dtype=np.float32)
    np.testing.assert_allclose(actual[1, :, 0], expected, rtol=1e-6, atol=1e-6)
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


@pytest.mark.parametrize("resolution", [512, 768, 1024, 1536])
def test_spatial_blur_tracks_image_scale_without_changing_sinc(resolution):
    gaussian = BlurConfig(
        sigma=(1.0, 1.0),
        probabilities=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        sinc_probability=0.0,
        motion_probability=0.0,
        defocus_probability=0.0,
    )
    kernel = mixed_blur_kernel(np.random.default_rng(23), gaussian, resolution)
    axis = np.arange(kernel.shape[0]) - kernel.shape[0] // 2
    measured_sigma = np.sqrt((kernel * axis[None, :] ** 2).sum())
    assert measured_sigma == pytest.approx(math.sqrt(resolution / 1024), rel=0.08)
    assert kernel.shape[0] % 2 == 1
    assert kernel.sum() == pytest.approx(1, abs=1e-6)
    sinc = BlurConfig(sinc_probability=1.0, motion_probability=0.0, defocus_probability=0.0)
    np.testing.assert_array_equal(
        mixed_blur_kernel(np.random.default_rng(23), sinc, resolution),
        mixed_blur_kernel(np.random.default_rng(23), sinc, 1024),
    )


def test_acquisition_blur_uses_short_edge_and_delivery_blur_uses_local_pixels(monkeypatch):
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
            DegradationConfig(first=StageConfig(blur_probability=1.0), second=SecondStageConfig(blur_probability=1.0)),
            7,
        )
        assert result.shape == image.shape
    assert resolutions == [64, 1024, 64, 1024]


@pytest.mark.parametrize("motion", [False, True])
def test_optical_exposure_kernels_are_centered_and_preserve_energy(motion):
    kernel = _optical_kernel(31, 2, 0, motion=motion)
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


def test_camera_noise_matches_shot_read_variance_in_linear_light():
    settings = NoiseConfig(
        probabilities=(0.0, 0.0, 1.0),
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
        assert float(decoded.var()) == pytest.approx(linear * 0.001 + 0.00001, rel=0.02)


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
        cameras.append(kwargs.get("camera"))
        return original_noise(image, settings, rng, **kwargs)

    monkeypatch.setattr(pipeline, "_compress", encode)
    monkeypatch.setattr(pipeline, "add_noise", noise)
    settings = DegradationConfig(
        scales=(4,),
        scenario_probabilities=tuple(float(i == scene) for i in range(3)),
        processing_probabilities=(1.0, 0.0, 0.0),
        first=StageConfig(noise_probability=1.0, compression_probability=1.0),
        second=SecondStageConfig(noise_probability=1.0, compression_probability=1.0),
    )
    degrade_image(torch.full((3, 128, 192), 0.5), settings, 7)
    assert len(codecs) == (2 if scene == 1 else 1)
    assert codecs[-1] == (32, 48)
    assert cameras == [scene == 2, False]


def test_degradation_rejects_invalid_scene_and_optical_probabilities():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="sum to one"):
        DegradationConfig(scenario_probabilities=(0.5, 0.4, 0.4))
    with pytest.raises(ValidationError, match="at most one"):
        BlurConfig(motion_probability=0.8, defocus_probability=0.8)


def test_scale_sampling_is_uniform_without_favoring_the_largest_factor(monkeypatch):
    from collections import Counter

    from unitunner.data.degradation import pipeline

    counts = Counter()
    scales = (2, 3, 4, 8)
    shapes = {(64 // scale, 96 // scale): scale for scale in scales}

    def capture(image, settings, rng, severity):
        counts[shapes[image.shape[:2]]] += 1
        assert 0 <= severity <= 0.5
        return image

    monkeypatch.setattr(pipeline, "_compress", capture)
    settings = DegradationConfig(
        scales=scales,
        scenario_probabilities=(1.0, 0.0, 0.0),
        processing_probabilities=(1.0, 0.0, 0.0),
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


def test_phone_processing_retains_grain_or_suppresses_texture_without_flattening_edges():
    from unitunner.data.degradation.pipeline import _processed_photo

    flat = np.full((128, 128, 3), 0.4, dtype=np.float32)
    grainy = _processed_photo(flat, NoiseConfig(), np.random.default_rng(7), 0.8, grainy=True)
    assert grainy.std() > 0.005
    assert abs(grainy.mean() - flat.mean()) < 0.01
    texture = flat.copy()
    texture[:, 64:] = 0.7
    texture += np.random.default_rng(8).normal(0, 0.015, texture.shape).astype(np.float32)
    smoothed = _processed_photo(texture, NoiseConfig(), np.random.default_rng(7), 0.8, grainy=False)
    assert smoothed[8:-8, 8:56].std() < texture[8:-8, 8:56].std() * 0.4
    assert smoothed[:, 68:].mean() - smoothed[:, :60].mean() > 0.25


@pytest.mark.parametrize("grainy", [False, True])
def test_phone_processing_is_encoded_on_delivery_grid(monkeypatch, grainy):
    from unitunner.data.degradation import pipeline

    encoded = []
    original = pipeline._compress

    def capture(image, settings, rng, severity):
        encoded.append(image.copy())
        return original(image, settings, rng, severity)

    monkeypatch.setattr(pipeline, "_compress", capture)
    cfg = DegradationConfig(
        scales=(4,),
        scenario_probabilities=(1.0, 0.0, 0.0),
        processing_probabilities=(0.0, float(grainy), float(not grainy)),
        second=SecondStageConfig(compression_probability=1.0),
    )
    low = degrade_image(torch.full((3, 128, 192), 0.4), cfg, 23)
    assert encoded[-1].shape[:2] == ((64, 96) if grainy else (32, 48))
    assert low.shape == (3, 128, 192)


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
