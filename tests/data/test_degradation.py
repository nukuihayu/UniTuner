"""Check official stage semantics, spatial adaptation and optional extensions."""

import math
import random

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.nn import functional as F

pytest.importorskip("cv2")
from unitunner.data.degradation.config import (
    BlurConfig,
    DegradationConfig,
    NoiseConfig,
    SecondStageConfig,
    StageConfig,
)
from unitunner.data.degradation.pipeline import (
    _resize,
    _resize_scale,
    _resolution_scale,
    add_noise,
    compress_image,
    degrade_image,
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
    assert trace[-1]["op"] == "alignment"


@pytest.mark.parametrize("scale", [2, 4, 6, 8, 16])
def test_native_lq_quantized_before_deterministic_alignment(scale):
    high = torch.from_numpy(np.random.default_rng(4).random((3, 129, 193), dtype=np.float32))
    cfg = DegradationConfig(scales=(float(scale),), output_mode="native")
    low = degrade_image(high, cfg, 29)
    assert low.shape == (3, 129 // scale, 193 // scale)
    torch.testing.assert_close(low * 255, (low * 255).round(), atol=2e-5, rtol=0)
    expected = F.interpolate(low[None], size=(129, 193), mode="bicubic", align_corners=False)[0].clamp(0, 1)
    aligned = degrade_image(high, cfg.model_copy(update={"output_mode": "aligned"}), 29)
    torch.testing.assert_close(aligned, expected, atol=0, rtol=0)


def test_fixed_two_stage_order_and_native_delivery_both_branches():
    cfg = DegradationConfig(
        scales=(4.0,),
        use_usm=False,
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
        ("delivery_resize", "final_sinc", "compression", "native_lq", "alignment"),
        ("compression", "delivery_resize", "final_sinc", "native_lq", "alignment"),
    }


def test_native_resize_matches_independent_torch_reference_without_damage():
    cfg = DegradationConfig(
        scales=(4.0,),
        use_usm=False,
        output_mode="native",
        final_sinc_probability=0,
        resize_modes=("bilinear",),
        first=StageConfig(blur_probability=0, noise_probability=0, compression_probability=0),
        second=SecondStageConfig(blur_probability=0, noise_probability=0, compression_probability=0),
    )
    high = torch.from_numpy(np.random.default_rng(3).random((3, 128, 192), dtype=np.float32))
    trace = []
    actual = degrade_image(high, cfg, 8, trace=trace)
    reference = high[None]
    for event in trace:
        if event["op"] in ("resize", "delivery_resize"):
            if event.get("stage") == 1:
                reference = F.interpolate(reference, scale_factor=event["factor"], mode="bilinear", align_corners=False)
            else:
                reference = F.interpolate(reference, size=event["shape"], mode="bilinear", align_corners=False)
    reference = (reference[0] * 255).round().clamp(0, 255) / 255
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("resolution,expected", [(100, 0.5), (400, 1), (1024, 1.6), (1280, math.sqrt(3.2)), (4096, 2)])
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
        use_usm=False,
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


@pytest.mark.parametrize("codec,quality", [("jpeg", 95), ("webp", 95), ("jpeg2000", 900)])
def test_codecs_preserve_rgb_and_tiny_grids(codec, quality):
    source = np.full((16, 24, 3), [0.8, 0.3, 0.1], dtype=np.float32)
    result = compress_image(source, codec, quality)
    assert result.shape == source.shape and result.dtype == np.float32
    np.testing.assert_allclose(result.mean((0, 1)), source.mean((0, 1)), atol=0.025)


def test_grain_is_exclusive_gray_correlated_and_bounded():
    source = np.full((128, 128, 3), 0.5, dtype=np.float32)
    cfg = NoiseConfig(grain_probability=1, grain_sigma=(4.0, 4.0), grain_correlation=(0.8, 0.8))
    trace = []
    result = add_noise(source, cfg, np.random.default_rng(0), trace=trace)
    residual = result - source
    assert trace[0]["kind"] == "grain" and len(trace) == 1
    np.testing.assert_array_equal(residual[..., 0], residual[..., 1])
    assert residual.std() * 255 == pytest.approx(4, rel=0.01)
    assert abs(residual.mean()) < 1e-6
    assert np.corrcoef(residual[1:, :, 0].ravel(), residual[:-1, :, 0].ravel())[0, 1] > 0.05


def test_noise_mixture_frequency():
    counts = {"grain": 0, "gaussian": 0, "poisson": 0}
    rng = np.random.default_rng(1)
    for _ in range(2000):
        trace = []
        add_noise(np.full((8, 8, 3), 0.5, dtype=np.float32), NoiseConfig(), rng, trace=trace)
        counts[trace[0]["kind"]] += 1
    assert counts["grain"] / 2000 == pytest.approx(0.05, abs=0.015)
    assert counts["gaussian"] / 2000 == pytest.approx(0.475, abs=0.035)
    assert counts["poisson"] / 2000 == pytest.approx(0.475, abs=0.035)


def test_noise_changes_do_not_change_geometry_or_codec_parameters():
    cfg = DegradationConfig(scales=(4.0,))
    data = cfg.model_dump()
    for stage in ("first", "second"):
        data[stage]["noise"]["grain_probability"] = 1.0
    grain_cfg = DegradationConfig.model_validate(data)
    source = torch.from_numpy(np.random.default_rng(4).random((3, 96, 144), dtype=np.float32))
    for seed in range(12):
        original, grain = [], []
        degrade_image(source, cfg, seed, trace=original)
        degrade_image(source, grain_cfg, seed, trace=grain)
        assert [e for e in original if e["op"] != "noise"] == [e for e in grain if e["op"] != "noise"]


def test_reference_kernel_does_not_depend_on_sr_factor():
    traces = []
    for scale in (2.0, 4.0, 8.0):
        trace = []
        cfg = DegradationConfig(scales=(scale,), resolution_exponent=0, first=StageConfig(sinc_probability=0))
        degrade_image(torch.full((3, 128, 192), 0.5), cfg, 17, trace=trace)
        traces.append(next(e["kernel"] for e in trace if e["op"] == "blur"))
    assert traces[0] == traces[1] == traces[2]


@pytest.mark.parametrize("scale", [2.0, 4.0, 6.0, 8.0, 16.0])
def test_high_scale_protection_preserves_x4_and_limits_secondary_downsampling(scale):
    source = torch.from_numpy(np.random.default_rng(1).random((3, 128, 192), dtype=np.float32))
    cfg = DegradationConfig(scales=(scale,))
    before, after = [], []
    old = degrade_image(source, cfg.model_copy(update={"high_scale_protection": False}), 13, trace=before)
    new = degrade_image(source, cfg, 13, trace=after)
    if scale <= 4:
        torch.testing.assert_close(old, new, atol=0, rtol=0)
        assert before == after
    else:
        resize = next(e for e in after if e["op"] == "resize" and e.get("stage") == 2)
        assert resize["factor"] >= 1 - 0.7 * 4 / scale
        previous = next(e for e in before if e["op"] == "resize" and e.get("stage") == 2)
        assert resize["shape"][0] > previous["shape"][0]
        assert [(e["kind"], e["parameter"]) for e in before if e["op"] == "noise"] == [
            (e["kind"], e["parameter"]) for e in after if e["op"] == "noise"
        ]


def test_gaussian_noise_uses_full_official_sigma():
    cfg = NoiseConfig(grain_probability=0, gaussian_probability=1, gaussian_sigma=(10.0, 10.0), gray_probability=1)
    result = add_noise(np.full((256, 256, 3), 0.5, dtype=np.float32), cfg, np.random.default_rng(1))
    assert result.std() * 255 == pytest.approx(10, rel=0.02)
    np.testing.assert_array_equal(result[..., 0], result[..., 1])


def test_optional_second_order_skip_keeps_final_lq_delivery():
    trace = []
    cfg = DegradationConfig(scales=(4.0,), second_order_probability=0)
    degrade_image(torch.full((3, 64, 96), 0.5), cfg, 7, trace=trace)
    assert len([e for e in trace if e["op"] == "noise"]) == 1
    assert next(e for e in trace if e["op"] == "native_lq")["shape"] == [16, 24]


@pytest.mark.parametrize("legacy", ["smear_strength", "grain_strength", "noise_strength", "late_noise_probability"])
def test_removed_heuristics_fail_instead_of_silently_applying(legacy):
    with pytest.raises(ValidationError):
        DegradationConfig(**{legacy: 0.5})


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
        torch.testing.assert_close(high, expected * 2 - 1, atol=0, rtol=0)
        expected_low = degrade_image(expected, settings, 19 + index * 1_000_003) if settings else expected
        torch.testing.assert_close(low, expected_low * 2 - 1, atol=0, rtol=0)
    assert torch.equal(before, torch.get_rng_state())
