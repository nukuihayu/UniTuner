"""Keep restoration pixels exact while avoiding redundant full-image allocations."""

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from unitunner.data.degradation.config import DegradationConfig
from unitunner.data.media import restoration


@pytest.mark.parametrize("mode", ("RGB", "RGBA", "L"))
@pytest.mark.parametrize("orientation", (1, 6, 8))
def test_restoration_rgb_matches_exif_and_color_conversion(tmp_path, mode, orientation):
    rgb = np.random.default_rng(42).integers(0, 256, (13, 19, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    exif = Image.Exif()
    exif[274] = orientation
    with Image.fromarray(rgb).convert(mode) as image:
        image.save(path, exif=exif)
    with Image.open(path) as image, ImageOps.exif_transpose(image) as oriented, oriented.convert("RGB") as image_rgb:
        expected = np.array(image_rgb, copy=True)
    np.testing.assert_array_equal(restoration._read_rgb(path), expected)
    actual, size = restoration.read_restoration_image(path, granularity=1)
    assert size == (expected.shape[1], expected.shape[0])
    pixels = torch.from_numpy(expected).permute(2, 0, 1).contiguous().float() / 127.5 - 1
    torch.testing.assert_close(actual, pixels, rtol=0, atol=0)


@pytest.mark.parametrize("synthetic", (False, True))
@pytest.mark.parametrize("size", (9, 64))
def test_pad_only_when_required_and_only_once_for_synthetic_hq(tmp_path, monkeypatch, synthetic, size):
    path = tmp_path / "image.png"
    Image.fromarray(np.full((size, size, 3), 127, dtype=np.uint8)).save(path)
    calls = []
    original = np.pad

    def pad(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(restoration.np, "pad", pad)
    monkeypatch.setattr(restoration, "degrade_image", lambda high, *args: high.clone())
    low, high = restoration.read_aligned_images(
        None if synthetic else path,
        path,
        resolution=(32, 32),
        seed=42,
        degradation=DegradationConfig(enabled=True) if synthetic else None,
    )
    assert len(calls) == (0 if size >= 32 else 1 if synthetic else 2)
    torch.testing.assert_close(low, high, rtol=0, atol=0)
