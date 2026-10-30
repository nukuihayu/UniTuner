"""Verify Lanczos image preprocessing and discrete mask interpolation."""

import numpy as np
import pytest
import torch
from PIL import Image

from unituner.data.media.image import read_image, resize_pixels


@pytest.mark.parametrize("source_size", [(96, 48), (24, 12)])
@pytest.mark.parametrize("channels", [3, 4])
def test_resize_and_decode_use_lanczos_for_both_scales(tmp_path, source_size, channels):
    width, height = source_size
    pixels = np.random.default_rng(15).integers(0, 256, (height, width, channels), dtype=np.uint8)
    original = pixels.copy()
    with Image.fromarray(pixels) as image, image.resize((64, 32), Image.Resampling.LANCZOS) as resized:
        expected = np.array(resized)[:, 16:48].copy()
    actual = resize_pixels(pixels, (32, 32))
    assert np.array_equal(actual, expected)
    assert np.array_equal(pixels, original) and not np.shares_memory(actual, pixels)
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)
    decoded = read_image(path, (32, 32), channels=channels)
    torch.testing.assert_close(decoded, torch.from_numpy(expected).permute(2, 0, 1).float() / 127.5 - 1, atol=0, rtol=0)


def test_mask_resampling_preserves_discrete_labels():
    pixels = np.zeros((48, 96, 3), dtype=np.uint8)
    pixels[:, 47:] = 255
    actual = resize_pixels(pixels, (32, 32), resample="mask")
    assert actual.shape == (32, 32, 3)
    assert set(np.unique(actual)) == {0, 255}
