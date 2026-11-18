"""Check FLUX3 EXIF/RGB reference geometry and reject empty multi-reference crops."""

import numpy as np
import pytest
import torch
from PIL import Image

from unituner.models.flux3.data.media import read_reference_images, reference_canvas


def test_flux3_reference_orientation_alpha_and_single_padding(tmp_path):
    pixels = np.arange(19 * 23 * 4, dtype=np.uint8).reshape(19, 23, 4)
    path = tmp_path / "image.png"
    exif = Image.Exif()
    exif[274] = 6
    with Image.fromarray(pixels) as image:
        image.save(path, exif=exif)
    (result,) = read_reference_images([path])
    assert result.pixels.shape == (3, 64, 64)
    assert result.crop_box == (22, 20, 41, 43)
    oriented = np.rot90(pixels[:, :, :3], k=3).copy()
    expected = np.pad(oriented, ((20, 21), (22, 23), (0, 0)), mode="reflect")
    torch.testing.assert_close(
        result.pixels, torch.from_numpy(expected).permute(2, 0, 1).float() / 127.5 - 1, atol=0, rtol=0
    )
    tiny = tmp_path / "tiny.png"
    with Image.new("RGB", (1, 3), (13, 27, 31)) as image:
        image.save(tiny)
    (actual,) = read_reference_images([tiny])
    assert actual.pixels.shape == (3, 64, 64)
    assert torch.isfinite(actual.pixels).all()
    assert read_reference_images([]) == ()
    with pytest.raises(ValueError, match=">=16"):
        read_reference_images([tiny, path])
    with pytest.raises(ValueError, match="at most ten"):
        read_reference_images([path] * 11)


def test_flux3_multi_reference_center_crop_and_canvas_budget(tmp_path):
    path = tmp_path / "image.png"
    pixels = np.arange(35 * 49 * 3, dtype=np.uint8).reshape(35, 49, 3)
    with Image.fromarray(pixels) as image:
        image.save(path)
    first, second = read_reference_images([path, path])
    assert first.crop_box is second.crop_box is None
    expected = torch.from_numpy(pixels[1:33, :48].copy()).permute(2, 0, 1).float() / 127.5 - 1
    torch.testing.assert_close(first.pixels, expected, atol=0, rtol=0)
    torch.testing.assert_close(first.pixels, second.pixels, atol=0, rtol=0)
    for width, height in ((10000, 10000), (10000, 317), (2049, 2049), (1, 1), (4096, 1025)):
        fit, canvas = reference_canvas(width, height)
        assert min(canvas) >= 64 and max(canvas) <= 4096
        assert canvas[0] * canvas[1] <= 4 * 1024 * 1024
        assert all(v % 16 == 0 for v in canvas)
        assert fit[0] <= width and fit[1] <= height
