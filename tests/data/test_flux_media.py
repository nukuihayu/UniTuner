"""Check native inference alpha-removal order independently of cache image semantics."""

import numpy as np
import pytest
import torch
from PIL import Image

from unitunner.data.media.image import read_image
from unitunner.models.flux.data.media import read_fill_condition, read_fill_image, read_reference_image


def test_flux_reference_alpha_is_removed_before_resizing_and_fill_masks_stay_aligned(tmp_path):
    rgb = np.arange(16 * 32 * 3, dtype=np.uint8).reshape(16, 32, 3)
    alpha = ((np.indices((16, 32))[1] % 2) * 255).astype(np.uint8)
    for name, pixels in (("rgb", rgb), ("rgba", np.concatenate((rgb, alpha[..., None]), axis=-1)), ("mask", alpha)):
        with Image.fromarray(pixels) as image:
            image.save(tmp_path / f"{name}.png")
    reference = read_reference_image(tmp_path / "rgb.png", preferred=True)
    actual = read_reference_image(tmp_path / "rgba.png", preferred=True)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    original = read_reference_image(tmp_path / "rgba.png", preferred=False)
    torch.testing.assert_close(original, torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1, atol=0, rtol=0)
    # Alpha and an equivalent L mask share geometry and RGB semantics.
    pixels, mask = read_fill_image(tmp_path / "rgba.png", None, (16, 32))
    explicit_pixels, explicit_mask = read_fill_image(tmp_path / "rgb.png", tmp_path / "mask.png", (16, 32))
    torch.testing.assert_close(pixels, explicit_pixels, atol=0, rtol=0)
    torch.testing.assert_close(mask, explicit_mask, atol=0, rtol=0)
    assert pixels.shape == (3, 32, 16) and mask.shape == (1, 32, 16)
    assert torch.all((mask == 0) | (mask == 1))
    assert torch.all(pixels.masked_select(mask.expand_as(pixels) == 1) == 0)


@pytest.mark.parametrize("size", [(257, 513), (513, 257), (2049, 1537)])
@pytest.mark.parametrize("resolution", [(1024, 1024), (512, 1024)])
def test_fill_training_and_inference_share_crop_without_alpha_contamination(tmp_path, size, resolution):
    width, height = size
    rgb = np.full((height, width, 3), (191, 97, 43), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 4 : height * 3 // 4, width // 3 : width * 2 // 3] = 255
    for name, array in [("target", rgb), ("source", np.dstack((rgb, mask))), ("mask", mask)]:
        Image.fromarray(array).save(tmp_path / f"{name}.png")
    target, source, explicit = (tmp_path / f"{name}.png" for name in ("target", "source", "mask"))
    actual, region = read_fill_condition(target, source, None, resolution)
    expected = read_image(target, resolution)
    assert actual.shape == expected.shape == (3, resolution[1], resolution[0])
    assert region.shape == (1, resolution[1], resolution[0])
    assert set(region.unique().tolist()) == {0, 1}
    torch.testing.assert_close(actual, expected * (1 - region), atol=0, rtol=0)
    for output, output_mask in (
        read_fill_condition(target, target, explicit, resolution),
        read_fill_image(source, None, resolution),
        read_fill_image(target, explicit, resolution),
    ):
        torch.testing.assert_close(actual, output, atol=0, rtol=0)
        torch.testing.assert_close(region, output_mask, atol=0, rtol=0)
    # Independent analytical center-crop landmark bounds, allowing nearest's
    # half-pixel rounding, detect shifting or stretching of the mask.
    scale = max(resolution[0] / width, resolution[1] / height)
    rw, rh = int(width * scale + 0.5), int(height * scale + 0.5)
    ys, xs = torch.where(region[0] > 0)
    left = max(0, width // 3 * rw / width - (rw - resolution[0]) // 2)
    top = max(0, height // 4 * rh / height - (rh - resolution[1]) // 2)
    assert abs(xs.min().item() - left) <= 1
    assert abs(ys.min().item() - top) <= 1
