"""Check native inference alpha-removal order independently of cache image semantics."""

import numpy as np
import torch
from PIL import Image

from unitunner.models.flux.data.media import read_fill_image, read_reference_image


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
    # Inference resizes both paths directly; alpha and an equivalent L mask agree.
    pixels, mask = read_fill_image(tmp_path / "rgba.png", None, (16, 32))
    explicit_pixels, explicit_mask = read_fill_image(tmp_path / "rgb.png", tmp_path / "mask.png", (16, 32))
    torch.testing.assert_close(pixels, explicit_pixels, atol=0, rtol=0)
    torch.testing.assert_close(mask, explicit_mask, atol=0, rtol=0)
    assert pixels.shape == (3, 32, 16) and mask.shape == (1, 32, 16)
    assert torch.all((mask == 0) | (mask == 1))
    assert torch.all(pixels.masked_select(mask.expand_as(pixels) == 1) == 0)
