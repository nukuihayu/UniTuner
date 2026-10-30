"""Audit paired mask landmarks and independent reference geometry at training sizes."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from unitunner.data.media.buckets import ImageBuckets
from unitunner.data.media.image import resize_pixels
from unitunner.models.qwen_image.data.media import reference_resolution

SIZES = [(600, 1200), (1200, 600), (900, 1200), (720, 1280), (1280, 720), (257, 513), (2049, 1537)]


def landmark(width, height):
    """A centered circle exposes anisotropic stretching; a nonzero border exposes padding."""
    image = Image.new("RGB", (width, height), (31, 73, 109))
    radius = min(width, height) // 6
    ImageDraw.Draw(image).ellipse(
        (width // 2 - radius, height // 2 - radius, width // 2 + radius, height // 2 + radius),
        fill=(255, 73, 109),
    )
    return image


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("bucketed", [False, True])
def test_paired_original_and_grayscale_mask_landmarks(tmp_path, size, bucketed):
    image = landmark(*size)
    path = tmp_path / "target.png"
    image.save(path)
    resolution = ImageBuckets((1024, 1024), alignment=32, enabled=bucketed).for_image(path)
    mask = image.getchannel("R").convert("RGBA")
    target = resize_pixels(np.array(image.convert("RGBA")), resolution)
    original = resize_pixels(np.array(image.convert("RGBA")), resolution)
    region = resize_pixels(np.array(mask), resolution)
    assert target.shape == original.shape == region.shape == (resolution[1], resolution[0], 4)
    assert np.array_equal(target, original)
    assert np.array_equal(target[..., 0], region[..., 0])


@pytest.mark.parametrize("size", SIZES)
def test_independent_reference_preserves_circle_and_aspect(size):
    width, height = size
    resolution = reference_resolution(width, height, area=1024**2, alignment=32)
    output = resize_pixels(np.array(landmark(width, height).convert("RGBA")), resolution)
    assert output.shape == (resolution[1], resolution[0], 4)
    assert all(value % 32 == 0 for value in resolution)
    assert resolution[0] * resolution[1] <= 1024**2
    assert (resolution[0] > resolution[1]) == (width > height)
    assert abs((resolution[0] / resolution[1]) / (width / height) - 1) < 0.05
    # Neither black nor transparent padding is introduced.
    assert np.all(output[[0, -1], :, :3] == (31, 73, 109))
    assert np.all(output[:, [0, -1], :3] == (31, 73, 109))
    assert output[..., 3].min() == 255
    ys, xs = np.where(output[..., 0] > 200)
    assert abs((xs.max() - xs.min()) - (ys.max() - ys.min())) <= 2
