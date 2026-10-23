"""Validate native H3 pixel ownership, gradients, canvas boundaries and reference images."""

import numpy as np
import pytest
import torch
from PIL import Image

from tests.support.assertions import invalid_input
from unitunner.models.minimax_h3.data.media.inputs import (
    adapt_reference_canvas,
    prepare_pixels,
    read_reference_image,
    resize_frames,
    round_to_canvas_multiple,
)


def test_float_pixels_preserve_caller_storage_and_rgb_gradients():
    pixels = torch.linspace(0, 1, 2 * 5 * 7 * 4).reshape(2, 5, 7, 4).requires_grad_()
    before = pixels.detach().clone()
    output = prepare_pixels(pixels)
    assert output.shape == (1, 3, 2, 5, 7) and output.is_contiguous() and output.dtype == torch.float32
    assert torch.equal(output, (pixels[..., :3] * 2 - 1).permute(3, 0, 1, 2).unsqueeze(0))
    output.sum().backward()
    assert torch.equal(pixels.detach(), before)
    assert pixels.grad is not None
    assert torch.equal(pixels.grad[..., :3], torch.full_like(pixels.grad[..., :3], 2.0))
    assert pixels.grad is not None
    assert torch.count_nonzero(pixels.grad[..., 3]) == 0
    readonly = before.numpy()
    readonly.flags.writeable = False
    assert torch.equal(prepare_pixels(readonly), output.detach())
    assert np.array_equal(readonly, before.numpy())


def test_native_uint8_float_cast_order_and_noncontiguous_views():
    pixels = torch.arange(2 * 5 * 7 * 4).remainder(256).to(torch.uint8).reshape(2, 5, 7, 4).transpose(1, 2)
    before = pixels.clone()
    output = prepare_pixels(pixels)
    expected = (pixels[..., :3].float() / 127.5 - 1).permute(3, 0, 1, 2).unsqueeze(0)
    assert torch.equal(output, expected) and torch.equal(pixels, before)
    floating = pixels.double() / 255
    assert torch.equal(prepare_pixels(floating), (floating[..., :3].float() * 2 - 1).permute(3, 0, 1, 2).unsqueeze(0))


def test_reference_image_native_area_rounding_and_alpha_drop(tmp_path):
    pixels = np.arange(40 * 70 * 4, dtype=np.uint8).reshape(40, 70, 4)
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)
    before = path.read_bytes()
    output = read_reference_image(path, target_size=(64, 32))
    assert output.shape == (1, 32, 64, 3)
    assert torch.equal(output, resize_frames([pixels], (64, 32)))
    assert path.read_bytes() == before
    assert adapt_reference_canvas(1920, 1080) == (1344, 768)
    assert adapt_reference_canvas(1080, 1920) == (768, 1344)
    assert round_to_canvas_multiple(48.0) == 64 and round_to_canvas_multiple(80.0) == 64


def test_invalid_media_values_fail_before_pil_or_vae_io():
    for pixels in (torch.empty(0, 32, 32, 3), torch.empty(1, 32, 32, 2), torch.ones(1, 3, 3, 3, dtype=torch.int64)):
        with pytest.raises(ValueError):
            prepare_pixels(pixels)
    for value in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError, match="finite"):
            prepare_pixels(torch.full((1, 3, 3, 3), value))
    with pytest.raises(ValueError, match="at least one"):
        resize_frames([], (32, 32))
    with pytest.raises(ValueError, match="uint8"):
        resize_frames(invalid_input([np.zeros((1, 1, 3), dtype=np.float32)]), (32, 32))
    for size in ((0, 32), (32, -1), (True, 32)):
        with pytest.raises(ValueError, match="positive integer"):
            adapt_reference_canvas(*size)
