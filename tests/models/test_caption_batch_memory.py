"""Keep caption batching independent and differentiable without redundant pads."""

import pytest
import torch

from unituner.models.components.conditioning import batch_captions


@pytest.mark.parametrize("lengths", [(3,), (3, 3), (3, 5)])
def test_caption_batch_values_storage_and_gradient(lengths):
    rows = [torch.randn(length, 8, requires_grad=True) for length in lengths]
    caption = batch_captions(rows)
    assert caption.lengths == lengths
    assert caption.features.shape == (len(rows), max(lengths), 8)
    for index, row in enumerate(rows):
        torch.testing.assert_close(caption.features[index, : len(row)], row)
        assert torch.count_nonzero(caption.features[index, len(row) :]) == 0
        assert caption.features.untyped_storage().data_ptr() != row.untyped_storage().data_ptr()
    caption.features.square().sum().backward()
    for row in rows:
        torch.testing.assert_close(row.grad, 2 * row.detach())
