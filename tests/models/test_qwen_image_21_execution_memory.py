"""Preserve Qwen2.1 gradients while sharing per-invocation geometry."""

from copy import deepcopy
from typing import cast

import pytest
import torch

from tests.models.test_qwen_image_21 import small_config
from unituner.models.qwen_image_2_1.layers import QwenImage21Block
from unituner.models.qwen_image_2_1.sequence import target_tokens
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("precision", ("fp32", "bf16"))
def test_shared_visibility_matches_per_block_masks_and_gradients(device, precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(614)
    model = QwenImage21Transformer(small_config()).to(device)
    reference = deepcopy(model)
    plans = []
    handles = [
        block.register_forward_pre_hook(lambda _block, args: plans.append(args[-1]))
        for block in model.transformer_blocks
    ]
    for block in reference.transformer_blocks:
        # Independent oracle retains mask construction inside every block.
        block.forward = lambda values, modulation, rotary, segments, mask, _plan, block=block: cast(
            QwenImage21Block, block
        ).transform(values, modulation, rotary, segments, mask)
    target = torch.randn(2, 6, 8, device=device)
    text = torch.randn(2, 5, 12, device=device)
    image = torch.randn(2, 4, 8, device=device)
    outputs, gradients = [], []
    try:
        for current in (model, reference):
            inputs = [value.clone().requires_grad_() for value in (target, text, image)]
            with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == "bf16"):
                output = current(
                    inputs[0],
                    inputs[1],
                    torch.tensor([0.2, 0.7], device=device),
                    (((1, 2, 2), (1, 2, 3)),),
                    text_lengths=(5, 4),
                    references=(inputs[2],),
                    image_slots=((2,), (1,)),
                )
            outputs.append(output)
            gradients.append(torch.autograd.grad(output.float().square().mean(), (*inputs, *current.parameters())))
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        for actual, expected in zip(*gradients, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert len(plans) == len(model.transformer_blocks)
        assert all(plan is plans[0] for plan in plans)
        assert plans[0] and all(mask.dtype == torch.bool for mask in plans[0])
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("offsets", ((2, 2), (1, 3)))
def test_target_spans_keep_training_storage_compact_and_borrow_uniform_inference(offsets):
    values = torch.randn(2, 9, 8, requires_grad=True)
    expected = torch.stack([values[row, offset : offset + 4] for row, offset in enumerate(offsets)])
    actual = target_tokens(values, offsets, 4)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    observed = torch.autograd.grad(actual, values, upstream)[0]
    reference = torch.autograd.grad(expected, values, upstream)[0]
    torch.testing.assert_close(observed, reference, atol=0, rtol=0)
    assert actual.untyped_storage().data_ptr() != values.untyped_storage().data_ptr()
    if offsets[0] == offsets[1]:
        with torch.no_grad():
            inference = target_tokens(values, offsets, 4)
        torch.testing.assert_close(inference, expected, atol=0, rtol=0)
        assert inference.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
