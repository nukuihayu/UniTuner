"""Verify the Torch training contract and explicit inference-only Kitchen selection."""

import pytest
import torch
from torch.nn import functional as F

from unitunner.execution.attention import inference_attention, scaled_dot_product_attention


def test_default_attention_keeps_training_gradients():
    torch.manual_seed(18)
    values = [torch.randn(2, heads, 7, 8, requires_grad=True) for heads in (4, 2, 2)]
    expected = F.scaled_dot_product_attention(values[0], values[1], values[2], enable_gqa=True)
    observed = scaled_dot_product_attention(values[0], values[1], values[2], enable_gqa=True)
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    a = torch.autograd.grad(expected.square().sum(), values)
    b = torch.autograd.grad(observed.square().sum(), values)
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_fullgraph_attention_context_is_restored_and_inference_policy_is_not_cached():
    from unitunner.execution import attention
    from unitunner.execution.compiler import CompilerInstallation

    class Model(torch.nn.Module):
        def attend(self, q, k, v):
            return scaled_dot_product_attention(q, k, v)

        forward = attend

    model = Model()
    values = [torch.randn(1, 2, 5, 8, requires_grad=True) for _ in range(3)]
    owner = CompilerInstallation(model, backend="eager", fullgraph=True)
    try:
        expected = F.scaled_dot_product_attention(values[0], values[1], values[2])
        actual = model(*values)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert not attention._compiled.active
        # Even after native attention was compiled, entering an inference-only
        # backend must reject training rather than silently reuse that graph.
        token = attention._backend.set(lambda *args, **kwargs: torch.zeros_like(args[0]))
        try:
            with pytest.raises(ValueError, match="inference-only"):
                model(*values)
            assert not attention._compiled.active
            with torch.no_grad():
                assert not torch.count_nonzero(model(*values))
        finally:
            attention._backend.reset(token)
        torch.testing.assert_close(model(*values), expected, atol=0, rtol=0)
        for left, right in zip(
            torch.autograd.grad(actual.sum(), values), torch.autograd.grad(expected.sum(), values), strict=True
        ):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    finally:
        owner.close()


def test_kitchen_matches_selected_inference_provider_and_restores_context():
    kitchen = pytest.importorskip("comfy_kitchen")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    values = [torch.randn(1, 2, 33, 128, device="cuda") for _ in range(3)]
    mask = torch.arange(33, device="cuda")[None, :] <= torch.arange(33, device="cuda")[:, None]
    with torch.no_grad():
        expected = kitchen.int8_attention(*values, attn_mask=mask)
        with inference_attention("kitchen-int8"):
            observed = scaled_dot_product_attention(values[0], values[1], values[2], attn_mask=mask)
            torch.testing.assert_close(observed, expected, atol=0, rtol=0)
            with pytest.raises(ValueError, match="causal"):
                scaled_dot_product_attention(values[0], values[1], values[2], is_causal=True)
        torch.testing.assert_close(
            scaled_dot_product_attention(values[0], values[1], values[2]),
            F.scaled_dot_product_attention(values[0], values[1], values[2]),
        )
    with pytest.raises(ValueError, match="inference-only"), inference_attention("kitchen-int8"):
        scaled_dot_product_attention(values[0], values[1], values[2])
    scaled_dot_product_attention(
        values[0].requires_grad_(), values[1].requires_grad_(), values[2].requires_grad_()
    ).sum().backward()
    assert all(value.grad is not None for value in values)
