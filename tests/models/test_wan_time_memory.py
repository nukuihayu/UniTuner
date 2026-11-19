"""Check shared-time projections against explicit token times and gradients."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from unituner.config.routing import TreadConfig
from unituner.models.wan.config.model import WanConfig
from unituner.models.wan.transformer import WanTransformer
from unituner.training.workflows.common.batches import batch_latents


@pytest.mark.parametrize("routing", [False, True])
@pytest.mark.parametrize("padding", [0, 2])
def test_shared_time_matches_token_time_outputs_and_gradients(routing, padding):
    torch.manual_seed(31)
    model = WanTransformer(
        WanConfig(
            version="2.2",
            hidden_size=48,
            num_heads=4,
            num_layers=2,
            ffn_dim=80,
            timestep_dim=16,
            text_dim=32,
            text_length=7,
        )
    )
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.03)
    if routing:
        model.token_routing = TreadConfig(end_layer=2)
    latents = (torch.randn(16, 2, 4, 4),)
    captions = (torch.randn(3, 32),)
    length = 8 + padding
    observations = []
    hook = model.time_embedding.register_forward_pre_hook(lambda _, args: observations.append(args[0].shape))

    def run(explicit):
        model.zero_grad(set_to_none=True)
        times = torch.tensor([401.0], requires_grad=True)
        token_times = times[:, None].expand(-1, length) if explicit else times
        torch.manual_seed(55)
        with sdpa_kernel(SDPBackend.MATH):
            result = model(latents, token_times, captions, sequence_length=length)[0]
            result.square().mean().backward()
        return result.detach(), times.grad.clone(), [p.grad.clone() for p in model.parameters()]

    try:
        shared, token = run(False), run(True)
    finally:
        hook.remove()
    assert observations == [torch.Size([1, 1, 16]), torch.Size([1, length, 16])]
    for left, right in zip(shared[:2], token[:2], strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=2e-4)
    for left, right in zip(shared[2], token[2], strict=True):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=5e-4)


def test_single_record_batch_reuses_storage_and_gradients():
    value = torch.randn(16, 2, 4, 4, requires_grad=True)
    before = value.detach().clone()
    batch = batch_latents([value])
    assert batch.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    batch.square().sum().backward()
    torch.testing.assert_close(value.grad, 2 * value.detach())
    torch.testing.assert_close(value.detach(), before)
    torch.testing.assert_close(batch_latents([value, value]), torch.stack([value, value]))
