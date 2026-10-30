"""Verify native flow interpolation, token geometry and every active gradient independently."""

import pytest
import torch

from tests.support.assertions import invalid_input
from unituner.models.qwen_image_2_1.conditioning import pack_latents, unpack_latents
from unituner.models.qwen_image_2_1.config.model import QwenImage21Config
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.training.integrations.qwen_image_2_1.binding import QwenImage21Binding, QwenImage21FlowRequest


def test_native_flow_layout_interpolation_reference_slots_and_gradients():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(971)
        model = QwenImage21Transformer(
            QwenImage21Config(
                in_channels=4,
                out_channels=4,
                context_in_dim=6,
                num_layers=2,
                attention_head_dim=8,
                num_attention_heads=2,
                axes_dims_rope=(2, 2, 4),
            )
        ).double()
        binding = QwenImage21Binding(model)
        clean = torch.randn(2, 4, 1, 2, 3, dtype=torch.float64, requires_grad=True)
        noise = torch.randn_like(clean, requires_grad=True)
        sigma = torch.tensor([0.217, 0.731], dtype=torch.float64, requires_grad=True)
        timesteps = torch.tensor([287.0, 831.0], dtype=torch.float64, requires_grad=True)
        captions = (
            torch.randn(3, 6, dtype=torch.float64, requires_grad=True),
            torch.randn(5, 6, dtype=torch.float64, requires_grad=True),
        )
        refs = (
            torch.randn(2, 4, 1, 3, 1, dtype=torch.float64, requires_grad=True),
            torch.randn(2, 4, 1, 1, 2, dtype=torch.float64, requires_grad=True),
        )
        slots = ((0, 2), (2, 2))
        request = QwenImage21FlowRequest(clean, noise, sigma, captions, refs, slots, timesteps)
        rng = torch.get_rng_state().clone()
        result = binding.predict(request)
        # Derive channel-last pixels and rectangular text padding independently.
        noisy = clean * (1 - sigma[:, None, None, None, None]) + noise * sigma[:, None, None, None, None]
        tokens = noisy[:, :, 0].permute(0, 2, 3, 1).reshape(2, 6, 4)
        context = torch.stack((torch.cat((captions[0], torch.zeros(2, 6, dtype=torch.float64))), captions[1]))
        reference_tokens = tuple(ref[:, :, 0].permute(0, 2, 3, 1).reshape(2, -1, 4) for ref in refs)
        expected = (
            model(
                tokens,
                context,
                timesteps / 1000,
                (((1, 3, 1), (1, 1, 2), (1, 2, 3)),),
                text_lengths=(3, 5),
                references=reference_tokens,
                image_slots=slots,
            )
            .reshape(2, 2, 3, 4)
            .permute(0, 3, 1, 2)
            .unsqueeze(2)
        )
        torch.testing.assert_close(result.prediction, expected, atol=1e-10, rtol=1e-8)
        assert torch.equal(result.target, noise - clean) and result.timesteps is timesteps
        loss = (expected.float() - (noise - clean).float()).square().mean()
        actual = binding.step(request)
        assert actual.sample_count == 2 and torch.equal(actual.loss, loss)
        inputs = (clean, noise, sigma, timesteps, *captions, *refs, *model.parameters())
        observed = torch.autograd.grad(actual.loss, inputs, retain_graph=True)
        expected_gradients = torch.autograd.grad(loss, inputs)
        for a, b in zip(observed, expected_gradients, strict=True):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
        assert torch.equal(rng, torch.get_rng_state()) and len(binding.adapter_targets()) == 14
        values = torch.arange(2 * 4 * 1 * 2 * 3, dtype=torch.float64).reshape(2, 4, 1, 2, 3).requires_grad_()
        assert torch.equal(unpack_latents(pack_latents(values), height=2, width=3), values)
        pack_latents(values).sum().backward()
        assert values.grad is not None
        assert torch.equal(values.grad, torch.ones_like(values))
        with pytest.raises(ValueError, match="slots"):
            binding.predict(
                QwenImage21FlowRequest(clean, noise, sigma, captions, refs, invalid_input(((0.1, 2), (2, 2))))
            )
    finally:
        torch.set_num_threads(previous)
