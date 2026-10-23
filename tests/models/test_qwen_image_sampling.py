"""Verify Qwen schedule, normalized CFG and edit-mask integration invariants."""

import pytest
import torch

from unitunner.models.components.conditioning import Caption
from unitunner.models.qwen_image.conditioning import pack_latents, unpack_latents
from unitunner.models.qwen_image.config.model import QwenImageConfig
from unitunner.models.qwen_image.inference.sampling import build_schedule, normalized_guidance, sample_latents
from unitunner.models.qwen_image.transformer import QwenImageTransformer


def test_qwen_cfg_norm_and_terminal_schedule():
    positive = torch.tensor([[[3.0, 4.0]]], requires_grad=True)
    negative = torch.tensor([[[0.0, 2.0]]], requires_grad=True)
    assert torch.equal(normalized_guidance(positive, negative, 0), torch.tensor([[[0.0, 5.0]]]))
    zero = normalized_guidance(positive, -positive, 0.5)
    assert torch.equal(zero, torch.zeros_like(zero))
    normalized_guidance(positive, negative, 2).sum().backward()
    assert negative.grad is not None and positive.grad is not None
    assert torch.isfinite(positive.grad).all() and torch.isfinite(negative.grad).all()
    expected = build_schedule(3, 6)
    assert expected.dtype == torch.float32 and expected[0] == 1 and expected[-1] == 0
    torch.testing.assert_close(expected[-2], torch.tensor(0.02), atol=1e-7, rtol=0)
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert torch.equal(build_schedule(3, 6), expected)
    finally:
        torch.set_default_dtype(original)
    for steps, shift in ((1, None), (3, float("nan")), (3, 1e30)):
        with pytest.raises(ValueError):
            build_schedule(steps, 6, shift=shift)


def test_qwen_mask_blends_after_prediction_without_mutating_inputs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = QwenImageTransformer(
            QwenImageConfig(
                variant="edit-2511",
                in_channels=8,
                out_channels=2,
                num_layers=1,
                attention_head_dim=16,
                num_attention_heads=2,
                joint_attention_dim=12,
                axes_dims_rope=(4, 6, 6),
            )
        )
        with torch.no_grad():
            model.proj_out.weight.zero_()
            model.proj_out.bias.zero_()
        noise = torch.randn(1, 2, 1, 2, 4)
        reference = torch.randn_like(noise)
        initial = noise.clone()
        source = reference.clone()
        caption = Caption(torch.randn(1, 3, 12))
        mask = torch.tensor([[[0.0], [1.0]]])
        rng = torch.get_rng_state().clone()
        result = sample_latents(
            model, noise, caption, references=(reference,), inpainting_mask=mask, steps=3, guidance=1
        )
        sigma = build_schedule(3, 2)[-2] * 1000 / 1000
        expected = pack_latents(noise) * mask + (
            pack_latents(reference) * (1 - sigma) + pack_latents(noise) * sigma
        ) * (1 - mask)
        expected = unpack_latents(expected, layers=1, height=2, width=4)
        assert torch.equal(result, expected)
        assert torch.equal(noise, initial) and torch.equal(reference, source)
        assert torch.equal(rng, torch.get_rng_state()) and model.training
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert torch.equal(
                result,
                sample_latents(
                    model, noise, caption, references=(reference,), inpainting_mask=mask, steps=3, guidance=1
                ),
            )
        with pytest.raises(ValueError, match="negative"):
            sample_latents(model, noise, caption, references=(reference,), steps=3, guidance=4)
        with pytest.raises(ValueError, match="mask"):
            sample_latents(model, noise, caption, references=(reference,), inpainting_mask=mask + 2, guidance=1)
    finally:
        torch.set_num_threads(previous)
