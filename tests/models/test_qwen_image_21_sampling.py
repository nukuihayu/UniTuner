"""Verify independent single-step plain CFG, terminal reference masks and RNG ownership."""

import pytest
import torch

from unitunner.models.qwen_image_2_1.conditioning import QwenImage21TextFeatures, pack_latents, unpack_latents
from unitunner.models.qwen_image_2_1.config.model import QwenImage21Config
from unitunner.models.qwen_image_2_1.inference.sampling import sample_latents
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer


def test_single_step_plain_cfg_mask_and_autocast_invariance():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1013)
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
        )
        noise = torch.randn(1, 4, 1, 2, 3)
        reference = torch.randn_like(noise)
        positive = QwenImage21TextFeatures(torch.randn(3, 6), (0,), ((2, 3),))
        negative = QwenImage21TextFeatures(torch.randn(5, 6), (2,), ((2, 3),))
        mask = torch.linspace(0, 1, 6).reshape(1, 6, 1)
        rng = torch.get_rng_state().clone()
        original = noise.clone()
        with torch.no_grad():
            p = model(
                pack_latents(noise),
                positive.features[None],
                torch.ones(1),
                (((1, 2, 3), (1, 2, 3)),),
                references=(pack_latents(reference),),
                image_slots=((0,),),
            )
            u = model(
                pack_latents(noise),
                negative.features[None],
                torch.ones(1),
                (((1, 2, 3), (1, 2, 3)),),
                references=(pack_latents(reference),),
                image_slots=((2,),),
            )
            expected = (pack_latents(noise) - (u + 2.2 * (p - u))) * mask + pack_latents(reference) * (1 - mask)
        actual = sample_latents(
            model,
            noise,
            positive,
            negative=negative,
            references=(reference,),
            inpainting_mask=mask,
            steps=1,
            guidance=2.2,
        )
        torch.testing.assert_close(actual, unpack_latents(expected, height=2, width=3), atol=1e-6, rtol=1e-5)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            repeated = sample_latents(
                model,
                noise,
                positive,
                negative=negative,
                references=(reference,),
                inpainting_mask=mask,
                steps=1,
                guidance=2.2,
            )
        assert torch.equal(repeated, actual)
        preserved = sample_latents(
            model, noise, positive, references=(reference,), inpainting_mask=torch.zeros_like(mask), steps=3, guidance=1
        )
        assert torch.equal(preserved, reference)
        assert torch.equal(rng, torch.get_rng_state()) and torch.equal(noise, original) and model.training
        with pytest.raises(ValueError, match="negative"):
            sample_latents(model, noise, positive, references=(reference,), guidance=2)
        with pytest.raises(ValueError, match="inpainting"):
            sample_latents(model, noise, positive, references=(reference,), inpainting_mask=mask + 1)
    finally:
        torch.set_num_threads(previous)
