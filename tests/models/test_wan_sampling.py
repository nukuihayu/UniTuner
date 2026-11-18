"""Check native Wan sampling isolation and explicit expert/CFG boundaries."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from unituner.models.wan.config.model import WanConfig
from unituner.models.wan.inference.sampling import sample_latents
from unituner.models.wan.transformer import WanTransformer


def test_sampling_preserves_model_rng_inputs_and_disables_ambient_autocast():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(553)
        model = WanTransformer(
            WanConfig(
                hidden_size=48, num_heads=4, num_layers=2, ffn_dim=80, timestep_dim=16, text_dim=32, text_length=7
            )
        )
        with torch.no_grad():
            model.head.head.weight.normal_(std=0.03)
        model.train()
        noise, captions = torch.randn(1, 16, 2, 4, 4), (torch.randn(4, 32),)
        original = noise.clone()
        parameters = {key: value.detach().clone() for key, value in model.state_dict().items()}
        state = torch.get_rng_state().clone()
        with sdpa_kernel(SDPBackend.MATH):
            expected = sample_latents(model, noise, captions, steps=3, guidance=None)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = sample_latents(model, noise, captions, steps=3, guidance=None)
        assert torch.equal(actual, expected)
        assert model.training and actual.dtype == torch.float32
        assert actual.data_ptr() != noise.data_ptr() and torch.equal(noise, original)
        assert torch.equal(torch.get_rng_state(), state)
        assert all(torch.equal(value, parameters[key]) for key, value in model.state_dict().items())
        with pytest.raises(ValueError, match="negative_captions"):
            sample_latents(model, noise, captions, steps=3, guidance=3.0)
        with pytest.raises(ValueError, match="dual experts require"):
            sample_latents(model, noise, captions, steps=3, guidance=None, high_noise_model=model, boundary=0.875)
    finally:
        torch.set_num_threads(previous)
