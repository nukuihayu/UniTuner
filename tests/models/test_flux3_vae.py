"""Check differentiable FLUX3 token geometry and frozen VAE state at the image boundary."""

from dataclasses import replace

import pytest
import torch

from unituner.models.flux2.config.vae import Flux2VAEConfig
from unituner.models.flux2.vae import Flux2VAE
from unituner.models.flux3.vae import decode_image, encode_image


def test_flux3_dense_image_vae_preserves_gradients_modes_rng_and_statistics():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vae = Flux2VAE(
            Flux2VAEConfig(
                base_channels=8, channel_multipliers=(1, 2), layers_per_block=1, latent_channels=2, norm_groups=4
            )
        )
        vae.encoder.mid.eval()
        assert vae.bn.running_mean is not None
        vae.bn.running_mean.copy_(torch.linspace(-0.2, 0.5, 8))
        assert vae.bn.running_var is not None
        vae.bn.running_var.copy_(torch.linspace(0.25, 1.5, 8))
        statistics = {name: value.clone() for name, value in vae.bn.state_dict().items()}
        modes = tuple(module.training for module in vae.modules())
        pixels = torch.randn(3, 8, 12, requires_grad=True)
        rng = torch.get_rng_state().clone()
        image = encode_image(vae, pixels)
        decoded = decode_image(vae, image)
        assert image.tokens.shape == (6, 8) and decoded.shape == pixels.shape
        assert image.ids.tolist() == [[0, r, c, 0] for r in range(2) for c in range(3)]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            encoded_autocast = encode_image(vae, pixels)
            decoded_autocast = decode_image(vae, encoded_autocast)
        torch.testing.assert_close(encoded_autocast.tokens, image.tokens, atol=0, rtol=0)
        torch.testing.assert_close(decoded_autocast, decoded, atol=0, rtol=0)
        decoded.square().mean().backward()
        assert pixels.grad is not None and pixels.grad.norm() > 0 and torch.isfinite(pixels.grad).all()
        for module in (vae.encoder.quant_conv, vae.decoder.post_quant_conv):
            assert module.weight.grad is not None and module.weight.grad.norm() > 0
        assert modes == tuple(module.training for module in vae.modules())
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        for name, value in vae.bn.state_dict().items():
            torch.testing.assert_close(value, statistics[name], atol=0, rtol=0)
        for invalid in (
            replace(image, ids=image.ids.flip(0)),
            replace(image, width=4),
            replace(image, ids=image.ids.float()),
        ):
            with pytest.raises(ValueError, match="canonical dense"):
                decode_image(vae, invalid)
    finally:
        torch.set_num_threads(previous)
