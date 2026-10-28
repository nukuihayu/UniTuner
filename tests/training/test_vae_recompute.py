"""Verify restoration VAE recomputation preserves values, gradients and ownership."""

import pytest
import torch

from tests.training.test_restoration_bf16 import FAMILIES, _vae
from unitunner.adapters.encoder import EncoderAdapters
from unitunner.inference.restoration.vae import decode_restoration_latents, encode_restoration_image
from unitunner.training.methods.vae_execution import checkpoint_vae_blocks
from unitunner.training.session import TrainingSession


@pytest.mark.parametrize("family", FAMILIES)
def test_vae_recompute_preserves_encoder_adapter_and_image_gradients(family):
    torch.manual_seed(92)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vae, _ = _vae(family)
        vae.eval().requires_grad_(False)
        source = torch.randn(1, 3, 32, 32)
        with TrainingSession() as adapters:
            encoder = adapters.own(EncoderAdapters(vae.encoder, rank=1))
            parameters = tuple(encoder.parameters())
            original = {id(m): m.forward for m in vae.modules()}
            outputs = []
            for recompute in (False, True):
                with TrainingSession() as execution:
                    if recompute:
                        checkpoint_vae_blocks(vae, execution)
                    image = source.clone().requires_grad_()
                    latent = encode_restoration_image(vae, image)
                    result = decode_restoration_latents(vae, latent)
                    gradients = torch.autograd.grad(result.square().mean(), (image, *parameters), allow_unused=True)
                    outputs.append((result.detach(), gradients))
                    with torch.no_grad():
                        reference = decode_restoration_latents(vae, encode_restoration_image(vae, source))
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)
                assert all(m.forward == original[id(m)] for m in vae.modules())
            torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
            for left, right in zip(outputs[0][1], outputs[1][1], strict=True):
                if left is None or right is None:
                    assert left is right
                else:
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
