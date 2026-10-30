"""Check native decode values and temporary VAE residency on success and failure."""

import pytest
import torch
from torch import Tensor

from unituner.inference.decoding import decode_on_device
from unituner.models.z_image.config.vae import ZImageVAEConfig
from unituner.models.z_image.vae import ZImageVAE


def small_vae() -> ZImageVAE:
    return (
        ZImageVAE(ZImageVAEConfig(block_channels=(8, 16), norm_groups=4, layers_per_block=1, latent_channels=2))
        .eval()
        .requires_grad_(False)
    )


def test_decode_preserves_native_values_parameters_and_rng() -> None:
    vae = small_vae()
    latents = torch.randn(1, 2, 3, 4, requires_grad=True)
    with torch.no_grad():
        expected = vae.decode(vae.denormalize(latents)).float()
    parameter_ids = tuple(id(parameter) for parameter in vae.parameters())
    rng = torch.get_rng_state().clone()
    flags = tuple(module.training for module in vae.modules())
    # A caller's autocast must not silently change the native FP32 decode.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = decode_on_device(vae, latents, device=torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device.type == "cpu" and actual.dtype == torch.float32 and not actual.requires_grad
    assert tuple(id(parameter) for parameter in vae.parameters()) == parameter_ids
    assert tuple(module.training for module in vae.modules()) == flags
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["transfer", "decode"])
def test_decode_parks_borrowed_vae_when_a_stage_fails(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    vae = small_vae()
    parked = False
    original_cpu = vae.cpu

    def park() -> ZImageVAE:
        nonlocal parked
        parked = True
        return original_cpu()

    def fail_transfer(device: torch.device) -> None:
        raise RuntimeError("transfer failed")

    def fail_decode(latents: Tensor) -> Tensor:
        raise RuntimeError("decode failed")

    monkeypatch.setattr(vae, "cpu", park)
    monkeypatch.setattr(
        vae, "to" if stage == "transfer" else "decode", fail_transfer if stage == "transfer" else fail_decode
    )
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        decode_on_device(vae, torch.zeros(1, 2, 3, 4), device=torch.device("cpu"))
    assert parked
    assert all(parameter.device.type == "cpu" for parameter in vae.parameters())
