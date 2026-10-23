"""Check packed VAE geometry, frozen statistics and strict native reload."""

import pytest
import torch

from unitunner.artifacts.tensors import save_tensors
from unitunner.cli.main import main
from unitunner.models.flux2.config.vae import Flux2VAEConfig
from unitunner.models.flux2.loading import load_flux2_vae, save_flux2_vae
from unitunner.models.flux2.vae import Flux2VAE


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_vae():
    vae = Flux2VAE(Flux2VAEConfig(base_channels=32, channel_multipliers=(1, 2), layers_per_block=1, latent_channels=2))
    assert vae.bn.running_mean is not None
    vae.bn.running_mean.copy_(torch.linspace(-0.5, 0.5, 8))
    assert vae.bn.running_var is not None
    vae.bn.running_var.copy_(torch.linspace(0.25, 1.5, 8))
    return vae


def test_flux2_vae_pack_order_frozen_stats_gradients_and_autocast():
    vae = small_vae().train()
    vae.decoder.eval()
    flags = [module.training for module in vae.modules()]
    pixels = torch.randn(2, 3, 8, 12, requires_grad=True)
    before = {key: value.clone() for key, value in vae.state_dict().items()}
    rng = torch.get_rng_state().clone()
    mean = vae.encoder(pixels).chunk(2, dim=1)[0]
    packed = torch.stack(
        [mean[:, channel, row::2, column::2] for channel in range(2) for row in range(2) for column in range(2)], 1
    )
    assert vae.bn.running_var is not None
    variance = vae.bn.running_var[None, :, None, None] + vae.config.normalization_eps
    assert vae.bn.running_mean is not None
    expected = (packed - vae.bn.running_mean[None, :, None, None]) / variance.sqrt()
    actual = vae.encode(pixels)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(vae.denormalize(actual), packed, atol=1e-6, rtol=1e-5)
    decoded = vae.decode(actual)
    assert decoded.shape == pixels.shape
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(vae.encode(pixels), actual, atol=0, rtol=0)
        torch.testing.assert_close(vae.decode(actual), decoded, atol=0, rtol=0)
    decoded.square().mean().backward()
    assert pixels.grad is not None and pixels.grad.count_nonzero()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in vae.parameters())
    assert [module.training for module in vae.modules()] == flags
    for key, value in vae.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    with pytest.raises(ValueError, match="divisible"):
        vae.encode(pixels[:, :, :-1])
    with pytest.raises(ValueError, match="FP32"):
        vae.to(torch.bfloat16).encode(pixels)


def test_flux2_vae_complete_native_reload_and_invalid_running_statistics(tmp_path):
    vae = small_vae()
    path = tmp_path / "vae.safetensors"
    save_flux2_vae(vae, path)
    topology = tmp_path / "config.json"
    topology.write_text(vae.config.model_dump_json())
    exported = tmp_path / "exported.safetensors"
    assert (
        main(
            [
                "weights",
                "import",
                str(path),
                str(exported),
                "--family",
                "moss-image",
                "--kind",
                "vae",
                "--topology",
                str(topology),
            ]
        )
        == 0
    )
    loaded = load_flux2_vae(exported)
    assert loaded.config == vae.config
    for key, value in vae.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    for name, invalid in (
        ("bn.running_var", -torch.ones(8)),
        ("bn.running_mean", torch.full((8,), float("nan"))),
        ("bn.num_batches_tracked", torch.tensor(0.0)),
    ):
        save_tensors({**vae.state_dict(), name: invalid}, path)
        with pytest.raises(ValueError, match=r"FLUX\.2"):
            load_flux2_vae(path, config=vae.config)
