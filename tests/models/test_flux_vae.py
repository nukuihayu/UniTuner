"""Verify native FLUX VAE posterior policy, strict artifacts and explicit RNG ownership."""

import pytest
import torch

from unituner.artifacts.tensors import save_tensors
from unituner.cli.main import main
from unituner.models.flux.config.vae import FluxVAEConfig
from unituner.models.flux.loading import load_flux_vae, save_flux_vae
from unituner.models.flux.vae import FluxVAE


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_flux_vae_explicit_posterior_scaling_gradients_and_autocast_isolation():
    torch.manual_seed(327)
    vae = FluxVAE(
        FluxVAEConfig(base_channels=8, channel_multipliers=(1, 2), norm_groups=4, layers_per_block=1, latent_channels=2)
    )
    vae.encoder.mid.eval()
    flags = tuple(module.training for module in vae.modules())
    buffers = {key: value.clone() for key, value in vae.named_buffers()}
    pixels = torch.randn(2, 3, 8, 12, requires_grad=True)
    noise = torch.randn(2, 2, 4, 6, requires_grad=True)
    rng = torch.get_rng_state().clone()
    posterior = vae.posterior(pixels)
    mean, sampled = vae.encode(pixels), vae.encode(pixels, noise=noise)
    torch.testing.assert_close(
        mean, (posterior.mean - vae.config.shift_factor) * vae.config.scaling_factor, atol=0, rtol=0
    )
    torch.testing.assert_close(
        sampled,
        (posterior.mean + torch.exp(0.5 * posterior.log_variance) * noise - vae.config.shift_factor)
        * vae.config.scaling_factor,
        atol=0,
        rtol=0,
    )
    decoded = vae.decode(sampled)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(vae.encode(pixels, noise=noise), sampled, atol=0, rtol=0)
        torch.testing.assert_close(vae.decode(sampled), decoded, atol=0, rtol=0)
    decoded.square().mean().backward()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all() for value in (pixels, noise, *vae.parameters())
    )
    assert tuple(module.training for module in vae.modules()) == flags
    for key, value in vae.named_buffers():
        torch.testing.assert_close(value, buffers[key], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    with pytest.raises(ValueError, match="noise must match"):
        vae.encode(pixels, noise=noise.bfloat16())
    with torch.no_grad():
        vae.encoder.conv_out.weight.zero_()
        assert vae.encoder.conv_out.bias is not None
        vae.encoder.conv_out.bias[2:].fill_(40)
    assert torch.equal(vae.posterior(pixels).log_variance, torch.full_like(noise, 40))


def test_flux_vae_strict_native_import_reload_and_invalid_artifacts(tmp_path):
    config = FluxVAEConfig(
        base_channels=8, channel_multipliers=(1, 2, 2), norm_groups=4, layers_per_block=1, latent_channels=2
    )
    vae = FluxVAE(config)
    source, destination, topology = (tmp_path / name for name in ("source.safetensors", "vae.safetensors", "vae.json"))
    topology.write_text(config.model_dump_json())
    save_tensors(vae.state_dict(), source)
    rng = torch.get_rng_state().clone()
    assert (
        main(
            [
                "weights",
                "import",
                str(source),
                str(destination),
                "--family",
                "flux",
                "--kind",
                "vae",
                "--source-format",
                "native",
                "--topology",
                str(topology),
            ]
        )
        == 0
    )
    loaded = load_flux_vae(destination)
    assert loaded.config == config
    for key, value in vae.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    with pytest.raises(ValueError, match="differs"):
        load_flux_vae(destination, config=FluxVAEConfig())
    state = dict(vae.state_dict())
    state.pop("encoder.conv_in.bias")
    save_tensors(state, source)
    with pytest.raises(ValueError, match="keys differ"):
        load_flux_vae(source, config=config)
    state = dict(vae.state_dict())
    state["decoder.conv_out.bias"] = torch.full((3,), float("nan"))
    save_tensors(state, source)
    with pytest.raises(ValueError, match="values invalid"):
        load_flux_vae(source, config=config)
    save_flux_vae(vae, destination)
    assert load_flux_vae(destination).config.spatial_factor == 4
