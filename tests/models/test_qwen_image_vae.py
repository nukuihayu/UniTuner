"""Verify causal VAE history ownership, layer semantics and strict native artifacts."""

import pytest
import torch

from unituner.artifacts.tensors import save_tensors
from unituner.cli.main import main
from unituner.models.qwen_image.config.vae import QwenImageVAEConfig
from unituner.models.qwen_image.loading import load_qwen_image_vae, save_qwen_image_vae
from unituner.models.qwen_image.vae import QwenImageVAE, QwenImageVAEExecution, QwenImageVAETiling


@pytest.fixture
def vae():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield QwenImageVAE(
        QwenImageVAEConfig(
            base_channels=4,
            latent_channels=2,
            channel_multipliers=(1, 2, 2, 2),
            layers_per_block=1,
            latents_mean=(-0.31, 0.73),
            latents_std=(1.73, 0.91),
        )
    )
    torch.set_num_threads(previous)


def test_vae_artifacts_and_import(vae, tmp_path):
    path = tmp_path / "vae.safetensors"
    rng = torch.get_rng_state().clone()
    save_qwen_image_vae(vae, path)
    restored = load_qwen_image_vae(path)
    assert torch.equal(rng, torch.get_rng_state())
    assert restored.config == vae.config
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in vae.state_dict().items())
    topology, source, output = (tmp_path / p for p in ("topology.json", "raw.safetensors", "imported.safetensors"))
    topology.write_text(vae.config.model_dump_json())
    save_tensors(vae.state_dict(), source)
    assert (
        main(
            [
                "weights",
                "import",
                str(source),
                str(output),
                "--family",
                "qwen-image",
                "--kind",
                "vae",
                "--topology",
                str(topology),
            ]
        )
        == 0
    )
    assert load_qwen_image_vae(output).config == vae.config
    with pytest.raises(ValueError, match=r"keys differ|invalid tensor"):
        load_qwen_image_vae(source)
    tensors = dict(vae.state_dict())
    tensors.pop(next(iter(tensors)))
    save_tensors(tensors, source)
    with pytest.raises(ValueError, match="keys differ"):
        load_qwen_image_vae(source, config=vae.config)
    tensors = dict(vae.state_dict())
    key = next(iter(tensors))
    tensors[key] = torch.full_like(tensors[key], float("nan"))
    save_tensors(tensors, source)
    with pytest.raises(ValueError, match="values"):
        load_qwen_image_vae(source, config=vae.config)


def test_temporal_history_is_local_and_causal(vae):
    pixels = torch.randn(1, 3, 9, 8, 8, requires_grad=True)
    other = torch.randn_like(pixels, requires_grad=True)
    rng = torch.get_rng_state().clone()
    first = vae.posterior(pixels)
    second = vae.posterior(other)
    repeat = vae.posterior(pixels)
    assert vae.training
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(first.mean, repeat.mean)
    prefix = vae.posterior(pixels[:, :, :5])
    torch.testing.assert_close(first.mean[:, :, :2], prefix.mean, atol=0, rtol=0)
    first.mean[:, :, -1].sum().backward()
    assert pixels.grad is not None and pixels.grad[:, :, :1].abs().sum() > 0
    second.mean.sum().backward()
    assert other.grad is not None
    latents = torch.randn(1, 2, 3, 1, 1, requires_grad=True)
    decoded = vae.decode_raw(latents)
    vae.decode_raw(torch.zeros_like(latents))
    assert torch.equal(decoded, vae.decode_raw(latents))
    decoded[:, :, -1].sum().backward()
    assert latents.grad is not None and latents.grad[:, :, :1].abs().sum() > 0


def test_independent_layers_explicit_noise_and_autocast(vae):
    pixels = torch.randn(1, 3, 3, 8, 8, requires_grad=True)
    encoded = vae.encode_layers(pixels)
    assert encoded.shape == (1, 2, 3, 1, 1)
    permutation = [2, 0, 1]
    assert torch.equal(vae.encode_layers(pixels[:, :, permutation]), encoded[:, :, permutation])
    decoded = vae.decode_layers(encoded)
    assert decoded.shape == pixels.shape
    noise = torch.randn_like(encoded)
    sampled = vae.encode_layers(pixels, posterior_noise=noise)
    assert not torch.equal(encoded, sampled)
    encoded[:, :, 1].sum().backward()
    assert pixels.grad is not None
    assert torch.count_nonzero(pixels.grad[:, :, [0, 2]]) == 0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(encoded, vae.encode_layers(pixels))
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert torch.equal(encoded, vae.encode_layers(pixels))
    finally:
        torch.set_default_dtype(previous)
    with pytest.raises(ValueError, match="1\\+4k"):
        vae.encode(pixels)
    with pytest.raises(ValueError, match="noise"):
        vae.encode_layers(pixels, posterior_noise=noise[:, :, :1])
    raw = torch.randn_like(encoded)
    reciprocal = 1 / torch.tensor(vae.config.latents_std).reshape(1, 2, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean).reshape(1, 2, 1, 1, 1)
    assert torch.equal(vae.normalize(raw), (raw - mean) * reciprocal)
    assert torch.equal(vae.denormalize(raw), raw / reciprocal + mean)


def test_posterior_clamps_log_variance_before_explicit_sampling(vae):
    with torch.no_grad():
        vae.quant_conv.weight.zero_()
        vae.quant_conv.bias.copy_(torch.tensor([0.2, -0.4, -50.0, 35.0]))
    pixels = torch.randn(1, 3, 1, 8, 8)
    posterior = vae.posterior(pixels)
    assert torch.equal(posterior.log_variance.flatten(), torch.tensor([-30.0, 20.0]))
    noise = torch.ones_like(posterior.mean)
    assert torch.equal(posterior.sample(noise), posterior.mean + torch.exp(0.5 * posterior.log_variance))
    with pytest.raises(ValueError, match="noise"):
        posterior.sample(noise.double())


def test_vae_tiling_and_slicing_are_per_operation(vae):
    policy = QwenImageVAEExecution(
        batch_slice_size=1, tiling=QwenImageVAETiling(height=16, width=32, stride_height=8, stride_width=8)
    )
    pixels = torch.randn(2, 3, 5, 24, 40, requires_grad=True)
    rng = torch.get_rng_state().clone()
    tiled = vae.encode(pixels, execution=policy)
    untiled = vae.encode(pixels)
    assert tiled.shape == untiled.shape == (2, 2, 2, 3, 5)
    assert not torch.equal(tiled, untiled)
    assert torch.equal(untiled, vae.encode(pixels))
    assert torch.equal(tiled, vae.encode(pixels, execution=policy))
    assert torch.equal(rng, torch.get_rng_state())
    decoded = vae.decode(tiled, execution=policy)
    assert decoded.shape == pixels.shape
    decoded.square().mean().backward()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    assert pixels.grad[:, :, :1].abs().sum() > 0
    sliced = QwenImageVAEExecution(batch_slice_size=1)
    expected = torch.cat([vae.encode(p) for p in pixels.detach().split(1)])
    assert torch.equal(vae.encode(pixels.detach(), execution=sliced), expected)
    invalid = QwenImageVAEExecution(tiling=QwenImageVAETiling(height=17, width=32, stride_height=8, stride_width=8))
    with pytest.raises(ValueError, match="spatial_factor"):
        vae.encode(pixels, execution=invalid)
    with pytest.raises(ValueError, match="strides"):
        QwenImageVAETiling(height=8, stride_height=16)
    # The native tiled branch deliberately returns unbounded pixels.
    with torch.no_grad():
        vae.decoder.conv_out.weight.zero_()
        vae.decoder.conv_out.bias.fill_(1.5)
    latents = torch.zeros(1, 2, 1, 3, 5)
    assert torch.equal(vae.decode_raw(latents), torch.ones(1, 3, 1, 24, 40))
    torch.testing.assert_close(
        vae.decode_raw(latents, execution=policy), torch.full((1, 3, 1, 24, 40), 1.5), atol=0, rtol=0
    )
