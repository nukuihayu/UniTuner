"""Validate H3 VAE artifacts, posterior ownership and malformed tile rejection."""

import pytest
import torch
from pydantic import ValidationError
from torch.nn.attention import SDPBackend, sdpa_kernel

from unitunner.artifacts.tensors import save_tensors
from unitunner.models.minimax_h3.conditioning import encode_condition_video
from unitunner.models.minimax_h3.config.video_vae import MiniMaxH3VideoDecoderConfig, MiniMaxH3VideoVAEConfig
from unitunner.models.minimax_h3.data.cache.video import encode_target_video, target_video_seed
from unitunner.models.minimax_h3.vae import MiniMaxH3VideoVAE, load_video_vae, save_video_vae


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with sdpa_kernel(SDPBackend.MATH):
        yield
    torch.set_num_threads(previous)


def _vae(*, include_decoder=True):
    config = MiniMaxH3VideoVAEConfig(
        channels=32,
        channel_multipliers=(1, 1, 1, 1, 1, 1),
        residual_blocks=(1, 1, 1, 1, 1, 1),
        tile_size=64,
        tile_overlap=32,
        tiling=False,
        decoder=MiniMaxH3VideoDecoderConfig(num_layers=2, num_heads=2, head_dim=8),
    )
    model = MiniMaxH3VideoVAE(config, include_decoder=include_decoder)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith(("scale1", "scale2", "register_tokens")):
                value.fill_(0.1)
        if model.decoder is not None:
            model.decoder.mask_token.fill_(0.02)
    return model


def test_vae_artifact_reload_and_explicit_decoder_inventory(tmp_path):
    model = _vae()
    source = tmp_path / "vae.safetensors"
    save_video_vae(model, source)
    before = source.read_bytes()
    rng = torch.get_rng_state().clone()
    loaded = load_video_vae(source)
    encoder = load_video_vae(source, include_decoder=False)
    assert torch.equal(rng, torch.get_rng_state()) and source.read_bytes() == before
    assert all(not value.is_meta for value in loaded.buffers())
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    assert encoder.decoder is None and encoder.post_quant_conv is None
    image = torch.randn(1, 3, 32, 32)
    assert torch.equal(encoder.encode(image), loaded.encode(image))
    assert torch.equal(loaded.decode(loaded.encode(image)), model.decode(model.encode(image)))
    target = tmp_path / "encoder.safetensors"
    save_video_vae(encoder, target)
    assert load_video_vae(target, include_decoder=False).decoder is None
    with pytest.raises(ValueError, match="inventory"):
        load_video_vae(target)
    with pytest.raises(RuntimeError, match="without its decoder"):
        encoder.decode(encoder.encode(image))
    broken = dict(model.state_dict())
    del broken["decoder.transformer_blocks.0.scale1"]
    save_tensors(broken, target)
    with pytest.raises(ValueError, match="inventory"):
        load_video_vae(target, config=model.config, include_decoder=False)
    broken = dict(model.state_dict())
    broken["latents_std"] = torch.zeros(24)
    save_tensors(broken, target)
    with pytest.raises(ValueError, match="latents_std"):
        load_video_vae(target, config=model.config)


def test_native_posterior_clamp_condition_rounding_and_global_rng_ownership():
    model = _vae(include_decoder=False)
    with torch.no_grad():
        model.quant_conv.weight.zero_()
        assert model.quant_conv.bias is not None
        model.quant_conv.bias[:24].fill_(1.0)
        assert model.quant_conv.bias is not None
        model.quant_conv.bias[24:36].fill_(-40.0)
        assert model.quant_conv.bias is not None
        model.quant_conv.bias[36:].fill_(30.0)
    pixels = torch.randn(1, 3, 1, 32, 32)
    posterior = model.posterior(pixels)
    assert torch.equal(posterior.mean, torch.ones_like(posterior.mean))
    assert torch.equal(posterior.log_variance[:, :12], torch.full_like(posterior.mean[:, :12], -30.0))
    assert torch.equal(posterior.log_variance[:, 12:], torch.full_like(posterior.mean[:, 12:], 20.0))
    rng = torch.get_rng_state().clone()
    target = encode_target_video(model, pixels, cache_seed=123, canonical_item_key="视频/a")
    assert torch.equal(target, encode_target_video(model, pixels, cache_seed=123, canonical_item_key="视频/a"))
    assert not torch.equal(target, encode_target_video(model, pixels, cache_seed=123, canonical_item_key="视频/b"))
    # Use explicit noise to check clamp and sample placement without a second RNG owner.
    noise = torch.ones_like(posterior.mean)
    expected = model.normalize(posterior.mean + torch.exp(0.5 * posterior.log_variance) * noise)
    assert torch.equal(model.encode(pixels, posterior_noise=noise), expected)
    generator = torch.Generator(device="cpu").manual_seed(42)
    condition_noise = torch.randn(posterior.mean.shape, generator=generator)
    expected = model.normalize(posterior.sample(condition_noise).half().float())
    condition = encode_condition_video(model, pixels)
    assert torch.equal(condition, expected)
    assert not target.requires_grad and not condition.requires_grad
    assert torch.equal(rng, torch.get_rng_state())
    assert target_video_seed(123, "视频/a") != target_video_seed(123, "视频/b")
    with pytest.raises(ValueError, match="noise"):
        posterior.sample(noise.half())
    with pytest.raises(ValueError, match="cache seed"):
        target_video_seed(True, "a")


def test_invalid_tile_geometry_and_pixel_axes_fail_before_computation():
    with pytest.raises(ValidationError, match="overlap"):
        MiniMaxH3VideoVAEConfig(tile_size=64, tile_overlap=64)
    with pytest.raises(ValidationError, match="aligned"):
        MiniMaxH3VideoVAEConfig(tile_size=63, tile_overlap=32)
    with pytest.raises(ValidationError, match="32-group"):
        MiniMaxH3VideoVAEConfig(channels=31)
    model = _vae(include_decoder=False)
    with pytest.raises(ValueError, match="align"):
        model.encode(torch.randn(1, 3, 1, 33, 32))
    with pytest.raises(ValueError, match="FP32"):
        model.encode(torch.randn(1, 3, 1, 32, 32).half())
