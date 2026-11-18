"""Validate native audio mean semantics, stereo ownership and exact artifacts."""

import pytest
import torch
from pydantic import ValidationError
from torch.nn.attention import SDPBackend, sdpa_kernel

from unituner.artifacts.tensors import save_tensors
from unituner.models.minimax_h3.conditioning import encode_audio
from unituner.models.minimax_h3.config.audio_vae import MiniMaxH3AudioVAEConfig
from unituner.models.minimax_h3.vae import MiniMaxH3AudioVAE, load_audio_vae, save_audio_vae


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with sdpa_kernel(SDPBackend.MATH):
        yield
    torch.set_num_threads(previous)


def _vae():
    model = MiniMaxH3AudioVAE(MiniMaxH3AudioVAEConfig(encoder_channels=2, hidden_channels=256, decoder_channels=128))
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith(".alpha"):
                value.fill_(1.0 if name.startswith("encoder.") else 0.0)
            elif name.endswith((".beta", "q_bias", "v_bias")):
                value.zero_()
        model.pre_block.attn.zero_k_bias.zero_()
        model.latents_mean.copy_(torch.linspace(-0.1, 0.1, 32))
        model.latents_std.copy_(torch.linspace(0.7, 1.3, 32))
    return model.eval().requires_grad_(False)


def test_mean_only_stereo_padding_and_differentiable_roundtrip():
    model = _vae()
    waveform = torch.randn(2, 2, 1601, requires_grad=True)
    before = waveform.detach().clone()
    rng = torch.get_rng_state().clone()
    encoded = model.encode(waveform)
    assert encoded.shape == (2, 32, 2, 3)
    assert torch.equal(model.encode(torch.nn.functional.pad(waveform, (0, 799))), encoded)
    assert torch.equal(model.encode(waveform.flip(1)), encoded.flip(2))
    with torch.no_grad():
        model.logs_proj.weight.fill_(999.0)
        assert model.logs_proj.bias is not None
        model.logs_proj.bias.fill_(999.0)
    assert torch.equal(model.encode(waveform), encoded)
    decoded = model.decode(encoded)
    assert decoded.shape == (2, 2, 2400) and decoded.min() >= -1 and decoded.max() <= 1
    decoded.square().mean().backward()
    assert waveform.grad is not None and torch.isfinite(waveform.grad).all() and torch.count_nonzero(waveform.grad)
    assert torch.equal(waveform.detach(), before) and torch.equal(rng, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_audio_vae_strict_reload_decoder_omission_and_cache_ownership(tmp_path):
    model = _vae()
    source = tmp_path / "vae.safetensors"
    save_audio_vae(model, source)
    before = source.read_bytes()
    rng = torch.get_rng_state().clone()
    loaded = load_audio_vae(source)
    encoder = load_audio_vae(source, include_decoder=False)
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    assert all(not value.is_meta for value in loaded.buffers())
    assert encoder.decoder is None and encoder.dec_in_proj is None
    assert torch.equal(rng, torch.get_rng_state()) and source.read_bytes() == before
    waveform = torch.randn(2, 807, dtype=torch.float64)
    saved = waveform.clone()
    rng = torch.get_rng_state().clone()
    cached = encode_audio(encoder, waveform)
    assert torch.equal(cached, loaded.encode(waveform.float().unsqueeze(0)))
    assert cached.shape == (1, 32, 2, 2) and not cached.requires_grad
    assert torch.equal(saved, waveform) and torch.equal(rng, torch.get_rng_state())
    target = tmp_path / "encoder.safetensors"
    save_audio_vae(encoder, target)
    assert load_audio_vae(target, include_decoder=False).decoder is None
    with pytest.raises(ValueError, match="inventory"):
        load_audio_vae(target)
    with pytest.raises(RuntimeError, match="without its decoder"):
        encoder.decode(cached)
    broken = dict(model.state_dict())
    del broken["decoder.conv_post.weight"]
    save_tensors(broken, target)
    with pytest.raises(ValueError, match="inventory"):
        load_audio_vae(target, config=model.config, include_decoder=False)
    broken = dict(model.state_dict())
    broken["latents_std"] = torch.zeros(32)
    save_tensors(broken, target)
    with pytest.raises(ValueError, match="latents_std"):
        load_audio_vae(target, config=model.config)


def test_invalid_audio_geometry_fails_at_native_boundary():
    with pytest.raises(ValidationError, match="800"):
        MiniMaxH3AudioVAEConfig(encoder_strides=(2, 4))
    with pytest.raises(ValidationError, match="eight"):
        MiniMaxH3AudioVAEConfig(hidden_channels=257)
    with pytest.raises(ValidationError, match="halving"):
        MiniMaxH3AudioVAEConfig(decoder_channels=129)
    model = _vae()
    with pytest.raises(ValueError, match="stereo"):
        encode_audio(model, torch.randn(1, 800))
    with pytest.raises(ValueError, match="FP32"):
        model.encode(torch.randn(1, 2, 800).half())
    with pytest.raises(ValueError, match="B32ST"):
        model.decode(torch.randn(1, 32, 1, 2))
