"""Exercise real H3 rendering, explicit FP32 decoding and borrowed-resource failure cleanup."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from unitunner.models.minimax_h3.config.audio_vae import MiniMaxH3AudioVAEConfig
from unitunner.models.minimax_h3.config.model import MiniMaxH3Config
from unitunner.models.minimax_h3.config.sampling import MiniMaxH3SamplingConfig
from unitunner.models.minimax_h3.config.video_vae import MiniMaxH3VideoDecoderConfig, MiniMaxH3VideoVAEConfig
from unitunner.models.minimax_h3.data.media.output import MiniMaxH3DecodedAV, MiniMaxH3DecodedImage
from unitunner.models.minimax_h3.inference.sampling import MiniMaxH3VideoInputs, render_minimax_h3
from unitunner.models.minimax_h3.layout import audio_latent_frames, build_layout
from unitunner.models.minimax_h3.transformer import MiniMaxH3Transformer
from unitunner.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE


@pytest.fixture(autouse=True)
def math_backend():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with sdpa_kernel(SDPBackend.MATH):
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def owners():
    model = (
        MiniMaxH3Transformer(
            MiniMaxH3Config(
                hidden_size=24,
                num_layers=2,
                token_refiner_num_layers=2,
                num_attention_heads=2,
                attention_head_dim=16,
                ffn_hidden_size=40,
                text_dim=12,
                timestep_input_dim=8,
                time_embed_hidden_size=16,
                time_embed_dim=8,
                rope_inv_freq_len=2,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    video = (
        MiniMaxH3VideoVAE(
            MiniMaxH3VideoVAEConfig(
                channels=32,
                channel_multipliers=(1, 1, 1, 1, 1, 1),
                residual_blocks=(1, 1, 1, 1, 1, 1),
                tile_size=64,
                tile_overlap=32,
                tiling=False,
                decoder=MiniMaxH3VideoDecoderConfig(num_layers=2, num_heads=2, head_dim=8),
            )
        )
        .eval()
        .requires_grad_(False)
    )
    audio = (
        MiniMaxH3AudioVAE(
            MiniMaxH3AudioVAEConfig(
                encoder_channels=2,
                hidden_channels=256,
                decoder_channels=128,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    with torch.no_grad():
        model.rope.inv_freq.copy_(torch.tensor([1.0, 0.01]))
        for name, value in video.named_parameters():
            if name.endswith(("scale1", "scale2", "register_tokens")):
                value.fill_(0.1)
        assert video.decoder is not None
        video.decoder.mask_token.fill_(0.02)
        for name, value in audio.named_parameters():
            if name.endswith(".alpha"):
                value.fill_(1.0 if name.startswith("encoder.") else 0.0)
            elif name.endswith((".beta", "q_bias", "v_bias")):
                value.zero_()
        audio.pre_block.attn.zero_k_bias.zero_()
        audio.latents_mean.zero_()
        audio.latents_std.fill_(1.0)
    return model, video, audio


def _request(frames, fps=24):
    sampling = MiniMaxH3SamplingConfig(
        width=32,
        height=32,
        frames=frames,
        fps=fps,
        steps=2,
        seed=71,
        temporal_fine_bands=1 if fps != 24 else 0,
        allow_experimental_duration=True,
    )
    layout = build_layout(
        task="t2va",
        text_length=3,
        target_video=(1 if frames == 1 else 2, 2, 2),
        target_audio_frames=2 if frames == 1 else audio_latent_frames(frames, output_fps=fps),
        one_frame=frames == 1,
        output_fps=fps,
        temporal_fine_bands=sampling.temporal_fine_bands,
    )
    return sampling, MiniMaxH3VideoInputs(torch.randn(1, 3, 12), torch.ones(1, 3, dtype=torch.int64), layout)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(("frames", "fps"), [(1, 24), (5, 24), (5, 12)])
def test_actual_render_owned_output_rng_modes_and_autocast(owners, device, frames, fps):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, video, audio = owners
    model.to(device)
    model.blocks[0].train()
    sampling, inputs = _request(frames, fps)
    parameters = [value for owner in owners for value in owner.parameters()]
    identities = [id(value) for value in parameters]
    weights = [{key: value.cpu().clone() for key, value in owner.state_dict().items()} for owner in owners]
    modes = [module.training for owner in owners for module in owner.modules()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    actual = render_minimax_h3(model, video, sampling, inputs, audio_vae=None if frames == 1 else audio)
    with torch.autocast(device, dtype=torch.bfloat16):
        repeated = render_minimax_h3(model, video, sampling, inputs, audio_vae=None if frames == 1 else audio)
    if frames == 1:
        assert isinstance(actual, MiniMaxH3DecodedImage) and isinstance(repeated, MiniMaxH3DecodedImage)
        assert actual.image.shape == (32, 32, 3) and actual.image.dtype == torch.uint8
        assert torch.equal(actual.image, repeated.image) and actual.image.data_ptr() != repeated.image.data_ptr()
    else:
        assert isinstance(actual, MiniMaxH3DecodedAV) and isinstance(repeated, MiniMaxH3DecodedAV)
        assert actual.video.shape == (frames, 32, 32, 3) and actual.video.dtype == torch.uint8
        samples = min(inputs.layout.target_audio_frames * 800, round(frames * 32000 / fps))
        assert actual.audio.shape == (2, samples) and actual.audio.dtype == torch.float32
        assert torch.equal(actual.video, repeated.video) and torch.equal(actual.audio, repeated.audio)
        assert (
            actual.video.data_ptr() != repeated.video.data_ptr()
            and actual.audio.data_ptr() != repeated.audio.data_ptr()
        )
    assert all(value.device.type == "cpu" for vae in [video, audio] for value in vae.parameters())
    assert identities == [id(value) for value in parameters]
    assert all(
        torch.equal(before[key], value.cpu())
        for before, owner in zip(weights, owners, strict=True)
        for key, value in owner.state_dict().items()
    )
    assert modes == [module.training for owner in owners for module in owner.modules()]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


@pytest.mark.parametrize("failing_vae", ["video", "audio"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_failure_restores_borrowed_vaes(owners, monkeypatch, failing_vae, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, video, audio = owners
    model.to(device)
    sampling, inputs = _request(5)

    def fail_decode(self, latents):
        raise RuntimeError("injected decode failure")

    vae_type = MiniMaxH3VideoVAE if failing_vae == "video" else MiniMaxH3AudioVAE
    monkeypatch.setattr(vae_type, "decode", fail_decode)
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="injected"):
        render_minimax_h3(model, video, sampling, inputs, audio_vae=audio)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(value.device.type == "cpu" for vae in [video, audio] for value in vae.parameters())


def test_invalid_layout_and_unowned_vae_reject_before_sampling(owners):
    model, video, audio = owners
    sampling, inputs = _request(5)
    with pytest.raises(ValueError, match="audio vae"):
        render_minimax_h3(model, video, sampling, inputs)
    video.train()
    with pytest.raises(ValueError, match="frozen eval CPU FP32"):
        render_minimax_h3(model, video, sampling, inputs, audio_vae=audio)
    with pytest.raises(ValueError, match="geometry/timeline"):
        render_minimax_h3(model, video, sampling.model_copy(update={"fps": 12}), inputs, audio_vae=audio)


@pytest.mark.parametrize("vae_name", ["video", "audio"])
def test_borrowed_vae_buffer_precision_rejects_without_rng_draw(owners, vae_name):
    model, video, audio = owners
    vae = video if vae_name == "video" else audio
    vae.register_buffer("invalid_precision", torch.ones(1, dtype=torch.float16))
    sampling, inputs = _request(5)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="CPU FP32"):
        render_minimax_h3(model, video, sampling, inputs, audio_vae=audio)
    assert torch.equal(rng, torch.get_rng_state())
