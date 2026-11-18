"""Check seeded H3 sampling ownership and callback/autocast isolation on a real joint model."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.inference.sampling import build_schedule, sample_latents, sample_seeded_latents
from unituner.models.minimax_h3.layout import build_layout
from unituner.models.minimax_h3.transformer import MiniMaxH3Transformer


def _model():
    model = MiniMaxH3Transformer(
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
    with torch.no_grad():
        model.rope.inv_freq.copy_(torch.tensor([1.0, 0.01]))
    return model


def test_seeded_rng_modes_inputs_and_callback_storage_are_preserved():
    torch.set_num_threads(1)
    torch.manual_seed(631)
    model = _model().train()
    model.token_refiner.eval()
    modes = [module.training for module in model.modules()]
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    layout = build_layout(
        task="fl2va",
        text_length=3,
        target_video=(2, 2, 4),
        target_audio_frames=8,
        visual_conditions=((1, 2, 4),),
        condition_roles=("last",),
    )
    text, tags = torch.randn(1, 3, 12), torch.tensor([[1, 0, 1]])
    condition = torch.randn(1, 24, 1, 2, 4)
    before = condition.clone()
    rng = torch.get_rng_state().clone()
    calls = []

    def corrupt_snapshot(index, video, audio):
        calls.append(index)
        video.zero_()
        audio.fill_(999)

    with sdpa_kernel(SDPBackend.MATH):
        expected = sample_seeded_latents(model, text, tags, layout, seed=71, steps=3, visual_conditions=(condition,))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = sample_seeded_latents(
                model,
                text,
                tags,
                layout,
                seed=71,
                steps=3,
                visual_conditions=(condition,),
                x0_callback=corrupt_snapshot,
            )
    assert calls == [0, 1, 2]
    assert torch.equal(expected.video, actual.video) and torch.equal(expected.audio, actual.audio)
    assert expected.video.dtype == expected.audio.dtype == torch.float32
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(before, condition)
    assert modes == [module.training for module in model.modules()]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items())
    actual.video.zero_()
    assert expected.video.count_nonzero()


def test_invalid_geometry_coefficients_and_schedule_reject_before_sampling():
    model = _model()
    layout = build_layout(task="t2va", text_length=3, target_video=(2, 2, 4), target_audio_frames=8)
    video, audio, text, tags = (
        torch.zeros(1, 24, 2, 2, 4),
        torch.zeros(1, 32, 2, 8),
        torch.zeros(1, 3, 12),
        torch.tensor([[1, 0, 1]]),
    )
    for steps in [0, True]:
        with pytest.raises(ValueError, match="steps"):
            build_schedule(steps)
    with pytest.raises(ValueError, match="shifts"):
        build_schedule(3, video_shift=101.0)
    with pytest.raises(ValueError, match="geometry"):
        sample_latents(model, video[:, :, :1], audio, text, tags, layout, steps=3)
    with pytest.raises(ValueError, match="clean"):
        sample_latents(model, video, audio, text, tags, layout, steps=3, visual_condition_clean=float("nan"))
