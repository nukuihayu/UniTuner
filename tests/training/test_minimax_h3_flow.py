"""Validate H3 supervision, native time units and shared engine ownership."""

from dataclasses import replace

import pytest
import torch
from pydantic import ValidationError
from torch.nn.attention import SDPBackend, sdpa_kernel

from unituner.adapters.config import AdapterConfig
from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.config.training import MiniMaxH3FlowConfig
from unituner.models.minimax_h3.layout import build_layout
from unituner.models.minimax_h3.packing import MiniMaxH3Targets
from unituner.models.minimax_h3.transformer import MiniMaxH3Transformer
from unituner.training.contracts import RunState
from unituner.training.engine import update_phase
from unituner.training.integrations.minimax_h3.binding import (
    MiniMaxH3Binding,
    MiniMaxH3FlowRequest,
    build_noisy_inputs,
)
from unituner.training.parameters import ParameterInstallation


def _request():
    return MiniMaxH3FlowRequest(
        torch.randn(1, 24, 2, 2, 2),
        torch.randn(1, 24, 2, 2, 2),
        torch.randn(1, 32, 2, 8),
        torch.randn(1, 32, 2, 8),
        torch.tensor([0.7339266538619995]),
        torch.randn(1, 3, 12),
        torch.tensor([[0, 1, 0]]),
        build_layout(task="t2va", text_length=3, target_video=(2, 2, 2), target_audio_frames=8),
    )


def _binding(config):
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
    return MiniMaxH3Binding(model, config)


def test_binding_shared_update_and_unsupervised_audio_skip():
    torch.set_num_threads(1)
    torch.manual_seed(191)
    binding = _binding(MiniMaxH3FlowConfig(audio_loss_weight=0.7))
    request = _request()
    state = RunState()
    weights = {key: value.clone() for key, value in binding.model.state_dict().items()}
    owner = ParameterInstallation(
        binding.model, adapter=AdapterConfig(rank=2, alpha=7.0), targets=binding.adapter_targets()
    )
    try:
        optimizer = torch.optim.AdamW(owner.branches.parameters(), lr=0.003)
        before = [p.detach().clone() for p in owner.branches.parameters()]
        rng = torch.get_rng_state().clone()
        with sdpa_kernel(SDPBackend.MATH):
            result = binding.predict(request)
            expected = torch.nn.functional.mse_loss(result.prediction.video, result.target.video)
            poisoned = replace(
                result,
                prediction=MiniMaxH3Targets(
                    result.prediction.video, torch.full_like(result.prediction.audio, float("nan"))
                ),
            )
            assert torch.equal(binding.loss(poisoned, audio_present=False), expected)
            zero_binding = MiniMaxH3Binding(binding.model, MiniMaxH3FlowConfig(audio_loss_weight=0.0))
            assert torch.equal(zero_binding.loss(poisoned, audio_present=True), expected)
            update = update_phase((binding.step(request) for _ in range(2)), optimizer, state, max_grad_norm=1.0)
        assert update.samples == 2 and state == RunState(updates=1, microbatches=2, samples=2)
        assert any(not torch.equal(old, new) for old, new in zip(before, owner.branches.parameters(), strict=True))
        assert all(torch.equal(weights[key], value) for key, value in binding.model.state_dict().items())
        assert torch.equal(rng, torch.get_rng_state())
        assert all(region.branch == "joint" for region in binding.regions.regions)
    finally:
        owner.close()


def test_noising_boundary_rejects_invalid_draws_and_unused_condition_noise():
    request, config = _request(), MiniMaxH3FlowConfig()
    for draw in (1.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="uniform"):
            build_noisy_inputs(replace(request, uniform=torch.tensor([draw])), config)
    with pytest.raises(ValueError, match="unused noise"):
        build_noisy_inputs(replace(request, audio_condition_noise=(torch.ones(1),)), config)
    with pytest.raises(ValueError, match="FP32"):
        build_noisy_inputs(replace(request, video=request.video.half()), config)
    with pytest.raises(ValidationError):
        MiniMaxH3FlowConfig(audio_loss_weight=float("nan"))
    with pytest.raises(ValidationError):
        MiniMaxH3FlowConfig(sigma_min=0.8, sigma_max=0.2)
