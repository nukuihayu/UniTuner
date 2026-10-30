"""Check MOSS flow math, trainable target inventory and native resumed recomputation."""

from dataclasses import asdict

import pytest
import torch

from unitunner.adapters.config import AdapterConfig
from unitunner.checkpoints.restore import restore_checkpoint
from unitunner.checkpoints.saving import save_checkpoint
from unitunner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unitunner.execution.recompute import RecomputeInstallation
from unitunner.models.moss_image.config.model import MossImageConfig
from unitunner.models.moss_image.transformer import MossImageTransformer
from unitunner.training.contracts import ImageFlowRequest, RunState
from unitunner.training.engine import update_phase
from unitunner.training.integrations.moss_image.binding import MossImageBinding
from unitunner.training.parameters import ParameterInstallation
from unitunner.training.session import TrainingSession

IDENTITY = ResumeIdentity(config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build(session, algorithm):
    model = MossImageTransformer(
        MossImageConfig(dim=32, ffn_dim=48, num_heads=2, num_layers=2, text_dim=12, freq_dim=8, latent_channels=8)
    )
    binding = MossImageBinding(model)
    targets = binding.adapter_targets()
    assert len(targets) == 22
    assert list(targets)[:5] == [
        f"text_embedder.blocks.0.{name}" for name in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")
    ]
    assert all("norm" not in name and "modulation" not in name for name in targets)
    adapter = None if algorithm == "full" else AdapterConfig(algorithm=algorithm, rank=2, alpha=2.0)
    installation = session.own(
        ParameterInstallation(model, adapter=adapter, targets=None if adapter is None else targets)
    )
    with torch.no_grad():
        for parameter in installation.branches.parameters():
            parameter.normal_(0, 0.03)
    modules = {"model": model, "adapter": installation.branches}
    optimizer = torch.optim.AdamW(installation.plan.parameters, lr=0.001)
    return binding, modules, optimizer


def request():
    return ImageFlowRequest(
        torch.randn(2, 8, 2, 3),
        torch.randn(2, 8, 2, 3),
        torch.tensor([0.244, 0.751]),
        torch.randn(2, 5, 12),
        torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
    )


def advance(binding, optimizer, state):
    update_phase((binding.step(request()) for _ in range(2)), optimizer, state, max_grad_norm=1.0)


def test_moss_binding_native_time_sign_interpolation_and_loss():
    with TrainingSession() as session:
        binding, _, _ = build(session, "full")
        inputs = request()
        result = binding.predict(inputs)
        fraction = inputs.sigma[:, None, None, None]
        expected = binding.model(
            (1 - fraction) * inputs.latents + fraction * inputs.noise,
            inputs.sigma * 1000,
            inputs.captions,
            inputs.caption_mask,
        )
        torch.testing.assert_close(result.prediction, expected, atol=0, rtol=0)
        torch.testing.assert_close(result.timesteps, inputs.sigma * 1000, atol=0, rtol=0)
        torch.testing.assert_close(result.target, inputs.noise - inputs.latents, atol=0, rtol=0)
        torch.testing.assert_close(
            binding.step(inputs).loss, (expected - (inputs.noise - inputs.latents)).square().mean(), atol=0, rtol=0
        )
        for sigma in (torch.tensor([-0.1, 0.5]), torch.tensor([0.5, float("nan")]), torch.tensor([0.5, 1.1])):
            with pytest.raises(ValueError, match="finite and in"):
                binding.predict(ImageFlowRequest(inputs.latents, inputs.noise, sigma, inputs.captions))


@pytest.mark.parametrize("algorithm", ["full", "lora", "loha", "lokr"])
def test_moss_full_and_adapters_resume_with_recompute_matches_continuous(tmp_path, algorithm):
    torch.manual_seed(83)
    with TrainingSession() as session, TrainingSession() as restored:
        binding, modules, optimizer = build(session, algorithm)
        identities = tuple(map(id, binding.model.parameters()))
        keys = tuple(binding.model.state_dict())
        state = RunState()
        before = [p.detach().clone() for group in optimizer.param_groups for p in group["params"]]
        advance(binding, optimizer, state)
        assert any(
            not torch.equal(a, b)
            for a, b in zip(before, [p for group in optimizer.param_groups for p in group["params"]], strict=True)
        )
        saved = tmp_path / "saved.safetensors"
        save_checkpoint(saved, modules, optimizer, CheckpointCursor(**asdict(state)), IDENTITY)
        advance(binding, optimizer, state)
        advance(binding, optimizer, state)
        expected = tmp_path / "expected.safetensors"
        save_checkpoint(expected, modules, optimizer, CheckpointCursor(**asdict(state)), IDENTITY)
        fresh, fresh_modules, fresh_optimizer = build(restored, algorithm)
        for region in fresh.regions.regions:
            restored.own(RecomputeInstallation(region))
        cursor = restore_checkpoint(saved, fresh_modules, fresh_optimizer, IDENTITY)
        fresh_state = RunState(**cursor.model_dump())
        advance(fresh, fresh_optimizer, fresh_state)
        advance(fresh, fresh_optimizer, fresh_state)
        actual = tmp_path / "actual.safetensors"
        save_checkpoint(actual, fresh_modules, fresh_optimizer, CheckpointCursor(**asdict(fresh_state)), IDENTITY)
        assert actual.read_bytes() == expected.read_bytes()
        assert state == fresh_state
        assert tuple(map(id, binding.model.parameters())) == identities
        assert tuple(binding.model.state_dict()) == keys
