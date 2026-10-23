"""Check explicit Wan noise/time units, immutable conditions and adapter resume."""

from dataclasses import asdict

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from unitunner.adapters.config import AdapterConfig
from unitunner.checkpoints.restore import restore_checkpoint
from unitunner.checkpoints.saving import save_checkpoint
from unitunner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unitunner.execution.recompute import RecomputeInstallation
from unitunner.models.wan.config.model import WanConfig
from unitunner.models.wan.data.cache.storage import WanCacheRecord
from unitunner.models.wan.transformer import WanTransformer
from unitunner.training.engine import update_phase
from unitunner.training.integrations.wan.binding import WanBinding, WanFlowRequest
from unitunner.training.parameters import ParameterInstallation
from unitunner.training.session import TrainingSession
from unitunner.training.state import RunState

IDENTITY = ResumeIdentity(config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with sdpa_kernel(SDPBackend.MATH):
        yield
    torch.set_num_threads(previous)


def config(**kwargs):
    return WanConfig(
        hidden_size=48, num_heads=4, num_layers=2, ffn_dim=80, timestep_dim=16, text_dim=32, text_length=7, **kwargs
    )


def test_independent_sigma_time_and_unmodified_control_conditions():
    model = WanTransformer(config(variant="i2v", in_channels=48))
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.03)
    binding = WanBinding(model)
    clean, noise = torch.randn(2, 16, 2, 4, 4), torch.randn(2, 16, 2, 4, 4)
    image = torch.randn(2, 20, 2, 4, 4, requires_grad=True)
    control = torch.randn_like(clean, requires_grad=True)
    times, sigma = torch.tensor([7.0, 1001.0]), torch.tensor([0.3, 0.8])
    before = image.detach().clone()
    request = WanFlowRequest(
        clean, noise, sigma, times, (torch.randn(3, 32), torch.randn(5, 32)), image, torch.randn(2, 257, 1280), control
    )
    rng = torch.get_rng_state().clone()
    result = binding.predict(request)
    assert torch.equal(
        result.noisy_latents, (1 - sigma[:, None, None, None, None]) * clean + sigma[:, None, None, None, None] * noise
    )
    assert result.timesteps is times and torch.equal(result.target, noise - clean)
    assert torch.equal(image.detach(), before)
    assert result.conditioning is not None
    assert torch.equal(result.conditioning[:, :16], control)
    assert result.conditioning is not None
    assert torch.equal(result.conditioning[:, 16:, :1], image[:, 4:, :1])
    assert result.conditioning is not None
    assert torch.count_nonzero(result.conditioning[:, 16:, 1:]) == 0
    result.prediction.square().mean().backward()
    assert image.grad is not None and torch.count_nonzero(image.grad[:, 4:, :1])
    assert torch.count_nonzero(image.grad[:, :4]) == 0 and torch.count_nonzero(image.grad[:, :, 1:]) == 0
    assert control.grad is not None and torch.count_nonzero(control.grad)
    assert torch.equal(rng, torch.get_rng_state())
    targets = binding.adapter_targets()
    assert len(targets) == 24 and "blocks.1.cross_attn.k_img" in targets
    assert all(
        name.startswith("blocks.") and not any(term in name for term in ("norm", "modulation", "embedding"))
        for name in targets
    )


def test_cached_single_frame_features_and_coordinates():
    binding = WanBinding(WanTransformer(config(variant="i2v", in_channels=36)))
    records = [
        WanCacheRecord(
            torch.randn(16, 2, 4, 4),
            torch.randn(3, 32),
            torch.randn(20, 2, 4, 4),
            torch.randn(1, 257, 1280),
            frame_indices=(0, 24),
            target_position=1,
        )
    ]
    latents = records[0].latents.unsqueeze(0)
    noise, sigma, times = torch.randn_like(latents), torch.tensor([0.4]), torch.tensor([401.0])
    assert records[0].image_condition is not None and records[0].image_features is not None
    expected = binding.step(
        WanFlowRequest(
            latents,
            noise,
            sigma,
            times,
            (records[0].caption,),
            records[0].image_condition.unsqueeze(0),
            records[0].image_features[0].unsqueeze(0),
            frame_indices=((0, 24),),
        )
    )
    actual = binding.cached_step(records, noise, sigma, times)
    assert actual.sample_count == 1 and torch.equal(actual.loss, expected.loss)
    with pytest.raises(ValueError, match="nonempty"):
        binding.cached_step([], noise, sigma, times)


def build(session, algorithm):
    model = WanTransformer(config(version="2.2"))
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.03)
    binding = WanBinding(model)
    installation = session.own(
        ParameterInstallation(
            model,
            adapter=None if algorithm == "full" else AdapterConfig(algorithm=algorithm, rank=2, alpha=2.0),
            targets={} if algorithm == "full" else binding.adapter_targets(),
        )
    )
    with torch.no_grad():
        for value in installation.branches.parameters():
            value.normal_(0, 0.03)
    return (
        binding,
        {"model": model, "adapter": installation.branches},
        torch.optim.AdamW(installation.plan.parameters, lr=0.001),
    )


def advance(binding, optimizer, state):
    def steps():
        for batch_size in (2, 1):
            clean = torch.randn(batch_size, 16, 2, 4, 4)
            sigma = torch.rand(batch_size)
            yield binding.step(
                WanFlowRequest(
                    clean,
                    torch.randn_like(clean),
                    sigma,
                    sigma * 1000 + 1,
                    tuple(torch.randn(3, 32) for _ in range(batch_size)),
                )
            )

    update_phase(steps(), optimizer, state, max_grad_norm=1.0)


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
def test_wan_nonzero_adapters_uneven_accumulation_and_resume(tmp_path, algorithm):
    torch.manual_seed(84)
    with TrainingSession() as session, TrainingSession() as restored:
        binding, modules, optimizer = build(session, algorithm)
        state = RunState()
        identities, keys = tuple(map(id, binding.model.parameters())), tuple(binding.model.state_dict())
        before = [parameter.detach().clone() for parameter in optimizer.param_groups[0]["params"]]
        advance(binding, optimizer, state)
        assert state.samples == 3 and any(
            not torch.equal(a, b) for a, b in zip(before, optimizer.param_groups[0]["params"], strict=True)
        )
        saved, expected, actual = (
            tmp_path / "saved.safetensors",
            tmp_path / "expected.safetensors",
            tmp_path / "actual.safetensors",
        )
        save_checkpoint(saved, modules, optimizer, CheckpointCursor(**asdict(state)), IDENTITY)
        advance(binding, optimizer, state)
        advance(binding, optimizer, state)
        save_checkpoint(expected, modules, optimizer, CheckpointCursor(**asdict(state)), IDENTITY)
        fresh, fresh_modules, fresh_optimizer = build(restored, algorithm)
        for region in fresh.regions.regions:
            restored.own(RecomputeInstallation(region))
        cursor = restore_checkpoint(saved, fresh_modules, fresh_optimizer, IDENTITY)
        fresh_state = RunState(**cursor.model_dump())
        advance(fresh, fresh_optimizer, fresh_state)
        advance(fresh, fresh_optimizer, fresh_state)
        save_checkpoint(actual, fresh_modules, fresh_optimizer, CheckpointCursor(**asdict(fresh_state)), IDENTITY)
        assert expected.read_bytes() == actual.read_bytes() and state == fresh_state
        assert tuple(map(id, binding.model.parameters())) == identities and tuple(binding.model.state_dict()) == keys
