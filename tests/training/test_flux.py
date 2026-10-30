"""Check FLUX adapters, native flow math and exact resumed double-stream recomputation."""

from dataclasses import asdict
from typing import Literal

import pytest
import torch

from unituner.adapters.config import AdapterConfig
from unituner.checkpoints.restore import restore_checkpoint
from unituner.checkpoints.saving import save_checkpoint
from unituner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unituner.execution.compiler import CompilerInstallation
from unituner.execution.recompute import RecomputeInstallation
from unituner.models.flux.conditioning import FluxCondition, image_ids, pack_latents
from unituner.models.flux.config.model import FluxConfig
from unituner.models.flux.transformer import FluxTransformer
from unituner.training.contracts import RunState
from unituner.training.engine import update_phase
from unituner.training.integrations.flux.binding import FluxBinding, FluxFlowRequest
from unituner.training.parameters import ParameterInstallation
from unituner.training.session import TrainingSession

IDENTITY = ResumeIdentity(config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build(session, algorithm, variant: Literal["dev", "kontext", "fill"] = "dev"):
    model = FluxTransformer(
        FluxConfig(
            variant=variant,
            in_channels=48 if variant == "fill" else 8,
            out_channels=8,
            vec_in_dim=6,
            context_in_dim=12,
            hidden_size=32,
            num_heads=2,
            depth=2,
            depth_single_blocks=2,
            axes_dim=(4, 6, 6),
            mlp_ratio=2.0,
        )
    )
    binding = FluxBinding(model)
    targets = binding.adapter_targets()
    assert len(targets) == 20
    assert all("norm" not in name and "mod" not in name for name in targets)
    installation = session.own(
        ParameterInstallation(
            model,
            adapter=None if algorithm == "full" else AdapterConfig(algorithm=algorithm, rank=2, alpha=2.0),
            targets={} if algorithm == "full" else targets,
        )
    )
    with torch.no_grad():
        for parameter in installation.branches.parameters():
            parameter.normal_(0, 0.03)
    modules = {"model": model, "adapter": installation.branches}
    optimizer = torch.optim.AdamW(installation.plan.parameters, lr=0.001)
    return binding, modules, optimizer


def request(variant: Literal["dev", "kontext", "fill"] = "dev"):
    condition = None
    if variant == "kontext":
        condition = FluxCondition(
            torch.randn(2, 2, 8), image_ids(2, 1, 2, device=torch.device("cpu"), reference=True), (0, 2)
        )
    elif variant == "fill":
        condition = FluxCondition(torch.randn(2, 6, 8), fill_mask=torch.rand(2, 6, 32))
    return FluxFlowRequest(
        torch.randn(2, 2, 4, 6),
        torch.randn(2, 2, 4, 6),
        torch.tensor([0.244, 0.751]),
        torch.randn(2, 5, 12),
        torch.randn(2, 6),
        torch.tensor([2.5, 3.5]),
        condition,
    )


def advance(binding, optimizer, state):
    update_phase(
        (binding.step(request(binding.model.config.variant)) for _ in range(2)), optimizer, state, max_grad_norm=1.0
    )


def test_flux_binding_native_clock_sign_and_packing():
    with TrainingSession() as session:
        binding, _, _ = build(session, "lora")
        inputs = request()
        result = binding.predict(inputs)
        fraction = inputs.sigma[:, None, None, None]
        packed = pack_latents((1 - fraction) * inputs.latents + fraction * inputs.noise)
        expected = binding.model(
            packed,
            image_ids(2, 2, 3, device=torch.device("cpu")),
            inputs.captions,
            torch.zeros(2, 5, 3),
            inputs.sigma,
            inputs.pooled,
            inputs.guidance,
        )
        torch.testing.assert_close(pack_latents(result.prediction), expected, atol=0, rtol=0)
        torch.testing.assert_close(result.timesteps, inputs.sigma, atol=0, rtol=0)
        torch.testing.assert_close(result.target, inputs.noise - inputs.latents, atol=0, rtol=0)


@pytest.mark.parametrize(("algorithm", "variant"), [("lora", "dev"), ("loha", "kontext"), ("lokr", "fill")])
def test_flux_adapters_resume_with_tuple_recompute_and_compiler_matches_continuous(tmp_path, algorithm, variant):
    torch.manual_seed(83)
    with TrainingSession() as session, TrainingSession() as restored:
        binding, modules, optimizer = build(session, algorithm, variant)
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
        fresh, fresh_modules, fresh_optimizer = build(restored, algorithm, variant)
        for region in fresh.regions.regions:
            restored.own(RecomputeInstallation(region))
            restored.own(CompilerInstallation(region.module, backend="eager", fullgraph=True))
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
