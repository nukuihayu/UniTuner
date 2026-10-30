"""Check FLUX.2 packed flow math and exact adapter resume through compiled recomputation."""

from dataclasses import asdict
from typing import Literal

import pytest
import torch

from unitunner.adapters.config import AdapterConfig
from unitunner.checkpoints.restore import restore_checkpoint
from unitunner.checkpoints.saving import save_checkpoint
from unitunner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unitunner.execution.compiler import CompilerInstallation
from unitunner.execution.recompute import RecomputeInstallation
from unitunner.models.flux2.conditioning import build_image_inputs, pack_text, unpack_image
from unitunner.models.flux2.config.model import Flux2Config
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.training.contracts import RunState
from unitunner.training.engine import update_phase
from unitunner.training.integrations.flux2.binding import Flux2Binding, Flux2FlowRequest
from unitunner.training.parameters import ParameterInstallation
from unitunner.training.session import TrainingSession

IDENTITY = ResumeIdentity(config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build(session, algorithm, variant: Literal["dev", "base", "distilled"] = "base"):
    model = Flux2Transformer(
        Flux2Config(
            version="dev" if variant == "dev" else "klein-4b",
            variant="distilled" if variant == "dev" else variant,
            hidden_size=32,
            num_heads=2,
            depth=1,
            depth_single_blocks=1,
            axes_dim=(4, 4, 4, 4),
            in_channels=8,
            context_in_dim=12,
        )
    )
    binding = Flux2Binding(model)
    targets = binding.adapter_targets()
    assert len(targets) == 10
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
    return Flux2FlowRequest(
        torch.randn(2, 8, 2, 3),
        torch.randn(2, 8, 2, 3),
        torch.tensor([0.244, 0.751]),
        (torch.randn(3, 12), torch.randn(5, 12)),
        ((torch.randn(8, 1, 2), torch.randn(8, 2, 1)), ()),
    )


def advance(binding, optimizer, state):
    update_phase((binding.step(request()) for _ in range(2)), optimizer, state, max_grad_norm=1.0)


@pytest.mark.parametrize("variant", ["dev", "base", "distilled"])
def test_flux2_binding_native_time_sign_interpolation_and_loss(variant):
    with TrainingSession() as session:
        binding, _, _ = build(session, "full", variant)
        inputs = request()
        result = binding.predict(inputs)
        fraction = inputs.sigma[:, None, None, None]
        noisy = (1 - fraction) * inputs.latents + fraction * inputs.noise
        predictions = []
        for index in range(2):
            image = build_image_inputs(noisy[index], inputs.references[index])
            context, ids = pack_text(inputs.captions[index])
            assert image.ids[0, :6, 0].tolist() == [0] * 6
            if index == 0:
                assert image.ids[0, 6:, 0].tolist() == [10, 10, 20, 20]
            guidance = torch.ones(1) if variant == "dev" else None
            prediction = binding.model(
                image.tokens, image.ids, inputs.sigma[index : index + 1], context[None], ids[None], guidance
            )
            predictions.append(unpack_image(prediction[0, :6], height=2, width=3))
        expected = torch.stack(predictions)
        torch.testing.assert_close(result.prediction, expected, atol=0, rtol=0)
        torch.testing.assert_close(result.timesteps, inputs.sigma, atol=0, rtol=0)
        torch.testing.assert_close(result.target, inputs.noise - inputs.latents, atol=0, rtol=0)
        torch.testing.assert_close(
            binding.step(inputs).loss, (result.prediction - result.target).square().mean(), atol=0, rtol=0
        )
        binding.step(inputs).loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in binding.model.parameters())
        for sigma in (torch.tensor([-0.1, 0.5]), torch.tensor([0.5, float("nan")]), torch.tensor([0.5, 1.1])):
            with pytest.raises(ValueError, match="finite B vector"):
                binding.predict(Flux2FlowRequest(inputs.latents, inputs.noise, sigma, inputs.captions))


@pytest.mark.parametrize("algorithm,variant", [("lora", "dev"), ("loha", "base"), ("lokr", "distilled")])
def test_flux2_full_and_adapters_resume_with_recompute_matches_continuous(tmp_path, algorithm, variant):
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
