"""Check FLUX3 packed flow math and exact adapter resume through compiled recomputation."""

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
from unituner.models.flux3.conditioning import Flux3Sample, image_tokens, pack_samples, reference_tokens, text_tokens
from unituner.models.flux3.config.model import Flux3Config
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.training.contracts import RunState
from unituner.training.engine import update_phase
from unituner.training.integrations.flux3.binding import Flux3Binding, Flux3FlowRequest
from unituner.training.parameters import ParameterInstallation
from unituner.training.session import TrainingSession

IDENTITY = ResumeIdentity(config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build(session, algorithm, variant: Literal["base", "distilled"] = "base"):
    model = Flux3Transformer(
        Flux3Config(
            variant=variant,
            hidden_size=32,
            num_heads=2,
            depth=1,
            depth_single=1,
            axes_dim=(4, 4, 4, 4),
            image_channels=8,
            text_channels=12,
        )
    )
    binding = Flux3Binding(model)
    targets = binding.adapter_targets()
    assert len(targets) == 24
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
    return Flux3FlowRequest(
        torch.randn(2, 8, 2, 3),
        torch.randn(2, 8, 2, 3),
        torch.tensor([0.244, 0.751]),
        (torch.randn(3, 12), torch.randn(5, 12)),
        ((torch.randn(8, 1, 2), torch.randn(8, 2, 1)), ()),
    )


def advance(binding, optimizer, state):
    update_phase((binding.step(request()) for _ in range(2)), optimizer, state, max_grad_norm=1.0)


@pytest.mark.parametrize("variant", ["base", "distilled"])
def test_flux3_binding_native_time_sign_interpolation_and_loss(variant):
    with TrainingSession() as session:
        binding, _, _ = build(session, "full", variant)
        inputs = request()
        result = binding.predict(inputs)
        fraction = inputs.sigma[:, None, None, None]
        noisy = (1 - fraction) * inputs.latents + fraction * inputs.noise
        # Construct each independent sample to check packing, clean conditions and output layout.
        predictions = []
        for index in range(2):
            packed = pack_samples(
                [
                    Flux3Sample(
                        image_tokens(noisy[index]),
                        text_tokens(inputs.captions[index]),
                        reference_tokens(inputs.references[index]),
                    )
                ],
                inputs.sigma[index : index + 1],
            )
            assert torch.equal(packed.timesteps["image"], inputs.sigma[index].expand(6))
            assert torch.count_nonzero(packed.timesteps["text"]) == 0
            assert torch.count_nonzero(packed.timesteps["image_cond"]) == 0
            destination = inputs.sigma[index : index + 1] if variant == "distilled" else None
            predictions.append(binding.model(packed, destination).reshape(2, 3, 8).permute(2, 0, 1))
        expected = torch.stack(predictions)
        torch.testing.assert_close(result.prediction, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result.timesteps, inputs.sigma, atol=0, rtol=0)
        torch.testing.assert_close(result.target, inputs.noise - inputs.latents, atol=0, rtol=0)
        torch.testing.assert_close(
            binding.step(inputs).loss,
            (result.prediction - (inputs.noise - inputs.latents)).square().mean(),
            atol=0,
            rtol=0,
        )
        # A missing reference branch is legitimate; its parameters must remain inactive.
        binding.step(Flux3FlowRequest(inputs.latents, inputs.noise, inputs.sigma, inputs.captions)).loss.backward()
        condition_parameters = dict(binding.model.content_mode_blocks["image_cond"].named_parameters())
        assert condition_parameters and all(p.grad is None for p in condition_parameters.values())
        assert binding.model.get_parameter("single_blocks.0.q_proj.weight").grad is not None
        for sigma in (torch.tensor([-0.1, 0.5]), torch.tensor([0.5, float("nan")]), torch.tensor([0.5, 1.1])):
            with pytest.raises(ValueError, match="finite B vector"):
                binding.predict(Flux3FlowRequest(inputs.latents, inputs.noise, sigma, inputs.captions))


@pytest.mark.parametrize(
    "algorithm,variant", [("full", "base"), ("lora", "distilled"), ("loha", "base"), ("lokr", "distilled")]
)
def test_flux3_full_and_adapters_resume_with_recompute_matches_continuous(tmp_path, algorithm, variant):
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
