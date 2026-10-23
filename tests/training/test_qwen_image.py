"""Verify Qwen Image native layer packing and exact full/adapter accumulated resume."""

import os
from dataclasses import asdict

import pytest
import torch

from unitunner.adapters.config import AdapterConfig
from unitunner.checkpoints.restore import restore_checkpoint
from unitunner.checkpoints.saving import save_checkpoint
from unitunner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unitunner.execution.compiler import CompilerInstallation
from unitunner.execution.recompute import RecomputeInstallation
from unitunner.models.qwen_image.conditioning import pack_latents, unpack_latents
from unitunner.models.qwen_image.config.model import QwenImageConfig
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.training.engine import update_phase
from unitunner.training.integrations.qwen_image.binding import QwenImageBinding, QwenImageFlowRequest
from unitunner.training.parameters import ParameterInstallation
from unitunner.training.session import TrainingSession
from unitunner.training.state import RunState

IDENTITY = ResumeIdentity(config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build(session, algorithm, variant):
    config = QwenImageConfig(
        variant=variant,
        in_channels=8,
        out_channels=2,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=12,
        axes_dims_rope=(4, 6, 6),
    )
    binding = QwenImageBinding(QwenImageTransformer(config).to(os.environ.get("UNITUNNER_TEST_DEVICE", "cpu")))
    adapter = None if algorithm == "full" else AdapterConfig(algorithm=algorithm, rank=2, alpha=2.0)
    installation = session.own(
        ParameterInstallation(
            binding.model, adapter=adapter, targets=None if adapter is None else binding.adapter_targets()
        )
    )
    with torch.no_grad():
        for parameter in installation.branches.parameters():
            parameter.normal_(0, 0.03)
    optimizer = torch.optim.AdamW(installation.plan.parameters, lr=0.001)
    return binding, {"model": binding.model, "adapter": installation.branches}, optimizer


def request(variant, remove_base):
    layers = 3 if variant == "layered" else 1
    device = os.environ.get("UNITUNNER_TEST_DEVICE", "cpu")
    return QwenImageFlowRequest(
        torch.randn(2, 2, layers, 4, 6, device=device),
        torch.randn(2, 2, layers, 4, 6, device=device),
        torch.tensor([0.2443, 0.7517], device=device),
        (torch.randn(3, 12, device=device), torch.randn(5, 12, device=device)),
        (torch.randn(2, 2, 1, 2, 4, device=device),) if variant == "edit-2511" else (),
        remove_base,
    )


def cuda_devices():
    return (
        (torch.cuda.current_device(),)
        if torch.device(os.environ.get("UNITUNNER_TEST_DEVICE", "cpu")).type == "cuda"
        else ()
    )


def advance(binding, optimizer, state, remove_base):
    update_phase(
        (binding.step(request(binding.model.config.variant, remove_base)) for _ in range(2)),
        optimizer,
        state,
        max_grad_norm=1.0,
    )


@pytest.mark.parametrize(
    "algorithm,variant,remove_base",
    [("full", "original", False), ("lora", "edit-2511", False), ("loha", "layered", False), ("lokr", "layered", True)],
)
def test_qwen_image_shared_flow_resume_and_compiled_recompute(tmp_path, algorithm, variant, remove_base):
    torch.manual_seed(83)
    with TrainingSession() as session, TrainingSession() as restored:
        binding, modules, optimizer = build(session, algorithm, variant)
        identities, keys = tuple(map(id, binding.model.parameters())), tuple(binding.model.state_dict())
        state = RunState()
        before = [p.detach().clone() for group in optimizer.param_groups for p in group["params"]]
        advance(binding, optimizer, state, remove_base)
        assert any(
            not torch.equal(a, b)
            for a, b in zip(before, [p for group in optimizer.param_groups for p in group["params"]], strict=True)
        )
        saved = tmp_path / "saved.safetensors"
        save_checkpoint(
            saved, modules, optimizer, CheckpointCursor(**asdict(state)), IDENTITY, cuda_devices=cuda_devices()
        )
        advance(binding, optimizer, state, remove_base)
        advance(binding, optimizer, state, remove_base)
        expected = tmp_path / "expected.safetensors"
        save_checkpoint(
            expected, modules, optimizer, CheckpointCursor(**asdict(state)), IDENTITY, cuda_devices=cuda_devices()
        )
        fresh, fresh_modules, fresh_optimizer = build(restored, algorithm, variant)
        for region in fresh.regions.regions:
            restored.own(RecomputeInstallation(region))
            restored.own(CompilerInstallation(region.module, backend="eager", fullgraph=True))
        cursor = restore_checkpoint(saved, fresh_modules, fresh_optimizer, IDENTITY, cuda_devices=cuda_devices())
        fresh_state = RunState(**cursor.model_dump())
        advance(fresh, fresh_optimizer, fresh_state, remove_base)
        advance(fresh, fresh_optimizer, fresh_state, remove_base)
        actual = tmp_path / "actual.safetensors"
        save_checkpoint(
            actual,
            fresh_modules,
            fresh_optimizer,
            CheckpointCursor(**asdict(fresh_state)),
            IDENTITY,
            cuda_devices=cuda_devices(),
        )
        assert actual.read_bytes() == expected.read_bytes()
        assert state == fresh_state
        assert tuple(map(id, binding.model.parameters())) == identities and tuple(binding.model.state_dict()) == keys


def test_qwen_image_explicit_single_channel_layer_packing_and_gradients():
    values = torch.arange(16, dtype=torch.float32).reshape(1, 1, 2, 2, 4).requires_grad_(True)
    packed = pack_latents(values)
    expected = torch.tensor([[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]], dtype=torch.float32)
    assert torch.equal(packed, expected)
    unpacked = unpack_latents(packed, layers=2, height=2, width=4)
    assert torch.equal(unpacked, values)
    unpacked.square().sum().backward()
    assert values.grad is not None
    assert torch.equal(values.grad, 2 * values)
    with pytest.raises(ValueError, match="even H/W"):
        pack_latents(torch.zeros(1, 2, 1, 3, 4))
    with pytest.raises(ValueError, match="exact positive"):
        unpack_latents(packed, layers=1, height=2, width=4)


def test_qwen_cached_samples_match_batched_loss_and_gradients():
    from unitunner.data.cache.storage import ImageCacheRecord

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(779)
        model = QwenImageTransformer(
            QwenImageConfig(
                variant="edit-2511",
                in_channels=8,
                out_channels=2,
                num_layers=2,
                attention_head_dim=16,
                num_attention_heads=2,
                joint_attention_dim=12,
                axes_dims_rope=(4, 6, 6),
            )
        )
        binding = QwenImageBinding(model)
        clean = torch.randn(2, 2, 1, 2, 4)
        noise = torch.randn_like(clean)
        sigma = torch.tensor([0.27, 0.81])
        captions = (torch.randn(3, 12), torch.randn(7, 12))
        references = (torch.randn(2, 2, 1, 2, 2), torch.randn(2, 2, 1, 4, 2))
        expected = binding.step(QwenImageFlowRequest(clean, noise, sigma, captions, references))
        expected.loss.backward()
        gradients = {
            name: None if parameter.grad is None else parameter.grad.clone()
            for name, parameter in model.named_parameters()
        }
        model.zero_grad(set_to_none=True)
        records = [
            ImageCacheRecord(clean[index], captions[index], references=tuple(value[index] for value in references))
            for index in range(2)
        ]
        observed = binding.cached_step(records, noise, sigma)
        assert observed.sample_count == expected.sample_count == 2
        torch.testing.assert_close(observed.loss, expected.loss, atol=1e-5, rtol=1e-4)
        observed.loss.backward()
        for name, parameter in model.named_parameters():
            if gradients[name] is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-5, rtol=1e-4)
    finally:
        torch.set_num_threads(previous)
