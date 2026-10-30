"""Qualify frozen feature math and continuation on compact native model fixtures."""

from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.execution.test_native_strategies import FAMILIES, _case, _move
from unitunner.checkpoints.restore import restore_checkpoint
from unitunner.checkpoints.saving import save_checkpoint
from unitunner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unitunner.config.execution import ExecutionConfig
from unitunner.config.features import LayerSyncConfig, RepaConfig
from unitunner.config.restoration import MethodConfig
from unitunner.config.routing import TreadConfig
from unitunner.data.cache.representation import (
    RepaManifest,
    RepaTargets,
    save_repa_manifest,
    save_repa_target,
)
from unitunner.data.cache.storage import CacheEntry
from unitunner.execution.installation import ExecutionInstallation
from unitunner.training.contracts import RunState
from unitunner.training.engine import update_phase
from unitunner.training.objectives import install_objectives
from unitunner.training.session import TrainingSession


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("compiled", [False, True])
def test_repa_layersync_tread_recompute_and_exact_resume(tmp_path, family, device_name, compiled):
    """Use synthetic frozen grids to test loss math, never encoder quality or full weights."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if compiled and device_name != "cuda":
        pytest.skip("Inductor training requires CUDA")
    torch.set_num_threads(1)
    torch.manual_seed(954)
    device = torch.device("cuda", torch.cuda.current_device()) if device_name == "cuda" else torch.device("cpu")
    cuda_devices = (device.index,) if device_name == "cuda" else ()
    with TrainingSession() as session, sdpa_kernel(SDPBackend.MATH):
        binding, modules, request = _case(family, session)
        for module in modules.values():
            module.to(device)
        request = _move(request, device)
        latents = request.video if family == "minimax-h3" else request.latents
        sources = tuple(
            CacheEntry(file=f"{index + 1:064x}.safetensors", sha256=f"{index + 1:064x}")
            for index in range(latents.shape[0])
        )
        # Deliberately different encoder/student grid sizes exercise interpolation.
        entries = tuple(
            save_repa_target(tmp_path, source, torch.randn(2, 3, 2, 7), encoder_sha256="a" * 64) for source in sources
        )
        save_repa_manifest(RepaManifest(encoder_sha256="a" * 64, encoder_dim=7, entries=entries), tmp_path)
        config = RepaConfig(weight=0.2, encoder_dim=7, encoder_sha256="a" * 64, cache_directory=str(tmp_path))
        targets = RepaTargets(config)
        targets.validate_sources(sources)
        alignment = install_objectives(
            binding.model,
            binding.regions,
            MethodConfig(layersync=LayerSyncConfig(weight=0.3), repa=config, tread=TreadConfig(end_layer=2)),
            session=session,
            targets=targets,
            device=device,
        )
        assert alignment is not None
        session.own(
            ExecutionInstallation(
                binding.regions,
                ExecutionConfig(
                    compile=compiled,
                    recompute="block",
                    residency="frozen-blocks" if device_name == "cuda" else "resident",
                    use_pinned_memory_for_block_swap=device_name == "cuda",
                ),
                device=device,
            )
        )
        assert alignment.projection is not None
        modules["repa"] = alignment.projection
        assert alignment.projection is not None
        parameters = tuple(modules["adapter"].parameters()) + tuple(alignment.projection.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=0.001)

        def step():
            with alignment.capture():
                alignment.set_targets(sources, latents)
                result = binding.step(request)
                return alignment.step(result)

        with alignment.capture():
            alignment.set_targets(sources, latents)
            result = binding.step(request)
            students, teachers = alignment.recorder.features[0], alignment.recorder.features[1]
            count = sum(value.shape[0] for value in students)
            expected = result.loss + 0.3 * sum(
                -F.cosine_similarity(left.float(), right.detach().float(), dim=-1).mean() * left.shape[0] / count
                for left, right in zip(students, teachers, strict=True)
            )
            rows = [row.unsqueeze(0) for capture in students for row in capture.unbind()]
            repa = []
            for row, (target, grid) in zip(rows, alignment._target_rows, strict=True):
                target = target.detach().unsqueeze(0).to(device).permute(0, 4, 1, 2, 3)
                resized = F.interpolate(
                    target, size=(grid.frames, grid.height, grid.width), mode="trilinear", align_corners=False
                )
                assert alignment.projection is not None
                projected = alignment.projection.layers(row.float())
                repa.append(
                    -F.cosine_similarity(projected, resized.permute(0, 2, 3, 4, 1).reshape_as(projected), dim=-1).mean()
                )
            expected = expected + 0.2 * (sum(repa) / len(repa))
            actual = alignment.step(result)
            torch.testing.assert_close(actual.loss, expected, atol=0, rtol=0)
        observed = torch.autograd.grad(actual.loss, parameters, allow_unused=True)
        assert all(value is None or torch.isfinite(value).all() for value in observed)
        assert any(value is not None and value.abs().sum() > 0 for value in observed[-4:])
        assert not alignment.recorder.features and not alignment._target_rows
        state = RunState()
        identity = ResumeIdentity(
            config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64
        )
        checkpoint = tmp_path / "checkpoint.safetensors"
        for update in range(3):
            update_phase((step() for _ in range(2)), optimizer, state, max_grad_norm=1.0)
            if update == 0:
                save_checkpoint(
                    checkpoint,
                    modules,
                    optimizer,
                    CheckpointCursor(**asdict(state)),
                    identity,
                    cuda_devices=cuda_devices,
                    reference_frozen_parameters=True,
                )
        expected_states = {
            name: {key: value.clone() for key, value in module.state_dict().items()} for name, module in modules.items()
        }
        expected_cpu = torch.get_rng_state().clone()
        expected_cuda = torch.cuda.get_rng_state(device).clone() if cuda_devices else None
        cursor = restore_checkpoint(checkpoint, modules, optimizer, identity, cuda_devices=cuda_devices)
        state = RunState(**cursor.model_dump())
        for _ in range(2):
            update_phase((step() for _ in range(2)), optimizer, state, max_grad_norm=1.0)
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, expected_states[name][key], atol=0, rtol=0)
        assert torch.equal(torch.get_rng_state(), expected_cpu)
        if cuda_devices:
            assert expected_cuda is not None
            assert torch.equal(torch.cuda.get_rng_state(device), expected_cuda)
        assert state.updates == 3 and state.microbatches == 6
    assert binding.model.feature_recorder is None
    assert binding.model.token_routing is None
