"""Exercise phase-aware snapshots and prevent pruning after publication failures."""

from pathlib import Path

import pytest
import torch

from unituner.artifacts.tensors import TensorReader
from unituner.checkpoints.restore import restore_checkpoint
from unituner.checkpoints.schema import build_resume_identity
from unituner.config.loading import load_training_config
from unituner.training import checkpointing
from unituner.training.contracts import RunState, StepResult
from unituner.training.engine import update_phases
from unituner.training.optimizers.adamw import create_adamw
from unituner.training.output import resume_path
from unituner.training.parameters import ParameterGroup

EXAMPLE = Path(__file__).parents[2] / "docs/examples/full.toml"


@pytest.mark.parametrize("failure", [None, "weights", "snapshot"])
def test_named_phase_snapshot_preserves_state_and_failure_keeps_previous_snapshot(tmp_path, monkeypatch, failure):
    config = load_training_config(EXAMPLE, overrides={"output": {"directory": str(tmp_path)}})
    identity = build_resume_identity(config, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)
    modules = {name: torch.nn.Linear(1, 1, bias=False) for name in ("generator", "discriminator")}
    optimizers = {
        name: create_adamw((ParameterGroup("model", tuple(model.parameters())),), lr=0.01)
        for name, model in modules.items()
    }
    state = RunState()
    events = []

    def weights(update):
        events.append(("weights", update))

    def advance():
        steps = [StepResult(model.weight.square().mean(), 1, phase=name) for name, model in modules.items()]
        update_phases(steps, optimizers, state)

    def save(writer=weights):
        checkpointing.save_training_update(
            config,
            modules=modules,
            optimizer=optimizers,
            state=state,
            identity=identity,
            device=torch.device("cpu"),
            weights_writer=writer,
        )

    advance()
    save()
    first = resume_path(config, 1)
    original_bytes = first.read_bytes()
    advance()
    actual_snapshot = checkpointing.save_checkpoint

    def snapshot(*args, **kwargs):
        events.append(("snapshot", state.updates))
        if failure == "snapshot":
            raise OSError("snapshot failed")
        actual_snapshot(*args, **kwargs)

    def failing_weights(update):
        weights(update)
        raise OSError("weights failed")

    monkeypatch.setattr(checkpointing, "save_checkpoint", snapshot)
    if failure:
        with pytest.raises(OSError, match=f"{failure} failed"):
            save(failing_weights if failure == "weights" else weights)
        assert first.read_bytes() == original_bytes
        assert not resume_path(config, 2).exists()
        assert events == (
            [("weights", 1), ("weights", 2)]
            if failure == "weights"
            else [("weights", 1), ("weights", 2), ("snapshot", 2)]
        )
    else:
        save()
        assert not first.exists()
        assert events == [("weights", 1), ("weights", 2), ("snapshot", 2)]
        restored = {name: torch.nn.Linear(1, 1, bias=False) for name in modules}
        restored_optimizers = {
            name: create_adamw((ParameterGroup("model", tuple(model.parameters())),), lr=0.01)
            for name, model in restored.items()
        }
        cursor = restore_checkpoint(resume_path(config, 2), restored, restored_optimizers, identity)
        assert (cursor.updates, cursor.microbatches, cursor.samples) == (2, 2, 2)
        for name, model in modules.items():
            torch.testing.assert_close(restored[name].weight, model.weight, atol=0, rtol=0)
            torch.testing.assert_close(
                restored_optimizers[name].state[restored[name].weight]["exp_avg"],
                optimizers[name].state[model.weight]["exp_avg"],
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("backend", ["single", "foreach"])
def test_image_workflow_uses_optimizer_settings_in_periodic_snapshots(tmp_path, backend):
    from tests.models.test_z_image import build_model
    from unituner.data.cache.schema import CacheIdentity
    from unituner.data.cache.storage import LatentManifest, save_latent_manifest, save_latent_record
    from unituner.models.z_image.loading import load_z_image, save_z_image
    from unituner.training.workflows.common.images import train

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        weights = tmp_path / "source.safetensors"
        save_z_image(build_model(), weights)
        cache = tmp_path / "cache"
        entry = save_latent_record(
            cache,
            CacheIdentity(
                source_sha256="a" * 64, encoder_sha256="b" * 64, transform_sha256="c" * 64, layout="z-image/CHW+LD"
            ),
            torch.zeros(2, 4, 6),
            torch.zeros(3, 8),
        )
        save_latent_manifest(LatentManifest(entries=(entry,)), cache)
        config = load_training_config(
            EXAMPLE,
            overrides={
                "model": {"weights": str(weights)},
                "dataset": {"groups": {"main": {"cache_directory": str(cache), "batch_size": 1}}},
                "train": {
                    "max_updates": 2,
                    "accumulation_steps": 1,
                    "optimizer": {
                        "backend": backend,
                        "betas": [0.8, 0.95],
                        "eps": 1e-6,
                    },
                },
                "output": {"directory": str(tmp_path / "output"), "save_every_updates": 1},
            },
        )
        state = train(config, device=torch.device("cpu"))
        assert state == RunState(updates=2, microbatches=2, samples=2)
        with TensorReader(resume_path(config, 2)) as reader:
            import json

            manifest = json.loads(reader.metadata()["unituner.checkpoint"])
            identity = build_resume_identity(
                config,
                dataset_manifest_sha256=manifest["identity"]["dataset_manifest_sha256"],
                model_artifact_sha256=manifest["identity"]["model_artifact_sha256"],
            )
            assert manifest["identity"] == identity.model_dump(mode="json")
        restored = load_z_image(weights, device=torch.device("cpu"))
        restored_optimizer = create_adamw((ParameterGroup("model", tuple(restored.parameters())),), lr=0.01)
        restore_checkpoint(
            resume_path(config, 2), {"model": restored, "adapter": torch.nn.ModuleList()}, restored_optimizer, identity
        )
        for group in restored_optimizer.param_groups:
            assert group["betas"] == (0.8, 0.95) and group["eps"] == 1e-6
            assert group["foreach"] == (backend == "foreach") and group["fused"] is False
        # A completed workflow resumes without taking another update.
        assert train(config, device=torch.device("cpu"), resume=resume_path(config, 2)) == state
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("changed", ["name", "scale"])
def test_checkpoint_rejects_group_policy_change_before_any_phase_mutation(tmp_path, changed):
    from unituner.checkpoints.saving import save_checkpoint
    from unituner.checkpoints.schema import CheckpointCursor

    config = load_training_config(EXAMPLE)
    identity = build_resume_identity(config, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)
    modules = {name: torch.nn.Linear(1, 1) for name in ("generator", "discriminator")}
    optimizers = {
        name: create_adamw((ParameterGroup("model", tuple(module.parameters()), 0.5),), lr=0.01)
        for name, module in modules.items()
    }
    path = tmp_path / "state.safetensors"
    save_checkpoint(path, modules, optimizers, CheckpointCursor(updates=0, microbatches=0, samples=0), identity)
    restored = {name: torch.nn.Linear(1, 1) for name in modules}
    restored_optimizers = {
        name: create_adamw(
            (
                ParameterGroup(
                    "changed" if changed == "name" and name == "discriminator" else "model",
                    tuple(module.parameters()),
                    2.0 if changed == "scale" and name == "discriminator" else 0.5,
                ),
            ),
            lr=0.1,
        )
        for name, module in restored.items()
    }
    before = {
        name: {key: value.clone() for key, value in module.state_dict().items()} for name, module in restored.items()
    }
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="group name or learning-rate scale"):
        restore_checkpoint(path, restored, restored_optimizers, identity)
    assert torch.equal(rng, torch.get_rng_state())
    for name, module in restored.items():
        assert not restored_optimizers[name].state
        assert restored_optimizers[name].param_groups[0]["lr"] == 0.1 * (
            2.0 if changed == "scale" and name == "discriminator" else 0.5
        )
        for key, value in module.state_dict().items():
            assert torch.equal(value, before[name][key])
