"""Exercise native audit observations, publication formats and resume continuity."""

import json
from contextlib import closing
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from unituner.artifacts.tensors import TensorReader
from unituner.checkpoints.restore import restore_checkpoint
from unituner.checkpoints.saving import save_checkpoint
from unituner.checkpoints.schema import CheckpointCursor, build_resume_identity
from unituner.config.loading import load_training_config
from unituner.training.audit import TrainingAudit, audit_training, current_audit
from unituner.training.output import prepare_training_output, publish_latest_weights, resume_path, weights_path
from unituner.training.parameters import ParameterInstallation
from unituner.training.publishing import publish_adapter
from unituner.training.timesteps import training_sigma_grid


def _config(directory, algorithm="lora", adapter_format="comfyui"):
    return prepare_training_output(
        load_training_config(
            Path(__file__).resolve().parents[2] / "docs/examples/flow.toml",
            overrides={
                "model": {"family": "qwen-image-2-1", "variant": "base"},
                "train": {"adapter": {"algorithm": algorithm, "rank": 2, "alpha": 2.0}},
                "output": {
                    "directory": str(directory),
                    "name": "concept",
                    "audit": True,
                    "adapter_format": adapter_format,
                },
            },
        )
    )


@pytest.mark.parametrize(
    "algorithm,adapter_format",
    [
        ("lora", "comfyui"),
        ("lora", "diffusers"),
        ("loha", "comfyui"),
        ("lokr", "comfyui"),
    ],
)
def test_native_update_checkpoint_resume_and_exports(tmp_path, algorithm, adapter_format):
    config = _config(tmp_path / "run", algorithm, adapter_format)
    assert config.train.adapter is not None
    model = nn.Linear(4, 4, bias=False)
    target = "transformer_blocks.0.attn.to_q"
    with (
        audit_training(config) as audit,
        closing(
            ParameterInstallation(
                model,
                adapter=config.train.adapter,
                targets={target: model},
            )
        ) as installation,
    ):
        optimizer = torch.optim.AdamW(installation.branches.parameters(), lr=1e-3)
        model(torch.ones(1, 4)).square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        identity = build_resume_identity(config, dataset_manifest_sha256="0" * 64, model_artifact_sha256="1" * 64)
        checkpoint = resume_path(config, 1)
        modules = {"adapter": installation.branches}
        save_checkpoint(
            checkpoint, modules, optimizer, CheckpointCursor(updates=1, microbatches=1, samples=1), identity
        )
        cursor = restore_checkpoint(checkpoint, modules, optimizer, identity)
        assert cursor.updates == 1
        training_sigma_grid("qwen-image-2-1", "shifted", device=torch.device("cpu"))
        publish_adapter(
            weights_path(config, update=1),
            branches=installation.branches,
            target_names=installation.plan.target_names,
            adapter=config.train.adapter,
            base_model="qwen-image-2-1",
            module_identifier="networks.lora",
            dtype=torch.float32,
            adapter_format=adapter_format,
            family="qwen-image-2-1",
        )
        publish_latest_weights(config, 1)
        assert audit is current_audit()
    assert current_audit() is None
    assert audit is not None
    directory = tmp_path / "run/training-audit"
    rows = json.loads((directory / "exports.json").read_text())
    assert {row["kind"] for row in rows} == {"stage", "final"}
    assert all(row["status"] == "publisher_keys_match" for row in rows)
    assert json.loads((directory / "manifest.json").read_text())["status"] == "complete"
    assert json.loads((directory / "targets.json").read_text())[0]["algorithm"] == algorithm
    assert len(json.loads((directory / "resume-files.json").read_text())) == 1
    assert len(json.loads((directory / "sigma-grid.json").read_text())["sigmas"]) == 1000
    updated = config.model_copy(
        update={
            "output": config.output.model_copy(
                update={
                    "directory": str(tmp_path / "resumed"),
                    "tensorboard": "other-logs",
                    "save_every_updates": 10,
                }
            )
        }
    )
    resumed = TrainingAudit(updated, resume=checkpoint)
    assert resumed.manifest["resume_audit"] == "matched"
    before = (directory / "manifest.json").read_bytes()
    for change in (
        {"include_layers": ("*.attn.*",)},
        {"adapter": config.train.adapter.model_copy(update={"rank": 4})},
        {"timestep_sampling": "uniform"},
    ):
        with pytest.raises(ValueError, match="resume identity differs"):
            TrainingAudit(
                config.model_copy(update={"train": config.train.model_copy(update=change)}), resume=checkpoint
            )
    assert (directory / "manifest.json").read_bytes() == before
    # Alter the real final header, retaining a valid safetensors file.
    with TensorReader(weights_path(config)) as reader:
        tensors = {key: reader.read(key) for key in reader.keys()}
    tensors.pop(next(iter(tensors)))
    save_file(tensors, weights_path(config))
    with pytest.raises(ValueError, match="differ from native publisher"):
        audit.finish()


def test_stage_weights_do_not_substitute_for_final(tmp_path):
    config = _config(tmp_path)
    assert config.train.adapter is not None
    model = nn.Linear(4, 4, bias=False)
    with pytest.raises(ValueError, match="without a final"), audit_training(config):
        with closing(
            ParameterInstallation(model, adapter=config.train.adapter, targets={"proj": model})
        ) as installation:
            publish_adapter(
                weights_path(config, update=1),
                branches=installation.branches,
                target_names=installation.plan.target_names,
                adapter=config.train.adapter,
                base_model="qwen-image-2-1",
                module_identifier="networks.lora",
                dtype=torch.float32,
            )
    assert current_audit() is None
    assert json.loads((tmp_path / "training-audit/manifest.json").read_text())["status"] == "failed"


def test_audit_preserves_original_exception_and_legacy_checkpoint_support(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="native failure"), audit_training(config):
        raise RuntimeError("native failure")
    assert current_audit() is None
    audit = TrainingAudit(config, resume=tmp_path / "legacy/resume/step.safetensors")
    # The current run is discoverable; a truly external old checkpoint has no audit.
    legacy = TrainingAudit(config, resume=Path("/legacy-checkpoints/step.safetensors"))
    assert audit.manifest["resume_audit"] == "matched"
    assert legacy.manifest["resume_audit"] == "legacy_checkpoint_native_validation"


def test_disabled_audit_does_not_create_files_or_change_execution(tmp_path):
    config = _config(tmp_path)
    config = config.model_copy(update={"output": config.output.model_copy(update={"audit": False})})
    with audit_training(config) as audit:
        assert audit is None and current_audit() is None
    assert not (tmp_path / "training-audit").exists()


def test_restoration_summary_marks_flow_controls_inapplicable(tmp_path):
    config = load_training_config(
        Path(__file__).resolve().parents[2] / "docs/examples/restoration.toml",
        overrides={
            "method": {"kind": "restoration"},
            "train": {"adapter": {"algorithm": "lora", "rank": 2, "alpha": 2.0}},
            "output": {"directory": str(tmp_path), "audit": True},
        },
    )
    audit = TrainingAudit(config)
    noise = json.loads((audit.directory / "manifest.json").read_text())["noise"]
    assert noise["applicable"] is False
    assert config.method.restoration is not None
    assert noise["midtimestep"] == config.method.restoration.midtimestep
    assert "shift" not in noise and "sigma_min" in noise["not_applicable"]


def test_legacy_wrapper_fingerprint_is_not_compared_to_native_schema(tmp_path):
    config = _config(tmp_path)
    directory = tmp_path / "training-audit"
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "engine": "unituner",
                "interface_version": 2,
                "resume_identity": "incompatible-wrapper-format",
            }
        )
    )
    audit = TrainingAudit(config, resume=tmp_path / "resume/step.safetensors")
    assert audit.manifest["resume_audit"] == "legacy_wrapper_native_validation"
