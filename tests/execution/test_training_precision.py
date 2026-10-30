"""Verify header precision selection and actual mixed-storage adapter training."""

from contextlib import closing
from importlib import import_module
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.execution.test_native_strategies import FAMILIES, _case
from unituner.adapters.config import AdapterConfig
from unituner.artifacts.formats.quantization import FORMATS
from unituner.artifacts.tensors import save_tensors
from unituner.checkpoints.restore import restore_checkpoint
from unituner.checkpoints.saving import save_checkpoint
from unituner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unituner.config.execution import ExecutionConfig
from unituner.config.loading import load_training_config
from unituner.execution.installation import training_base_dtype
from unituner.execution.precision import artifact_precision, resolve_training_precision
from unituner.training.contracts import RunState, StepResult
from unituner.training.engine import update_phase
from unituner.training.parameters import ParameterInstallation
from unituner.training.session import TrainingSession


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    ("dtype", "name"), [(torch.float32, "fp32"), (torch.float16, "fp16"), (torch.bfloat16, "bf16")]
)
def test_auto_reads_matrix_dtype_without_materializing_payload(tmp_path, monkeypatch, dtype, name):
    path = tmp_path / "weights.safetensors"
    save_tensors({"weight": torch.ones(8, 8, dtype=dtype), "norm": torch.ones(8)}, path)
    monkeypatch.setattr("unituner.artifacts.tensors.TensorReader.read", lambda *a, **k: pytest.fail("payload read"))
    assert artifact_precision(path) == name
    config = ExecutionConfig(precision=name)
    assert training_base_dtype(config, adapter=True) == dtype
    assert training_base_dtype(config, adapter=False) == torch.float32


def test_auto_resolves_shards_and_explicit_precision_skips_detection(tmp_path, monkeypatch):
    for index in (1, 2):
        save_tensors(
            {f"block{index}.weight": torch.ones(8, 8, dtype=torch.bfloat16), f"norm{index}": torch.ones(8)},
            tmp_path / f"model-0000{index}-of-00002.safetensors",
        )
    config = load_training_config(
        Path(__file__).resolve().parents[2] / "docs/examples/flow.toml",
        overrides={"model": {"weights": str(tmp_path)}, "execution": {"precision": "auto"}},
    )
    assert resolve_training_precision(config).execution.precision == "bf16"
    explicit = config.model_copy(update={"execution": ExecutionConfig(precision="fp16")})
    monkeypatch.setattr(
        "unituner.execution.precision.artifact_precision", lambda *a: pytest.fail("explicit precision must win")
    )
    assert resolve_training_precision(explicit) is explicit


@pytest.mark.parametrize("precision", [name for name, spec in FORMATS.items() if not spec.calibrated])
def test_kitchen_precision_maps_to_quantization(precision):
    config = ExecutionConfig(precision=precision)
    assert config.precision == "auto" and config.quantization == precision


@pytest.mark.parametrize("name", [name for name, spec in FORMATS.items() if spec.calibrated])
@pytest.mark.parametrize("field", ["precision", "quantization"])
def test_calibrated_formats_are_not_training_precision_options(name, field):
    with pytest.raises(ValueError, match="offline-calibrated"):
        ExecutionConfig(**{field: name})


@pytest.mark.parametrize(("alias", "canonical"), [("fp8", "float8_e4m3fn"), ("convrot-int8", "int8_convrot")])
def test_precision_aliases_agree_without_rewriting_explicit_quantization(alias, canonical):
    assert ExecutionConfig(precision=alias, quantization=canonical).quantization == canonical
    assert ExecutionConfig(precision=canonical, quantization=alias).quantization == canonical
    assert ExecutionConfig(quantization=alias).quantization == alias
    with pytest.raises(ValueError, match="different Kitchen formats"):
        ExecutionConfig(precision=alias, quantization="nvfp4")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_low_precision_storage_with_fp32_adapter_update(tmp_path, family, dtype):
    torch.manual_seed(931)
    module_name = family.replace("-", "_")
    stem = "qwen_image_21" if family == "qwen-image-2-1" else module_name
    loading = import_module(f"unituner.models.{module_name}.loading")
    with TrainingSession() as session:
        binding, _, request = _case(family, session, "full")
        path = tmp_path / "model.safetensors"
        getattr(loading, f"save_{stem}")(binding.model, path)
        loaded = getattr(loading, f"load_{stem}")(path, dtype=dtype)
        assert any(p.dtype == dtype for p in loaded.parameters())
        binding.model = loaded
        owner = session.own(
            ParameterInstallation(
                loaded,
                adapter=AdapterConfig(algorithm="lora", rank=2),
                targets=binding.adapter_targets(),
                adapter_dtype=torch.float32,
            )
        )
        assert all(p.dtype == torch.float32 for p in owner.plan.parameters)
        before = [p.detach().clone() for p in owner.plan.parameters]
        optimizer = torch.optim.AdamW(owner.plan.parameters, lr=1e-3)
        with torch.autocast("cpu", dtype=dtype, cache_enabled=False):
            result = binding.step(request)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in owner.plan.parameters)
        optimizer.step()
        assert any(not torch.equal(a, b) for a, b in zip(before, owner.plan.parameters, strict=True))
        assert all(p.grad is None for p in loaded.parameters())


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_low_precision_base_fp32_factors_shared_optimizer_step(algorithm, precision):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    model = nn.Linear(8, 8, dtype=dtype)
    with closing(
        ParameterInstallation(
            model,
            adapter=AdapterConfig(algorithm=algorithm, rank=2),
            targets={"linear": model},
            adapter_dtype=torch.float32,
        )
    ) as owner:
        optimizer = torch.optim.SGD(owner.plan.parameters, lr=0.1)
        before = [p.detach().clone() for p in owner.plan.parameters]

        def steps():
            yield StepResult(model(torch.ones(2, 8)).float().square().mean(), 1)

        state = RunState()
        result = update_phase(steps(), optimizer, state, precision=precision)
        assert state.updates == 1 and torch.isfinite(result.mean_loss)
        assert any(not torch.equal(a, b) for a, b in zip(before, owner.plan.parameters, strict=True))
        assert all(p.dtype == torch.float32 for p in owner.plan.parameters)


def test_mixed_storage_checkpoint_restores_weights_and_optimizer_masters(tmp_path):
    model = nn.Linear(8, 8, dtype=torch.bfloat16)
    with closing(
        ParameterInstallation(
            model,
            adapter=AdapterConfig(algorithm="lora", rank=2),
            targets={"linear": model},
            adapter_dtype=torch.float32,
        )
    ) as owner:
        optimizer = torch.optim.AdamW(owner.plan.parameters, lr=0.01)

        def steps():
            yield StepResult(model(torch.ones(1, 8)).float().square().mean(), 1)

        update_phase(steps(), optimizer, RunState(), precision="bf16")
        modules = {"model": model, "adapter": owner.branches}
        expected = {
            role: {name: value.clone() for name, value in module.state_dict().items()}
            for role, module in modules.items()
        }
        identity = ResumeIdentity(
            config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64
        )
        path = tmp_path / "state.safetensors"
        save_checkpoint(path, modules, optimizer, CheckpointCursor(updates=1, microbatches=1, samples=1), identity)
        with torch.no_grad():
            for parameter in (*model.parameters(), *owner.branches.parameters()):
                parameter.add_(1)
        cursor = restore_checkpoint(path, modules, optimizer, identity)
        assert cursor.updates == 1
        for role, module in modules.items():
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, expected[role][name], rtol=0, atol=0)
        assert all(optimizer.state[p]["exp_avg"].dtype == torch.float32 for p in owner.plan.parameters)
