"""Observe native training boundaries without replacing sampler or optimizer behavior."""

import json
import subprocess
import sys
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from safetensors import safe_open

from unituner.artifacts.atomic import atomic_output
from unituner.checkpoints.schema import build_resume_identity
from unituner.config.training import TrainingConfig

if TYPE_CHECKING:
    from torch import Tensor, nn

    from unituner.adapters.config import AdapterConfig
    from unituner.training.parameters import ParameterInstallation


class TargetRecord(TypedDict):
    """Describe one selected native projection or full parameter."""

    name: str
    mode: str
    algorithm: str | None
    rank: int | None
    alpha: float | None
    category: str
    refiner: bool
    trainable_parameters: int
    state_keys: list[str]


AUDIT_VERSION = 1
_CURRENT: ContextVar["TrainingAudit | None"] = ContextVar("training_audit", default=None)


def _write_json(path: Path, value: object) -> None:
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _source_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _category(name: str) -> str:
    if "qkv_mlp" in name or ("single_blocks" in name and name.endswith(("linear1", "linear2"))):
        return "fused_attention_mlp"
    if any(word in name for word in ("mlp", "ffn", "feed_forward")):
        return "mlp"
    if any(word in name for word in ("attn", "attention", "q_proj", "k_proj", "v_proj", "qkv")):
        return "attention"
    if name.startswith("text_embedder.blocks.") and name.endswith(".proj"):
        return "attention"
    if ".proj." in name and name.rsplit(".", 1)[-1] in {"q", "k", "v", "o"}:
        return "attention"
    return "other"


class TrainingAudit:
    """Record native selections and verify publisher keys using output-owned paths."""

    def __init__(self, config: TrainingConfig, *, resume: Path | None = None) -> None:
        self.config = config
        self.directory = Path(config.output.directory) / "training-audit"
        self.targets: list[TargetRecord] = []
        self.publications: dict[str, list[str]] = {}
        self.latest: dict[str, str] = {}
        # Reuse native numerical identity. Placeholder content digests perform no model IO;
        # the actual checkpoint restore separately verifies real dataset/model identities.
        identity = build_resume_identity(
            config, dataset_manifest_sha256="0" * 64, model_artifact_sha256="0" * 64
        ).config_sha256
        previous = None
        if resume is not None:
            previous = next(
                (
                    parent / "training-audit" / "manifest.json"
                    for parent in resume.parents
                    if (parent / "training-audit" / "manifest.json").is_file()
                ),
                None,
            )
        resume_audit = "legacy_checkpoint_native_validation" if resume else None
        if previous is not None:
            old = json.loads(previous.read_text(encoding="utf-8"))
            if "audit_version" not in old:
                # Earlier notebook wrappers used a different, over-strict fingerprint.
                # Do not compare schemas; native checkpoint restore remains authoritative.
                resume_audit = "legacy_wrapper_native_validation"
            elif old.get("audit_version") != AUDIT_VERSION or old.get("resume_identity") != identity:
                raise ValueError(
                    "Native training audit resume identity differs (training configuration or audit version)"
                )
            else:
                resume_audit = "matched"
        restoration = config.method.kind == "restoration"
        restoration_config = config.method.restoration
        if restoration:
            if restoration_config is None:
                raise ValueError("Restoration audit requires resolved restoration settings")
            noise = {
                "applicable": False,
                "midtimestep": restoration_config.midtimestep,
                "sigma": restoration_config.sigma,
                "calibrate": restoration_config.calibrate,
                "posterior": restoration_config.posterior,
                "not_applicable": ["timestep_sampling", "shift", "sigma_min", "sigma_max"],
            }
        else:
            noise = {
                "applicable": True,
                "sampling": config.train.timestep_sampling,
                "implementation": "native training_sigma_grid/sample_sigmas",
            }
        self.manifest: dict[str, object] = {
            "audit_version": AUDIT_VERSION,
            "status": "configured",
            "resume_identity": identity,
            "trainer_commit": _source_commit(),
            "noise": noise,
            "resume_audit": resume_audit,
            "model_references": config.model.model_dump(mode="json"),
        }
        _write_json(self.directory / "resolved-config.json", config.model_dump(mode="json"))
        _write_json(
            self.directory / "runtime-info.json",
            {
                "resume": resume,
                "output": config.output.model_dump(mode="json"),
                "argv": sys.argv,
            },
        )
        self.save()

    def save(self) -> None:
        """Atomically publish the current audit summary."""
        _write_json(self.directory / "manifest.json", self.manifest)

    def record_installation(
        self, installation: "ParameterInstallation", model: "nn.Module", adapter: "AdapterConfig | None"
    ) -> None:
        """Describe actual trainable native targets after successful attachment."""
        plan = installation.plan
        by_name = dict(model.named_parameters())
        for index, name in enumerate(plan.target_names):
            branch = installation.branches[index] if plan.mode == "adapter" else None
            parameters = list(branch.parameters()) if branch is not None else [by_name[name]]
            self.targets.append(
                {
                    "name": name,
                    "mode": plan.mode,
                    "algorithm": adapter.algorithm if adapter else None,
                    "rank": adapter.rank if adapter else None,
                    "alpha": adapter.alpha if adapter else None,
                    "category": _category(name),
                    "refiner": "refiner" in name or name.startswith("text_embedder.blocks."),
                    "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
                    "state_keys": list(branch.state_dict()) if branch is not None else [name],
                }
            )
        # Aggregate from native groups to avoid counting non-trainable base weights.
        self.manifest["target_count"] = len(self.targets)
        self.manifest["trainable_parameters"] = sum(row["trainable_parameters"] for row in self.targets)
        self.manifest["parameters_by_category"] = {
            category: sum(row["trainable_parameters"] for row in self.targets if row["category"] == category)
            for category in {str(row["category"]) for row in self.targets}
        }
        total = sum(row["trainable_parameters"] for row in self.targets)
        self.manifest["parameter_fractions"] = {
            category: sum(row["trainable_parameters"] for row in self.targets if row["category"] == category) / total
            if total
            else 0
            for category in {row["category"] for row in self.targets}
        }
        refiner = sum(row["trainable_parameters"] for row in self.targets if row["refiner"])
        self.manifest["refiner_fraction"] = refiner / total if total else 0
        _write_json(self.directory / "targets.json", self.targets)
        self.save()

    def record_noise_grid(self, family: str, sampling: str, grid: "Tensor | None") -> None:
        """Record the actual native grid without replacing its construction or draws."""
        _write_json(
            self.directory / "sigma-grid.json",
            {
                "family": family,
                "sampling": sampling,
                "sigmas": None if grid is None else grid.tolist(),
                "distribution": "continuous_uniform" if grid is None else "uniform_grid_index",
            },
        )

    def record_export(self, path: Path, keys: Iterable[str]) -> None:
        """Capture exact keys assembled by the native algorithm/layout publisher."""
        self.publications[str(path.resolve())] = sorted(keys)
        _write_json(self.directory / "published-keys.json", self.publications)

    def record_latest(self, path: Path, source: Path) -> None:
        """Associate a final alias with its exact native step publication."""
        self.latest[str(path.resolve())] = str(source.resolve())

    def finish(self) -> None:
        """Check final, stage and continuation artifacts without reading tensor payloads."""
        from unituner.training.output import weights_path

        root = Path(self.config.output.directory)
        expected_final = [weights_path(self.config)]
        if self.config.wan_experts is not None:
            expected_final.append(weights_path(self.config, expert="high"))
        if not self.targets:
            raise ValueError("Native training audit recorded no trainable targets")
        if any(not path.is_file() for path in expected_final):
            raise ValueError("Training returned successfully without a final safetensors export in weights/")
        rows = []
        name = self.config.output.name
        for path in sorted((root / "weights").glob("*.safetensors")):
            stem = path.stem.removesuffix(".high_noise").removesuffix(".encoder")
            suffix = stem.removeprefix(name + "-")
            if stem != name and (not stem.startswith(name + "-") or not suffix.isdecimal()):
                continue
            with safe_open(path, framework="pt", device="cpu") as reader:
                keys = sorted(reader.keys())
            key = str(path.resolve())
            expected = self.publications.get(self.latest.get(key, key))
            if expected is not None and keys != expected:
                raise ValueError(f"Exported keys differ from native publisher: {path}")
            rows.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "kind": "final" if stem == name else "stage",
                    "component": "encoder"
                    if ".encoder." in path.name
                    else "high"
                    if ".high_noise." in path.name
                    else "low",
                    "status": "publisher_keys_match" if expected is not None else "header_recorded",
                    "keys": keys,
                }
            )
        _write_json(self.directory / "exports.json", rows)
        _write_json(
            self.directory / "resume-files.json",
            [
                path.relative_to(root).as_posix()
                for path in sorted((root / "resume").glob("*.safetensors"))
                if path.name.startswith(name + ".state-")
            ],
        )


def current_audit() -> TrainingAudit | None:
    """Return the audit owned by the current native run, if explicitly enabled."""
    return _CURRENT.get()


@contextmanager
def audit_training(config: TrainingConfig, *, resume: Path | None = None) -> Generator[TrainingAudit | None]:
    """Scope native observations without intercepting CLI exits or modifying numerical APIs."""
    if not config.output.audit:
        yield None
        return
    audit = TrainingAudit(config, resume=resume)
    token = _CURRENT.set(audit)
    try:
        yield audit
        audit.finish()
        audit.manifest["status"] = "complete"
        audit.save()
    except BaseException as error:
        audit.manifest.update(status="failed", error=str(error))
        audit.save()
        raise
    finally:
        _CURRENT.reset(token)
