"""Record lightweight, non-executable provenance for future offline weight analysis."""

import json
from pathlib import Path

from unituner.artifacts.atomic import atomic_output


def record_context(config, targets, identity, *, encoder_targets=()):
    """Persist exact native target order without copying tensors or private dataset content."""
    document = {
        "schema": 1,
        "family": str(config.model.family),
        "variant": config.model.variant,
        "method": config.method.kind,
        "parameter_mode": config.train.parameter_mode,
        "base_weights": str(Path(config.model.weights).resolve()),
        "encoder_base": str(Path(config.model.vae).resolve()) if config.model.vae else None,
        "adapter_targets": list(targets),
        "encoder_targets": list(encoder_targets),
        "identity": identity.model_dump(mode="json"),
        "tensorboard": str(Path(config.output.tensorboard).resolve()) if config.output.tensorboard else None,
        "run_name": config.output._run_name,
    }
    path = Path(config.output.directory) / "training-analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def publication_metadata(config, identity, update):
    """Attach actual completed update and numerical identity to portable exports."""
    return {
        "unituner.analysis.step": str(config.train.max_updates if update is None else update),
        "unituner.analysis.identity": json.dumps(identity.model_dump(mode="json"), sort_keys=True),
    }
