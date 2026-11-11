"""Resolve one-step sigma from matching model, adapter and encoder artifacts."""

import math
from pathlib import Path

from unituner.artifacts.tensors import TensorReader, resolve_shards
from unituner.config.restoration_inference import RestorationInferenceConfig


def restoration_sigma(config: RestorationInferenceConfig) -> float:
    paths = [config.model.weights]
    if config.adapter is not None:
        paths.append(config.adapter.weights)
    if config.encoder_weights is not None:
        paths.append(config.encoder_weights)
    recorded = []
    for path in paths:
        for shard in resolve_shards(Path(path)):
            with TensorReader(shard) as reader:
                value = reader.metadata().get("unituner.restoration.sigma")
            if value is not None:
                try:
                    sigma = float(value)
                except ValueError as error:
                    raise ValueError("Restoration artifact has invalid sigma metadata") from error
                if not math.isfinite(sigma) or not 0 < sigma < 1:
                    raise ValueError("Restoration artifact sigma must be finite and strictly inside (0,1)")
                recorded.append(sigma)
    if recorded and any(value != recorded[0] for value in recorded):
        raise ValueError("Restoration model/adapter/encoder artifacts record conflicting sigma values")
    if config.sigma is not None:
        if recorded and config.sigma != recorded[0]:
            raise ValueError("Explicit restoration sigma differs from the trained artifact")
        return config.sigma
    return recorded[0] if recorded else 0.244
