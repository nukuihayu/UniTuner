"""Resolve automatic training precision from checkpoint headers before model loading."""

from collections import Counter
from math import prod
from pathlib import Path

from unitunner.artifacts.formats.precision import FloatingPrecision
from unitunner.artifacts.tensors import TensorReader, resolve_shards
from unitunner.config.training import TrainingConfig


def artifact_precision(path: Path) -> FloatingPrecision:
    """Choose the dominant matrix dtype; quantized bases default to BF16 compute.

    Count elements rather than keys so small FP32 norms do not override BF16
    projections. Inspect all shard headers without allocating weight tensors.
    """
    counts: Counter[str] = Counter()
    quantized = False
    for shard in resolve_shards(path):
        with TensorReader(shard) as reader:
            quantized |= "_quantization_metadata" in reader.metadata()
            for key in reader.keys():
                dtype = reader.dtype(key)
                shape = reader.shape(key)
                quantized |= key.endswith("comfy_quant") or dtype.startswith("F8")
                if len(shape) >= 2 and dtype in ("F32", "F16", "BF16"):
                    counts[dtype] += prod(shape)
    if quantized:
        return "bf16"
    if not counts:
        raise ValueError(f"Cannot infer training precision from {path}; set execution.precision explicitly")
    choices: dict[str, FloatingPrecision] = {"F32": "fp32", "F16": "fp16", "BF16": "bf16"}
    return choices[counts.most_common(1)[0][0]]


def resolve_training_precision(config: TrainingConfig) -> TrainingConfig:
    """Resolve auto once; explicit CLI/TOML precision always wins over artifact dtype."""
    if config.execution.precision != "auto":
        return config
    precision = artifact_precision(Path(config.model.weights))
    return config.model_copy(update={"execution": config.execution.model_copy(update={"precision": precision})})
