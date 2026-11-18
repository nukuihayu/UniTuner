"""Select the attention-only adapter training policy shared by Qwen Image models."""

from collections.abc import Mapping

from torch import nn


def attention_adapter_targets(projections: Mapping[str, nn.Linear]) -> dict[str, nn.Linear]:
    """Match SimpleTuner's default Q/K/V/output LoRA targets without changing inference."""
    suffixes = (".attn.to_q", ".attn.to_k", ".attn.to_v", ".attn.to_out.0")
    return {name: projection for name, projection in projections.items() if name.endswith(suffixes)}
