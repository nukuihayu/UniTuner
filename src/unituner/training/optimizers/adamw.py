"""Build AdamW with explicit numerical options and shared backend selection."""

from collections.abc import Iterable

import torch

from unituner.config.optimizer import DEFAULT_ADAMW_CONFIG, AdamWConfig
from unituner.training.parameters import ParameterGroup, validate_parameter_groups


def create_adamw(
    groups: Iterable[ParameterGroup],
    *,
    lr: float,
    weight_decay: float = 0.01,
    config: AdamWConfig = DEFAULT_ADAMW_CONFIG,
) -> torch.optim.Optimizer:
    """Preserve named groups and their learning-rate multipliers on every backend."""
    groups = tuple(groups)
    validate_parameter_groups(groups)
    named_groups = [
        {
            "params": group.parameters,
            "name": group.name,
            "learning_rate_scale": group.learning_rate_scale,
            "lr": lr * group.learning_rate_scale,
        }
        for group in groups
    ]
    if config.implementation == "adv_optm":
        try:
            from unituner.training.optimizers.advanced import ResumableAdamW
        except ImportError as error:
            raise ImportError("adv_optm optimizer requires the project's base dependencies; run 'uv sync --locked'") from error
        return ResumableAdamW(
            named_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=config.betas,
            eps=config.eps,
            state_precision=config.state_precision,
            factored_2nd=config.factored_2nd,
            stochastic_rounding=False,
        )
    parameters = tuple(parameter for group in groups for parameter in group.parameters)
    supports_fused = bool(parameters) and all(
        parameter.device.type == "cuda" and parameter.dtype in (torch.float32, torch.float16, torch.bfloat16)
        for parameter in parameters
    )
    if config.backend == "fused" and not supports_fused:
        raise ValueError("AdamW fused backend requires CUDA FP32, FP16 or BF16 parameters")
    fused = supports_fused if config.backend == "auto" else config.backend == "fused"
    foreach = config.backend == "foreach" if config.backend in ("foreach", "single") else None
    return torch.optim.AdamW(
        named_groups,
        lr=lr,
        weight_decay=weight_decay,
        betas=config.betas,
        eps=config.eps,
        fused=fused,
        foreach=foreach,
    )
