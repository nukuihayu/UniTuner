"""Apply CUDA backend choices for one application call and restore caller state."""

from collections.abc import Callable
from functools import wraps
from typing import Concatenate, Protocol

import torch

from unitunner.config.runtime import RuntimeConfig


class RuntimeOwner(Protocol):
    @property
    def runtime(self) -> RuntimeConfig: ...


def runtime_scope[C: RuntimeOwner, **P, R](entry: Callable[Concatenate[C, P], R]) -> Callable[Concatenate[C, P], R]:
    """Use scoped Musubi-style TF32 and cuDNN options for CUDA entry points."""

    @wraps(entry)
    def run(config: C, *args: P.args, **kwargs: P.kwargs) -> R:
        device = torch.device(str(kwargs.get("device", "cpu")))
        if device.type != "cuda":
            return entry(config, *args, **kwargs)
        policy = config.runtime
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = policy.cuda_allow_tf32
            with torch.backends.cudnn.flags(
                enabled=policy.cudnn_enabled,
                benchmark=policy.cudnn_benchmark,
                deterministic=policy.cudnn_deterministic,
                allow_tf32=policy.cuda_allow_tf32,
            ):
                return entry(config, *args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    return run
