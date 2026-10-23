"""Verify application-wide backend lifetime and resume/cache configuration identity."""

from types import SimpleNamespace

import pytest
import torch

from unitunner.config.runtime import RuntimeConfig
from unitunner.execution.runtime import runtime_scope


@pytest.mark.parametrize("fail", [False, True])
def test_backend_scope_applies_and_restores_even_on_failure(fail):
    before = (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    config = SimpleNamespace(
        runtime=RuntimeConfig(cudnn_enabled=False, cudnn_benchmark=True, cudnn_deterministic=True, cuda_allow_tf32=True)
    )

    @runtime_scope
    def entry(config, *, device):
        assert not torch.backends.cudnn.enabled
        assert torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32
        if fail:
            raise ValueError("failure")
        return 7

    if fail:
        with pytest.raises(ValueError, match="failure"):
            entry(config, device="cuda")
    else:
        assert entry(config, device="cuda") == 7
    after = (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    assert after == before


def test_cpu_runtime_does_not_change_cuda_backend_state():
    before = torch.backends.cudnn.benchmark

    @runtime_scope
    def entry(config, *, device):
        assert torch.backends.cudnn.benchmark == before

    entry(SimpleNamespace(runtime=RuntimeConfig(cudnn_benchmark=not before)), device="cpu")
