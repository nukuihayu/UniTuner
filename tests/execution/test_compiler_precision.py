"""Keep compiled backward outside forward AMP, matching the shared training engine."""

import pytest
import torch
from torch import nn

from unituner.execution.compiler import CompilerInstallation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_compiled_fp32_island_backward_runs_outside_forward_amp():
    """AOT must not introduce BF16 into the gradient of an FP32 protected projection."""

    class Projection(nn.Linear):
        def forward(self, input):
            with torch.autocast("cuda", enabled=False):
                return super().forward(input.float())

    torch.manual_seed(31)
    module = Projection(256, 128).cuda()
    values = torch.randn(32, 256, device="cuda")
    probe = torch.randn(32, 128, device="cuda")
    observations = []
    for compiled in (False, True):
        inputs = values.detach().clone().requires_grad_()
        owner = CompilerInstallation(module, precision="bf16", fullgraph=True) if compiled else None
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                output = module(inputs)
            gradients = torch.autograd.grad(output, (inputs, *module.parameters()), probe)
            observations.append((output.detach(), *gradients))
        finally:
            if owner is not None:
                owner.close()
    for eager, compiled in zip(*observations, strict=True):
        torch.testing.assert_close(compiled, eager, atol=1e-5, rtol=1e-4)
