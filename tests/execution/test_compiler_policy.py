"""Verify scoped Inductor memory policy and compiled training gradients."""

import copy

import pytest
import torch
from torch import nn
from torch._inductor import config as inductor_config

from unitunner.execution import compiler


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_compile_policy_is_per_graph_and_covers_all_precisions(monkeypatch, precision):
    providers = []
    options = []

    def capture_compile(fn, **kwargs):
        providers.append(kwargs["backend"])
        return fn

    def capture_inductor(graph, inputs, **kwargs):
        options.append(kwargs["options"])
        return graph.forward

    monkeypatch.setattr(torch, "compile", capture_compile)
    monkeypatch.setattr(compiler, "compile_inductor", capture_inductor)
    # This test captures provider wiring without executing a compiler. Keep the
    # production version guard intact; real compilation is tested on CUDA below.
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    model = nn.Linear(4, 4)
    with inductor_config.patch({"triton.cudagraphs": True, "epilogue_fusion": False}):
        owner = compiler.CompilerInstallation(model, precision=precision)
        try:
            providers[0](torch.fx.symbolic_trace(nn.Linear(4, 4)), [torch.ones(2, 4)])
            assert options == [
                {
                    "emulate_precision_casts": precision != "fp32",
                    "triton.cudagraphs": False,
                    "epilogue_fusion": True,
                }
            ]
            assert inductor_config.triton.cudagraphs is True
            assert inductor_config.epilogue_fusion is False
        finally:
            owner.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_inductor_memory_policy_preserves_forward_and_backward(precision):
    torch.manual_seed(82)
    model = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32)).cuda().train()
    reference = copy.deepcopy(model)
    owner = compiler.CompilerInstallation(model, fullgraph=True, precision=precision)
    try:
        for rows in (16, 24):
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            x = torch.randn(rows, 32, device="cuda", requires_grad=True)
            y = x.detach().clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                actual, expected = model(x), reference(y)
                actual_loss, expected_loss = actual.float().square().mean(), expected.float().square().mean()
            actual_loss.backward()
            expected_loss.backward()
            tolerance = 0.02 if precision == "bf16" else 1e-4
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance * 0.01)
            for a, b in zip((x, *model.parameters()), (y, *reference.parameters()), strict=True):
                torch.testing.assert_close(a.grad, b.grad, rtol=tolerance, atol=tolerance * 0.01)
    finally:
        owner.close()
