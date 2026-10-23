"""Compare nonzero adapter predictions and every derivative to independent dense formulas."""

import pytest
import torch
from torch.nn import functional as F

from unitunner.adapters.loha import LoHa
from unitunner.adapters.lokr import LoKr
from unitunner.adapters.lora import LoRA


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr", "lokr-full"])
def test_nonzero_forward_and_all_gradients(algorithm):
    torch.manual_seed(41)
    if algorithm == "lora":
        branch = LoRA(6, 8, rank=2, alpha=3.0, dtype=torch.float64)
    elif algorithm == "loha":
        branch = LoHa(6, 8, rank=2, alpha=3.0, dtype=torch.float64)
    else:
        branch = LoKr((2, 3), (2, 4), rank=2, alpha=3.0, full_right=algorithm == "lokr-full", dtype=torch.float64)
    with torch.no_grad():
        for parameter in branch.parameters():
            parameter.normal_(0, 0.3)
    inputs = torch.randn(3, 2, 6, dtype=torch.float64, requires_grad=True)
    actual = branch(inputs)
    factors = {name: value.detach().clone().requires_grad_() for name, value in branch.named_parameters()}
    reference_input = inputs.detach().clone().requires_grad_()
    if algorithm == "lora":
        weight = torch.einsum("or,ri->oi", factors["up"], factors["down"]) * 1.5
    elif algorithm == "loha":
        weight = torch.einsum("or,ri,os,si->oi", factors["a1"], factors["b1"], factors["a2"], factors["b2"]) * 1.5
    else:
        right = (
            factors["right"]
            if algorithm == "lokr-full"
            else torch.einsum("or,ri->oi", factors["right_up"], factors["right_down"])
        )
        weight = torch.einsum("ab,cd->acbd", factors["left"], right).reshape(8, 6)
        if algorithm != "lokr-full":
            weight = weight * 1.5
    reference = F.linear(reference_input, weight)
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-8)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * probe).sum(), (inputs, *branch.parameters()))
    reference_grads = torch.autograd.grad((reference * probe).sum(), (reference_input, *factors.values()))
    for actual_grad, reference_grad in zip(actual_grads, reference_grads, strict=True):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, reference_grad, atol=1e-10, rtol=1e-8)
