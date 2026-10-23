"""Compare feature losses and custom gradients with independent eager formulas."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.assertions import require_type
from unitunner.training.losses.dists import dists_loss
from unitunner.training.losses.frequency_distribution import FrequencyDistributionLoss, _SortedL1, _StableAngle


def test_dists_matches_centered_channel_uniform_formula_and_target_detach():
    torch.manual_seed(93)
    left = tuple(torch.randn(2, channels, 3, 4, requires_grad=True) for channels in (3, 5))
    right = tuple(torch.randn_like(value).requires_grad_() for value in left)
    contributions = []
    for x, y in zip(left, right, strict=True):
        x, y = x.flatten(2), y.detach().flatten(2)
        mx, my = x.mean(-1), y.mean(-1)
        vx, vy = x.var(-1, correction=0), y.var(-1, correction=0)
        covariance = ((x - mx[..., None]) * (y - my[..., None])).mean(-1)
        contributions.append(
            (2 * mx * my + 1e-6) / (mx.square() + my.square() + 1e-6) + (2 * covariance + 1e-6) / (vx + vy + 1e-6)
        )
    expected = 1 - torch.cat(contributions, dim=1).mean() / 2
    actual = dists_loss(left, right)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    gradients = torch.autograd.grad(actual, (*left, *right), retain_graph=True, allow_unused=True)
    reference = torch.autograd.grad(expected, left)
    for actual_gradient, expected_gradient in zip(gradients[:2], reference, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-5)
    assert gradients[2:] == (None, None)
    constant = torch.ones(2, 3, 4, 4, requires_grad=True)
    torch.testing.assert_close(dists_loss((constant,), (constant,)), torch.zeros(()), atol=0, rtol=0)


def test_frequency_distribution_matches_full_sort_amplitude_and_phase_gradients():
    torch.manual_seed(95)
    stages = nn.ModuleList([nn.Conv2d(3, 2, 1)])
    rng = torch.get_rng_state().clone()
    loss = FrequencyDistributionLoss(stages, (2,), seed=19)
    assert torch.equal(rng, torch.get_rng_state())
    generated = torch.randn(1, 3, 6, 7, requires_grad=True)
    target = torch.randn_like(generated).requires_grad_()
    actual = loss(generated, target)
    x, y = loss.features(generated)[0], loss.features(target.detach())[0]
    fx, fy = torch.fft.fft2(x), torch.fft.fft2(y)
    terms = []
    for a, b in ((fx.abs(), fy.abs()), (fx.angle(), fy.angle())):
        projected_a = F.conv2d(a, loss.projections[0]).flatten(2).sort(dim=-1).values
        projected_b = F.conv2d(b, loss.projections[0]).flatten(2).sort(dim=-1).values
        terms.append((projected_a - projected_b).abs().mean())
    expected = require_type(sum(terms), torch.Tensor)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual_grad, target_grad = torch.autograd.grad(actual, (generated, target), allow_unused=True, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, generated)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
    assert target_grad is None
    assert all(not parameter.requires_grad for parameter in loss.parameters())


def test_frequency_zero_angle_and_sorted_ties_have_finite_gradients():
    value = torch.tensor([0j, 1e-30 + 2e-30j, 1 + 2j], requires_grad=True)
    _StableAngle.apply(value).sum().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert value.grad is not None
    assert value.grad[0] == 0
    left = torch.ones(1, 2, 4, requires_grad=True)
    right = torch.zeros_like(left).requires_grad_()
    actual = _SortedL1.apply(left, right).sum()
    actual.backward()
    torch.testing.assert_close(left.grad, torch.full_like(left, 1 / 8), rtol=0, atol=0)
    assert right.grad is None


def test_feature_losses_reject_empty_levels():
    with pytest.raises(ValueError):
        dists_loss((torch.empty(1, 0, 4, 4),), (torch.empty(1, 0, 4, 4),))
    with pytest.raises(ValueError):
        FrequencyDistributionLoss(nn.ModuleList(), (), seed=0)


@pytest.mark.parametrize("bf16", [False, True])
def test_frequency_chunks_recompute_sort_without_retaining_indices(monkeypatch, bf16):
    torch.manual_seed(513)
    loss = FrequencyDistributionLoss(nn.ModuleList([nn.Identity()]), (2,), seed=1, bf16_compute=bf16)
    # An uneven final chunk must retain its correct contribution to the mean.
    bank = loss.projections[0][:19].detach()
    loss.projections = nn.ParameterList([nn.Parameter(bank, requires_grad=False)])
    left = torch.randn(2, 2, 9, 11, requires_grad=True)
    right = torch.randn_like(left).requires_grad_()
    fx, fy = torch.fft.fft2(left), torch.fft.fft2(right.detach())
    terms = []
    for a, b in ((fx.abs(), fy.abs()), (fx.angle(), fy.angle())):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            a, b = F.conv2d(a, bank).float(), F.conv2d(b, bank).float()
        terms.append((a.flatten(2).sort().values - b.flatten(2).sort().values).abs().mean())
    expected = sum(terms)
    expected_grad = torch.autograd.grad(expected, left)[0]
    widths, saved_dtypes = [], []
    conv = F.conv2d

    def record_conv(value, weight, *args, **kwargs):
        widths.append(weight.shape[0])
        return conv(value, weight, *args, **kwargs)

    def pack(tensor):
        saved_dtypes.append(tensor.dtype)
        return tensor

    monkeypatch.setattr(F, "conv2d", record_conv)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = loss.forward_features([left], [right])
    assert torch.int8 not in saved_dtypes and torch.int32 not in saved_dtypes
    forward_calls = len(widths)
    actual_grad, target_grad = torch.autograd.grad(actual, (left, right), allow_unused=True)
    assert len(widths) > forward_calls  # Backward recomputes projections instead of storing sort maps.
    assert max(widths) <= 8
    assert target_grad is None
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0.03 if bf16 else 1e-5, atol=1e-3 if bf16 else 1e-6)
