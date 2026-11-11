"""Compare feature losses and custom gradients with independent eager formulas."""

import pytest
import torch

from unituner.training.losses.dists import dists_loss


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


def test_feature_losses_reject_empty_levels():
    with pytest.raises(ValueError):
        dists_loss((torch.empty(1, 0, 4, 4),), (torch.empty(1, 0, 4, 4),))
