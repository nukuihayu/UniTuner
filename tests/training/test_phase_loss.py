"""Check periodic residual supervision against an explicit regularized local-fit reference."""

import pytest
import torch

from unitunner.config.restoration import RestorationConfig, RestorationLossWeights
from unitunner.training.losses.phase import phase_loss
from unitunner.training.losses.restoration import generator_loss


def dense_reference(error, target, period, cycles, epsilon):
    size = period * cycles
    coordinate = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    basis = torch.stack((torch.ones_like(xx), xx, yy), -1).reshape(-1, 3)
    starts = [sorted({*range(0, n - size + 1, size), n - size}) for n in error.shape[-2:]]
    losses = []
    for top in starts[0]:
        for left in starts[1]:
            tile = error[..., top : top + size, left : left + size]
            flat = tile.flatten(-2).double()
            reference = target.detach()[..., top : top + size, left : left + size].flatten(-2).double()
            design = torch.cat((basis.expand(*flat.shape[:-1], -1, -1), reference[..., None]), -1)
            ridge = torch.diag(torch.tensor([0.0, 0.0, 0.0, 1e-6 * size * size], dtype=torch.float64))
            gram = design.transpose(-1, -2) @ design
            coefficients = torch.linalg.solve(gram + ridge, design.transpose(-1, -2) @ flat[..., None])
            residual = (flat - (design @ coefficients).squeeze(-1)).reshape_as(tile)
            phases = torch.stack(
                [residual[..., a::period, b::period].mean((-2, -1)) for a in range(period) for b in range(period)],
                -1,
            )
            phases = phases - phases.mean(-1, keepdim=True)
            losses.append((torch.sqrt(phases.square() + epsilon**2) - epsilon).mean())
    return torch.stack(losses).mean()


@pytest.mark.parametrize("shape", [(16, 24), (17, 24), (16, 25), (17, 25)])
def test_matches_explicit_lighting_and_target_fit_including_edges(shape):
    torch.manual_seed(9)
    generated = torch.randn(2, 3, *shape)
    if shape == (17, 25):
        generated = generated.to(memory_format=torch.channels_last)
    generated.requires_grad_()
    target = torch.randn_like(generated).requires_grad_()
    actual = phase_loss(generated, target, period=4, cycles=2)
    expected = dense_reference(generated - target.detach(), target, 4, 2, 1e-3)
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-5)
    actual_grad, target_grad = torch.autograd.grad(actual, (generated, target), allow_unused=True)
    expected_grad = torch.autograd.grad(expected, generated)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-5)
    assert target_grad is None
    assert actual_grad[..., -1, -1].abs().sum() > 0


def test_affine_invariance_and_correct_periodic_texture():
    torch.manual_seed(7)
    target = torch.randn(1, 3, 16, 16)
    x = torch.arange(16).float()[None, :]
    y = x.T
    affine = (0.2 + 0.003 * x + 0.002 * y)[None, None]
    assert phase_loss(target, target, period=4, cycles=4) == 0
    assert phase_loss(target + affine, target, period=4, cycles=4) < 1e-10
    pattern = (0.01 * torch.cos(2 * torch.pi * x / 4)).expand_as(target)
    generated = (target + pattern).requires_grad_()
    loss = phase_loss(generated, target, period=4, cycles=4)
    shifted = phase_loss(generated + affine, target, period=4, cycles=4)
    torch.testing.assert_close(loss, shifted, atol=1e-7, rtol=1e-5)
    gradient = torch.autograd.grad(loss, generated)[0]
    assert loss > 1e-3
    assert abs(gradient.sum()) < 1e-6
    assert abs((gradient * x).sum()) < 1e-6
    assert abs((gradient * y).sum()) < 1e-6
    assert phase_loss(generated - gradient, target, period=4, cycles=4) < loss


def test_native_loss_weighting_and_disabled_small_images():
    weights = RestorationLossWeights(lrr=0.0, l1=0.5, dv3d=0.0, gan=0.0, fdl=0.0, phase=4.0)
    options = RestorationConfig(phase_period=2, phase_cycles=2, phase_epsilon=0.01)
    generated = torch.randn(1, 3, 8, 8).requires_grad_()
    target = torch.randn_like(generated)
    output = generator_loss(
        generated,
        target,
        generated,
        None,
        weights,
        phase_period=options.phase_period,
        phase_cycles=options.phase_cycles,
        phase_epsilon=options.phase_epsilon,
    )
    expected = 0.5 * (generated - target).abs().mean() + 4 * phase_loss(
        generated, target, period=2, cycles=2, epsilon=0.01
    )
    torch.testing.assert_close(output.total, expected)
    assert not output.metrics["phase"].requires_grad
    disabled = weights.model_copy(update={"phase": 0.0})
    plain = generator_loss(generated, target, generated, None, disabled)
    torch.testing.assert_close(plain.total, 0.5 * (generated - target).abs().mean(), atol=0, rtol=0)
    with pytest.raises(ValueError, match="window"):
        phase_loss(generated, target)
    with pytest.raises(ValueError):
        RestorationConfig(phase_epsilon=float("nan"))


def test_bf16_inputs_use_float_statistics():
    generated = torch.randn(1, 3, 16, 16).bfloat16().requires_grad_()
    target = torch.randn_like(generated)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = phase_loss(generated, target, period=4, cycles=4)
    expected = phase_loss(generated.float(), target.float(), period=4, cycles=4)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()


@pytest.mark.parametrize("epsilon", [1e-12, 1.0])
def test_zero_residual_has_finite_zero_gradient_at_epsilon_limits(epsilon):
    generated = torch.zeros(1, 3, 16, 16, requires_grad=True)
    loss = phase_loss(generated, generated.detach(), period=4, cycles=4, epsilon=epsilon)
    gradient = torch.autograd.grad(loss, generated)[0]
    assert loss == 0
    assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("epsilon", [0.0, 1e-30, 2.0, float("inf"), float("nan")])
def test_invalid_epsilon_fails_before_float32_underflow(epsilon):
    with pytest.raises(ValueError):
        RestorationConfig(phase_epsilon=epsilon)
    with pytest.raises(ValueError, match="epsilon"):
        phase_loss(torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 16, 16), epsilon=epsilon)


def test_target_contrast_is_protected_but_new_flat_region_pattern_is_penalized():
    coordinate = torch.arange(128).float()
    target = (
        0.3 * torch.cos(2 * torch.pi * coordinate[None, :] / 16)
        + 0.2 * torch.sin(2 * torch.pi * coordinate[:, None] / 16)
    )[None, None].expand(1, 3, -1, -1)
    generated = (target * 0.7 + 0.1).requires_grad_()
    assert phase_loss(generated, target) < 1e-8
    pattern = (0.01 * torch.cos(2 * torch.pi * coordinate / 16))[None, None, None, :].expand_as(target)
    assert phase_loss(pattern, torch.zeros_like(pattern)) > 1e-3
