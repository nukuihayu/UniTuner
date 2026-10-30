"""Verify MS-SSIM values, gradients, configuration and native loss integration."""

from pathlib import Path

import pytest
import torch
from pydantic import JsonValue
from torch.nn import functional as F

from unituner.checkpoints.schema import build_resume_identity
from unituner.config.loading import ConfigError, load_training_config
from unituner.config.restoration import RestorationLossWeights
from unituner.training.losses.ms_ssim import ms_ssim_loss
from unituner.training.losses.restoration import generator_loss


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def dense_reference(x, y):
    """Independent FP64 2D Gaussian implementation of the five-scale definition."""
    x, y = (x.double() + 1) / 2, (y.detach().double() + 1) / 2
    coordinates = torch.arange(-5, 6, dtype=torch.float64)
    gaussian = torch.exp(-(coordinates[:, None] ** 2 + coordinates[None, :] ** 2) / 4.5)
    kernel = (gaussian / gaussian.sum())[None, None].repeat(3, 1, 1, 1)
    factors = []
    for level, exponent in enumerate((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
        mx, my, xx, yy, xy = [F.conv2d(value, kernel, groups=3) for value in (x, y, x * x, y * y, x * y)]
        cs = (2 * (xy - mx * my) + 0.0009) / (xx - mx * mx + yy - my * my + 0.0009)
        score = cs if level < 4 else cs * (2 * mx * my + 0.0001) / (mx * mx + my * my + 0.0001)
        factors.append(score.mean((-2, -1)).clamp(1e-6, 1) ** exponent)
        if level < 4:
            pad = (x.shape[-2] % 2, x.shape[-1] % 2)
            x, y = [F.avg_pool2d(value, 2, padding=pad) for value in (x, y)]
    return 1 - torch.stack(factors).prod(0).mean()


def test_dense_reference_value_and_gradient_on_odd_rectangular_images():
    rng = torch.Generator().manual_seed(31)
    target = (torch.rand(2, 3, 161, 177, generator=rng) * 2 - 1).requires_grad_()
    prediction = (target.detach() * 0.8 + torch.rand(target.shape, generator=rng) * 0.2).requires_grad_()
    actual, expected = ms_ssim_loss(prediction, target), dense_reference(prediction, target)
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=2e-4)
    actual_grad, target_grad = torch.autograd.grad(actual, (prediction, target), allow_unused=True)
    expected_grad = torch.autograd.grad(expected, prediction)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=3e-3)
    assert target_grad is None


@pytest.mark.parametrize("kind", ["constant", "anticorrelated", "overshoot", "bf16"])
def test_extreme_inputs_have_finite_loss_and_gradients(kind):
    target = torch.rand(1, 3, 176, 192) * 2 - 1
    prediction = -target if kind == "anticorrelated" else target.clone()
    if kind == "constant":
        target.zero_()
        prediction.fill_(0.2)
    elif kind == "overshoot":
        prediction *= 3
    elif kind == "bf16":
        prediction, target = prediction.bfloat16(), target.bfloat16()
    prediction.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = ms_ssim_loss(prediction, target)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    if kind == "bf16":
        assert loss.item() == pytest.approx(0, abs=1e-6)


def test_native_weighting_and_disabled_small_images():
    target = torch.rand(1, 3, 176, 176) * 2 - 1
    prediction = (target * 0.8).requires_grad_()
    weights = RestorationLossWeights(lrr=0.0, l1=0.5, dv3d=0.0, gan=0.0, fdl=0.0, ms_ssim=0.2)
    loss = generator_loss(prediction, target, prediction, None, weights)
    expected = 0.5 * F.l1_loss(prediction, target) + 0.2 * ms_ssim_loss(prediction, target)
    torch.testing.assert_close(loss.total, expected)
    assert not loss.metrics["ms_ssim"].requires_grad
    small = target[..., :32, :32]
    disabled = generator_loss(small, small, small, None, weights.model_copy(update={"ms_ssim": 0.0}))
    assert disabled.total.item() == 0
    with pytest.raises(ValueError, match=">160"):
        ms_ssim_loss(small, small)


def test_config_allows_ms_ssim_only_and_pins_resume_weight():
    example = Path(__file__).resolve().parents[2] / "docs/examples/restoration.toml"
    weights: dict[str, JsonValue] = {
        "lrr": 0.0,
        "l1": 0.0,
        "dv3d": 0.0,
        "gan": 0.0,
        "fdl": 0.0,
        "histogram": 0.0,
        "phase": 0.0,
        "ms_ssim": 0.2,
    }
    config = load_training_config(example, overrides={"losses": weights})
    changed = load_training_config(example, overrides={"losses": {**weights, "ms_ssim": 0.3}})
    kwargs = {"dataset_manifest_sha256": "a" * 64, "model_artifact_sha256": "b" * 64}
    assert build_resume_identity(config, **kwargs) != build_resume_identity(changed, **kwargs)
    with pytest.raises(ConfigError, match="MS-SSIM"):
        load_training_config(
            example, overrides={"losses": weights, "dataset": {"groups": {"main": {"resolution": [128, 256]}}}}
        )
