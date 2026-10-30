"""Check DWT orientation, boundary handling, gradients and actual native integration."""

import pytest
import torch
from torch.autograd import gradcheck

from tests.execution.test_native_strategies import FAMILIES, _case
from unituner.config.wavelet import WaveletLossConfig
from unituner.training.losses.wavelet import WaveletLoss
from unituner.training.session import TrainingSession


def test_haar_band_orientation_and_zero_boundary():
    loss = WaveletLoss(WaveletLossConfig(wavelet="haar", levels=1), device=torch.device("cpu"))
    value = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    observed = loss._decompose(value)
    for band, expected in zip(observed, (5.0, -2.0, -1.0, 0.0), strict=True):
        torch.testing.assert_close(band, torch.tensor([[[[expected]]]]), atol=1e-6, rtol=1e-5)
    odd = torch.ones(1, 1, 3, 3)
    ll, *_ = loss._decompose(odd)
    torch.testing.assert_close(ll, torch.tensor([[[[2.0, 1.0], [1.0, 0.5]]]]), atol=1e-6, rtol=1e-5)


def test_wavelet_gradients_and_frame_reduction():
    torch.manual_seed(825)
    loss = WaveletLoss(WaveletLossConfig(levels=2), device=torch.device("cpu"))
    inputs = [torch.randn(1, 1, 5, 7, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    assert gradcheck(loss, inputs)
    prediction = torch.randn(2, 3, 2, 7, 9, requires_grad=True)
    target = torch.randn_like(prediction, requires_grad=True)
    sigma = torch.tensor([0.1, 0.8])
    expected = sum(loss(prediction[:, :, i], target[:, :, i], sigma) for i in range(2)) / 2
    actual = loss(prediction, target, sigma)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.backward()
    assert prediction.grad is not None and target.grad is not None
    assert torch.isfinite(prediction.grad).all() and torch.isfinite(target.grad).all()


@pytest.mark.parametrize("family", FAMILIES[:8])
def test_native_image_wavelet_loss_and_gradients(family):
    torch.set_num_threads(1)
    torch.manual_seed(826)
    with TrainingSession() as session:
        binding, modules, request = _case(family, session)
        result = binding.predict(request)
        expected = (result.prediction.float() - result.target.float()).square().mean()
        binding.wavelet = WaveletLoss(WaveletLossConfig(levels=2), device=torch.device("cpu"))
        expected = expected + binding.wavelet(result.prediction, result.target, request.sigma)
        observed = binding.step(request).loss
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        parameters = tuple(modules["adapter"].parameters())
        left = torch.autograd.grad(expected, parameters, allow_unused=True)
        right = torch.autograd.grad(observed, parameters, allow_unused=True)
        for a, b in zip(left, right, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_normalized_singleton_band_remains_finite():
    loss = WaveletLoss(WaveletLossConfig(wavelet="haar", levels=2, normalize_bands=True), device=torch.device("cpu"))
    value = torch.ones(1, 1, 2, 2, requires_grad=True)
    target = torch.zeros_like(value)
    result = loss(value, target)
    result.backward()
    assert value.grad is not None
    assert torch.isfinite(result) and torch.isfinite(value.grad).all()
