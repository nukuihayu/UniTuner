"""Skip target assembly outside selected feature layers without changing gradients."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.execution.test_native_strategies import FAMILIES, _case
from unituner.models.components.features import FeatureRecorder
from unituner.training.session import TrainingSession


@pytest.mark.parametrize("family", FAMILIES)
def test_only_selected_active_layers_assemble_features(family):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with TrainingSession() as session, sdpa_kernel(SDPBackend.MATH):
            binding, modules, request = _case(family, session)
            parameters = tuple(modules["adapter"].parameters())
            baseline = binding.step(request).loss
            expected = torch.autograd.grad(baseline, parameters, allow_unused=True)
            recorder = FeatureRecorder(frozenset({0}))
            calls = []
            record = recorder.record

            def observe(index, values):
                calls.append(index)
                record(index, values)

            recorder.record = observe
            binding.model.feature_recorder = recorder
            inactive = binding.step(request).loss
            assert not calls and not recorder.features
            recorder.active = True
            actual = binding.step(request).loss
            assert calls and set(calls) == {0} and set(recorder.features) == {0}
            torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
            torch.testing.assert_close(inactive, baseline, atol=0, rtol=0)
            for left, right in zip(torch.autograd.grad(actual, parameters, allow_unused=True), expected, strict=True):
                if right is None:
                    assert left is None
                else:
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
            recorder.clear()
            binding.model.feature_recorder = None
    finally:
        torch.set_num_threads(previous)
