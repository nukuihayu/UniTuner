"""Regress checkpoint/offload ownership without requiring CUDA allocations."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from unituner.config.execution import ExecutionConfig
from unituner.execution.activations import AttentionOffloadInstallation, offloaded_attention
from unituner.execution.recompute import RecomputeInstallation
from unituner.execution.regions import ModelRegion


class _Owner:
    def __init__(self):
        self.saved_bytes = 0

    def pack(self, value):
        self.saved_bytes += value.nbytes
        return value.detach()

    def unpack(self, value):
        return value


@pytest.mark.parametrize("interval,index,expected_saved", [(1, 0, False), (2, 1, True)])
def test_checkpoint_owns_sdpa_saves_and_preserves_gradients(interval, index, expected_saved):
    owner = _Owner()

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(16, 48)
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            q, k, v = self.projection(value).chunk(3, dim=-1)
            return offloaded_attention(owner, q[:, None], k[:, None], v[:, None], None, 0.0, False, None, False)

    torch.manual_seed(42)
    module = Attention()
    reference = deepcopy(module)
    x = torch.randn(1, 32, 16, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    q, k, v = reference.projection(expected_x).chunk(3, dim=-1)
    expected = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])
    expected.square().sum().backward()
    region = ModelRegion(module, index, "visual")
    offload = AttentionOffloadInstallation(region, owner)
    recompute = RecomputeInstallation(region, interval=interval, attention_offload=True)
    try:
        for _ in range(3):
            module.zero_grad(set_to_none=True)
            x.grad = None
            owner.saved_bytes = 0
            before = module.calls
            actual = module(x)
            actual.square().sum().backward()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(x.grad, expected_x.grad)
            for left, right in zip(module.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(left.grad, right.grad)
            assert (owner.saved_bytes > 0) == expected_saved
            assert module.calls - before == (1 if expected_saved else 2)
    finally:
        recompute.close()
        offload.close()


def test_pinned_masters_remain_an_explicit_choice():
    assert ExecutionConfig(residency="frozen-blocks").use_pinned_memory_for_block_swap is None
    assert ExecutionConfig(
        residency="frozen-blocks", use_pinned_memory_for_block_swap=True
    ).use_pinned_memory_for_block_swap


def test_expert_installation_shares_one_cache_and_does_not_pin_by_default(monkeypatch):
    import unituner.execution.installation as implementation
    from unituner.execution.regions import ModelRegions

    caches, blocks = [], []

    class Cache:
        def __init__(self, **kwargs):
            caches.append(self)

        def close(self):
            pass

    class Prefetch:
        def __init__(self, device):
            pass

        def close(self):
            pass

    class Block:
        def __init__(self, region, **kwargs):
            blocks.append((region, kwargs))

        def close(self):
            pass

    monkeypatch.setattr(implementation, "H2dCache", Cache)
    monkeypatch.setattr(implementation, "WeightPrefetch", Prefetch)
    monkeypatch.setattr(implementation, "FrozenBlockInstallation", Block)
    low = ModelRegions((ModelRegion(torch.nn.Linear(4, 4).requires_grad_(False), 0, "visual"),))
    high = ModelRegions((ModelRegion(torch.nn.Linear(4, 4).requires_grad_(False), 0, "visual"),))
    owner = implementation.ExecutionInstallation(
        low,
        ExecutionConfig(residency="frozen-blocks", precision="bf16"),
        device=torch.device("cuda", 0),
        additional_regions=(high,),
    )
    try:
        assert len(caches) == 1 and len(blocks) == 2
        assert all(options["cache"] is caches[0] for _, options in blocks)
        assert all(options["use_pinned_memory"] is False for _, options in blocks)
    finally:
        owner.close()


def test_attention_host_budget_and_release_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import gc

    from unituner.execution.activations import AttentionActivations

    values = torch.randn(128, device="cuda", requires_grad=True) * 2
    owner = AttentionActivations(values.device, max_host_bytes=values.nbytes)
    try:
        packet = owner.pack(values)
        assert owner.active_host_bytes == values.nbytes
        with pytest.raises(MemoryError, match="attention_activation_offload_mb"):
            owner.pack(values)
        torch.testing.assert_close(owner.unpack(packet), values)
        del packet
        gc.collect()
        assert owner.active_host_bytes == 0
        second = owner.pack(values)
        assert owner.peak_host_bytes == values.nbytes
        del second
    finally:
        owner.close()


@pytest.mark.parametrize(
    "family",
    ["wan", "qwen-image-2-1", "flux", "flux2", "flux3", "z-image", "moss-image", "qwen-image", "krea2", "minimax-h3"],
)
def test_model_adapter_updates_match_with_checkpoint_owned_attention(family):
    from tests.execution.test_native_strategies import _case
    from unituner.training.session import TrainingSession

    torch.manual_seed(927)
    with TrainingSession() as session:
        binding, modules, request = _case(family, session, "lora")
        initial = {name: deepcopy(module.state_dict()) for name, module in modules.items()}
        parameters = tuple(modules["adapter"].parameters())
        observations = []
        for optimized in (False, True):
            for name, module in modules.items():
                module.load_state_dict(initial[name])
            optimizer = torch.optim.AdamW(parameters, lr=0.001)
            owners = []
            policy = _Owner()
            if optimized:
                for region in binding.regions.regions:
                    owners.append(AttentionOffloadInstallation(region, policy))
                    owners.append(RecomputeInstallation(region, attention_offload=True))
            try:
                torch.manual_seed(911)
                losses = []
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    result = binding.step(request)
                    result.loss.backward()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
                    optimizer.step()
                    losses.append(result.loss.detach())
                observations.append((losses, tuple(p.detach().clone() for p in parameters)))
                assert policy.saved_bytes == 0
            finally:
                for owner in reversed(owners):
                    owner.close()
        for expected_group, actual_group in zip(observations[0], observations[1], strict=True):
            for expected, actual in zip(expected_group, actual_group, strict=True):
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_pageable_frozen_payload_reuses_original_cpu_storage():
    from unituner.execution.frozen_blocks import _pack_weights

    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.Linear(64, 16)).requires_grad_(False)
    parameters = tuple(model.named_parameters())
    original = {name: value.data_ptr() for name, value in parameters}
    payloads = _pack_weights(parameters, pinned=False)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload.host is None
    assert payload.source.nbytes == sum(value.nbytes for _, value in parameters)
    assert len(payload.source.chunks) == len(parameters)
    for (name, value), chunk in zip(parameters, payload.source.chunks, strict=True):
        assert chunk.data_ptr() == original[name] == value.data_ptr()
        torch.testing.assert_close(chunk, value.detach().reshape(-1), atol=0, rtol=0)


def test_weight_staging_gathers_chunks_with_bounded_reusable_windows(monkeypatch):
    from unituner.execution.weight_source import WeightSource, WeightStaging

    class Event:
        def synchronize(self):
            pass

    class Stream:
        def record_event(self):
            return Event()

    original_empty = torch.empty

    def pageable_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", pageable_empty)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    source = WeightSource((torch.arange(7, dtype=torch.float32), torch.arange(13, dtype=torch.float32) + 20))
    destination = torch.empty(20)
    staging = WeightStaging(device=torch.device("cpu"))
    staging._window_bytes = 16
    try:
        for _ in range(3):
            staging.copy(source, destination)
            torch.testing.assert_close(destination, torch.cat(source.chunks), atol=0, rtol=0)
            assert staging.pinned_bytes == 32
            assert staging.registered_bytes == 0
    finally:
        staging.close()
    assert staging.pinned_bytes == 0
