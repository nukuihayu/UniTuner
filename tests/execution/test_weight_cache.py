"""Exercise actual CUDA cache reuse, speculative pressure and graph-safe retirement."""

from types import SimpleNamespace
from typing import Literal, cast

import pytest
import torch

from unituner.execution.prefetch import WeightPrefetch
from unituner.execution.weight_cache import H2dCache, H2dWeights
from unituner.execution.weight_lease import FrozenWeights, PrefetchUnavailable
from unituner.execution.weight_source import WeightSource

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def test_cancelled_prefetch_orders_recycling_without_host_synchronization(monkeypatch):
    events = []

    class CopyReady:
        def synchronize(self):
            pytest.fail("Cancelling an unused copy must not block the CPU")

    ready = cast(torch.Event, CopyReady())
    stream = SimpleNamespace(wait_event=lambda event: events.append(("wait", event)))
    owner = cast(
        FrozenWeights,
        SimpleNamespace(device=torch.device("cuda", 0), release=lambda leases: events.append(("release", leases))),
    )
    prefetch = WeightPrefetch(owner.device)
    prefetch._pending.append((owner, (), ready))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    prefetch.cancel()
    assert events == [("wait", ready), ("release", ())]
    assert prefetch._pending == []


@pytest.mark.parametrize("pinned", [False, True])
def test_bidirectional_prefetch_reuses_turnaround_and_cross_step_weights(pinned):
    device = torch.device("cuda", torch.cuda.current_device())
    hosts = [torch.full((262144,), float(index), pin_memory=pinned) for index in range(4)]
    cache = H2dCache(device=device, workspace_bytes=hosts[0].nbytes, retained_bytes=2 * hosts[0].nbytes)
    weights = [H2dWeights((WeightSource((host,)),), cache) for host in hosts]
    prefetch = WeightPrefetch(device)
    for owner in weights:
        prefetch.register(owner)
    try:
        for step in range(4):
            before = cache.transfers
            for backward in (False, True):
                order = range(3, -1, -1) if backward else range(4)
                for index in order:
                    leases = prefetch.acquire(weights[index])
                    # Submit an actual read before the next copy overwrites an idle buffer.
                    observed = leases[0].tensor.clone()
                    prefetch.next(weights[index], backward=backward)
                    weights[index].release(leases)
                    torch.testing.assert_close(observed.cpu(), hosts[index], atol=0, rtol=0)
            prefetch.cancel()
            assert cache.transfers - before == (5 if step == 0 else 2)
            assert cache.statistics().allocated_bytes == 3 * hosts[0].nbytes
    finally:
        prefetch.close()
        for owner in weights:
            owner.close()
        cache.close()


@pytest.mark.parametrize("sizes", [(1,) * 8, (1, 2, 3, 2, 1, 2, 3, 1)])
def test_distributed_residents_survive_lookahead_across_both_directions(sizes):
    """Retained blocks must survive speculative misses beyond a cached neighbor."""
    device = torch.device("cuda", torch.cuda.current_device())
    hosts = [torch.full((size * 65536,), float(index), pin_memory=True) for index, size in enumerate(sizes)]
    workspace = max(host.nbytes for host in hosts)
    cache = H2dCache(
        device=device,
        workspace_bytes=workspace,
        retained_bytes=sum(host.nbytes for host in hosts) * 5 // 8 - workspace,
    )
    owners = [H2dWeights((WeightSource((host,)),), cache) for host in hosts]
    prefetch = WeightPrefetch(device)
    for owner in owners:
        prefetch.register(owner)
    residents = [owner for owner in owners if owner.prefetch_past_resident]
    crossed = False
    try:
        assert 0 < len(residents) < len(owners)
        for step in range(3):
            for backward in (False, True):
                for index in range(len(owners) - 1, -1, -1) if backward else range(len(owners)):
                    leases = prefetch.acquire(owners[index])
                    observed = leases[0].tensor.clone()
                    prefetch.next(owners[index], backward=backward)
                    crossed |= any(
                        abs(next(i for i, candidate in enumerate(owners) if candidate is owner) - index) > 1
                        for owner, _, _ in prefetch._pending
                    )
                    owners[index].release(leases)
                    torch.testing.assert_close(observed.cpu(), hosts[index], atol=0, rtol=0)
                    if step:
                        assert all(not owner.needs_transfer() for owner in residents)
            prefetch.cancel()
            assert cache.statistics().allocated_bytes <= cache.statistics().capacity_bytes
        assert crossed
    finally:
        prefetch.close()
        for owner in owners:
            owner.close()
        cache.close()


def test_active_weights_are_never_overwritten_and_speculation_cannot_exceed_budget():
    device = torch.device("cuda", torch.cuda.current_device())
    hosts = [torch.full((262144,), float(index)) for index in range(3)]
    cache = H2dCache(device=device, workspace_bytes=hosts[0].nbytes, retained_bytes=0)
    owners = [H2dWeights((WeightSource((host,)),), cache) for host in hosts]
    try:
        held = owners[0].acquire()
        shared = owners[0].acquire()
        assert held[0].tensor.data_ptr() == shared[0].tensor.data_ptr()
        with pytest.raises(PrefetchUnavailable):
            owners[1].acquire(speculative=True)
        extra = owners[1].acquire()
        torch.testing.assert_close(held[0].tensor.cpu(), hosts[0], atol=0, rtol=0)
        assert extra[0].tensor.data_ptr() != held[0].tensor.data_ptr()
        owners[0].release(held)
        third = owners[2].acquire()
        torch.testing.assert_close(shared[0].tensor.cpu(), hosts[0], atol=0, rtol=0)
        for owner, leases in ((owners[0], shared), (owners[1], extra), (owners[2], third)):
            owner.release(leases)
        assert cache.statistics().active_bytes == 0
        assert cache.statistics().allocated_bytes <= cache.statistics().capacity_bytes
    finally:
        for owner in owners:
            owner.close()
        cache.close()


def test_failed_copy_does_not_publish_partial_cached_contents(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    host = torch.ones(262144)
    cache = H2dCache(device=device, workspace_bytes=host.nbytes, retained_bytes=0)
    owner = H2dWeights((WeightSource((host,)),), cache)
    original = torch.Tensor.copy_

    def fail(destination, source, **kwargs):
        destination.fill_(-1)
        raise RuntimeError("incomplete frozen copy")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "copy_", fail)
            with pytest.raises(RuntimeError, match="incomplete frozen copy"):
                owner.acquire()
        assert cache.transfers == 0
        assert cache.statistics().active_bytes == 0
        assert torch.Tensor.copy_ is original
        leases = owner.acquire()
        torch.testing.assert_close(leases[0].tensor.cpu(), host, atol=0, rtol=0)
        assert cache.transfers == 1
        owner.release(leases)
    finally:
        owner.close()
        cache.close()


def test_automatic_capacity_releases_idle_storage_after_memory_pressure(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    size = 1024**2
    cache = H2dCache(device=device, workspace_bytes=size)
    owners = [H2dWeights((WeightSource((torch.full((size // 4,), float(index)),)),), cache) for index in range(3)]
    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "mem_get_info", lambda device: (4 * size, 8 * size))
            patch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
            patch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
            patch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
            cache._completed_backward()
            for owner in owners:
                leases = owner.acquire()
                owner.release(leases)
            assert cache.statistics().allocated_bytes == 3 * size
            patch.setattr(torch.cuda, "mem_get_info", lambda device: (0, 8 * size))
            cache.prepare_forward(owners[-1]._sources[0])
            leases = owners[-1].acquire()
            torch.testing.assert_close(leases[0].tensor.cpu(), torch.full((size // 4,), 2.0), atol=0, rtol=0)
            owners[-1].release(leases)
            assert cache.statistics().allocated_bytes <= 2 * size
    finally:
        for owner in owners:
            owner.close()
        cache.close()


@pytest.mark.parametrize("allocation_workspace", [0, 1024**2])
def test_peak_calibration_reserves_contiguous_allocation_workspace(monkeypatch, allocation_workspace):
    device = torch.device("cuda", torch.cuda.current_device())
    size = 1024**2
    cache = H2dCache(device=device, workspace_bytes=size, allocation_workspace_bytes=allocation_workspace)
    owner = H2dWeights((WeightSource((torch.ones(size // 4),)),), cache)
    allocated = 100 * size
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8 * size, 128 * size))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: allocated)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 106 * size)
    try:
        cache.prepare_forward(owner._sources[0])
        leases = owner.acquire()
        owner.release(leases)
        # Three MiB of measured transient work plus allocation slack must remain
        # available even though the allocator reports eight reusable MiB.
        allocated = 103 * size
        cache._completed_backward()
        assert cache.statistics().capacity_bytes == 6 * size - allocation_workspace
    finally:
        owner.close()
        cache.close()


@pytest.mark.parametrize("compiled", [False, True])
def test_branching_outstanding_graphs_and_repeated_eager_backward(compiled):
    from copy import deepcopy

    from torch import nn

    from unituner.config.execution import ExecutionConfig
    from unituner.execution.installation import ExecutionInstallation
    from unituner.execution.regions import ModelRegion, ModelRegions

    device = torch.device("cuda", torch.cuda.current_device())

    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(nn.Linear(16, 16, device=device) for _ in range(4))
            for layer in self.layers:
                layer.weight.requires_grad_(False)

        def forward(self, value, skip):
            for index, layer in enumerate(self.layers):
                if skip and index == 1:
                    continue
                value = layer(value).sin()
            return value

    torch.manual_seed(81)
    reference = Branch()
    staged = deepcopy(reference)
    owners = []
    try:
        for model, residency in ((reference, "resident"), (staged, "frozen-blocks")):
            regions = ModelRegions(
                tuple(ModelRegion(layer, index, "visual") for index, layer in enumerate(model.layers))
            )
            owners.append(
                ExecutionInstallation(
                    regions,
                    ExecutionConfig(
                        residency=cast(Literal["resident", "frozen-blocks", "trainable-blocks"], residency),
                        compile=compiled,
                        recompute="block",
                        frozen_weight_cache_mb=0,
                    ),
                    device=device,
                )
            )
        inputs = [torch.randn(2, 16, device=device, requires_grad=True) for _ in range(2)]
        copies = [value.detach().clone().requires_grad_() for value in inputs]
        expected = [reference(value, bool(index)) for index, value in enumerate(inputs)]
        actual = [staged(value, bool(index)) for index, value in enumerate(copies)]
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            left.square().mean().backward(retain_graph=not compiled)
            right.square().mean().backward(retain_graph=not compiled)
        if not compiled:
            expected[0].sum().backward()
            actual[0].sum().backward()
        for left, right in zip(inputs, copies, strict=True):
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
        for left, right in zip(reference.parameters(), staged.parameters(), strict=True):
            if left.requires_grad:
                torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
        assert owners[-1].h2d_statistics().active_bytes == 0
    finally:
        for owner in reversed(owners):
            owner.close()


@pytest.mark.parametrize("explicit", [False, True])
def test_new_workload_recalibrates_automatic_cache_without_discarding_live_leases(monkeypatch, explicit):
    device = torch.device("cuda", torch.cuda.current_device())
    size = 1024**2
    cache = H2dCache(device=device, workspace_bytes=size, retained_bytes=3 * size if explicit else None)
    owners = [H2dWeights((WeightSource((torch.full((size // 4,), float(i)),)),), cache) for i in range(3)]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8 * size, 16 * size))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    try:
        first = owners[0]._sources[0]
        cache.prepare_forward(first, signature=(0, (2, 16)), entry=True)
        monkeypatch.setattr("unituner.execution.weight_cache.after_backward", lambda callback: None)
        cache.observe_backward(source=first, signature=(0, (2, 16)))
        cache._completed_backward()
        live = owners[0].acquire()
        for owner in owners[1:]:
            leases = owner.acquire()
            owner.release(leases)
        capacity = cache.statistics().capacity_bytes
        # Same workload retains calibrated storage; a larger bucket or another
        # expert starts from streaming capacity even while older graphs are live.
        cache.prepare_forward(first, signature=(0, (2, 16)), entry=True)
        assert cache.statistics().capacity_bytes == capacity
        for source, signature in [(first, (0, (8, 16))), (owners[1]._sources[0], (1, (8, 16)))]:
            cache.prepare_forward(source, signature=signature, entry=True)
            assert cache.statistics().capacity_bytes == (capacity if explicit else size)
            torch.testing.assert_close(live[0].tensor.cpu(), torch.zeros(size // 4), atol=0, rtol=0)
        owners[0].release(live)
        assert cache.statistics().allocated_bytes <= (capacity if explicit else size)
    finally:
        for owner in owners:
            owner.close()
        cache.close()


def test_each_expert_entry_recalibrates_for_changed_tensor_shapes(monkeypatch):
    from copy import deepcopy

    from torch import nn

    from unituner.config.execution import ExecutionConfig
    from unituner.execution.installation import ExecutionInstallation
    from unituner.execution.regions import ModelRegion, ModelRegions

    device = torch.device("cuda", torch.cuda.current_device())
    models = [nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 16)).to(device) for _ in range(2)]
    for model in models:
        for layer in (model[0], model[2]):
            layer.weight.requires_grad_(False)
    references = deepcopy(models)
    groups = tuple(ModelRegions(tuple(ModelRegion(model[i], i, "visual") for i in (0, 2))) for model in models)
    owner = ExecutionInstallation(
        groups[0],
        ExecutionConfig(residency="frozen-blocks", recompute="block"),
        device=device,
        additional_regions=(groups[1],),
    )
    cache = owner._cache
    assert cache is not None
    entries = []
    prepare = cache.prepare_forward

    def observe(source, **kwargs):
        prepare(source, **kwargs)
        if kwargs.get("entry"):
            entries.append((kwargs["signature"], cache.statistics().capacity_bytes))

    monkeypatch.setattr(cache, "prepare_forward", observe)
    try:
        for expert, batch in [(0, 2), (0, 8), (1, 8), (0, 8), (1, 8), (0, 2)]:
            value = torch.randn(batch, 16, device=device, requires_grad=True)
            copied = value.detach().clone().requires_grad_()
            expected, actual = references[expert](value), models[expert](copied)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            expected.sum().backward()
            actual.sum().backward()
            torch.testing.assert_close(copied.grad, value.grad, atol=0, rtol=0)
            for left, right in zip(references[expert].parameters(), models[expert].parameters(), strict=True):
                if left.requires_grad:
                    torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
        assert len(entries) == 6
        assert all(entries[index][1] == cache._workspace for index in (0, 1, 2, 5))
        assert all(entries[index][1] > cache._workspace for index in (3, 4))
        assert entries[0][0] != entries[1][0]
        assert entries[1][0] != entries[2][0]
    finally:
        owner.close()


def test_automatic_cache_only_reuses_completed_workloads_and_still_shrinks_under_pressure(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    size = 1024**2
    cache = H2dCache(device=device, workspace_bytes=size)
    source = WeightSource((torch.zeros(size // 4),))
    free = [8 * size]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free[0], 16 * size))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    try:
        cache.prepare_forward(source, signature=(1, (2, 16)), entry=True)
        assert not cache._calibrated_signatures
        cache._capacity = 4 * size
        cache.prepare_forward(source, signature=(1, (2, 16)), entry=True)
        assert cache.statistics().capacity_bytes == size
        monkeypatch.setattr("unituner.execution.weight_cache.after_backward", lambda callback: None)
        cache.observe_backward(source=source, signature=(1, (2, 16)))
        cache._completed_backward()
        capacity = cache.statistics().capacity_bytes
        cache.prepare_forward(source, signature=(1, (2, 16)), entry=True)
        assert cache.statistics().capacity_bytes == capacity
        free[0] = size
        cache.prepare_forward(source, signature=(1, (2, 16)), entry=True)
        assert cache.statistics().capacity_bytes == size
    finally:
        cache.close()


def test_outstanding_forward_is_not_calibrated_by_another_graphs_backward():
    from torch import nn

    from unituner.config.execution import ExecutionConfig
    from unituner.execution.installation import ExecutionInstallation
    from unituner.execution.regions import ModelRegion, ModelRegions

    device = torch.device("cuda", torch.cuda.current_device())
    model = nn.Linear(16, 16, device=device)
    model.weight.requires_grad_(False)
    owner = ExecutionInstallation(
        ModelRegions((ModelRegion(model, 0, "visual"),)),
        ExecutionConfig(residency="frozen-blocks", recompute="block"),
        device=device,
    )
    cache = owner._cache
    assert cache is not None
    try:
        small = model(torch.randn(2, 16, device=device, requires_grad=True))
        large = model(torch.randn(8, 16, device=device, requires_grad=True))
        assert not cache._calibrated_signatures
        small.sum().backward()
        assert len(cache._calibrated_signatures) == 1
        assert next(iter(cache._calibrated_signatures.values()))[1] == ((2, 16), torch.float32)
        large.sum().backward()
        assert next(iter(cache._calibrated_signatures.values()))[1] == ((8, 16), torch.float32)
    finally:
        owner.close()
