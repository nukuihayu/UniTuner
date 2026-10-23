"""Exercise real AIMDO residency through native gradients, updates and checkpoint replay."""

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.application import test_flux_flow
from tests.application.test_qwen_image_21_flow import test_public_training_three_updates_and_exact_resume as public_case
from tests.execution.test_frozen_blocks import test_mixed_dtype_strided_weights_and_outstanding_graphs as mixed_case
from tests.execution.test_native_strategies import FAMILIES
from tests.execution.test_native_strategies import test_native_h2d_forward_backward_three_updates as native_case
from tests.support.text_assets import main_with_local_tokenizers
from unitunner.config.execution import ExecutionConfig
from unitunner.execution.allocator_bootstrap import vbar_initialized
from unitunner.execution.frozen_blocks import FrozenBlockInstallation
from unitunner.execution.installation import ExecutionInstallation
from unitunner.execution.mapped_weights import VbarArena, VbarWeights
from unitunner.execution.prefetch import WeightPrefetch
from unitunner.execution.regions import ModelRegion, ModelRegions
from unitunner.execution.weight_source import WeightSource

pytestmark = pytest.mark.skipif(not vbar_initialized(), reason="Run the VBAR comparison driver before importing Torch")
flux_assets = test_flux_flow.assets


@pytest.mark.parametrize("fail_at", [1, 2])
def test_failed_tensor_bridge_releases_faulted_pages(monkeypatch, fail_at):
    """A fault preceding a tensor-view failure must not leave an unowned page pin."""
    host = torch.ones(1024, pin_memory=True)
    owner = VbarWeights((host, host), device=torch.device("cuda", 0))
    try:
        calls = 0
        make_view = owner._bridge.aimdo_to_tensor

        def fail_view(*args):
            nonlocal calls
            calls += 1
            if calls == fail_at:
                raise RuntimeError("injected tensor bridge failure")
            return make_view(*args)

        with monkeypatch.context() as patch:
            patch.setattr(owner._bridge, "aimdo_to_tensor", fail_view)
            with pytest.raises(RuntimeError, match="tensor bridge failure"):
                owner.acquire()
        assert owner.statistics().active_leases == 0
        assert owner._owner.free_memory(32 * 1024**2) == 32 * 1024**2
        owner._owner.prioritize()
        leases = owner.acquire()
        for lease in leases:
            torch.testing.assert_close(lease.tensor.cpu(), host, atol=0, rtol=0)
        owner.release(leases)
    finally:
        owner.close()
        owner.close()


def test_speculative_headroom_backs_off_only_after_allocator_retries(monkeypatch):
    size = 1024**2
    arena = VbarArena(32 * size, device=torch.device("cuda", 0), allocation_workspace_bytes=size)
    try:
        arena.allocate((4 * size,))
        retries = arena._allocation_retries
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "memory_stats", lambda device: {"num_alloc_retries": retries})
            arena._completed_backward()
            assert arena._allocation_workspace_bytes == size
            retries += 1
            arena._completed_backward()
            assert arena._allocation_workspace_bytes == 4 * size
            arena._completed_backward()
            assert arena._allocation_workspace_bytes == 4 * size
    finally:
        arena.close()


def test_vbar_cold_cuda_context_keeps_weights_resident():
    """Fault resident pages before any Torch CUDA allocation in a fresh process."""
    program = """
from unitunner.execution.allocator_bootstrap import initialize_vbar
initialize_vbar()
import torch
from unitunner.execution import mapped_weights
host = torch.ones(1024 * 1024)
owner = mapped_weights.VbarWeights((host,), device=torch.device("cuda", 0))
try:
    leases = owner.acquire()
    assert leases[0].allocation is not None, "Cold context incorrectly forced fallback"
    torch.testing.assert_close(leases[0].tensor.cpu(), host, atol=0, rtol=0)
    owner.release(leases)
    repeated = owner.acquire()
    assert owner.statistics().transfers == 1
    assert owner.statistics().hits == 1
    owner.release(repeated)
finally:
    owner.close()
print(mapped_weights.__file__)
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    module_file = sys.modules[VbarArena.__module__].__file__
    assert module_file is not None
    assert str(Path(module_file).resolve()) in result.stdout


def test_vbar_preserves_mmap_sources_and_only_stages_misses(tmp_path):
    """Keep CPU storage and parameter identity through cache hits and forced misses."""
    path = tmp_path / "weights.bin"
    path.write_bytes(bytes(16 * 16 * 4))
    source = torch.from_file(str(path), size=16 * 16, dtype=torch.float32).reshape(16, 16)
    source.copy_(torch.arange(source.numel()).reshape_as(source) / source.numel())
    model = torch.nn.Linear(16, 16, bias=False)
    model.weight = torch.nn.Parameter(source, requires_grad=False)
    identity, pointer = id(model.weight), model.weight.data_ptr()
    inputs = torch.randn(2, 16, device="cuda", requires_grad=True)
    expected = torch.nn.functional.linear(inputs, source.cuda())
    owner = FrozenBlockInstallation(ModelRegion(model, 0, "visual"), device=inputs.device, use_vbar=True)
    try:
        assert id(model.weight) == identity and model.weight.data_ptr() == pointer
        assert not model.weight.is_pinned()
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), expected, atol=0, rtol=0)
            first = owner.vbar_statistics()
            torch.testing.assert_close(model(inputs), expected, atol=0, rtol=0)
        second = owner.vbar_statistics()
        assert first.transfers == second.transfers == 1
        assert second.hits == first.hits + 1
        assert 0 < second.pinned_staging_bytes <= 128 * 1024**2
        assert owner._vbar is not None
        owner._vbar._owner.set_watermark(0)
        actual = model(inputs)
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert owner.vbar_statistics().fallbacks > 0
        assert owner.vbar_statistics().registered_source_bytes > 0
        assert model.weight.is_pinned()
    finally:
        owner.close()
    assert id(model.weight) == identity and model.weight.data_ptr() == pointer
    assert not model.weight.is_pinned()


def test_vbar_bounded_staging_and_retired_buffer_reuse():
    """Reuse both host windows and fallback storage without corrupting earlier readers."""
    device = torch.device("cuda", torch.cuda.current_device())
    # Cross both 64MiB windows and source boundaries, including a partial tail.
    chunks = tuple(torch.full((17 * 1024**2 + index,), index + 0.25) for index in range(3))
    owner = VbarWeights((WeightSource(chunks),), device=device)
    try:
        owner._arena.staging._registration_budget = 0
        owner._owner.set_watermark(0)
        first = owner.acquire()
        pointer = first[0].tensor.data_ptr()
        offset = 0
        for chunk in chunks:
            torch.testing.assert_close(first[0].tensor[offset : offset + chunk.numel()].cpu(), chunk, atol=0, rtol=0)
            offset += chunk.numel()
        earlier = first[0].tensor.sum()
        owner.release(first)
        second = owner.acquire()
        assert second[0].tensor.data_ptr() == pointer
        assert owner.statistics().pinned_staging_bytes == 128 * 1024**2
        torch.testing.assert_close(second[0].tensor.sum(), earlier, atol=0, rtol=0)
        owner.release(second)
    finally:
        owner.close()
    assert owner.statistics().pinned_staging_bytes == 0


def test_vbar_source_registration_handles_shared_partial_pages():
    """A small registered view must not make a larger overlapping source appear pinned."""
    device = torch.device("cuda", torch.cuda.current_device())
    host = torch.arange(1024 * 1024, dtype=torch.float32)
    owner = VbarWeights((WeightSource((host[:17],)), WeightSource((host[17:],))), device=device)
    try:
        owner._owner.set_watermark(0)
        leases = owner.acquire()
        torch.testing.assert_close(leases[0].tensor.cpu(), host[:17], atol=0, rtol=0)
        torch.testing.assert_close(leases[1].tensor.cpu(), host[17:], atol=0, rtol=0)
        assert host.nbytes <= owner.statistics().registered_source_bytes < host.nbytes + 8192
        assert owner.statistics().pinned_staging_bytes == 0
        owner.release(leases)
    finally:
        owner.close()
    assert not host.is_pinned()


def test_vbar_source_registration_preserves_neighbouring_cpu_copies():
    """Registering a small view must not break copies into an adjacent malloc range."""
    device = torch.device("cuda", torch.cuda.current_device())
    host = torch.zeros(1024 * 1024, dtype=torch.float32)
    owner = VbarWeights((host[:17],), device=device)
    try:
        owner._owner.set_watermark(0)
        leases = owner.acquire()
        adjacent = torch.ones(host[17:].shape, device=device)
        host[17:].copy_(adjacent)
        torch.testing.assert_close(host[17:], torch.ones_like(host[17:]), atol=0, rtol=0)
        torch.testing.assert_close(leases[0].tensor.cpu(), host[:17], atol=0, rtol=0)
        owner.release(leases)
    finally:
        owner.close()
    assert not host.is_pinned()


def test_vbar_fallback_cache_source_identity_and_best_fit():
    """Keep immutable contents across direction changes and invalidate overwritten buffers."""
    device = torch.device("cuda", torch.cuda.current_device())
    arena = VbarArena(4 * 1024**2, device=device, defer_release=True)
    hosts = [torch.full((count,), value) for count, value in ((262144, 1.0), (524288, 2.0), (131072, 3.0))]
    owners = [VbarWeights((host,), device=device, arena=arena) for host in hosts]
    try:
        arena.staging._registration_budget = 0
        arena._owner.set_watermark(0)
        first, larger = owners[0].acquire(), owners[1].acquire()
        pointer = first[0].tensor.data_ptr()
        owners[0].release(first)
        owners[1].release(larger)
        for _ in range(2):
            reused = owners[0].acquire()
            assert reused[0].tensor.data_ptr() == pointer
            torch.testing.assert_close(reused[0].tensor.cpu(), hosts[0], atol=0, rtol=0)
            owners[0].release(reused)
        assert owners[0].statistics().transfers == 1
        assert owners[0].statistics().hits == 2
        replacement = owners[2].acquire()
        assert replacement[0].tensor.data_ptr() == pointer
        torch.testing.assert_close(replacement[0].tensor.cpu(), hosts[2], atol=0, rtol=0)
        owners[2].release(replacement)
        restored = owners[0].acquire()
        torch.testing.assert_close(restored[0].tensor.cpu(), hosts[0], atol=0, rtol=0)
        assert owners[0].statistics().transfers == 2
        owners[0].release(restored)
    finally:
        for owner in owners:
            owner.close()
        arena.close()


def test_vbar_failed_fallback_copy_never_caches_partial_contents(monkeypatch):
    """A failed transfer must invalidate a returned buffer before the next demand load."""
    device = torch.device("cuda", torch.cuda.current_device())
    host = torch.ones(1024)
    owner = VbarWeights((host,), device=device)
    try:
        owner._arena.staging._registration_budget = 0
        owner._owner.set_watermark(0)

        def fail_copy(source, destination, *, streaming=False):
            destination.fill_(-1)
            raise RuntimeError("injected incomplete weight copy")

        with monkeypatch.context() as patch:
            patch.setattr(owner._arena.staging, "copy", fail_copy)
            with pytest.raises(RuntimeError, match="incomplete weight copy"):
                owner.acquire()
        assert owner.statistics().active_leases == 0
        restored = owner.acquire()
        torch.testing.assert_close(restored[0].tensor.cpu(), host, atol=0, rtol=0)
        assert owner.statistics().transfers == 1
        assert owner.statistics().hits == 0
        owner.release(restored)
    finally:
        owner.close()


def test_vbar_prefetch_skips_resident_regions_with_distributed_addresses():
    """Prefetch the next actual miss while intervening resident blocks keep computing."""
    device = torch.device("cuda", torch.cuda.current_device())
    size = 8 * 1024**2
    arena = VbarArena(16 * size, device=device, defer_release=True)
    arena.plan_regions(((size,),) * 16)
    hosts = [torch.full((size // 4,), float(index)) for index in range(16)]
    weights = [VbarWeights((host,), device=device, arena=arena) for host in hosts]
    prefetch = WeightPrefetch(arena.device, arena=arena)
    for weight in weights:
        prefetch.register(weight)
    try:
        arena._owner.set_watermark(12 * size)
        for weight, host in zip(weights, hosts, strict=True):
            leases = prefetch.acquire(weight)
            torch.testing.assert_close(leases[0].tensor.cpu(), host, atol=0, rtol=0)
            prefetch.next(weight, backward=False)
            weight.release(leases)
        prefetch.cancel()
        first = prefetch.acquire(weights[0])
        prefetch.next(weights[0], backward=False)
        prepared = prefetch._pending[0] if prefetch._pending else None
        assert prepared is not None and prepared[0] is weights[3]
        weights[0].release(first)
        for index in (1, 2):
            leases = prefetch.acquire(weights[index])
            torch.testing.assert_close(leases[0].tensor.cpu(), hosts[index], atol=0, rtol=0)
            prefetch.next(weights[index], backward=False)
            assert prefetch._pending[0] is prepared
            weights[index].release(leases)
        streamed = prefetch.acquire(weights[3])
        assert streamed is prepared[1]
        torch.testing.assert_close(streamed[0].tensor.cpu(), hosts[3], atol=0, rtol=0)
        weights[3].release(streamed)
    finally:
        prefetch.close()
        for weight in weights:
            weight.close()
        arena.close()


@pytest.mark.parametrize("compiled", [False, True])
def test_vbar_lookahead_branches_outstanding_and_retained_graphs(compiled):
    """An unexpected region order and retained backward must not consume stale pages."""
    from copy import deepcopy

    torch.manual_seed(817)
    device = torch.device("cuda", torch.cuda.current_device())
    blocks = [torch.nn.Linear(16, 16, device=device) for _ in range(3)]
    for block in blocks:
        block.weight.requires_grad_(False)
    references = deepcopy(blocks)
    owner = ExecutionInstallation(
        ModelRegions(tuple(ModelRegion(block, index, "visual") for index, block in enumerate(blocks))),
        ExecutionConfig(compile=compiled, residency="vbar"),
        device=device,
    )
    try:
        assert owner._arena is not None
        owner._arena._owner.set_watermark(0)
        inputs = [torch.randn(3, 16, device=device, requires_grad=True) for _ in range(2)]
        copies = [value.detach().clone().requires_grad_() for value in inputs]
        actual = [blocks[2](blocks[0](inputs[0])), blocks[1](blocks[2](inputs[1]))]
        expected = [references[2](references[0](copies[0])), references[1](references[2](copies[1]))]
        for value, reference in zip(reversed(actual), reversed(expected), strict=True):
            for _ in range(2):
                value.square().sum().backward(retain_graph=True)
                reference.square().sum().backward(retain_graph=True)
                for a, b in zip(
                    (*inputs, *(block.bias for block in blocks)),
                    (*copies, *(block.bias for block in references)),
                    strict=True,
                ):
                    assert (a.grad is None) == (b.grad is None)
                    if a.grad is not None:
                        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4)
                assert owner.vbar_statistics().active_leases == 0
        with torch.no_grad():
            blocks[0](inputs[0])
        assert owner.vbar_statistics().fallbacks > 0
    finally:
        owner.close()
        owner.close()
    assert all("forward" not in block.__dict__ for block in blocks)


@pytest.mark.parametrize("family", FAMILIES)
def test_native_vbar_updates_and_resume(tmp_path, family):
    native_case(tmp_path, family, True, 0, residency="vbar")


def test_vbar_mixed_weights_and_two_outstanding_graphs():
    mixed_case(True, 0, vbar=True)


def test_vbar_pin_eviction_fallback_and_repopulation(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    host = torch.randn(1024, pin_memory=True)
    owner = VbarWeights((host,), device=device)
    try:
        first = owner.acquire()
        torch.testing.assert_close(first[0].tensor, host.to(device), atol=0, rtol=0)
        assert owner.statistics().transfers == 1
        owner.release(first)
        first = owner.acquire()
        assert owner.statistics().hits == 1
        # A live backward lease prevents forced eviction of the referenced page.
        assert owner._owner.free_memory(32 * 1024**2) == 0
        inputs = torch.randn(32, device=device, requires_grad=True)
        output = inputs @ first[0].tensor.reshape(32, 32)
        output.square().sum().backward()
        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()
        owner.release(first)
        # Pressure lowered the watermark while pinned; unpin now frees the page.
        assert owner.statistics().loaded_bytes == 0
        fallback = owner.acquire()
        assert fallback[0].allocation is None
        torch.testing.assert_close(fallback[0].tensor, host.to(device), atol=0, rtol=0)
        owner.release(fallback)
        # Simulate a later free-capacity increase without allocating most of the
        # shared GPU merely to manufacture pressure. The actual reload is native.
        free_bytes, total = torch.cuda.mem_get_info(device)
        owner._arena._recovery_check_seconds = 0.0
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "mem_get_info", lambda device: (free_bytes + 32 * 1024**2, total))
            reloaded = owner.acquire()
        torch.testing.assert_close(reloaded[0].tensor, host.to(device), atol=0, rtol=0)
        assert owner.statistics().transfers == 3
        owner.release(reloaded)
        assert owner.statistics().active_leases == 0
    finally:
        owner.close()
        owner.close()
    assert owner.statistics().loaded_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        owner.acquire()


def test_vbar_retained_graph_releases_each_backward():
    model = torch.nn.Linear(8, 8, device="cuda")
    model.weight.requires_grad_(False)
    inputs = torch.randn(2, 8, device="cuda", requires_grad=True)
    expected = model(inputs).square().sum()
    gradients = torch.autograd.grad(expected, (inputs, model.bias))
    owner = FrozenBlockInstallation(ModelRegion(model, 0, "visual"), device=inputs.device, use_vbar=True)
    try:
        loss = model(inputs).square().sum()
        for _ in range(2):
            loss.backward(retain_graph=True)
            torch.testing.assert_close(inputs.grad, gradients[0], atol=0, rtol=0)
            torch.testing.assert_close(model.bias.grad, gradients[1], atol=0, rtol=0)
            assert owner.vbar_statistics().active_leases == 0
            inputs.grad = model.bias.grad = None
    finally:
        owner.close()


def test_vbar_failed_recompute_drains_leases_and_hooks():
    class FailsOnReplay(torch.nn.Linear):
        calls = 0
        fail = True

        def forward(self, input):
            self.calls += 1
            if self.calls > 1 and self.fail:
                raise RuntimeError("deliberate recompute failure")
            return super().forward(input).square()

    model = FailsOnReplay(8, 8, device="cuda")
    model.weight.requires_grad_(False)
    inputs = torch.randn(2, 8, device="cuda")
    owner = FrozenBlockInstallation(ModelRegion(model, 0, "visual"), device=inputs.device, use_vbar=True)
    with pytest.raises(RuntimeError, match="deliberate"):
        model(inputs).square().sum().backward()
    assert owner.vbar_statistics().active_leases > 0
    owner.close()
    assert owner.vbar_statistics().active_leases == 0
    model.fail = False
    model(inputs).square().sum().backward()
    assert model.bias.grad is not None
    assert torch.isfinite(model.bias.grad).all()


def test_vbar_rejects_higher_order_graph_before_releasing_live_weights():
    model = torch.nn.Linear(8, 8, device="cuda")
    model.weight.requires_grad_(False)
    inputs = torch.randn(2, 8, device="cuda", requires_grad=True)
    owner = FrozenBlockInstallation(ModelRegion(model, 0, "visual"), device=inputs.device, use_vbar=True)
    try:
        loss = model(inputs).square().sum()
        with pytest.raises(ValueError, match="first-order"):
            torch.autograd.grad(loss, (inputs, model.bias), create_graph=True)
    finally:
        owner.close()


@pytest.mark.parametrize("recompute", ["block"])
def test_vbar_public_training_and_cold_cli_resume(tmp_path, monkeypatch, recompute):
    # Cross-process byte equality also requires deterministic CUDA backward kernels.
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with sdpa_kernel(SDPBackend.MATH):
            _public_training_and_cold_cli_resume(tmp_path, monkeypatch, recompute=recompute)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _public_training_and_cold_cli_resume(tmp_path, monkeypatch, *, recompute):
    monkeypatch.setenv("UNITUNNER_TEST_DEVICE", "cuda")
    public_case(tmp_path, "lora", 2, residency="vbar", recompute=recompute, attention_offload=True)
    # The application reference also verifies changed-content resume rejection;
    # restore its original caption before testing a fresh CLI process.
    (tmp_path / "images/a.txt").write_text("short caption")
    assert main_with_local_tokenizers(["cache", str(tmp_path / "train.toml"), "--device", "cuda"]) == 0
    artifact = tmp_path / "run/weights/model.safetensors"
    expected = artifact.read_bytes()
    checkpoint = tmp_path / "run/resume/model.state-00000003.safetensors"
    expected_checkpoint = checkpoint.read_bytes()
    spec = find_spec("comfy_aimdo.control")
    assert spec is not None and spec.origin is not None
    environment = os.environ.copy()
    environment["PYTHONPATH"] = environment.get("PYTHONPATH", "") + os.pathsep + str(Path(spec.origin).parent.parent)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from unitunner.execution.allocator_bootstrap import initialize_vbar; initialize_vbar(); "
            "from tests.support.text_assets import main_with_local_tokenizers; import sys; "
            "import torch; torch.use_deterministic_algorithms(True); "
            "from torch.nn.attention import SDPBackend, sdpa_kernel\n"
            "with sdpa_kernel(SDPBackend.MATH): raise SystemExit(main_with_local_tokenizers(sys.argv[1:]))",
            "train",
            str(tmp_path / "train.toml"),
            "--device",
            "cuda",
            "--resume",
            str(tmp_path / "run/resume/model.state-00000001.safetensors"),
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert artifact.read_bytes() == expected
    assert checkpoint.read_bytes() == expected_checkpoint


def test_vbar_flux_public_training_previews_and_resume(flux_assets, monkeypatch):
    monkeypatch.setenv("UNITUNNER_TEST_DEVICE", "cuda")
    test_flux_flow.test_flux_real_variant_cache_training_resume_and_content_identity(
        flux_assets, "dev", "lora", residency="vbar"
    )
