"""Qualify native block recompute and saved attention offload across all families."""

from copy import deepcopy
from typing import Literal

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.execution.test_native_strategies import FAMILIES, _case, _move
from unitunner.config.execution import ExecutionConfig
from unitunner.execution.activations import AttentionActivations
from unitunner.execution.allocator_bootstrap import vbar_initialized
from unitunner.execution.installation import ExecutionInstallation
from unitunner.training.contracts import RunState
from unitunner.training.engine import update_phase
from unitunner.training.session import TrainingSession


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    ("device_name", "recompute", "residency", "offload", "compiled"),
    [
        ("cpu", "block", "resident", False, False),
        ("cuda", "block", "resident", True, False),
        ("cuda", "block", "frozen-blocks", True, False),
        pytest.param(
            "cuda",
            "block",
            "vbar",
            True,
            False,
            marks=pytest.mark.skipif(not vbar_initialized(), reason="VBAR bootstrap"),
        ),
    ],
)
def test_native_activation_gradients_three_updates(
    family,
    device_name,
    recompute,
    residency,
    offload,
    compiled,
    full_training=False,
    precision: Literal["fp32", "bf16"] = "fp32",
    reference_compiled=None,
):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(927)
    device = torch.device(device_name, 0) if device_name == "cuda" else torch.device("cpu")
    with TrainingSession() as session, sdpa_kernel(SDPBackend.MATH):
        binding, modules, request = _case(family, session, "full" if full_training else "lora")
        for module in modules.values():
            module.to(device)
        request = _move(request, device)
        initial = {name: deepcopy(module.state_dict()) for name, module in modules.items()}
        parameters = tuple(modules["model" if full_training else "adapter"].parameters())
        observations = []
        for optimized in (False, True):
            for name, module in modules.items():
                module.load_state_dict(initial[name])
            optimizer = torch.optim.AdamW(parameters, lr=0.001)
            state = RunState()
            config = ExecutionConfig(
                precision=precision,
                compile=compiled if optimized or reference_compiled is None else reference_compiled,
                recompute=recompute if optimized else "none",
                residency=residency if optimized else "resident",
                attention_activation_offload=offload and optimized,
                use_pinned_memory_for_block_swap=optimized and residency == "frozen-blocks",
            )
            owner = ExecutionInstallation(binding.regions, config, device=device)
            values = {}

            def capture(opt, args, kwargs, values=values, state=state):
                for index, parameter in enumerate(parameters):
                    values[f"gradient.{state.updates}.{index}"] = (
                        None if parameter.grad is None else parameter.grad.detach().clone()
                    )

            handle = optimizer.register_step_pre_hook(capture)
            try:
                for index in range(3):
                    result = update_phase(
                        (binding.step(request) for _ in range(1)),
                        optimizer,
                        state,
                        max_grad_norm=1.0,
                        precision=precision,
                    )
                    values[f"loss.{index}"] = result.mean_loss
                if offload and optimized:
                    assert owner._activations is not None
                    assert owner._activations.offloaded_bytes > 0
                    assert owner._activations.restored_bytes > 0
            finally:
                handle.remove()
                owner.close()
            for index, parameter in enumerate(parameters):
                values[f"updated.{index}"] = parameter.detach().clone()
            observations.append(values)
        assert observations[0].keys() == observations[1].keys()
        for name, expected in observations[0].items():
            actual = observations[1][name]
            assert (actual is None) == (expected is None), name
            if expected is not None:
                torch.testing.assert_close(
                    actual, expected, atol=1e-5, rtol=1e-4, msg=lambda message, name=name: name + ": " + message
                )


@pytest.mark.parametrize(
    "family,residency,recompute",
    [
        ("wan", "resident", "block"),
        ("qwen-image-2-1", "vbar", "block"),
        ("flux3", "frozen-blocks", "block"),
    ],
)
def test_compiled_activation_combination(family, residency, recompute):
    if residency == "vbar" and not vbar_initialized():
        pytest.skip("VBAR bootstrap required")
    test_native_activation_gradients_three_updates(family, "cuda", recompute, residency, True, True)


@pytest.mark.parametrize("family", FAMILIES)
def test_compiled_vbar_block_three_updates(family):
    if not vbar_initialized():
        pytest.skip("VBAR bootstrap required")
    test_native_activation_gradients_three_updates(family, "cuda", "block", "vbar", True, True)


def test_activation_offload_strides_and_retained_graph():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda", 0)
    owner = AttentionActivations(device)
    inputs = torch.randn(4, 6, device=device, requires_grad=True)
    values = (inputs * 2).T[:, ::2]
    packet = owner.pack(values)
    try:
        restored = owner.unpack(packet)
        assert restored.stride() == values.stride()
        torch.testing.assert_close(restored, values, atol=0, rtol=0)
        with torch.autograd.graph.saved_tensors_hooks(owner.pack, owner.unpack):
            loss = values.square().sum()
        for _ in range(2):
            gradient = torch.autograd.grad(loss, inputs, retain_graph=True)[0]
            expected = torch.autograd.grad(values.square().sum(), inputs, retain_graph=True)[0]
            torch.testing.assert_close(gradient, expected, atol=0, rtol=0)
    finally:
        owner.close()
        owner.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_full_training_activation_strategies(family):
    test_native_activation_gradients_three_updates(family, "cuda", "block", "resident", True, False, True)


def test_fp32_time_and_output_islands_under_amp():
    from unitunner.models.minimax_h3.layers import TimeEmbedder

    module = TimeEmbedder(16, 32, 24).cuda()
    times = torch.tensor([0.23, 0.72], device="cuda", requires_grad=True)
    expected = module(times)
    expected_grad = torch.autograd.grad(expected.square().sum(), (times, *module.parameters()))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = module(times)
    assert actual.dtype == torch.float32
    actual_grad = torch.autograd.grad(actual.square().sum(), (times, *module.parameters()))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)

    with TrainingSession() as session:
        for family, names in (
            ("wan", ("time_embedding", "time_projection")),
            ("minimax-h3", ("video_patch_proj", "audio_patch_proj", "final_layer.video_out", "final_layer.audio_out")),
        ):
            binding, modules, request = _case(family, session)
            for value in modules.values():
                value.cuda()
            outputs = {}
            handles = []
            for name in names:

                def capture(module, arguments, output, name=name, outputs=outputs):
                    outputs[name] = output.dtype

                handles.append(binding.model.get_submodule(name).register_forward_hook(capture))
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    binding.step(_move(request, torch.device("cuda", 0)))
                assert outputs == dict.fromkeys(names, torch.float32)
            finally:
                for handle in handles:
                    handle.remove()


@pytest.mark.parametrize("residency,recompute", [("resident", "block"), ("vbar", "block")])
def test_public_activation_strategies_resume(tmp_path, monkeypatch, residency, recompute):
    if residency == "vbar" and not vbar_initialized():
        pytest.skip("VBAR bootstrap required")
    from tests.application.test_qwen_image_21_flow import test_public_training_three_updates_and_exact_resume

    monkeypatch.setenv("UNITUNNER_TEST_DEVICE", "cuda")
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with sdpa_kernel(SDPBackend.MATH):
            test_public_training_three_updates_and_exact_resume(
                tmp_path, "lora", 2, residency, recompute=recompute, attention_offload=True
            )
    finally:
        torch.use_deterministic_algorithms(deterministic)


def test_attention_replay_releases_completed_cuda_graphs():
    from unitunner.execution.attention import scaled_dot_product_attention
    from unitunner.execution.regions import ModelRegion, ModelRegions

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(64, 192, device="cuda")

        def forward(self, inputs):
            q, k, v = self.projection(inputs).chunk(3, dim=-1)
            return scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])

    module = Attention()
    owner = ExecutionInstallation(
        ModelRegions((ModelRegion(module, 0, "visual"),)),
        ExecutionConfig(compile=False, recompute="block", attention_activation_offload=True),
        device=torch.device("cuda", 0),
    )
    allocated = []
    try:
        with sdpa_kernel(SDPBackend.MATH):
            for index in range(14):
                values = torch.randn(1, 512, 64, device="cuda", requires_grad=True)
                result = module(values).square().sum()
                result.backward()
                module.zero_grad(set_to_none=True)
                del result, values
                torch.cuda.synchronize()
                if index in (1, 13):
                    allocated.append(torch.cuda.memory_allocated())
        assert allocated[1] <= allocated[0] + 1024**2, allocated
    finally:
        owner.close()


@pytest.mark.parametrize("family", ["flux", "qwen-image-2-1", "wan"])
@pytest.mark.parametrize("compiled", [False, True])
def test_vbar_block_bf16_three_updates(family, compiled):
    if not vbar_initialized():
        pytest.skip("VBAR bootstrap required")
    test_native_activation_gradients_three_updates(family, "cuda", "block", "vbar", True, compiled, precision="bf16")


@pytest.mark.parametrize("family", ["flux", "qwen-image-2-1", "wan"])
def test_compiled_vbar_block_bf16_against_native_eager(family):
    if not vbar_initialized():
        pytest.skip("VBAR bootstrap required")
    test_native_activation_gradients_three_updates(
        family, "cuda", "block", "vbar", True, True, precision="bf16", reference_compiled=False
    )


@pytest.mark.parametrize("family", ["flux", "qwen-image-2-1"])
@pytest.mark.parametrize("compiled", [False, True])
def test_vbar_block_outstanding_graphs_and_eviction(family, compiled):
    if not vbar_initialized():
        pytest.skip("VBAR bootstrap required")
    torch.manual_seed(991)
    device = torch.device("cuda", 0)
    with TrainingSession() as session, sdpa_kernel(SDPBackend.MATH):
        binding, modules, request = _case(family, session)
        for module in modules.values():
            module.to(device)
        request = _move(request, device)
        parameters = tuple(modules["adapter"].parameters())
        expected = torch.autograd.grad(binding.step(request).loss, parameters, allow_unused=True)
        owner = ExecutionInstallation(
            binding.regions, ExecutionConfig(compile=compiled, recompute="block", residency="vbar"), device=device
        )
        try:
            losses = [binding.step(request).loss, binding.step(request).loss]
            assert owner._arena is not None
            owner._arena._owner.set_watermark(0)
            for loss in reversed(losses):
                # Compiled graphs use native AOT donation and one backward per
                # graph; eager additionally qualifies repeated first-order VJPs.
                for _ in range(1 if compiled else 2):
                    actual = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=not compiled)
                    for value, reference in zip(actual, expected, strict=True):
                        assert (value is None) == (reference is None)
                        if value is not None:
                            torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-4)
                    assert owner.vbar_statistics().active_leases == 0
        finally:
            owner.close()
