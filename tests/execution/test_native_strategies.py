"""Compare shared block execution against all ten native model families on CUDA."""

from dataclasses import asdict, fields, is_dataclass, replace
from typing import Any, Literal, cast

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.models import test_z_image
from tests.training import (
    test_flux,
    test_flux2,
    test_flux3,
    test_minimax_h3_flow,
    test_moss_image,
    test_qwen_image,
    test_wan,
)
from unitunner.adapters.config import AdapterConfig
from unitunner.checkpoints.restore import restore_checkpoint
from unitunner.checkpoints.saving import save_checkpoint
from unitunner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unitunner.config.execution import ExecutionConfig
from unitunner.execution.allocator_bootstrap import vbar_initialized
from unitunner.execution.frozen_blocks import FrozenModelPlacement
from unitunner.execution.installation import ExecutionInstallation
from unitunner.models.krea2.config.model import Krea2Config
from unitunner.models.krea2.transformer import Krea2Transformer
from unitunner.models.minimax_h3.config.training import MiniMaxH3FlowConfig
from unitunner.models.qwen_image_2_1.config.model import QwenImage21Config
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unitunner.training.engine import update_phase
from unitunner.training.integrations.krea2.binding import Krea2Binding, Krea2FlowRequest
from unitunner.training.integrations.qwen_image_2_1.binding import QwenImage21Binding, QwenImage21FlowRequest
from unitunner.training.integrations.wan.binding import WanFlowRequest
from unitunner.training.integrations.z_image.binding import ZImageBinding
from unitunner.training.parameters import ParameterInstallation
from unitunner.training.session import TrainingSession
from unitunner.training.state import RunState

FAMILIES = [
    "flux",
    "flux2",
    "flux3",
    "moss-image",
    "qwen-image",
    "qwen-image-2-1",
    "z-image",
    "krea2",
    "wan",
    "minimax-h3",
]


def test_krea_derived_time_matches_explicit_and_rejects_invalid_inputs():
    """Avoid rechecking derived times while retaining the explicit input boundary."""
    with TrainingSession() as session:
        binding, _, request = _case("krea2", session)
        derived = binding.step(request)
        explicit = binding.step(replace(request, timesteps=request.sigma * 1000))
        torch.testing.assert_close(derived.loss, explicit.loss, atol=0, rtol=0)
        for value in (float("nan"), float("inf"), -1.0, 1001.0):
            with pytest.raises(ValueError, match="timesteps"):
                binding.step(replace(request, timesteps=torch.full_like(request.sigma, value)))


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _move(getattr(value, field.name), device) for field in fields(value)})
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _case(family, session, mode="lora") -> tuple[Any, dict[str, Any], Any]:
    helpers = {"flux": test_flux, "flux2": test_flux2, "flux3": test_flux3, "moss-image": test_moss_image}
    if family in helpers:
        helper = helpers[family]
        binding, modules, _optimizer = helper.build(session, mode)
        return binding, modules, helper.request()
    if family == "qwen-image":
        binding, modules, _optimizer = test_qwen_image.build(session, mode, "edit-2511")
        return binding, modules, test_qwen_image.request("edit-2511", False)
    if family == "wan":
        binding, modules, _optimizer = test_wan.build(session, mode)
        clean = torch.randn(2, 16, 2, 4, 4)
        sigma = torch.tensor([0.244, 0.751])
        return (
            binding,
            modules,
            WanFlowRequest(
                clean, torch.randn_like(clean), sigma, sigma * 1000 + 1, (torch.randn(3, 32), torch.randn(5, 32))
            ),
        )
    if family == "z-image":
        binding = ZImageBinding(test_z_image.build_model())
        request = test_z_image.request()
    elif family == "minimax-h3":
        binding = test_minimax_h3_flow._binding(MiniMaxH3FlowConfig())
        request = test_minimax_h3_flow._request()
    elif family == "qwen-image-2-1":
        binding = QwenImage21Binding(
            QwenImage21Transformer(
                QwenImage21Config(
                    in_channels=4,
                    out_channels=4,
                    context_in_dim=6,
                    num_layers=2,
                    attention_head_dim=8,
                    num_attention_heads=2,
                    axes_dims_rope=(2, 2, 4),
                )
            )
        )
        clean = torch.randn(2, 4, 1, 2, 3)
        request = QwenImage21FlowRequest(
            clean, torch.randn_like(clean), torch.tensor([0.244, 0.751]), (torch.randn(3, 6), torch.randn(5, 6))
        )
    else:
        binding = Krea2Binding(
            Krea2Transformer(
                Krea2Config(
                    hidden_size=64,
                    timestep_dim=16,
                    text_dim=32,
                    num_heads=4,
                    num_kv_heads=2,
                    mlp_multiplier=2,
                    num_layers=2,
                    text_layers=3,
                    text_heads=4,
                    text_kv_heads=2,
                )
            )
        )
        clean = torch.randn(2, 16, 1, 4, 6)
        request = Krea2FlowRequest(
            clean, torch.randn_like(clean), torch.tensor([0.244, 0.751]), (torch.randn(3, 3, 32), torch.randn(5, 3, 32))
        )
    if mode == "full":
        session.own(ParameterInstallation(binding.model, adapter=None, targets={}))
        return binding, {"model": binding.model}, request
    owner = session.own(
        ParameterInstallation(
            binding.model, adapter=AdapterConfig(algorithm="lora", rank=2), targets=binding.adapter_targets()
        )
    )
    with torch.no_grad():
        for parameter in owner.branches.parameters():
            parameter.normal_(std=0.03)
    return binding, {"model": binding.model, "adapter": owner.branches}, request


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(("pinned", "cache_mb"), [(False, 0), (True, 0), (True, 1)])
def test_native_h2d_forward_backward_three_updates(
    tmp_path,
    family,
    pinned,
    cache_mb,
    residency: Literal["resident", "frozen-blocks", "vbar", "trainable-blocks"] = "frozen-blocks",
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(941)
    device = torch.device("cuda", torch.cuda.current_device())
    with TrainingSession() as session, sdpa_kernel(SDPBackend.MATH):
        binding, modules, request = _case(family, session)
        model, adapter = modules["model"], modules["adapter"]
        model.to(device)
        adapter.to(device)
        request = _move(request, device)
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        initial_adapter = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
        identities = [id(parameter) for parameter in model.parameters()]
        parameters = tuple(adapter.parameters())
        expected = binding.step(request)
        expected.loss.backward()
        gradients = [None if parameter.grad is None else parameter.grad.clone() for parameter in parameters]
        for parameter in parameters:
            parameter.grad = None
        config = ExecutionConfig(
            compile=False,
            residency=residency,
            recompute="block",
            use_pinned_memory_for_block_swap=pinned,
            frozen_weight_cache_mb=cache_mb,
        )
        model.cpu()
        placement = FrozenModelPlacement(model, binding.regions, device=device)
        owner = ExecutionInstallation(binding.regions, config, device=device)
        try:
            actual = binding.step(request)
            actual.loss.backward()
            torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
            for before, parameter in zip(gradients, parameters, strict=True):
                assert (before is None) == (parameter.grad is None)
                if before is not None:
                    torch.testing.assert_close(parameter.grad, before, atol=0, rtol=0)
        finally:
            owner.close()
            placement.close()
            model.to(device)
        for parameter in parameters:
            parameter.grad = None
        del actual, expected, gradients
        snapshots = []
        for strategy in ("resident", residency):
            model.load_state_dict(initial)
            adapter.load_state_dict(initial_adapter)
            optimizer = torch.optim.AdamW(parameters, lr=0.001)
            state = RunState()
            execution = ExecutionInstallation(
                binding.regions,
                ExecutionConfig(
                    compile=False,
                    residency=cast(Literal["resident", "frozen-blocks", "vbar", "trainable-blocks"], strategy),
                    recompute="block",
                    use_pinned_memory_for_block_swap=pinned if strategy != "resident" else False,
                    frozen_weight_cache_mb=cache_mb if strategy != "resident" else 0,
                ),
                device=device,
            )
            try:
                identity = ResumeIdentity(
                    config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64
                )
                checkpoint = tmp_path / (strategy + ".safetensors")
                for update in range(3):
                    update_phase((binding.step(request) for _ in range(2)), optimizer, state, max_grad_norm=1.0)
                    if update == 0:
                        save_checkpoint(
                            checkpoint,
                            modules,
                            optimizer,
                            CheckpointCursor(**asdict(state)),
                            identity,
                            cuda_devices=(device.index,),
                        )
                continuous = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
                cursor = restore_checkpoint(checkpoint, modules, optimizer, identity, cuda_devices=(device.index,))
                state = RunState(**cursor.model_dump())
                for _ in range(2):
                    update_phase((binding.step(request) for _ in range(2)), optimizer, state, max_grad_norm=1.0)
                assert all(torch.equal(value, adapter.state_dict()[key]) for key, value in continuous.items())
                assert state == RunState(updates=3, microbatches=6, samples=6 * (1 if family == "minimax-h3" else 2))
                snapshots.append({key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()})
            finally:
                execution.close()
        assert identities == [id(parameter) for parameter in model.parameters()]
        assert snapshots[0].keys() == snapshots[1].keys()
        assert all(torch.equal(value, snapshots[1][key]) for key, value in snapshots[0].items())


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    ("residency", "cache_mb"),
    [
        ("resident", 0),
        ("frozen-blocks", 0),
        ("frozen-blocks", 1),
        pytest.param(
            "vbar", 0, marks=pytest.mark.skipif(not vbar_initialized(), reason="Requires early VBAR bootstrap")
        ),
    ],
)
def test_native_inductor_forward_and_every_active_gradient(family, residency, cache_mb):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(941)
    device = torch.device("cuda", torch.cuda.current_device())
    with TrainingSession() as session, sdpa_kernel(SDPBackend.MATH):
        binding, modules, request = _case(family, session)
        modules["model"].to(device)
        modules["adapter"].to(device)
        request = _move(request, device)
        parameters = tuple(modules["adapter"].parameters())
        expected = binding.step(request)
        expected.loss.backward()
        gradients = [None if value.grad is None else value.grad.clone() for value in parameters]
        for parameter in parameters:
            parameter.grad = None
        owner = ExecutionInstallation(
            binding.regions,
            ExecutionConfig(compile=True, recompute="block", residency=residency, frozen_weight_cache_mb=cache_mb),
            device=device,
        )
        try:
            actual = binding.step(request)
            actual.loss.backward()
            # Declared FP32 model budget for compiled versus accepted eager math.
            torch.testing.assert_close(actual.loss, expected.loss, atol=1e-5, rtol=1e-4)
            for before, parameter in zip(gradients, parameters, strict=True):
                assert (before is None) == (parameter.grad is None)
                if before is not None:
                    torch.testing.assert_close(parameter.grad, before, atol=1e-5, rtol=1e-4)
        finally:
            owner.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_native_layersync_target_capture_backward_and_h2d(family, device_name):
    from unitunner.config.features import LayerSyncConfig
    from unitunner.training.objectives import FeatureAlignment

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(941)
    device = torch.device("cuda", torch.cuda.current_device()) if device_name == "cuda" else torch.device("cpu")
    with TrainingSession() as session, sdpa_kernel(SDPBackend.MATH):
        binding, modules, request = _case(family, session)
        modules["model"].to(device)
        modules["adapter"].to(device)
        request = _move(request, device)
        alignment = FeatureAlignment(modules["model"], LayerSyncConfig(weight=0.3), num_layers=2)
        strategy = (
            ExecutionConfig(residency="frozen-blocks", recompute="block")
            if device_name == "cuda"
            else ExecutionConfig(recompute="block")
        )
        execution = ExecutionInstallation(binding.regions, strategy, device=device)
        try:
            with alignment.capture():
                result = binding.step(request)
                students = alignment.recorder.features[0]
                teachers = alignment.recorder.features[1]
                assert len(students) == len(teachers)
                count = sum(value.shape[0] for value in students)
                expected = result.loss + 0.3 * sum(
                    -torch.nn.functional.cosine_similarity(left.float(), right.detach().float(), dim=-1).mean()
                    * left.shape[0]
                    / count
                    for left, right in zip(students, teachers, strict=True)
                )
                parameters = tuple(modules["adapter"].parameters())
                expected_value = expected.detach()
                expected_gradients = torch.autograd.grad(expected, parameters, allow_unused=True)
            assert not alignment.recorder.features and not alignment.recorder.active
            # Compare independent graphs, as production performs one backward
            # per microbatch. Reusing a donated AOT graph is not a loss test.
            with alignment.capture():
                actual = alignment.step(binding.step(request))
                torch.testing.assert_close(actual.loss, expected_value, atol=0, rtol=0)
            observed = torch.autograd.grad(actual.loss, parameters, allow_unused=True)
            for left, right in zip(observed, expected_gradients, strict=True):
                assert (left is None) == (right is None)
                if left is not None:
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
        finally:
            execution.close()
            alignment.close()
            alignment.close()
        assert modules["model"].feature_recorder is None


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("compiled", [False, True])
def test_native_tread_resident_vs_h2d_three_updates(family, compiled):
    """Require exact eager replay and qualify compiled gradients AND AdamW trajectories.

    Independent Inductor graphs may fuse/reassociate FP32 reductions differently.
    Do not add runtime graph breaks just to make those graphs bitwise identical.
    RNG and gradient participation must still match exactly for both providers.
    """
    from unitunner.config.routing import TreadConfig
    from unitunner.training.objectives import TokenRoutingInstallation

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(941)
    device = torch.device("cuda", torch.cuda.current_device())
    with TrainingSession() as session, sdpa_kernel(SDPBackend.MATH):
        binding, modules, request = _case(family, session)
        model, adapter = modules["model"], modules["adapter"]
        model.to(device)
        adapter.to(device)
        request = _move(request, device)
        routing = TokenRoutingInstallation(model, TreadConfig(end_layer=2), num_layers=2)
        initial = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
        observations = []
        try:
            for strategy in ["resident", "frozen-blocks"]:
                adapter.load_state_dict(initial)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, device)
                execution = ExecutionInstallation(
                    binding.regions,
                    ExecutionConfig(
                        compile=compiled,
                        residency=cast(Literal["resident", "frozen-blocks", "vbar", "trainable-blocks"], strategy),
                        recompute="block",
                    ),
                    device=device,
                )
                optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001)
                trajectory = []
                try:
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        loss = binding.step(request).loss
                        loss.backward()
                        gradients = [
                            None if value.grad is None else value.grad.clone() for value in adapter.parameters()
                        ]
                        optimizer.step()
                        trajectory.append(
                            (
                                loss.detach(),
                                gradients,
                                [value.detach().clone() for value in adapter.parameters()],
                                torch.get_rng_state().clone(),
                                torch.cuda.get_rng_state(device).clone(),
                            )
                        )
                    observations.append(trajectory)
                finally:
                    execution.close()
                    optimizer.zero_grad(set_to_none=True)
        finally:
            routing.close()
        for left, right in zip(*observations, strict=True):
            torch.testing.assert_close(left[0], right[0], atol=1e-6 if compiled else 0, rtol=1e-5 if compiled else 0)
            assert torch.equal(left[3], right[3])
            assert torch.equal(left[4], right[4])
            for x, y in zip(left[1], right[1], strict=True):
                assert (x is None) == (y is None)
                if x is not None:
                    if compiled:
                        # A scale-invariant FP32 error budget handles cancellation
                        # near zero without accepting all small gradients outright.
                        reference = y.double()
                        torch.testing.assert_close(
                            (x.double() - reference).norm(),
                            reference.new_zeros(()),
                            atol=1e-5 * reference.norm().item(),
                            rtol=0,
                        )
                    else:
                        torch.testing.assert_close(x, y, atol=0, rtol=0)
            for x, y in zip(left[2], right[2], strict=True):
                torch.testing.assert_close(x, y, atol=2e-6 if compiled else 0, rtol=1e-5 if compiled else 0)
        assert model.token_routing is None


@pytest.mark.parametrize("family", ["flux", "flux2", "flux3", "qwen-image"])
def test_native_preview_sampling_resident_vs_cpu_loaded_h2d(family):
    from unitunner.config.routing import TreadConfig
    from unitunner.models.components.conditioning import Caption
    from unitunner.models.flux.inference.sampling import sample_latents as sample_flux
    from unitunner.models.flux2.inference.sampling import sample_latents as sample_flux2
    from unitunner.models.flux3.inference.sampling import sample_latents as sample_flux3
    from unitunner.models.qwen_image.inference.sampling import sample_latents as sample_qwen
    from unitunner.training.objectives import TokenRoutingInstallation
    from unitunner.training.previews.context import preview_context

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(941)
    device = torch.device("cuda", torch.cuda.current_device())
    with TrainingSession() as session, sdpa_kernel(SDPBackend.MATH):
        binding, modules, request = _case(family, session)
        model, adapter = modules["model"], modules["adapter"]
        model.to(device)
        adapter.to(device)
        request = _move(request, device)
        session.own(TokenRoutingInstallation(model, TreadConfig(end_layer=2), num_layers=2))

        def sample(allow_cpu_frozen):
            if family == "flux":
                return sample_flux(
                    model, request.noise, request.captions, request.pooled, steps=2, allow_cpu_frozen=allow_cpu_frozen
                )
            if family == "flux2":
                return sample_flux2(
                    model,
                    request.noise,
                    request.captions,
                    negative=request.captions if model.config.variant == "base" else None,
                    references=request.references,
                    steps=2,
                    allow_cpu_frozen=allow_cpu_frozen,
                )
            if family == "flux3":
                return sample_flux3(
                    model,
                    request.noise,
                    request.captions,
                    negative=request.captions,
                    references=request.references,
                    steps=2,
                    allow_cpu_frozen=allow_cpu_frozen,
                )
            return sample_qwen(
                model,
                request.noise[:1],
                Caption(request.captions[0].unsqueeze(0)),
                references=tuple(value[:1] for value in request.references),
                steps=2,
                guidance=1.0,
                allow_cpu_frozen=allow_cpu_frozen,
            )

        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
        with preview_context(model, additional_models=(adapter,), cuda_devices=(device.index,)):
            expected = sample(False)
        model.cpu()
        placement = FrozenModelPlacement(model, binding.regions, device=device)
        execution = ExecutionInstallation(
            binding.regions,
            ExecutionConfig(residency="frozen-blocks", recompute="block", use_pinned_memory_for_block_swap=True),
            device=device,
        )
        try:
            with preview_context(model, additional_models=(adapter,), cuda_devices=(device.index,)):
                with pytest.raises(ValueError, match="weights"):
                    sample(False)
                actual = sample(True)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert model.training and adapter.training
            assert torch.equal(torch.get_rng_state(), cpu_rng)
            assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
        finally:
            execution.close()
            placement.close()
