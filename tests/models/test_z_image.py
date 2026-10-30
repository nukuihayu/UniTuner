"""Verify native Z-Image math, artifacts and real optimizer/resource integration."""

from dataclasses import asdict

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from unituner.adapters.config import AdapterConfig
from unituner.artifacts.tensors import save_tensors
from unituner.checkpoints.restore import restore_checkpoint
from unituner.checkpoints.saving import save_checkpoint
from unituner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unituner.execution.compiler import CompilerInstallation
from unituner.execution.recompute import RecomputeInstallation
from unituner.models.components.conditioning import batch_captions
from unituner.models.z_image.config.model import ZImageConfig
from unituner.models.z_image.config.vae import ZImageVAEConfig
from unituner.models.z_image.inference.sampling import build_schedule, sample_latents
from unituner.models.z_image.layers import ZImageAttention, apply_rotary_embedding
from unituner.models.z_image.loading import load_z_image, load_z_image_vae, save_z_image, save_z_image_vae
from unituner.models.z_image.transformer import ZImageTransformer
from unituner.models.z_image.vae import SpatialAttention, ZImageVAE
from unituner.training.contracts import ImageFlowRequest, RunState
from unituner.training.engine import update_phase
from unituner.training.integrations.z_image.binding import ZImageBinding
from unituner.training.parameters import ParameterInstallation
from unituner.training.session import TrainingSession


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build_model():
    return ZImageTransformer(
        ZImageConfig(
            in_channels=2,
            dim=24,
            num_layers=2,
            num_refiner_layers=1,
            num_heads=4,
            num_kv_heads=2,
            caption_dim=8,
            timestep_hidden_dim=32,
            axes_dims=(2, 2, 2),
            axes_lengths=(128, 32, 32),
        )
    )


def request(batch=2):
    return ImageFlowRequest(
        torch.randn(batch, 2, 4, 6),
        torch.randn(batch, 2, 4, 6),
        torch.linspace(0.2, 0.8, batch),
        torch.randn(batch, 35, 8),
        torch.arange(35)[None] < torch.arange(1, batch + 1)[:, None] * 17,
    )


def test_grouped_attention_matches_independent_softmax_and_gradients():
    torch.manual_seed(8)
    attention = ZImageAttention(build_model().config)
    x = torch.randn(2, 5, 24, requires_grad=True)
    frequencies = torch.polar(torch.ones(2, 5, 3), torch.randn(2, 5, 3))
    mask = torch.tensor([[True, True, False, False, False], [True, False, True, True, False]])
    with sdpa_kernel(SDPBackend.MATH):
        actual = attention(x, frequencies, mask)
    assert attention.norm_q is not None and attention.norm_k is not None
    q = attention.norm_q(attention.to_q(x).reshape(2, 5, 4, 6))
    k = attention.norm_k(attention.to_k(x).reshape(2, 5, 2, 6))
    q = apply_rotary_embedding(q, frequencies).transpose(1, 2)
    k = apply_rotary_embedding(k, frequencies).repeat_interleave(2, dim=2).transpose(1, 2)
    v = attention.to_v(x).reshape(2, 5, 2, 6).repeat_interleave(2, dim=2).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / (6**0.5)
    values = (scores.masked_fill(~mask[:, None, None], float("-inf")).softmax(-1) @ v).transpose(1, 2).flatten(2)
    expected = attention.to_out[0](values)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    parameters = (x, *attention.parameters())
    actual_gradients = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-5)


def test_native_artifact_exact_reload_and_reject_partial(tmp_path):
    model = build_model()
    binding = ZImageBinding(model)
    inputs = request()
    expected = binding.predict(inputs).prediction
    path = tmp_path / "model.safetensors"
    save_z_image(model, path)
    loaded = load_z_image(path)
    assert loaded.config == model.config
    torch.testing.assert_close(ZImageBinding(loaded).predict(inputs).prediction, expected, atol=0, rtol=0)
    state = model.state_dict()
    state.pop("cap_pad_token")
    save_tensors(state, path)
    with pytest.raises(ValueError, match="missing"):
        load_z_image(path, config=model.config)


def test_sparse_caption_and_empty_mask_have_finite_backward():
    model = build_model()
    inputs = request()
    mask = torch.zeros(2, 35, dtype=torch.bool)
    mask[0, 34] = True
    prediction = model(inputs.latents.unsqueeze(2), 1 - inputs.sigma, inputs.captions, mask)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    # Late real text must influence the prediction even when earlier positions are holes.
    captions = inputs.captions.clone()
    captions[0, 34] += torch.arange(8)
    changed = model(inputs.latents.unsqueeze(2), 1 - inputs.sigma, captions, mask)
    assert not torch.equal(prediction[0], changed[0])
    torch.testing.assert_close(prediction[1], changed[1], atol=0, rtol=0)


@pytest.mark.parametrize("algorithm", [None, "lora", "loha", "lokr"])
def test_real_full_and_adapter_recompute_compile_updates(algorithm):
    torch.manual_seed(9)
    eager = build_model()
    optimized = build_model()
    optimized.load_state_dict(eager.state_dict(), strict=True)
    original_ids = [id(parameter) for parameter in optimized.parameters()]
    original_keys = tuple(optimized.state_dict())
    eager_binding, optimized_binding = ZImageBinding(eager), ZImageBinding(optimized)
    config = None if algorithm is None else AdapterConfig(algorithm=algorithm, rank=2, alpha=3.0)
    inputs = request()
    with TrainingSession() as session:
        baseline = session.own(
            ParameterInstallation(
                eager, adapter=config, targets=None if config is None else eager_binding.adapter_targets()
            )
        )
        installed = session.own(
            ParameterInstallation(
                optimized, adapter=config, targets=None if config is None else optimized_binding.adapter_targets()
            )
        )
        if config:
            # All factors are nonzero, so every active adapter factor is exercised.
            with torch.no_grad():
                for parameter in baseline.branches.parameters():
                    parameter.normal_(std=0.1)
            installed.branches.load_state_dict(baseline.branches.state_dict(), strict=True)
        for region in optimized_binding.regions.regions:
            session.own(RecomputeInstallation(region))
            session.own(CompilerInstallation(region.module, backend="eager"))
        first = torch.optim.AdamW(baseline.plan.parameters, lr=0.001)
        second = torch.optim.AdamW(installed.plan.parameters, lr=0.001)
        state, optimized_state = RunState(), RunState()
        for _ in range(3):
            update_phase([eager_binding.step(inputs)], first, state)
            update_phase([optimized_binding.step(inputs)], second, optimized_state)
        for actual, expected in zip(installed.plan.parameters, baseline.plan.parameters, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        assert state == optimized_state == RunState(updates=3, microbatches=3, samples=6)
    assert original_ids == [id(parameter) for parameter in optimized.parameters()]
    assert original_keys == tuple(optimized.state_dict())
    assert all(parameter.requires_grad for parameter in optimized.parameters())


@pytest.mark.parametrize("adapter", [False, True])
def test_native_training_resume_is_exact(tmp_path, adapter):
    identity = ResumeIdentity(config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64)
    torch.manual_seed(7)
    model, fresh = build_model(), build_model()
    binding, fresh_binding = ZImageBinding(model), ZImageBinding(fresh)
    config = AdapterConfig(rank=2, alpha=3.0) if adapter else None
    with TrainingSession() as session:
        installation = session.own(
            ParameterInstallation(model, adapter=config, targets=binding.adapter_targets() if adapter else None)
        )
        restored = session.own(
            ParameterInstallation(fresh, adapter=config, targets=fresh_binding.adapter_targets() if adapter else None)
        )
        optimizer = torch.optim.AdamW(installation.plan.parameters, lr=0.001)
        fresh_optimizer = torch.optim.AdamW(restored.plan.parameters, lr=0.001)
        modules = {"model": model, "adapter": installation.branches}
        fresh_modules = {"model": fresh, "adapter": restored.branches}
        state = RunState()
        update_phase([binding.step(request())], optimizer, state)
        path = tmp_path / "checkpoint.safetensors"
        save_checkpoint(path, modules, optimizer, CheckpointCursor(**asdict(state)), identity)
        for _ in range(2):
            update_phase([binding.step(request())], optimizer, state)
        cursor = restore_checkpoint(path, fresh_modules, fresh_optimizer, identity)
        restored_state = RunState(**cursor.model_dump())
        for _ in range(2):
            update_phase([fresh_binding.step(request())], fresh_optimizer, restored_state)
        assert state == restored_state
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, fresh_modules[name].state_dict()[key], atol=0, rtol=0)


def test_vae_attention_preserves_single_head_outputs_and_gradients(monkeypatch):
    attention = SpatialAttention(16, 4).double()
    pixels = torch.randn(2, 16, 3, 5, dtype=torch.float64, requires_grad=True)
    tokens = attention.group_norm(pixels).flatten(2).transpose(1, 2)
    query, key, value = [projection(tokens) for projection in (attention.to_q, attention.to_k, attention.to_v)]
    probabilities = (query @ key.transpose(-1, -2) / 4).softmax(-1)
    expected = pixels + attention.to_out[0](probabilities @ value).transpose(1, 2).reshape_as(pixels)
    original = torch.nn.functional.scaled_dot_product_attention

    def explicit_head_sdpa(query, key, value, **kwargs):
        assert query.shape == key.shape == value.shape == (2, 1, 15, 16)
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", explicit_head_sdpa)
    actual = attention(pixels)
    torch.testing.assert_close(actual, expected)
    targets = (pixels, *attention.parameters())
    upstream = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(expected, targets, upstream)
    actual_gradients = torch.autograd.grad(actual, targets, upstream)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_vae_explicit_posterior_and_artifact_roundtrip(tmp_path):
    vae = ZImageVAE(ZImageVAEConfig(block_channels=(8, 16), norm_groups=4, layers_per_block=1, latent_channels=2))
    pixels = torch.randn(2, 3, 8, 12, requires_grad=True)
    posterior = vae.encode(pixels)
    noise = torch.randn_like(posterior.mean)
    torch.testing.assert_close(
        posterior.sample(noise), posterior.mean + (0.5 * posterior.log_variance).exp() * noise, atol=0, rtol=0
    )
    decoded = vae.decode(posterior.mean)
    assert decoded.shape == pixels.shape
    decoded.square().mean().backward()
    assert pixels.grad is not None
    assert torch.isfinite(pixels.grad).all()
    torch.testing.assert_close(vae.denormalize(vae.normalize(posterior.mean)), posterior.mean, atol=1e-6, rtol=1e-5)
    path = tmp_path / "vae.safetensors"
    save_z_image_vae(vae, path)
    loaded = load_z_image_vae(path)
    assert loaded.config == vae.config
    torch.testing.assert_close(loaded.decode(posterior.mean), decoded, atol=0, rtol=0)


def test_caption_batch_and_sampling_preserve_inputs_and_rng():
    model = build_model()
    rows = [torch.randn(3, 8), torch.randn(35, 8)]
    caption = batch_captions(rows)
    assert caption.mask is not None
    assert caption.mask.sum(1).tolist() == [3, 35]
    noise = torch.randn(2, 2, 4, 6)
    before = noise.clone()
    rng = torch.get_rng_state().clone()
    first = sample_latents(model, noise, caption, steps=3)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    torch.testing.assert_close(noise, before, atol=0, rtol=0)
    second = sample_latents(model, noise, caption, steps=3)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    schedule = build_schedule(3).sigmas
    assert schedule[0] == 1 and schedule[-1] == 0
    assert torch.all(schedule[1:] < schedule[:-1])
