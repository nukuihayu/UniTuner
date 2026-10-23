"""Verify FLUX native grids, guidance semantics and conditioned sampling isolation."""

import pytest
import torch

from unitunner.models.flux.conditioning import FluxCondition, image_ids
from unitunner.models.flux.config.model import FluxConfig
from unitunner.models.flux.inference.sampling import build_schedule, sample_latents
from unitunner.models.flux.transformer import FluxTransformer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_flux_schedule_preserves_native_grid_and_rejects_nonfinite_resolution_shift():
    original = torch.get_default_dtype()
    baseline = build_schedule(4, 6)
    try:
        torch.set_default_dtype(torch.float64)
        assert build_schedule(4, 6) == baseline
    finally:
        torch.set_default_dtype(original)
    assert build_schedule(4, 6, shift=1.0) == (1.0, 0.75, 0.5, 0.25, 0.0)
    for steps, tokens, shift in (
        (True, 6, None),
        (4, 0, None),
        (4, 10**9, None),
        (4, 6, float("nan")),
        (4, 6, 0.0),
        (4, 6, 1e-50),
    ):
        with pytest.raises(ValueError):
            build_schedule(steps, tokens, shift=shift)


@pytest.mark.parametrize("variant", ["dev", "kontext", "fill"])
def test_flux_sampling_preserves_noise_conditions_modes_and_rng_under_autocast(variant):
    torch.manual_seed(291)
    model = FluxTransformer(
        FluxConfig(
            variant=variant,
            in_channels=48 if variant == "fill" else 8,
            out_channels=8,
            vec_in_dim=6,
            context_in_dim=12,
            hidden_size=32,
            num_heads=2,
            depth=1,
            depth_single_blocks=1,
            axes_dim=(4, 6, 6),
            mlp_ratio=2.0,
        )
    )
    model.double_blocks[0].eval()
    noise = torch.randn(2, 2, 4, 6)
    captions, pooled = torch.randn(2, 5, 12), torch.randn(2, 6)
    condition = None
    if variant == "kontext":
        condition = FluxCondition(
            torch.randn(2, 2, 8), image_ids(2, 1, 2, device=torch.device("cpu"), reference=True), (0, 2)
        )
    if variant == "fill":
        condition = FluxCondition(torch.randn(2, 6, 8), fill_mask=torch.rand(2, 6, 32))
    before = noise.clone()
    before_condition = None if condition is None else condition.tokens.clone()
    rng = torch.get_rng_state().clone()
    flags = tuple(module.training for module in model.modules())
    output = sample_latents(model, noise, captions, pooled, condition=condition, steps=3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = sample_latents(model, noise, captions, pooled, condition=condition, steps=3)
    torch.testing.assert_close(actual, output, atol=0, rtol=0)
    torch.testing.assert_close(noise, before, atol=0, rtol=0)
    if condition is not None:
        torch.testing.assert_close(condition.tokens, before_condition, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert tuple(module.training for module in model.modules()) == flags
    assert output.shape == noise.shape and not output.requires_grad
    if variant == "kontext":
        first = sample_latents(model, noise[:1], captions[:1], pooled[:1], steps=3)
        assert condition is not None and condition.ids is not None
        second = sample_latents(
            model,
            noise[1:],
            captions[1:],
            pooled[1:],
            condition=FluxCondition(condition.tokens[1:], condition.ids[1:]),
            steps=3,
        )
        torch.testing.assert_close(output, torch.cat((first, second)), atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="guidance"):
        sample_latents(model, noise, captions, pooled, condition=condition, steps=3, guidance=float("inf"))
