"""Check FLUX.2 native token geometry and sampler resource/precision ownership."""

import pytest
import torch

from unitunner.config.sampling import SamplingConfig
from unitunner.inference.sampling import generate_latents
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux2.conditioning import build_image_inputs, pack_image, pack_text, unpack_image
from unitunner.models.flux2.config.model import Flux2Config
from unitunner.models.flux2.inference.sampling import build_schedule, sample_latents
from unitunner.models.flux2.transformer import Flux2Transformer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_flux2_dense_packing_reference_order_and_explicit_geometry():
    latent = torch.arange(24).reshape(2, 3, 4).float().requires_grad_()
    tokens, ids = pack_image(latent)
    expected = torch.tensor([[latent[c, y, x].item() for c in range(2)] for y in range(3) for x in range(4)])
    assert torch.equal(tokens, expected)
    assert ids.tolist() == [[0, y, x, 0] for y in range(3) for x in range(4)]
    refs = (torch.randn(2, 1, 2), torch.randn(2, 2, 1))
    image = build_image_inputs(latent, refs)
    assert image.target_count == 12
    assert image.ids[0, 12:].tolist() == [[10, 0, 0, 0], [10, 0, 1, 0], [20, 0, 0, 0], [20, 1, 0, 0]]
    assert torch.equal(unpack_image(tokens, height=3, width=4), latent)
    tokens.sum().backward()
    assert latent.grad is not None
    assert torch.equal(latent.grad, torch.ones_like(latent))
    text = torch.randn(5, 12)
    context, positions = pack_text(text)
    assert context is text
    assert positions.tolist() == [[0, 0, 0, index] for index in range(5)]
    with pytest.raises(ValueError, match="geometry"):
        unpack_image(tokens, height=2, width=4)
    with pytest.raises(ValueError, match="channels"):
        build_image_inputs(latent, (torch.randn(3, 1, 1),))
    expected = build_schedule(4, 4096)
    prior = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert build_schedule(4, 4096) == expected
    finally:
        torch.set_default_dtype(prior)
    for shift in (0.0, float("nan"), 1e30):
        with pytest.raises(ValueError):
            build_schedule(4, 4096, shift=shift)


@pytest.mark.parametrize("version,variant", [("dev", "distilled"), ("klein-4b", "base"), ("klein-9b", "distilled")])
def test_flux2_sampler_matches_individual_rows_and_preserves_resources(version, variant):
    model = Flux2Transformer(
        Flux2Config(
            version=version,
            variant=variant,
            in_channels=8,
            context_in_dim=12,
            hidden_size=32,
            num_heads=2,
            depth=1,
            depth_single_blocks=1,
            axes_dim=(4, 4, 4, 4),
        )
    )
    model.double_blocks[0].eval()
    modes = tuple(module.training for module in model.modules())
    noise = torch.randn(2, 8, 2, 3)
    saved = noise.clone()
    captions = (torch.randn(3, 12), torch.randn(5, 12))
    negative = (torch.randn(2, 12), torch.randn(4, 12)) if variant == "base" else None
    references = ((torch.randn(8, 1, 2), torch.randn(8, 2, 1)), ())
    rng = torch.get_rng_state().clone()
    actual = sample_latents(model, noise, captions, negative=negative, references=references, steps=3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = sample_latents(model, noise, captions, negative=negative, references=references, steps=3)
    assert torch.equal(actual, mixed)
    for index in range(2):
        expected = sample_latents(
            model,
            noise[index : index + 1],
            captions[index : index + 1],
            negative=None if negative is None else negative[index : index + 1],
            references=references[index : index + 1],
            steps=3,
        )
        assert torch.equal(actual[index], expected[0])
    assert not torch.equal(actual, noise)
    assert torch.equal(noise, saved)
    assert torch.equal(torch.get_rng_state(), rng)
    assert tuple(module.training for module in model.modules()) == modes
    assert all(p.grad is None for p in model.parameters())
    if variant == "base":
        with pytest.raises(ValueError, match="negative"):
            sample_latents(model, noise, captions, steps=3)
    elif version != "dev":
        with pytest.raises(ValueError, match="guidance1"):
            sample_latents(model, noise, captions, steps=3, guidance=4.0)


def test_flux2_public_generation_preserves_native_bchw_noise_rounding_and_default_dtype():
    model = Flux2Transformer(
        Flux2Config(
            in_channels=8,
            context_in_dim=12,
            hidden_size=32,
            num_heads=2,
            depth=1,
            depth_single_blocks=1,
            axes_dim=(4, 4, 4, 4),
        )
    )
    caption = Caption(torch.randn(1, 5, 12))
    sampling = SamplingConfig(width=32, height=16, steps=2, seed=31)
    generator = torch.Generator(device="cpu").manual_seed(31)
    expected_noise = (
        torch.randn((1, 8, 1, 2), generator=generator, dtype=torch.float32, device="cpu").to(torch.bfloat16).float()
    )
    expected = sample_latents(model, expected_noise, (caption.features[0],), steps=2)
    state = torch.get_rng_state().clone()
    actual = generate_latents(model, caption, sampling)
    assert torch.equal(actual, expected)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert torch.equal(generate_latents(model, caption, sampling), expected)
    finally:
        torch.set_default_dtype(previous)
    assert torch.equal(torch.get_rng_state(), state)
