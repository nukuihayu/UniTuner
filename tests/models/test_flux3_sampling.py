"""Check FLUX3 flow-map destinations, packed CFG and inference resource isolation."""

import numpy as np
import pytest
import torch
from PIL import Image

from unitunner.config.sampling import SamplingConfig
from unitunner.data.media.image import save_image
from unitunner.inference.sampling import generate_latents
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux3.conditioning import Flux3Sample, image_tokens, pack_samples, reference_tokens, text_tokens
from unitunner.models.flux3.config.model import Flux3Config
from unitunner.models.flux3.inference.sampling import build_schedule, sample_latents
from unitunner.models.flux3.transformer import Flux3Transformer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_flux3_schedule_pins_dtype_and_native_grid():
    schedule = build_schedule(4)
    grid = torch.linspace(1, 0, 5)
    torch.testing.assert_close(schedule, 16.0 * grid / (1 + 15.0 * grid), atol=0, rtol=0)
    torch.testing.assert_close(build_schedule(4, shift=6.93), 6.93 * grid / (1 + 5.93 * grid), atol=0, rtol=0)
    before = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.testing.assert_close(build_schedule(4), schedule, atol=0, rtol=0)
    finally:
        torch.set_default_dtype(before)
    for steps, shift in ((True, 1.0), (0, 1.0), (4, 0.0), (4, float("nan")), (4, 1e-50), (4, 1e30)):
        with pytest.raises(ValueError):
            build_schedule(steps, shift=shift)


@pytest.mark.parametrize("variant", ["base", "distilled"])
def test_flux3_sampler_matches_individual_euler_and_preserves_inputs(variant):
    torch.manual_seed(812)
    model = Flux3Transformer(
        Flux3Config(
            variant=variant,
            hidden_size=32,
            num_heads=2,
            depth=1,
            depth_single=1,
            axes_dim=(4, 4, 4, 4),
            image_channels=8,
            text_channels=12,
        )
    )
    model.txt_mode_blocks[0].eval()
    noise = torch.randn(2, 8, 2, 3)
    positive = (torch.randn(3, 12), torch.randn(5, 12))
    negative = (torch.randn(4, 12), torch.randn(2, 12)) if variant == "base" else None
    refs = ((torch.randn(8, 1, 2), torch.randn(8, 2, 1)), ())
    tensors = (noise, *positive, *(negative or ()), *refs[0])
    before = [t.clone() for t in tensors]
    rng, flags = torch.get_rng_state().clone(), tuple(m.training for m in model.modules())
    result = sample_latents(model, noise, positive, negative=negative, references=refs, steps=3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = sample_latents(model, noise, positive, negative=negative, references=refs, steps=3)
    torch.testing.assert_close(actual, result, atol=0, rtol=0)
    expected = noise.clone()
    schedule = build_schedule(3)
    with torch.no_grad():
        for step in range(3):
            for i in range(2):
                sample = Flux3Sample(image_tokens(expected[i]), text_tokens(positive[i]), reference_tokens(refs[i]))
                packed = pack_samples([sample], schedule[step : step + 1])
                destination = schedule[step + 1 : step + 2] if variant == "distilled" else None
                velocity = model(packed, destination)
                if negative is not None:
                    uncond = model(
                        pack_samples(
                            [Flux3Sample(sample.image, text_tokens(negative[i]), sample.references)],
                            schedule[step : step + 1],
                        )
                    )
                    velocity = uncond + 4.0 * (velocity - uncond)
                expected[i] += (schedule[step + 1] - schedule[step]) * velocity.reshape(2, 3, 8).permute(2, 0, 1)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)
    for old, new in zip(before, tensors, strict=True):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert flags == tuple(m.training for m in model.modules())
    assert not result.requires_grad
    with pytest.raises(ValueError, match=r"CFG|guidance"):
        sample_latents(model, noise, positive, steps=3, guidance=float("nan"))
    with pytest.raises(ValueError, match="FP32 weights"):
        sample_latents(model.to(torch.bfloat16), noise, positive, negative=negative, steps=3)


def test_flux3_public_generation_uses_official_token_noise_rounding_and_truncated_png(tmp_path):
    model = Flux3Transformer(
        Flux3Config(
            variant="distilled",
            hidden_size=32,
            num_heads=2,
            depth=1,
            depth_single=1,
            axes_dim=(4, 4, 4, 4),
            image_channels=8,
            text_channels=12,
        )
    )
    caption = Caption(torch.randn(1, 3, 12))
    sampling = SamplingConfig(width=32, height=16, steps=2, seed=72)
    noise = (
        torch.randn((2, 8), generator=torch.Generator().manual_seed(72), dtype=torch.float32)
        .to(torch.bfloat16)
        .float()
        .T.reshape(1, 8, 1, 2)
    )
    expected = sample_latents(model, noise, (caption.features[0],), steps=2)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        actual = generate_latents(model, caption, sampling, vae_factor=16)
    finally:
        torch.set_default_dtype(previous)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    pixels = torch.tensor([-1.01, -1, -0.999, 0, 0.001, 1, 1.1], dtype=torch.float32)[None, None].expand(3, 1, 7)
    path = tmp_path / "out.png"
    save_image(pixels, path, rounding="truncate")
    with Image.open(path) as image:
        actual = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    torch.testing.assert_close(
        actual,
        torch.tensor([0, 0, 0, 127, 127, 255, 255], dtype=torch.uint8)[None, None].expand(3, 1, 7),
        atol=0,
        rtol=0,
    )
