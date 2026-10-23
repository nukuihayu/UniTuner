"""Verify FLUX native geometry, strict artifacts and heterogeneous reference semantics."""

from typing import Literal

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from unitunner.artifacts.tensors import save_tensors
from unitunner.models.flux.conditioning import (
    FluxCondition,
    build_inputs,
    image_ids,
    pack_fill_mask,
    pack_latents,
    unpack_latents,
)
from unitunner.models.flux.config.model import FluxConfig
from unitunner.models.flux.loading import load_flux, save_flux
from unitunner.models.flux.transformer import FluxTransformer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(variant: Literal["dev", "kontext", "fill"] = "dev"):
    return FluxConfig(
        variant=variant,
        in_channels=48 if variant == "fill" else 8,
        out_channels=8,
        vec_in_dim=6,
        context_in_dim=12,
        hidden_size=32,
        num_heads=2,
        depth=2,
        depth_single_blocks=2,
        axes_dim=(4, 6, 6),
        mlp_ratio=2.0,
    )


def test_flux_packing_mask_order_and_variant_conditions():
    values = torch.arange(2 * 4 * 6).reshape(1, 2, 4, 6).float().requires_grad_()
    packed = pack_latents(values)
    expected = torch.tensor(
        [
            [
                [values[0, c, y + dy, x + dx].item() for c in range(2) for dy in range(2) for dx in range(2)]
                for y in range(0, 4, 2)
                for x in range(0, 6, 2)
            ]
        ]
    )
    torch.testing.assert_close(packed, expected, atol=0, rtol=0)
    torch.testing.assert_close(unpack_latents(packed, 4, 6), values, atol=0, rtol=0)
    packed.sum().backward()
    assert values.grad is not None
    assert torch.equal(values.grad, torch.ones_like(values))
    mask = torch.arange(16 * 32).reshape(1, 1, 16, 32).float() / 511
    mask_tokens = pack_latents(pack_fill_mask(mask))
    expected_mask = torch.tensor(
        [
            [
                [
                    mask[0, 0, y + dy * 8 + py, x + dx * 8 + px].item()
                    for py in range(8)
                    for px in range(8)
                    for dy in range(2)
                    for dx in range(2)
                ]
                for y in range(0, 16, 16)
                for x in range(0, 32, 16)
            ]
        ]
    )
    torch.testing.assert_close(mask_tokens, expected_mask, atol=0, rtol=0)
    ids = image_ids(1, 2, 3, device=torch.device("cpu"))
    torch.testing.assert_close(ids[0, :, 1:], torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]).float())
    with pytest.raises(ValueError, match="Dev"):
        build_inputs("dev", packed, ids, FluxCondition(packed))
    with pytest.raises(ValueError, match="cannot consume"):
        build_inputs("fill", packed, ids, FluxCondition(packed, ids))
    with pytest.raises(ValueError, match="reference lengths"):
        build_inputs("kontext", packed, ids, FluxCondition(packed, ids, (2,)))
    for kwargs in ({"axes_dim": (3, 6, 7)}, {"variant": "fill"}, {"hidden_size": 33}):
        with pytest.raises(ValueError):
            config().model_validate({**config().model_dump(), **kwargs})


def test_flux_native_reload_preserves_rng_and_rejects_incomplete_or_unsafe_weights(tmp_path):
    model = FluxTransformer(config())
    path = tmp_path / "flux.safetensors"
    rng = torch.get_rng_state().clone()
    save_flux(model, path)
    loaded = load_flux(path)
    assert loaded.config == model.config
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="differs"):
        load_flux(path, config=config("kontext"))
    state = dict(model.state_dict())
    missing = dict(state)
    missing.pop("img_in.bias")
    save_tensors(missing, path)
    with pytest.raises(ValueError, match="keys differ"):
        load_flux(path, config=config())
    state["img_in.bias"] = torch.full_like(state["img_in.bias"], 1e8)
    save_tensors(state, path)
    with pytest.raises(ValueError, match="dtype range"):
        load_flux(path, config=config(), dtype=torch.float16)
    state["img_in.bias"] = torch.full_like(state["img_in.bias"], float("nan"))
    save_tensors(state, path)
    with pytest.raises(ValueError, match="values invalid"):
        load_flux(path, config=config())


def test_flux_heterogeneous_kontext_matches_individual_reference_prefixes():
    torch.manual_seed(74)
    model = FluxTransformer(config("kontext"))
    img = torch.randn(2, 6, 8, requires_grad=True)
    txt = torch.randn(2, 3, 12, requires_grad=True)
    ids = torch.randn(2, 6, 3)
    txt_ids = torch.zeros(2, 3, 3)
    times, pooled, guidance = torch.tensor([0.3, 0.7]), torch.randn(2, 6), torch.tensor([2.5, 3.5])
    with sdpa_kernel(SDPBackend.MATH):
        batched = model(img, ids, txt, txt_ids, times, pooled, guidance, control_lengths=(0, 2))[:, :4]
        individual = torch.cat(
            [
                model(
                    img[i : i + 1, :length],
                    ids[i : i + 1, :length],
                    txt[i : i + 1],
                    txt_ids[i : i + 1],
                    times[i : i + 1],
                    pooled[i : i + 1],
                    guidance[i : i + 1],
                )[:, :4]
                for i, length in enumerate((4, 6))
            ]
        )
        torch.testing.assert_close(batched, individual, atol=1e-6, rtol=1e-5)
        variables = (img, txt, *model.parameters())
        left = torch.autograd.grad(batched.square().mean(), variables, retain_graph=True)
        right = torch.autograd.grad(individual.square().mean(), variables)
    for actual, expected in zip(left, right, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert not left[0][0, 4:].count_nonzero()
