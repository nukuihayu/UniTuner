"""Compare model boundaries with the upstream low-precision timestep pipelines."""

import math

import pytest
import torch

from unituner.models.components.conditioning import Caption
from unituner.models.flux3.conditioning import Flux3Sample, image_tokens, pack_samples, text_tokens
from unituner.models.flux3.config.model import Flux3Config
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.models.qwen_image.config.model import QwenImageConfig
from unituner.models.qwen_image.transformer import QwenImageTransformer
from unituner.models.qwen_image_2_1.config.model import QwenImage21Config
from unituner.models.qwen_image_2_1.restoration import restore_latents
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.training.integrations.qwen_image_2_1.binding import QwenImage21Binding, QwenImage21FlowRequest

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("variant", ["original", "edit-2511", "layered"])
@pytest.mark.parametrize("native_time", [0.0, 127.25, 503.5, 817.75, 1000.0])
def test_qwen_native_clock_rounding_matches_diffusers_pipeline(device, dtype, variant, native_time):
    model = QwenImageTransformer(
        QwenImageConfig(
            variant=variant,
            in_channels=8,
            out_channels=2,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=12,
            axes_dims_rope=(4, 6, 6),
        )
    ).to(device)
    clock = torch.tensor([native_time], device=device)
    sigma = (clock / 1000).requires_grad_()
    times = []
    features = []

    def capture_time(_module, args):
        times.append(args[0].detach().clone())

    def capture_features(_module, args):
        features.append(args[0].detach().clone())

    with (
        model.time_text_embed.register_forward_pre_hook(capture_time),
        model.time_text_embed.timestep_embedder.register_forward_pre_hook(capture_features),
        torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32),
    ):
        output = model(
            torch.randn(1, 2, 8, device=device),
            torch.randn(1, 2, 12, device=device),
            sigma,
            ((1, 1, 2),),
            layer_condition=torch.zeros(1, device=device, dtype=torch.int64) if variant == "layered" else None,
        )
    expected = clock.to(dtype) / 1000
    if model.config.zero_reference_time:
        expected = torch.cat((expected, expected.new_zeros(1)))
    torch.testing.assert_close(times[0], expected, atol=0, rtol=0)
    exponent = -math.log(10000) * torch.arange(128, device=device, dtype=torch.float32)
    phase = 1000 * (expected[:, None].float() * torch.exp(exponent / 128)[None])
    torch.testing.assert_close(features[0], torch.cat((phase.cos(), phase.sin()), -1).to(dtype), atol=0, rtol=0)
    output.float().square().mean().backward()
    assert sigma.grad is not None and torch.isfinite(sigma.grad).all() and sigma.grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("entry", ["model", "flow", "explicit-flow", "restoration"])
@pytest.mark.parametrize("native_time", [127.25, 503.5, 817.75])
def test_qwen21_native_clock_rounding_before_normalization(device, dtype, entry, native_time):
    model = QwenImage21Transformer(
        QwenImage21Config(
            in_channels=4,
            out_channels=4,
            context_in_dim=6,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            axes_dims_rope=(2, 2, 4),
        )
    ).to(device)
    latent = torch.randn(1, 4, 1, 1, 2, device=device)
    text = torch.randn(1, 2, 6, device=device)
    clock = torch.tensor([native_time], device=device)
    sigma = (clock / 1000).requires_grad_()
    native = clock.to(dtype)
    expected_time = torch.cat(((native / 1000).to(dtype), native.new_zeros(1)))
    observed = {}

    def capture_time(_module, args):
        observed["time"] = args[0].detach().clone()

    def capture_features(_module, args):
        observed["features"] = args[0].detach().clone()

    with (
        model.time_text_embed.register_forward_pre_hook(capture_time),
        model.time_text_embed.timestep_embedder.register_forward_pre_hook(capture_features),
        torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32),
    ):
        if entry == "restoration":
            result = restore_latents(model, latent, Caption(text), sigma=native_time / 1000)
        elif entry in ("flow", "explicit-flow"):
            result = (
                QwenImage21Binding(model)
                .predict(
                    QwenImage21FlowRequest(
                        latent,
                        torch.randn_like(latent),
                        sigma,
                        (text[0],),
                        timesteps=sigma * 1000 if entry == "explicit-flow" else None,
                    )
                )
                .prediction
            )
        else:
            result = model(latent[:, :, 0].flatten(2).transpose(1, 2), text, sigma, (((1, 1, 2),),))
    torch.testing.assert_close(observed["time"], expected_time, atol=0, rtol=0)
    frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128).to(device)
    phase = 1000 * expected_time.float()[:, None] * frequencies
    expected_features = torch.cat((phase.cos(), phase.sin()), dim=-1).to(dtype)
    torch.testing.assert_close(observed["features"], expected_features, atol=0, rtol=0)
    result.float().square().mean().backward()
    assert model.time_text_embed.timestep_embedder.linear_1.weight.grad is not None
    if entry != "restoration":
        assert sigma.grad is not None and torch.isfinite(sigma.grad).all() and sigma.grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("variant", ["base", "distilled"])
@pytest.mark.parametrize("source_time,target_time", [(0.173, 0.0625), (0.507, 0.318), (0.829, 0.703)])
def test_flux3_rounds_source_and_destination_fractions_before_scaling(device, dtype, variant, source_time, target_time):
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
            mlp_ratio=2.0,
        )
    ).to(device)
    sigma = torch.tensor([source_time], device=device, requires_grad=True)
    destination = torch.tensor([target_time], device=device, requires_grad=True) if variant == "distilled" else None
    packed = pack_samples(
        [
            Flux3Sample(
                image_tokens(torch.randn(8, 1, 2, device=device)), text_tokens(torch.randn(2, 12, device=device))
            )
        ],
        sigma,
    )
    observed = []
    destinations = []

    def capture(_module, args):
        observed.append(args[0].detach().clone())

    def capture_destination(_module, args):
        destinations.append(args[0].detach().clone())

    handle = model.time_in.register_forward_pre_hook(capture)
    ref_handle = None if model.ref_time_in is None else model.ref_time_in.register_forward_pre_hook(capture_destination)
    try:
        with torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32):
            result = model(packed, destination)
    finally:
        handle.remove()
        if ref_handle is not None:
            ref_handle.remove()
    frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32, device=device) / 128)

    def reference(time):
        phase = (time.to(dtype) * 1000).float()[:, None] * frequencies
        return torch.cat((phase.cos(), phase.sin()), dim=-1).to(dtype)

    torch.testing.assert_close(observed[0], reference(sigma).expand(2, -1), atol=0, rtol=0)
    for features in observed[1:]:
        torch.testing.assert_close(features, reference(sigma.new_zeros(features.shape[0])), atol=0, rtol=0)
    if destination is not None:
        torch.testing.assert_close(destinations[0], reference(destination), atol=0, rtol=0)
    assert packed.timesteps["image"].dtype == torch.float32
    torch.testing.assert_close(packed.timesteps["image"], sigma.detach().expand(2), atol=0, rtol=0)
    result.float().square().mean().backward()
    assert sigma.grad is not None and torch.isfinite(sigma.grad).all() and sigma.grad.abs().sum() > 0
    if destination is not None:
        assert (
            destination.grad is not None and torch.isfinite(destination.grad).all() and destination.grad.abs().sum() > 0
        )
