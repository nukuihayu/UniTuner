"""Verify image-only Qwen2.1 VAE ownership, posterior and spatial execution contracts."""

import numpy as np
import pytest
import torch
from PIL import Image

from tests.support.assertions import invalid_input
from unitunner.artifacts.formats.qwen_image import image21_vae_key_map
from unitunner.artifacts.tensors import save_tensors
from unitunner.cli.main import main
from unitunner.models.qwen_image.vae import QwenImageVAEExecution, QwenImageVAETiling
from unitunner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unitunner.models.qwen_image_2_1.data.encoding import encode_images, image_tensor
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21_vae, save_qwen_image_21_vae
from unitunner.models.qwen_image_2_1.vae import (
    AverageDownsample,
    DuplicateUpsample,
    ImageConv2d,
    QwenImage21VAE,
    fold_patches,
    unfold_patches,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def vae():
    return QwenImage21VAE(
        QwenImage21VAEConfig(
            base_channels=2,
            decoder_base_channels=4,
            latent_channels=2,
            channel_multipliers=(1, 2, 4, 4, 4),
            layers_per_block=1,
            latents_mean=(-0.31, 0.73),
            latents_std=(1.73, 0.91),
        )
    )


def test_strict_native_vae_artifact_and_public_import(vae, tmp_path):
    source, destination, topology = (
        tmp_path / name for name in ("source.safetensors", "native.safetensors", "topology.json")
    )
    rng = torch.get_rng_state().clone()
    save_qwen_image_21_vae(vae, source)
    restored = load_qwen_image_21_vae(source)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in vae.state_dict().items())
    topology.write_text(vae.config.model_dump_json())
    save_tensors(vae.state_dict(), source)
    command = [
        "weights",
        "import",
        str(source),
        str(destination),
        "--family",
        "qwen-image-2-1",
        "--kind",
        "vae",
        "--topology",
        str(topology),
    ]
    before = source.read_bytes()
    assert main(command) == 0
    assert source.read_bytes() == before and load_qwen_image_21_vae(destination).config == vae.config
    accepted = destination.read_bytes()
    tensors = dict(vae.state_dict())
    tensors.pop(next(iter(tensors)))
    save_tensors(tensors, source)
    with pytest.raises(ValueError, match="keys differ"):
        load_qwen_image_21_vae(source, config=vae.config)
    tensors = dict(vae.state_dict())
    tensors["encoder.conv_in.weight"] = torch.full_like(tensors["encoder.conv_in.weight"], float("nan"))
    save_tensors(tensors, source)
    assert main(command) == 2 and destination.read_bytes() == accepted
    save_qwen_image_21_vae(vae, source)
    with pytest.raises(ValueError, match="topology differs"):
        load_qwen_image_21_vae(source, config=vae.config.model_copy(update={"input_channels": 3}))
    pixels, latents = torch.randn(1, 4, 1, 16, 32), torch.randn(1, 2, 1, 1, 2)
    assert torch.equal(vae.encode(pixels), restored.encode(pixels))
    assert torch.equal(vae.decode_raw(latents), restored.decode_raw(latents))


@pytest.mark.parametrize("source_format", ["auto", "diffusers", "comfyui"])
@pytest.mark.parametrize("prefix", ["", "vae."])
def test_released_vae_bf16_temporal_kernels_and_nested_keys(vae, tmp_path, source_format, prefix):
    keys = image21_vae_key_map(tuple(vae.state_dict()), num_blocks=5, layers_per_block=1, is_residual=True)
    assert keys["encoder.down_blocks.1.downsampler.time_conv.weight"] == (
        "encoder.downsamples.1.downsamples.1.time_conv.weight"
    )
    assert keys["decoder.up_blocks.0.upsampler.resample.1.weight"] == (
        "decoder.upsamples.0.upsamples.2.resample.1.weight"
    )
    spatial = {f"{name}.weight" for name, module in vae.named_modules() if isinstance(module, ImageConv2d)}
    tensors = {
        prefix + (keys[name] if source_format != "diffusers" else name): (
            value.unsqueeze(2) if name in spatial else value
        ).bfloat16()
        for name, value in vae.state_dict().items()
    }
    path = tmp_path / "external.safetensors"
    save_tensors(tensors, path)
    rng = torch.get_rng_state().clone()
    restored = load_qwen_image_21_vae(path, config=vae.config, source_format=source_format, prefix=prefix)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in vae.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value.bfloat16().float())
    reference = vae.to(dtype=torch.bfloat16).float()
    pixels = torch.zeros(1, 4, 1, 16, 32)
    with torch.no_grad():
        latent = reference.encode(pixels)
        assert torch.equal(restored.encode(pixels), latent)
        assert torch.equal(restored.decode(latent), reference.decode(latent))
    if source_format == "comfyui" and prefix == "vae.":
        topology, destination = tmp_path / "topology.json", tmp_path / "imported.safetensors"
        topology.write_text(vae.config.model_dump_json())
        assert (
            main(
                [
                    "weights",
                    "import",
                    str(path),
                    str(destination),
                    "--family",
                    "qwen-image-2-1",
                    "--kind",
                    "vae",
                    "--source-format",
                    "comfyui",
                    "--prefix",
                    "vae.",
                    "--topology",
                    str(topology),
                ]
            )
            == 0
        )
        imported = load_qwen_image_21_vae(destination)
        assert torch.equal(imported.encode(pixels), latent)
    key = prefix + (keys["encoder.conv_in.weight"] if source_format != "diffusers" else "encoder.conv_in.weight")
    tensors[key] = tensors[key].expand(-1, -1, 3, -1, -1).contiguous()
    save_tensors(tensors, path)
    with pytest.raises(ValueError, match="invalid tensor shape"):
        load_qwen_image_21_vae(path, config=vae.config, source_format=source_format, prefix=prefix)


def test_image_posterior_gradients_normalization_and_ambient_autocast(vae):
    pixels = torch.randn(2, 4, 1, 16, 32, requires_grad=True)
    noise = torch.randn(2, 2, 1, 1, 2, requires_grad=True)
    modes = tuple(module.training for module in vae.modules())
    rng = torch.get_rng_state().clone()
    posterior = vae.posterior(pixels)
    sampled = vae.encode(pixels, posterior_noise=noise)
    mean = pixels.new_tensor(vae.config.latents_mean).reshape(1, -1, 1, 1, 1)
    std = pixels.new_tensor(vae.config.latents_std).reshape(1, -1, 1, 1, 1)
    assert torch.equal(vae.normalize(posterior.mean), (posterior.mean - mean) / std)
    assert torch.equal(vae.denormalize(sampled), sampled * std + mean)
    assert not torch.equal(sampled, vae.encode(pixels))
    decoded = vae.decode(sampled)
    (decoded.square().mean() + posterior.mean.square().mean() + posterior.log_variance.square().mean()).backward()
    assert pixels.grad is not None and pixels.grad.abs().sum() > 0
    assert noise.grad is not None and noise.grad.abs().sum() > 0
    inactive = [name for name, parameter in vae.named_parameters() if parameter.grad is None]
    assert inactive and all(".time_conv." in name for name in inactive)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for name, parameter in vae.named_parameters()
        if name not in inactive
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(sampled, vae.encode(pixels, posterior_noise=noise))
        assert torch.equal(decoded, vae.decode(sampled))
    assert torch.equal(rng, torch.get_rng_state()) and modes == tuple(module.training for module in vae.modules())
    with pytest.raises(ValueError, match="BC1HW"):
        vae.encode(pixels.expand(-1, -1, 3, -1, -1))
    with pytest.raises(ValueError, match="BC1HW"):
        vae.decode(sampled.expand(-1, -1, 3, -1, -1))
    with torch.no_grad():
        vae.quant_conv.weight.zero_()
        vae.quant_conv.bias.copy_(torch.tensor([0.0, 0.0, 40.0, -40.0]))
    clamped = vae.posterior(pixels)
    assert torch.equal(clamped.log_variance[:, 0], torch.full_like(clamped.log_variance[:, 0], 20))
    assert torch.equal(clamped.log_variance[:, 1], torch.full_like(clamped.log_variance[:, 1], -30))


def test_spatial_slicing_tiling_patch_order_and_final_clamp(vae):
    policy = QwenImageVAEExecution(
        batch_slice_size=1, tiling=QwenImageVAETiling(height=32, width=48, stride_height=16, stride_width=16)
    )
    pixels = torch.randn(2, 4, 1, 48, 80, requires_grad=True)
    rng = torch.get_rng_state().clone()
    tiled = vae.encode(pixels, execution=policy)
    assert tiled.shape == (2, 2, 1, 3, 5)
    assert torch.equal(tiled, vae.encode(pixels, execution=policy))
    whole = vae.encode(pixels)
    assert not torch.equal(tiled, whole)
    assert torch.equal(whole, vae.encode(pixels))
    decoded = vae.decode(tiled, execution=policy)
    decoded.square().mean().backward()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        vae.decoder.conv_out.weight.zero_()
        vae.decoder.conv_out.bias.fill_(1.5)
    assert torch.equal(vae.decode_raw(tiled.detach(), execution=policy), torch.ones_like(pixels))
    with pytest.raises(ValueError, match="spatial_factor"):
        vae.encode(
            pixels,
            execution=QwenImageVAEExecution(
                tiling=QwenImageVAETiling(height=33, width=48, stride_height=16, stride_width=16)
            ),
        )
    original = torch.arange(1 * 3 * 1 * 4 * 6, dtype=torch.float32).reshape(1, 3, 1, 4, 6).requires_grad_()
    packed = fold_patches(original, 2)
    expected = torch.stack([original[:, :, :, dy::2, dx::2] for dx in range(2) for dy in range(2)], dim=2).flatten(1, 2)
    assert torch.equal(packed, expected) and torch.equal(unfold_patches(packed, 2), original)
    unfold_patches(packed, 2).sum().backward()
    assert original.grad is not None
    assert torch.equal(original.grad, torch.ones_like(original))


def test_resampling_shortcuts_and_published_meta_topology():
    values = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(1, 2, 1, 4, 6).requires_grad_()
    observed = AverageDownsample(2, 4, temporal_factor=2, spatial_factor=2)(values)
    cells = (
        values[:, :, :, 0::2, 0::2]
        + values[:, :, :, 1::2, 0::2]
        + values[:, :, :, 0::2, 1::2]
        + values[:, :, :, 1::2, 1::2]
    )
    expected = torch.cat(
        (torch.zeros_like(cells[:, :1]), cells[:, :1] / 4, torch.zeros_like(cells[:, 1:]), cells[:, 1:] / 4), dim=1
    )
    assert torch.equal(observed, expected)
    up_input = torch.randn(1, 4, 1, 2, 3, requires_grad=True)
    up = DuplicateUpsample(4, 2, temporal_factor=2)(up_input)
    expected_up = up_input[:, [1, 3]].repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    assert torch.equal(up, expected_up)
    (observed.sum() + up.sum()).backward()
    assert values.grad is not None and up_input.grad is not None
    with torch.device("meta"):
        model = QwenImage21VAE(QwenImage21VAEConfig())
    assert model.config.spatial_factor == 16 and model.config.latent_channels == 64
    assert model.encoder.conv_in.weight.shape == (96, 4, 3, 3)
    assert model.decoder.conv_in.weight.shape == (1152, 64, 3, 3)
    assert model.decoder.conv_out.weight.shape == (4, 144, 3, 3)
    assert len(model.config.latents_mean) == len(model.config.latents_std) == 64


def test_grouped_media_means_preserve_alpha_order_and_source_pixels(vae):
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    rgb[..., 0] = 173
    rgba = np.zeros((64, 32, 4), dtype=np.uint8)
    rgba[..., 1] = 81
    rgba[..., 3] = 47
    pil = Image.fromarray(np.full((32, 64, 4), 211, dtype=np.uint8))
    original = rgb.copy(), rgba.copy(), pil.tobytes()
    rng = torch.get_rng_state().clone()
    latents = encode_images(vae, [rgb, rgba, pil])
    assert len(latents) == 3 and all(not value.requires_grad for value in latents)
    for image, latent in zip((rgb, rgba, pil), latents, strict=True):
        expected = vae.encode(image_tensor(image, channels=4))[0]
        torch.testing.assert_close(latent, expected, atol=1e-6, rtol=1e-5)
    assert np.array_equal(rgb, original[0]) and np.array_equal(rgba, original[1]) and pil.tobytes() == original[2]
    assert torch.equal(image_tensor(rgb, channels=4)[:, 3], torch.ones(1, 1, 32, 64))
    assert torch.equal(image_tensor(rgba, channels=4)[:, 3], torch.full((1, 1, 64, 32), 47 / 127.5 - 1))
    assert torch.equal(rng, torch.get_rng_state()) and encode_images(vae, []) == []
    with pytest.raises(ValueError, match="uint8 HWC"):
        image_tensor(invalid_input(rgb.astype(np.float32)), channels=4)
    with pytest.raises(ValueError, match="multiples of32"):
        encode_images(vae, [Image.new("RGBA", (31, 32))])
