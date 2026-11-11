"""Small native image model tests; no downloaded weights needed."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.image import (
    ImageFinalLayer,
    MiniMaxH3ImageTransformer,
    load_image_transformer,
    sample_image,
    save_image_transformer,
)
from unituner.models.minimax_h3.image_inference import encode_image_condition, generate_image
from unituner.models.minimax_h3.layout import MiniMaxH3VideoGeometry
from unituner.models.minimax_h3.packing import _unpack_video_rows


def _config() -> MiniMaxH3Config:
    return MiniMaxH3Config(
        hidden_size=24,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=12,
        ffn_hidden_size=32,
        text_dim=8,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        rope_inv_freq_len=2,
    )


def test_image_model_forward_and_reference() -> None:
    model = MiniMaxH3ImageTransformer(_config(), pdd_sigmas=(1.0, 0.75, 0.0), dtype=torch.float32)
    assert not any("audio" in key for key in model.state_dict())
    noise = torch.randn(1, 24, 1, 4, 6)
    text = torch.randn(1, 3, 8)
    tags = torch.tensor([[1, 0, 1]])
    reference = torch.randn(1, 24, 1, 2, 4)
    prediction = model.predict(noise, text, tags, sigma=1.0, sigma_next=0.75, references=(reference,))
    assert prediction.shape == noise.shape
    assert torch.isfinite(prediction).all()
    rng = torch.random.get_rng_state()
    repeat = model.predict(noise, text, tags, sigma=1.0, sigma_next=0.75, references=(reference,))
    torch.testing.assert_close(prediction, repeat)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    result = sample_image(model, noise, text, tags, references=(reference,))
    assert result.shape == noise.shape
    assert torch.isfinite(result).all()


def test_image_projection_restores_patch_order() -> None:
    model = MiniMaxH3ImageTransformer(_config(), dtype=torch.float32)
    with torch.no_grad():
        model.final_layer.video_out.weight.zero_()
        model.final_layer.video_out.bias.copy_(torch.arange(96))
    noise = torch.zeros(1, 24, 1, 4, 6)
    actual = model.predict(noise, torch.zeros(1, 1, 8), torch.ones(1, 1, dtype=torch.long), sigma=1.0, sigma_next=0.0)
    rows = torch.arange(96, dtype=torch.float32).repeat(1, 6, 1)
    expected = _unpack_video_rows(rows, MiniMaxH3VideoGeometry(1, 4, 6))
    torch.testing.assert_close(actual, -expected)


def test_pdd_head_weights() -> None:
    layer = ImageFinalLayer(_config(), 3, dtype=torch.float32, device=None)
    with torch.no_grad():
        layer.norm.weight.fill_(1)
        layer.adaln_proj.linear.weight.zero_()
        layer.adaln_proj.linear.bias.zero_()
        layer.video_out.weight.zero_()
        layer.video_out.bias.copy_(torch.tensor([2.0, 3.0, 5.0]).repeat_interleave(96))
    hidden = torch.ones(1, 2, 24)
    times = torch.zeros(1, 8)
    result = layer(hidden, times, 0, 2, (1.0, 0.75, 0.25, 0.0))
    assert torch.allclose(result, torch.full_like(result, 2.0 + 3.0 * (2.0 / 3.0)))


def test_continuous_training_sigma_uses_its_pdd_interval() -> None:
    model = MiniMaxH3ImageTransformer(_config(), pdd_sigmas=(1.0, 0.75, 0.5, 0.0), dtype=torch.float32)
    with torch.no_grad():
        model.final_layer.video_out.weight.zero_()
        model.final_layer.video_out.bias.copy_(torch.tensor([2.0, 3.0, 5.0]).repeat_interleave(96))
    noise = torch.zeros(1, 24, 1, 2, 2)
    text = torch.zeros(1, 1, 8)
    tags = torch.ones(1, 1, dtype=torch.long)
    prediction = model.predict(noise, text, tags, sigma=0.8, sigma_next=0.75)
    torch.testing.assert_close(prediction, torch.full_like(prediction, -2.0))
    at_grid = model.predict(noise, text, tags, sigma=0.75, sigma_next=0.5)
    torch.testing.assert_close(at_grid, torch.full_like(at_grid, -5.0))


def test_image_checkpoint_roundtrip(tmp_path) -> None:
    model = MiniMaxH3ImageTransformer(_config(), pdd_sigmas=(1.0, 0.0), dtype=torch.float32)
    path = tmp_path / "image.safetensors"
    schedule = tmp_path / "pdd_config.json"
    schedule.write_text(json.dumps({"num_inference_steps": 1, "sigmas": [1.0, 0.0]}))
    save_image_transformer(model, path)
    loaded = load_image_transformer(path, config=_config(), pdd_config=schedule, dtype=torch.float32)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])


def test_official_fused_pdd_heads(tmp_path) -> None:
    model = MiniMaxH3ImageTransformer(_config(), pdd_sigmas=(1.0, 0.5, 0.0), dtype=torch.float32)
    path = tmp_path / "image.safetensors"
    heads = tmp_path / "pdd_fused_heads.safetensors"
    schedule = tmp_path / "pdd_config.json"
    save_image_transformer(model, path)
    save_file(
        {
            "video_out.weight": torch.zeros(2, 96, 24),
            "video_out.bias": torch.ones(2, 96),
        },
        heads,
    )
    schedule.write_text(json.dumps({"num_inference_steps": 2, "sigmas": [1.0, 0.5, 0.0]}))
    loaded = load_image_transformer(
        path,
        config=_config(),
        pdd_config=schedule,
        pdd_heads=heads,
        dtype=torch.float32,
    )
    assert loaded.fused_pdd_heads
    assert loaded.final_layer.video_out.weight.shape == (192, 24)
    assert torch.all(loaded.final_layer.video_out.bias == 1)
    native = tmp_path / "native.safetensors"
    save_image_transformer(loaded, native)
    restored = load_image_transformer(native, config=_config(), pdd_config=schedule, dtype=torch.float32)
    assert restored.fused_pdd_heads
    torch.testing.assert_close(restored.final_layer.video_out.bias, loaded.final_layer.video_out.bias)


def test_generate_image_entrypoint() -> None:
    model = MiniMaxH3ImageTransformer(_config(), dtype=torch.float32)

    class SmallVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def decode(self, latents):
            return latents[:, :3].repeat_interleave(16, -2).repeat_interleave(16, -1)

    class SmallTextEncoder:
        def encode(self, presentation):
            assert presentation.text == "test prompt"
            return SimpleNamespace(hidden_states=torch.zeros(1, 8), token_tags=torch.ones(1, dtype=torch.long))

    pixels = generate_image(model, SmallVAE(), SmallTextEncoder(), "test prompt", width=32, height=32, seed=1)
    assert pixels.shape == (1, 3, 1, 32, 32)


def test_edit_uses_comfy_picture_label_and_reference_aspect(tmp_path, monkeypatch) -> None:
    from unituner.models.minimax_h3 import image_inference

    image = tmp_path / "reference.png"
    Image.fromarray(np.zeros((32, 64, 3), dtype=np.uint8)).save(image)
    monkeypatch.setattr(
        image_inference, "_resize_reference", lambda source, low, high: torch.zeros(1, 32, 64, 3, dtype=torch.uint8)
    )

    class SmallVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def encode(self, pixels):
            return torch.zeros(1, 24, 1, 2, 4)

        def decode(self, latents):
            return torch.zeros(1, 3, 1, latents.shape[-2] * 16, latents.shape[-1] * 16)

    class SmallTextEncoder:
        def encode(self, presentation):
            assert presentation.processor_text.startswith("Picture 1: ")
            assert not presentation.processor_text.startswith("<Picture 1>:")
            return SimpleNamespace(hidden_states=torch.zeros(1, 8), token_tags=torch.ones(1, dtype=torch.long))

    model = MiniMaxH3ImageTransformer(_config(), dtype=torch.float32)
    pixels = generate_image(model, SmallVAE(), SmallTextEncoder(), "edit", images=(image,))
    assert pixels.shape == (1, 3, 1, 32, 64)


def test_reference_resize_uses_comfy_h3_image_short_edge_bands() -> None:
    from unituner.models.minimax_h3.image_inference import _resize_reference

    small = np.zeros((64, 128, 3), dtype=np.uint8)
    assert _resize_reference(small, 1024, 1024).shape == (1, 1024, 2048, 3)
    large = np.zeros((2048, 4096, 3), dtype=np.uint8)
    resized = _resize_reference(large, 1024, 2048)
    assert resized.shape == (1, 2048, 4096, 3)
    resized = _resize_reference(large, 1024, 1024)
    assert resized.shape == (1, 1024, 2048, 3)


def test_image_condition_defaults_and_official_reference_limit(tmp_path, monkeypatch) -> None:
    from unituner.models.minimax_h3 import image_inference

    image = tmp_path / "ref.png"
    Image.fromarray(np.zeros((32, 64, 3), dtype=np.uint8)).save(image)
    monkeypatch.setattr(
        image_inference, "_resize_reference", lambda source, low, high: torch.zeros(1, 32, 64, 3, dtype=torch.uint8)
    )

    class SmallVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def encode(self, pixels):
            return torch.zeros(1, 24, 1, 2, 4)

    class SmallText:
        def encode(self, presentation):
            return SimpleNamespace(hidden_states=torch.zeros(1, 8), token_tags=torch.ones(1, dtype=torch.long))

    device = torch.device("cpu")
    vae, text = SmallVAE(), SmallText()
    assert encode_image_condition(vae, text, "prompt", model_device=device)[-1] == (2048, 2048)
    assert encode_image_condition(vae, text, "prompt", size_policy="official", model_device=device)[-1] == (4096, 4096)
    assert encode_image_condition(vae, text, "edit", images=(image,), model_device=device)[-1] == (64, 32)
    official = encode_image_condition(vae, text, "edit", size_policy="official", images=(image,), model_device=device)[
        -1
    ]
    assert official == (5792, 2912)
    assert encode_image_condition(
        vae, text, "edit", size_policy="official", width=64, height=32, images=(image,), model_device=device
    )[-1] == (64, 32)
    assert encode_image_condition(vae, text, "edit", height=96, images=(image,), model_device=device)[-1] == (64, 96)
    assert len(encode_image_condition(vae, text, "edit", images=(image,) * 16, model_device=device)[2]) == 16
    with pytest.raises(ValueError, match="at most 16"):
        encode_image_condition(vae, text, "edit", images=(image,) * 17, model_device=device)
