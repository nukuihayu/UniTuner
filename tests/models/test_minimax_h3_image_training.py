"""Exercise image flow and restoration through the native training binding."""

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from unituner.config.features import LayerSyncConfig
from unituner.config.restoration import MethodConfig
from unituner.config.routing import TreadConfig
from unituner.inference.restoration.vae import encode_restoration_image
from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.config.video_vae import MiniMaxH3VideoDecoderConfig, MiniMaxH3VideoVAEConfig
from unituner.models.minimax_h3.image import MiniMaxH3ImageTransformer
from unituner.models.minimax_h3.image_vae import MiniMaxH3ImageVAE
from unituner.training.integrations.minimax_h3.image import (
    MiniMaxH3ImageBinding,
    MiniMaxH3ImageFlowRequest,
    prepare_image_flow_request,
)
from unituner.training.objectives import align_steps, install_objectives
from unituner.training.session import TrainingSession


def _binding() -> MiniMaxH3ImageBinding:
    config = MiniMaxH3Config(
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
    return MiniMaxH3ImageBinding(MiniMaxH3ImageTransformer(config, pdd_sigmas=(1.0, 0.5, 0.0), dtype=torch.float32))


def test_flow_request_encodes_edit_with_inference_condition(tmp_path, monkeypatch) -> None:
    from unituner.models.minimax_h3 import image_inference

    monkeypatch.setattr(
        image_inference, "_resize_reference", lambda source, low, high: torch.zeros(1, 32, 64, 3, dtype=torch.uint8)
    )
    image = tmp_path / "reference.png"
    Image.fromarray(np.zeros((32, 64, 3), dtype=np.uint8)).save(image)

    class SmallVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def encode(self, pixels):
            return torch.zeros(1, 24, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)

    class SmallTextEncoder:
        def encode(self, presentation):
            assert presentation.processor_text.startswith("Picture 1: ")
            return SimpleNamespace(hidden_states=torch.zeros(2, 8), token_tags=torch.ones(2, dtype=torch.long))

    request = prepare_image_flow_request(
        SmallVAE(),
        SmallTextEncoder(),
        torch.zeros(1, 3, 32, 64),
        "edit",
        noise=torch.randn(1, 24, 1, 2, 4),
        sigma=torch.tensor([0.5]),
        model_device=torch.device("cpu"),
        images=(image,),
    )
    assert request.latents.shape == (1, 24, 1, 2, 4)
    assert len(request.references) == 1
    assert request.references[0].shape == (1, 24, 1, 2, 4)
    assert _binding().step(request).loss.isfinite()


def test_flow_step_and_restoration_keep_gradients() -> None:
    binding = _binding()
    low = torch.randn(1, 24, 1, 2, 4, requires_grad=True)
    high = torch.randn_like(low)
    noise = torch.randn_like(low)
    text = torch.randn(1, 3, 8)
    tags = torch.tensor([[1, 0, 1]])
    reference = (torch.randn(1, 24, 1, 2, 4),)
    request = MiniMaxH3ImageFlowRequest(high, noise, torch.tensor([0.75]), text, tags, reference)
    prediction = binding.predict(request)
    torch.testing.assert_close(prediction.target, high - noise)
    result = binding.step(request)
    assert result.sample_count == 1
    result.loss.backward()
    assert binding.model.blocks[0].attn.qkv_proj.weight.grad is not None
    binding.model.zero_grad(set_to_none=True)
    restored = binding.restore(low, text, tags, sigma=0.5, references=reference)
    assert restored.shape == low.shape
    loss = binding.restoration_step(low, high, text, tags, sigma=0.5, references=reference)
    loss.loss.backward()
    assert low.grad is not None and torch.isfinite(low.grad).all()
    assert binding.model.blocks[0].attn.qkv_proj.weight.grad is not None
    assert set(binding.adapter_targets()) == {
        "blocks.0.attn.qkv_proj",
        "blocks.0.attn.out_proj",
        "blocks.0.mlp.fc1",
        "blocks.0.mlp.fc2",
    }


def test_h3_image_tread_and_layersync_keep_target_gradients() -> None:
    config = _binding().model.config.model_copy(update={"num_layers": 2})
    model = MiniMaxH3ImageTransformer(config, dtype=torch.float32)
    binding = MiniMaxH3ImageBinding(model)
    request = MiniMaxH3ImageFlowRequest(
        torch.randn(1, 24, 1, 2, 4),
        torch.randn(1, 24, 1, 2, 4),
        torch.tensor([0.5]),
        torch.randn(1, 3, 8),
        torch.tensor([[1, 0, 1]]),
        (torch.randn(1, 24, 1, 2, 4),),
    )
    method = MethodConfig(
        layersync=LayerSyncConfig(student_layer=0, teacher_layer=1),
        tread=TreadConfig(drop_fraction=0.5, start_layer=0, end_layer=1),
    )
    with TrainingSession() as session:
        alignment = install_objectives(
            model, binding.regions, method, session=session, targets=None, device=torch.device("cpu")
        )
        result = next(align_steps(lambda: (binding.step(request) for _ in range(1)), alignment))
        assert torch.isfinite(result.loss) and "layersync" in result.metrics
        result.loss.backward()
        assert model.blocks[0].attn.qkv_proj.weight.grad is not None
    assert model.feature_recorder is None and model.token_routing is None


def test_pixel_restoration_step_preserves_encoder_and_denoiser_gradients() -> None:
    binding = _binding()

    class SmallVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_weight = torch.nn.Parameter(torch.ones(()))

        def encode(self, pixels):
            return pixels[:, :1].repeat(1, 24, 1, 1, 1) * self.encoder_weight

        def decode(self, latents):
            return latents[:, :3]

    vae = SmallVAE()
    low = torch.randn(1, 3, 2, 4)
    high = torch.zeros_like(low)
    result = binding.restoration_image_step(
        vae, low, high, torch.randn(1, 2, 8), torch.ones(1, 2, dtype=torch.long), sigma=0.5
    )
    assert result.phase == "generator"
    result.loss.backward()
    assert vae.encoder_weight.grad is not None
    assert binding.model.blocks[0].attn.qkv_proj.weight.grad is not None


def test_pixel_restoration_uses_native_image_vae_geometry() -> None:
    config = MiniMaxH3VideoVAEConfig(
        channels=32,
        channel_multipliers=(1,) * 6,
        residual_blocks=(1,) * 6,
        tiling=False,
        token_drop=0,
        decoder=MiniMaxH3VideoDecoderConfig(num_layers=1, num_heads=2, head_dim=16, rope_dim_ratio=0.5, rope_dims=2),
    )
    vae = MiniMaxH3ImageVAE(config)
    with torch.no_grad():
        for name, parameter in vae.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.zero_()
    vae.initialize_nonpersistent_buffers(torch.device("cpu"))
    pixels = torch.zeros(1, 3, 32, 64)
    mean = encode_restoration_image(vae, pixels, posterior="mean")
    torch.manual_seed(7)
    sampled = encode_restoration_image(vae, pixels, posterior="sample")
    assert mean.shape == sampled.shape == (1, 24, 1, 2, 4)
    assert not torch.equal(mean, sampled)
    binding = _binding()
    low = torch.zeros(1, 3, 32, 64)
    high = torch.zeros_like(low)
    result = binding.restoration_image_step(
        vae, low, high, torch.zeros(1, 2, 8), torch.ones(1, 2, dtype=torch.long), sigma=0.5
    )
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert binding.model.blocks[0].attn.qkv_proj.weight.grad is not None
