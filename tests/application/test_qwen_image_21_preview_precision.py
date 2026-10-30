"""Publish real Qwen2.1 PNG previews from mixed precision live training resources."""

import pytest
import torch
from PIL import Image

from tests.support.qwen_image_21 import write_text_assets
from tests.support.text_assets import local_text_config
from unitunner.adapters.attachment import AdapterAttachment
from unitunner.adapters.lora import LoRA
from unitunner.config.loading import load_training_config
from unitunner.models.qwen_image_2_1.config.model import QwenImage21Config
from unitunner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unitunner.models.qwen_image_2_1.loading import save_qwen_image_21_vae
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE
from unitunner.precision import FLOAT_DTYPES
from unitunner.training.integrations.qwen_image_2_1 import previews
from unitunner.training.previews.images import PreviewImages


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16", "auto"])
@pytest.mark.parametrize("adapter", [False, True])
def test_native_preview_publishes_without_converting_live_weights(tmp_path, monkeypatch, precision, adapter):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        effective = ("bf16" if adapter else "fp32") if precision == "auto" else precision
        write_text_assets(tmp_path / "text")
        save_qwen_image_21_vae(
            QwenImage21VAE(
                QwenImage21VAEConfig(
                    base_channels=2, decoder_base_channels=4, channel_multipliers=(1, 2, 4, 4, 4), layers_per_block=1
                )
            ),
            tmp_path / "vae.safetensors",
        )
        model = QwenImage21Transformer(
            QwenImage21Config(
                in_channels=64,
                out_channels=64,
                num_layers=1,
                context_in_dim=16,
                attention_head_dim=16,
                num_attention_heads=2,
                axes_dims_rope=(4, 6, 6),
            )
        )
        if adapter:
            model.to(FLOAT_DTYPES[effective]).requires_grad_(False)
        branch = LoRA(model.img_in.in_features, model.img_in.out_features, rank=2, alpha=1)
        owner = AdapterAttachment(model.img_in, branch)
        Image.new("RGBA", (32, 32), (51, 97, 193, 255)).save(tmp_path / "reference.png")
        Image.new("L", (32, 32), 127).save(tmp_path / "mask.png")
        config_path = tmp_path / "preview.toml"
        config_path.write_text(f'''[model]
family = "qwen-image-2-1"
variant = "base"
weights = "{tmp_path.as_posix()}/unused.safetensors"
vae = "{tmp_path.as_posix()}/vae.safetensors"
text_encoder = "{tmp_path.as_posix()}/text"
[dataset.groups.main]
source = "{tmp_path.as_posix()}/unused-images"
cache_directory = "{tmp_path.as_posix()}/unused-cache"
[train]
parameter_mode = "full"
[execution]
precision = "{precision}"
[output]
directory = "{tmp_path.as_posix()}/run"
[preview]
every_updates = 1
[[preview.samples]]
prompt = "edit"
width = 32
height = 32
steps = 2
guidance = 1.0
conditioning_resize = "output"
reference_images = ["{tmp_path.as_posix()}/reference.png"]
mask_image = "{tmp_path.as_posix()}/mask.png"
''')
        config = load_training_config(config_path)
        config = config.model_copy(
            update={
                "model": config.model.model_copy(
                    update={"text": local_text_config(tmp_path / "text", base=config.model.text)}
                )
            }
        )
        actual_generate = previews.generate_latents
        observed = []

        def capture(*args, **kwargs):
            latents = actual_generate(*args, **kwargs)
            observed.append(latents.dtype)
            return latents

        monkeypatch.setattr(previews, "generate_latents", capture)
        before = [(p, p.clone(), p.dtype) for p in (*model.parameters(), *branch.parameters())]
        rng = torch.get_rng_state().clone()
        try:
            PreviewImages(config, None)(0, model, branch, None)
            with Image.open(tmp_path / "run/previews/step-00000000-00.png") as image:
                assert image.size == (32, 32)
            assert observed == [FLOAT_DTYPES[effective]]
            assert all(torch.equal(p, value) and p.dtype == dtype for p, value, dtype in before)
            assert torch.equal(rng, torch.get_rng_state()) and model.training and branch.training
        finally:
            owner.close()
    finally:
        torch.set_num_threads(previous)
