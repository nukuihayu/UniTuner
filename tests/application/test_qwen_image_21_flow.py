"""Exercise public Qwen2.1 cached full/adapter training and exact update-boundary resume."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from tests.support.qwen_image_21 import write_text_assets
from tests.support.text_assets import main_with_local_tokenizers
from unitunner.artifacts.tensors import TensorReader
from unitunner.models.qwen_image_2_1.config.model import QwenImage21Config
from unitunner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21, save_qwen_image_21, save_qwen_image_21_vae
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE


def test_public_training_independent_reference_geometry(tmp_path):
    test_public_training_three_updates_and_exact_resume(tmp_path, "lora", 2, independent=True)


@pytest.mark.parametrize("algorithm,count", [("full", 0), ("lora", 2), ("loha", 1), ("lokr", 2)])
def test_public_training_three_updates_and_exact_resume(
    tmp_path,
    algorithm,
    count,
    residency="resident",
    *,
    precision="fp32",
    wavelet=False,
    recompute=None,
    attention_offload=False,
    independent=False,
):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(983)
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
                num_layers=2,
                context_in_dim=16,
                attention_head_dim=16,
                num_attention_heads=2,
                axes_dims_rope=(4, 6, 6),
            )
        )
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        save_qwen_image_21(model, tmp_path / "model.safetensors")
        for directory in ["images", "ref0", "ref1"]:
            (tmp_path / directory).mkdir()
            for i, name in enumerate(["a", "b"]):
                h, w = (32, 64) if directory == "images" or (directory == "ref0") == (i == 0) else (64, 32)
                c = 4 if i == 0 else 3
                pixels = (np.arange(h * w * c, dtype=np.uint32).reshape(h, w, c) * 13 + i * 7).astype(np.uint8)
                Image.fromarray(pixels).save(tmp_path / directory / f"{name}.png")
        (tmp_path / "images/a.txt").write_text("short caption")
        (tmp_path / "images/b.txt").write_text("long caption about objects and their texture")
        body = f'''[model]
family = "qwen-image-2-1"
variant = "base"
weights = "{tmp_path.as_posix()}/model.safetensors"
vae = "{tmp_path.as_posix()}/vae.safetensors"
text_encoder = "{tmp_path.as_posix()}/text"
[dataset.groups.main]
source = "{tmp_path.as_posix()}/images"
cache_directory = "{tmp_path.as_posix()}/cache"
resolution = [64,32]
batch_size = 2
reference_sources = {json.dumps([str(tmp_path / f"ref{i}") for i in range(count)])}
{"reference_resolution = [128,64]" if independent else ""}
[train]
parameter_mode = "{"full" if algorithm == "full" else "adapter"}"
max_updates = 3
accumulation_steps = 2
seed = 991
learning_rate = 0.001
'''
        if algorithm != "full":
            body += f'''[train.adapter]
algorithm = "{algorithm}"
rank = 2
alpha = 7.0
'''
        if wavelet:
            body += "[method.wavelet]\nweight = 0.5\nlevels = 2\n"
        selected_recompute = recompute if recompute is not None else ("block" if residency != "vbar" else None)
        body += f'''[execution]
precision = "{precision}"
residency = "{residency}"
{f'recompute = "{selected_recompute}"' if selected_recompute is not None else ""}
attention_activation_offload = {str(attention_offload).lower()}
[output]
max_resume_checkpoints = 0
directory = "{tmp_path.as_posix()}/run"
name = "model"
save_every_updates = 1
'''
        config = tmp_path / "train.toml"
        config.write_text(body)
        device = os.environ.get("UNITUNNER_TEST_DEVICE", "cpu")
        assert main_with_local_tokenizers(["cache", str(config), "--device", device]) == 0
        assert main_with_local_tokenizers(["train", str(config), "--device", device]) == 0
        checkpoint = tmp_path / "run/resume/model.state-00000003.safetensors"
        artifact = tmp_path / "run/weights/model.safetensors"
        expected = checkpoint.read_bytes(), artifact.read_bytes()
        assert (
            main_with_local_tokenizers(
                [
                    "train",
                    str(config),
                    "--device",
                    device,
                    "--resume",
                    str(tmp_path / "run/resume/model.state-00000001.safetensors"),
                ]
            )
            == 0
        )
        assert expected == (checkpoint.read_bytes(), artifact.read_bytes())
        if algorithm == "full":
            trained = load_qwen_image_21(artifact)
            assert trained.config == model.config
            assert any(not torch.equal(value, initial[key]) for key, value in trained.state_dict().items())
        else:
            with TensorReader(artifact) as reader:
                keys = reader.keys()
                metadata = reader.metadata()
                prefix = "transformer." if algorithm == "lora" else "lycoris_"
                assert all(key.startswith(prefix) for key in keys)
                assert (
                    metadata["base_model_version"] == "qwen_image_21" and metadata["qwen_image_model_version"] == "2.1"
                )
                assert len(keys) == 14 * ({"lora": 3, "loha": 5, "lokr": 4}[algorithm])
                assert all(torch.isfinite(reader.read(key)).all() for key in keys)
        sample = f"""prompt = "test image"
negative_prompt = ""
width = 64
height = 32
steps = {1 if algorithm == "full" else 3}
seed = 1009
guidance = 2.0
reference_images = {json.dumps([str(tmp_path / f"ref{i}/a.png") for i in range(count)])}
conditioning_resize = "output"
"""
        if algorithm == "lokr":
            Image.fromarray(np.arange(32 * 64 * 3, dtype=np.uint32).reshape(32, 64, 3).astype(np.uint8)).save(
                tmp_path / "mask.png"
            )
            sample += f'mask_image = "{tmp_path.as_posix()}/mask.png"\n'
        preview_config = tmp_path / "preview.toml"
        preview_config.write_text(
            body + "[preview]\nevery_updates = 1\nat_start = true\n[[preview.samples]]\n" + sample
        )
        assert main_with_local_tokenizers(["train", str(preview_config), "--device", device]) == 0
        assert expected == (checkpoint.read_bytes(), artifact.read_bytes())
        header = body.split("[dataset.groups.main]")[0]
        if algorithm == "full":
            header = header.replace((tmp_path / "model.safetensors").as_posix(), artifact.as_posix())
        generation = f'output = "{tmp_path.as_posix()}/generated.png"\n' + header + "[sampling]\n" + sample
        if algorithm != "full":
            generation += f'''[adapter]
weights = "{artifact.as_posix()}"
[adapter.config]
algorithm = "{algorithm}"
rank = 2
alpha = 7.0
'''
        generate_config = tmp_path / "generate.toml"
        generate_config.write_text(generation)
        rng = torch.get_rng_state().clone()
        assert main_with_local_tokenizers(["generate", str(generate_config), "--device", device]) == 0
        image = (tmp_path / "generated.png").read_bytes()
        assert image == (tmp_path / "run/previews/step-00000003-00.png").read_bytes()
        assert main_with_local_tokenizers(["generate", str(generate_config), "--device", device]) == 0
        assert image == (tmp_path / "generated.png").read_bytes() and torch.equal(rng, torch.get_rng_state())
        with Image.open(tmp_path / "generated.png") as output:
            assert output.mode == "RGBA" and output.size == (64, 32)
        # Content changes must invalidate both cache reuse and checkpoint identity.
        (tmp_path / "images/a.txt").write_text("modified conditioning")
        assert main_with_local_tokenizers(["cache", str(config), "--device", device]) == 0
        assert (
            main_with_local_tokenizers(
                [
                    "train",
                    str(config),
                    "--device",
                    device,
                    "--resume",
                    str(tmp_path / "run/resume/model.state-00000001.safetensors"),
                ]
            )
            == 2
        )
        assert expected == (checkpoint.read_bytes(), artifact.read_bytes())
    finally:
        torch.set_num_threads(previous)
