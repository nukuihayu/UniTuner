"""Exercise actual Krea2 public cache, adapter training/export and update-boundary resume."""

import os
from contextlib import closing

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.support.qwen_image_21 import write_text_assets
from tests.support.text_assets import main_with_local_tokenizers
from unitunner.adapters.config import AdapterConfig
from unitunner.adapters.loading import LoadedAdapter
from unitunner.artifacts.formats.krea2_adapters import krea2_adapter_prefixes
from unitunner.artifacts.tensors import TensorReader
from unitunner.data.cache.storage import load_latent_manifest, read_latent_record
from unitunner.models.krea2.config.model import Krea2Config
from unitunner.models.krea2.loading import load_krea2, save_krea2
from unitunner.models.krea2.projections import adapter_projections
from unitunner.models.krea2.transformer import Krea2Transformer
from unitunner.models.qwen_image.config.vae import QwenImageVAEConfig
from unitunner.models.qwen_image.loading import save_qwen_image_vae
from unitunner.models.qwen_image.vae import QwenImageVAE


@pytest.fixture(autouse=True, scope="module")
def deterministic_execution():
    """Pin deterministic kernels for exact compiled training/preview/resume comparisons."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.backends.cudnn.flags(deterministic=True, benchmark=False):
            yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr", "full"])
def test_public_cache_train_export_and_exact_resume(tmp_path, algorithm):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1051)
        write_text_assets(tmp_path / "text", language_layers=3, feature_layers=(0, 2))
        save_qwen_image_vae(
            QwenImageVAE(QwenImageVAEConfig(base_channels=2, layers_per_block=1)), tmp_path / "vae.safetensors"
        )
        model = Krea2Transformer(
            Krea2Config(
                hidden_size=64,
                timestep_dim=16,
                text_dim=16,
                num_heads=4,
                num_kv_heads=2,
                mlp_multiplier=2,
                num_layers=1,
                text_layers=2,
                text_heads=2,
                text_kv_heads=1,
            )
        )
        initial = {name: value.clone() for name, value in model.state_dict().items()}
        save_krea2(model, tmp_path / "model.safetensors")
        (tmp_path / "images").mkdir()
        for i, name in enumerate(["a", "b"]):
            Image.fromarray(
                (np.arange(16 * 32 * 3, dtype=np.uint32).reshape(16, 32, 3) * 13 + i * 7).astype(np.uint8)
            ).save(tmp_path / f"images/{name}.png")
        (tmp_path / "images/a.txt").write_text("")
        (tmp_path / "images/b.txt").write_text("longer caption about objects and flowers")
        body = f'''[model]
family = "krea2"
variant = "raw"
weights = "{tmp_path}/model.safetensors"
vae = "{tmp_path}/vae.safetensors"
text_encoder = "{tmp_path}/text"
[model.text]
max_length = 256
[dataset.groups.main]
source = "{tmp_path}/images"
cache_directory = "{tmp_path}/cache"
resolution = [32,16]
batch_size = 2
[train]
max_updates = 3
accumulation_steps = 2
seed = 1057
learning_rate = 0.001
[train.adapter]
algorithm = "{algorithm}"
rank = 2
alpha = 7.0
[execution]
precision = "fp32"
recompute = "block"
compile = {str(os.environ.get("UNITUNNER_TEST_COMPILE") == "1").lower()}
[output]
max_resume_checkpoints = 0
directory = "{tmp_path}/run"
name = "model"
save_every_updates = 1
'''
        if algorithm == "full":
            body = body.replace(
                '[train.adapter]\nalgorithm = "full"\nrank = 2\nalpha = 7.0\n', 'parameter_mode = "full"\n'
            )
        config = tmp_path / "train.toml"
        config.write_text(body)
        selected = os.environ.get("UNITUNNER_TEST_DEVICE", "cpu")
        with sdpa_kernel(SDPBackend.MATH):
            assert main_with_local_tokenizers(["cache", str(config), "--device", selected]) == 0
            manifest = load_latent_manifest(tmp_path / "cache")
            assert manifest.family == "krea2"
            rows = [
                read_latent_record(tmp_path / "cache", entry, device=torch.device("cpu")) for entry in manifest.entries
            ]
            assert all(row.latents.shape == (16, 1, 2, 4) and row.caption.shape[1:] == (2, 16) for row in rows)
            assert rows[0].caption.shape[0] != rows[1].caption.shape[0]
            cached = (tmp_path / "cache/manifest.json").read_bytes()
            assert main_with_local_tokenizers(["cache", str(config), "--device", selected]) == 0
            assert (tmp_path / "cache/manifest.json").read_bytes() == cached
            assert main_with_local_tokenizers(["train", str(config), "--device", selected]) == 0
            checkpoint = tmp_path / "run/resume/model.state-00000003.safetensors"
            artifact = tmp_path / "run/weights/model.safetensors"
            expected = checkpoint.read_bytes(), artifact.read_bytes()
            assert (
                main_with_local_tokenizers(
                    [
                        "train",
                        str(config),
                        "--device",
                        selected,
                        "--resume",
                        str(tmp_path / "run/resume/model.state-00000001.safetensors"),
                    ]
                )
                == 0
            )
            assert expected == (checkpoint.read_bytes(), artifact.read_bytes())
        preview_config = tmp_path / "preview.toml"
        preview_config.write_text(
            body
            + """[preview]
every_updates = 1
at_start = true
[[preview.samples]]
prompt = "a flower"
negative_prompt = "blurry"
width = 32
height = 16
steps = 2
guidance = 2.5
"""
        )
        with sdpa_kernel(SDPBackend.MATH):
            assert main_with_local_tokenizers(["train", str(preview_config), "--device", selected]) == 0
        assert expected == (checkpoint.read_bytes(), artifact.read_bytes())
        previews = sorted((tmp_path / "run/previews").glob("*.png"))
        assert len(previews) == 4
        for image in previews:
            with Image.open(image) as frame:
                assert frame.mode == "RGB" and frame.size == (32, 16)
        generation = tmp_path / "generate.toml"
        generation_text = f'''output = "{tmp_path}/generated.png"
[model]
family = "krea2"
variant = "raw"
weights = "{tmp_path}/model.safetensors"
vae = "{tmp_path}/vae.safetensors"
text_encoder = "{tmp_path}/text"
[model.text]
max_length = 256
[sampling]
prompt = "a flower"
negative_prompt = "blurry"
width = 32
height = 16
steps = 2
guidance = 2.5
[adapter]
weights = "{artifact}"
[adapter.config]
algorithm = "{algorithm}"
rank = 2
alpha = 1.0
'''
        if algorithm == "full":
            generation_text = generation_text.split("[adapter]")[0].replace(
                f'weights = "{tmp_path}/model.safetensors"', f'weights = "{artifact}"'
            )
        generation.write_text(generation_text)
        with sdpa_kernel(SDPBackend.MATH):
            assert main_with_local_tokenizers(["generate", str(generation), "--device", selected]) == 0
            generated = (tmp_path / "generated.png").read_bytes()
            assert main_with_local_tokenizers(["generate", str(generation), "--device", selected]) == 0
            assert (tmp_path / "generated.png").read_bytes() == generated
        assert generated == previews[-1].read_bytes()
        if algorithm == "full":
            trained = load_krea2(artifact)
            assert any(not torch.equal(value, initial[name]) for name, value in trained.state_dict().items())
        else:
            with TensorReader(artifact) as reader:
                metadata = reader.metadata()
                assert metadata["base_model_version"] == "krea2" and metadata["network_alpha"] == "7.0"
                assert not any(key.startswith("ss_") for key in metadata)
                assert all(
                    key.startswith("transformer." if algorithm == "lora" else "lycoris_") for key in reader.keys()
                )
                assert len(reader.keys()) == 48 * {"lora": 3, "loha": 5, "lokr": 4}[algorithm]
            targets = adapter_projections(model)
            with closing(
                LoadedAdapter(
                    artifact,
                    AdapterConfig(algorithm=algorithm, rank=2, alpha=1.0),
                    targets,
                    external_prefixes=krea2_adapter_prefixes(targets, algorithm=algorithm),
                )
            ) as loaded:
                assert any(value.abs().sum() > 0 for value in loaded.branches.parameters())
        for name, value in model.state_dict().items():
            assert torch.equal(value, initial[name])
        (tmp_path / "images/a.txt").write_text("edited caption")
        assert main_with_local_tokenizers(["cache", str(config), "--device", selected]) == 0
        assert (tmp_path / "cache/manifest.json").read_bytes() != cached
        assert (
            main_with_local_tokenizers(
                [
                    "train",
                    str(config),
                    "--device",
                    selected,
                    "--resume",
                    str(tmp_path / "run/resume/model.state-00000001.safetensors"),
                ]
            )
            == 2
        )
    finally:
        torch.set_num_threads(previous)
