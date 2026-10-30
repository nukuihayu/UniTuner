"""Exercise public Qwen cache/train/preview/generate and exact native update-boundary resume."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from tests.support.qwen_image import write_text_assets
from tests.support.text_assets import main_with_local_tokenizers
from unituner.data.cache.storage import load_latent_manifest, read_latent_record
from unituner.models.qwen_image.config.model import QwenImageConfig
from unituner.models.qwen_image.config.vae import QwenImageVAEConfig
from unituner.models.qwen_image.loading import save_qwen_image, save_qwen_image_vae
from unituner.models.qwen_image.transformer import QwenImageTransformer
from unituner.models.qwen_image.vae import QwenImageVAE


def command(arguments):
    return main_with_local_tokenizers([*arguments, "--device", os.environ.get("UNITUNER_TEST_DEVICE", "cpu")])


@pytest.mark.parametrize(
    "variant,algorithm,count,remove_base",
    [
        ("original", "full", 0, False),
        ("edit", "lora", 1, False),
        ("edit-2509", "loha", 0, False),
        ("edit-2511", "lokr", 2, False),
        ("layered", "lora", 0, False),
        ("layered", "full", 0, True),
    ],
)
def test_qwen_cache_train_resume_preview_and_generation(tmp_path, variant, algorithm, count, remove_base):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(761)
        root = tmp_path
        write_text_assets(root / "text")
        save_qwen_image_vae(
            QwenImageVAE(
                QwenImageVAEConfig(
                    input_channels=4 if variant == "layered" else 3,
                    base_channels=4,
                    latent_channels=2,
                    channel_multipliers=(1, 2, 2, 2),
                    layers_per_block=1,
                    latents_mean=(-0.31, 0.73),
                    latents_std=(1.73, 0.91),
                )
            ),
            root / "vae.safetensors",
        )
        save_qwen_image(
            QwenImageTransformer(
                QwenImageConfig(
                    variant=variant,
                    in_channels=8,
                    out_channels=2,
                    num_layers=2,
                    joint_attention_dim=16,
                    attention_head_dim=16,
                    num_attention_heads=2,
                    axes_dims_rope=(4, 6, 6),
                )
            ),
            root / "model.safetensors",
        )
        for directory in ("images", "ref0", "ref1", "foreground0", "foreground1"):
            (root / directory).mkdir()
            for index, name in enumerate(("a", "b")):
                height, width = (35 + 16 * index, 49 + 16 * index) if directory.startswith("ref") else (24, 40)
                channels = 4 if directory.startswith("foreground") or directory == "ref0" else 3
                pixels = (
                    np.arange(height * width * channels, dtype=np.uint32).reshape(height, width, channels) * 13
                    + index * 37
                ).astype(np.uint8)
                with Image.fromarray(pixels) as image:
                    image.save(root / directory / f"{name}.png")
        (root / "images/a.txt").write_text("short caption")
        (root / "images/b.txt").write_text("long caption " * 7)
        header = f'''[model]
family = "qwen-image"
variant = "{variant}"
weights = "{root}/model.safetensors"
vae = "{root}/vae.safetensors"
text_encoder = "{root}/text"
text_source_format = "qwen"
'''
        reference_dirs = [str(root / f"ref{index}") for index in range(count)]
        layer_dirs = [str(root / f"foreground{index}") for index in range(2)] if variant == "layered" else []
        train = (
            header
            + f'''[dataset.groups.main]
source = "{root}/images"
cache_directory = "{root}/cache"
resolution = [32,16]
batch_size = 2
reference_sources = {json.dumps(reference_dirs)}
layer_sources = {json.dumps(layer_dirs)}
[train]
parameter_mode = "{"full" if algorithm == "full" else "adapter"}"
max_updates = 3
accumulation_steps = 2
seed = 763
learning_rate = 0.001
'''
        )
        if algorithm != "full":
            train += f'''[train.adapter]
algorithm = "{algorithm}"
rank = 2
alpha = 3.0
'''
        if variant == "layered":
            train += f"""[method.qwen_image]
remove_base = {str(remove_base).lower()}
"""
        train += f'''[execution]
precision = "fp32"
residency = "resident"
recompute = "block"
[output]
max_resume_checkpoints = 0
directory = "{root}/run"
name = "model"
save_every_updates = 1
'''
        (root / "train.toml").write_text(train)
        rng = torch.get_rng_state().clone()
        assert command(["cache", str(root / "train.toml")]) == 0
        assert torch.equal(rng, torch.get_rng_state())
        manifest = load_latent_manifest(root / "cache")
        assert manifest.family == "qwen-image" and manifest.variant == variant
        records = [read_latent_record(root / "cache", entry, device=torch.device("cpu")) for entry in manifest.entries]
        assert all(record.latents.shape == (2, 3 if variant == "layered" else 1, 2, 4) for record in records)
        assert all(len(record.references) == count for record in records)
        assert records[0].caption.shape[0] != records[1].caption.shape[0]
        if count:
            assert all(reference.shape == record.latents.shape for record in records for reference in record.references)
        content = (root / "cache/manifest.json").read_bytes()
        if variant == "layered":
            foreground = root / "foreground0/a.png"
            original = foreground.read_bytes()
            with Image.new("RGB", (40, 24), (11, 23, 59)) as image:
                image.save(foreground)
            assert command(["cache", str(root / "train.toml")]) == 2
            assert (root / "cache/manifest.json").read_bytes() == content
            foreground.write_bytes(original)
        rng = torch.get_rng_state().clone()
        assert command(["cache", str(root / "train.toml")]) == 0
        assert (root / "cache/manifest.json").read_bytes() == content
        assert torch.equal(rng, torch.get_rng_state())
        assert command(["train", str(root / "train.toml")]) == 0
        checkpoints = {
            path.relative_to(root / "run").as_posix(): path.read_bytes()
            for path in (root / "run").rglob("*.safetensors")
        }
        references = [str(root / f"ref{index}/a.png") for index in range(count)]
        if variant == "edit-2509":
            references = [str(root / "ref0/a.png")]
        if variant == "layered":
            references = [str(root / "images/a.png")]
        sample = f"""prompt = "test image"
negative_prompt = ""
width = 32
height = 16
steps = 3
seed = 767
guidance = 2.0
reference_images = {json.dumps(references)}
conditioning_resize = "output"
"""
        if variant == "layered":
            sample += "output_layers = 2\n"
        (root / "preview.toml").write_text(
            train + "[preview]\nevery_updates = 1\nat_start = true\n[[preview.samples]]\n" + sample
        )
        assert command(["train", str(root / "preview.toml")]) == 0
        assert checkpoints == {
            path.relative_to(root / "run").as_posix(): path.read_bytes()
            for path in (root / "run").rglob("*.safetensors")
        }
        assert (
            command(
                [
                    "train",
                    str(root / "preview.toml"),
                    "--resume",
                    str(root / "run/resume/model.state-00000001.safetensors"),
                ]
            )
            == 0
        )
        for name in ("resume/model.state-00000003.safetensors", "weights/model.safetensors"):
            assert (root / "run" / name).read_bytes() == checkpoints[name]
        generation = 'output = "' + str(root / "generated.png") + '"\n'
        generation += (
            header.replace(str(root / "model.safetensors"), str(root / "run/weights/model.safetensors"))
            if algorithm == "full"
            else header
        )
        generation += "[sampling]\n" + sample
        if algorithm != "full":
            generation += f'''[adapter]
weights = "{root}/run/weights/model.safetensors"
[adapter.config]
algorithm = "{algorithm}"
rank = 2
alpha = 3.0
'''
        (root / "generate.toml").write_text(generation)
        assert command(["generate", str(root / "generate.toml")]) == 0
        for layer in range(3 if variant == "layered" else 1):
            suffix = "" if layer == 0 else f".layer-{layer:03d}"
            assert (root / f"generated{suffix}.png").read_bytes() == (
                root / f"run/previews/step-00000003-00{suffix}.png"
            ).read_bytes()
        if count or variant == "layered":
            directory = "ref0" if count else "foreground0"
            with Image.new("RGBA", (40, 24), (11, 23, 59, 173)) as image:
                image.save(root / directory / "a.png")
            assert command(["cache", str(root / "train.toml")]) == 0
            changed = load_latent_manifest(root / "cache")
            assert changed.entries[0] != manifest.entries[0] and changed.entries[1] == manifest.entries[1]
            assert (
                command(
                    [
                        "train",
                        str(root / "train.toml"),
                        "--resume",
                        str(root / "run/resume/model.state-00000001.safetensors"),
                    ]
                )
                == 2
            )
    finally:
        torch.set_num_threads(previous)
