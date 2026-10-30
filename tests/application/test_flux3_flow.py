"""Exercise FLUX3 local cache, full/adapters, resumed training and public generated previews."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2Tokenizer, Qwen2VLImageProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from tests.support.text_assets import main_with_local_tokenizers
from unituner.config.loading import load_training_config
from unituner.data.cache.storage import load_latent_manifest, read_latent_record
from unituner.data.media.image import read_image
from unituner.models.flux2.config.vae import Flux2VAEConfig
from unituner.models.flux2.loading import load_flux2_vae, save_flux2_vae
from unituner.models.flux2.vae import Flux2VAE
from unituner.models.flux3.config.model import Flux3Config
from unituner.models.flux3.loading import save_flux3
from unituner.models.flux3.transformer import Flux3Transformer


def command(arguments):
    return main_with_local_tokenizers([*arguments, "--device", os.environ.get("UNITUNER_TEST_DEVICE", "cpu")])


@pytest.fixture
def assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(921)
    text = tmp_path / "text"
    text.mkdir()
    vocabulary = {value: index for index, value in enumerate(sorted(ByteLevel.alphabet()))}
    vocabulary.update(
        {
            "<|endoftext|>": 256,
            "<|im_start|>": 257,
            "<|im_end|>": 258,
            "<|image_pad|>": 259,
            "<|video_pad|>": 260,
            "<|vision_start|>": 261,
            "<|vision_end|>": 262,
        }
    )
    tokenizer = Qwen2Tokenizer(vocab=vocabulary, merges=[], pad_token="<|endoftext|>")
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen3VLVideoProcessor(),
        tokenizer=tokenizer,
        chat_template="{{ '<|im_start|>user\\n' + messages[0]['content'] + '<|im_end|>\\n<|im_start|>assistant\\n' }}",
    )
    processor.save_pretrained(text)
    text_config = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 263,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000, "mrope_section": [1, 1, 2]},
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "out_hidden_size": 16,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "deepstack_visual_indexes": [],
    }
    Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=259,
            video_token_id=260,
            vision_start_token_id=261,
            vision_end_token_id=262,
            tie_word_embeddings=False,
        )
    ).save_pretrained(text)
    vae = Flux2VAE(
        Flux2VAEConfig(
            base_channels=8, channel_multipliers=(1, 1, 1, 1), layers_per_block=1, latent_channels=2, norm_groups=4
        )
    )
    save_flux2_vae(vae, tmp_path / "vae.safetensors")
    for directory in ("images", "ref0", "ref1"):
        (tmp_path / directory).mkdir()
        for index, name in enumerate(("a", "b")):
            height, width = (16, 32) if directory == "images" else (35, 49) if directory == "ref0" else (67, 83)
            pixels = (
                np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3) * 13
                + index * 37
                + (11 if directory == "ref0" else 67)
            ).astype(np.uint8)
            with Image.fromarray(pixels) as image:
                image.save(tmp_path / directory / f"{name}.png")
    (tmp_path / "images/a.txt").write_text("short")
    (tmp_path / "images/b.txt").write_text("long caption " * 14)
    yield tmp_path
    torch.set_num_threads(previous)


def configs(root, algorithm, variant, count):
    model = Flux3Transformer(
        Flux3Config(
            variant=variant,
            hidden_size=32,
            num_heads=2,
            depth=1,
            depth_single=1,
            axes_dim=(4, 4, 4, 4),
            image_channels=8,
            text_channels=128,
        )
    )
    save_flux3(model, root / "model.safetensors")
    header = f'''[model]
family = "flux3"
variant = "{variant}"
weights = "{root}/model.safetensors"
vae = "{root}/vae.safetensors"
text_encoder = "{root}/text"
'''
    references = [str(root / f"ref{index}") for index in range(count)]
    train = (
        header
        + f'''[dataset.groups.main]
source = "{root}/images"
cache_directory = "{root}/cache"
resolution = [32,16]
batch_size = 2
reference_sources = {json.dumps(references)}
[train]
parameter_mode = "{"full" if algorithm == "full" else "adapter"}"
max_updates = 3
accumulation_steps = 2
seed = 712
learning_rate = 0.001
'''
    )
    if algorithm != "full":
        train += f'''[train.adapter]
algorithm = "{algorithm}"
rank = 2
alpha = 3.0
'''
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
    sample = f"""prompt = "a short generated image"
width = 32
height = 16
steps = 2
seed = 81
reference_images = {json.dumps([str(root / "ref0/a.png"), str(root / "ref1/a.png")])}
"""
    (root / "preview.toml").write_text(
        train
        + f"""[preview]
every_updates = 1
at_start = true
[[preview.samples]]
{sample}"""
    )
    exported_header = (
        header.replace(str(root / "model.safetensors"), str(root / "run/weights/model.safetensors"))
        if algorithm == "full"
        else header
    )
    generation = f'output = "{root}/generated.png"\n' + exported_header + "[sampling]\n" + sample
    if algorithm != "full":
        generation += f'''[adapter]
weights = "{root}/run/weights/model.safetensors"
[adapter.config]
algorithm = "{algorithm}"
rank = 2
alpha = 3.0
'''
    (root / "generate.toml").write_text(generation)
    return root / "train.toml"


@pytest.mark.parametrize(
    "algorithm,variant,count",
    [("full", "base", 2), ("lora", "distilled", 1), ("loha", "base", 0), ("lokr", "distilled", 2)],
)
def test_flux3_cache_train_generate_preview_and_resume(assets, algorithm, variant, count):
    root = assets
    train = configs(root, algorithm, variant, count)
    assert command(["cache", str(train)]) == 0
    manifest = load_latent_manifest(root / "cache")
    assert manifest.family == "flux3" and manifest.variant == variant
    records = [read_latent_record(root / "cache", entry, device=torch.device("cpu")) for entry in manifest.entries]
    assert all(len(record.references) == count for record in records)
    assert records[0].caption.shape[0] == 80 and records[1].caption.shape[0] > 80
    selected = torch.device(os.environ.get("UNITUNER_TEST_DEVICE", "cpu"))
    vae = load_flux2_vae(root / "vae.safetensors", device=selected).eval()
    with torch.no_grad():
        expected = vae.encode(read_image(root / "images/a.png", (32, 16))[None].to(selected))[0].cpu()
    torch.testing.assert_close(records[0].latents, expected, atol=1e-5, rtol=1e-4)
    with torch.no_grad():
        expected_refs = tuple(
            vae.encode(read_image(root / f"ref{index}/a.png", (32, 16))[None].to(selected))[0]
            .to(torch.bfloat16)
            .float()
            .cpu()
            for index in range(count)
        )
    for actual, expected in zip(records[0].references, expected_refs, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    content = (root / "cache/manifest.json").read_bytes()
    assert command(["cache", str(train)]) == 0
    assert (root / "cache/manifest.json").read_bytes() == content
    assert command(["train", str(train)]) == 0
    checkpoints = {
        file.relative_to(root / "run").as_posix(): file.read_bytes() for file in (root / "run").rglob("*.safetensors")
    }
    assert command(["generate", str(root / "generate.toml")]) == 0
    first = (root / "generated.png").read_bytes()
    assert command(["generate", str(root / "generate.toml")]) == 0
    assert (root / "generated.png").read_bytes() == first
    with Image.open(root / "generated.png") as image:
        assert image.size == (32, 16) and image.mode == "RGB"
    assert command(["train", str(root / "preview.toml")]) == 0
    assert checkpoints == {
        file.relative_to(root / "run").as_posix(): file.read_bytes() for file in (root / "run").rglob("*.safetensors")
    }
    assert (root / "run/previews/step-00000003-00.png").read_bytes() == first
    assert (
        command(
            ["train", str(root / "preview.toml"), "--resume", str(root / "run/resume/model.state-00000001.safetensors")]
        )
        == 0
    )
    for name in ("resume/model.state-00000003.safetensors", "weights/model.safetensors"):
        assert (root / "run" / name).read_bytes() == checkpoints[name]
    # A reference content edit must invalidate its associated target record, even at the same path.
    if count:
        with Image.new("RGB", (35, 49), (11, 23, 59)) as image:
            image.save(root / "ref0/a.png")
        assert command(["cache", str(train)]) == 0
        changed = load_latent_manifest(root / "cache")
        assert changed.entries[0] != manifest.entries[0] and changed.entries[1] == manifest.entries[1]
        assert (
            command(["train", str(train), "--resume", str(root / "run/resume/model.state-00000001.safetensors")]) == 2
        )
    config = load_training_config(train)
    assert len(config.dataset.groups["main"].reference_sources) == count
