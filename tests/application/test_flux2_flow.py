"""Exercise FLUX.2 local cache, full/adapters, resumed training and public generated previews."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    PixtralImageProcessor,
    PixtralProcessor,
    Qwen2Tokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from tests.support.text_assets import main_with_local_tokenizers
from unitunner.artifacts.tensors import save_tensors
from unitunner.config.loading import load_training_config
from unitunner.data.cache.storage import load_latent_manifest, read_latent_record
from unitunner.data.media.image import read_image
from unitunner.models.flux2.config.model import Flux2Config
from unitunner.models.flux2.config.vae import Flux2VAEConfig
from unitunner.models.flux2.loading import load_flux2_vae, save_flux2_vae
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.models.flux2.vae import Flux2VAE


def command(arguments):
    return main_with_local_tokenizers([*arguments, "--device", os.environ.get("UNITUNNER_TEST_DEVICE", "cpu")])


@pytest.fixture
def assets(tmp_path, version):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(921)
    text = tmp_path / "text"
    text.mkdir()
    vocabulary = {value: index for index, value in enumerate(sorted(ByteLevel.alphabet()))}
    vocabulary.update({"<|endoftext|>": 256, "<|im_start|>": 257, "<|im_end|>": 258})
    tokenizer = Qwen2Tokenizer(vocab=vocabulary, merges=[], pad_token="<|endoftext|>")
    if version == "dev":
        template = "{% for message in messages %}{{ message['role'] + ': ' }}{% for content in message['content'] %}{{ content['text'] }}{% endfor %}{{ '\\n' }}{% endfor %}"
        processor = PixtralProcessor(
            image_processor=PixtralImageProcessor(), tokenizer=tokenizer, chat_template=template
        )
        processor.save_pretrained(text)
        config = Mistral3Config(
            text_config={
                "model_type": "mistral",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "vocab_size": 259,
            },
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 8,
                "patch_size": 2,
            },
            image_token_index=259,
        )
        full = Mistral3ForConditionalGeneration(config).eval().requires_grad_(False)
    else:
        tokenizer.chat_template = "{{ '<|im_start|>user\\n' + messages[0]['content'] + '<|im_end|>' }}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}{% if enable_thinking %}{{ '<think>' }}{% endif %}"
        tokenizer.save_pretrained(text)
        full = (
            Qwen3ForCausalLM(
                Qwen3Config(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=28,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    vocab_size=259,
                )
            )
            .eval()
            .requires_grad_(False)
        )
    full.save_pretrained(text)
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


def configs(root, algorithm, version, variant, count):
    model = Flux2Transformer(
        Flux2Config(
            version=version,
            variant=variant,
            hidden_size=32,
            num_heads=2,
            depth=1,
            depth_single_blocks=1,
            axes_dim=(4, 4, 4, 4),
            in_channels=8,
            context_in_dim=48,
        )
    )
    save_tensors(model.state_dict(), root / "raw.safetensors")
    (root / "topology.json").write_text(model.config.model_dump_json())
    assert (
        main_with_local_tokenizers(
            [
                "weights",
                "import",
                str(root / "raw.safetensors"),
                str(root / "model.safetensors"),
                "--family",
                "flux2",
                "--kind",
                "transformer",
                "--topology",
                str(root / "topology.json"),
                "--source-format",
                "native",
            ]
        )
        == 0
    )
    header = f'''[model]
family = "flux2"
version = "{version}"
variant = "{variant}"
weights = "{root}/model.safetensors"
vae = "{root}/vae.safetensors"
text_encoder = "{root}/text"
'''
    references = [str(root / f"ref{index}") for index in range(count)]
    train = (
        header
        + f'''[method]
{"guidance = 2.0" if version == "dev" else ""}
[dataset.groups.main]
source = "{root}/images"
cache_directory = "{root}/cache"
resolution = [32,16]
batch_size = 2
reference_sources = {json.dumps(references)}
[train]
parameter_mode = "{"full" if algorithm == "full" else "adapter"}"
max_updates = 3
accumulation_steps = {4 if version != "dev" and variant == "distilled" else 2}
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
steps = {4 if version != "dev" and variant == "distilled" else 2}
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
    "algorithm,version,variant,count",
    [
        ("lora", "dev", "distilled", 2),
        ("loha", "klein-4b", "base", 1),
        ("lokr", "klein-9b", "distilled", 0),
        ("full", "dev", "distilled", 2),
        ("full", "klein-4b", "base", 1),
    ],
)
def test_flux2_cache_train_generate_preview_and_resume(assets, algorithm, version, variant, count):
    root = assets
    train = configs(root, algorithm, version, variant, count)
    assert command(["cache", str(train)]) == 0
    manifest = load_latent_manifest(root / "cache")
    assert manifest.family == "flux2" and manifest.variant == variant and manifest.version == version
    records = [read_latent_record(root / "cache", entry, device=torch.device("cpu")) for entry in manifest.entries]
    assert all(len(record.references) == count for record in records)
    assert all(record.caption.shape == (512, 48) for record in records)
    selected = torch.device(os.environ.get("UNITUNNER_TEST_DEVICE", "cpu"))
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
    wrong = root / "wrong-version.toml"
    wrong_version = "klein-4b" if version != "klein-4b" else "klein-9b"
    wrong.write_text(train.read_text().replace(f'version = "{version}"', f'version = "{wrong_version}"'))
    assert command(["train", str(wrong)]) == 2
    config = load_training_config(train)
    assert len(config.dataset.groups["main"].reference_sources) == count
