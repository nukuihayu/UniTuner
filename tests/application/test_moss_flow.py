"""Exercise local Qwen3.5 final features and native MOSS image application lifecycles."""

import os
from importlib import import_module

import numpy as np
import pytest
import torch

from tests.support.text_assets import local_text_config, main_with_local_tokenizers
from unitunner.models.flux2.config.vae import Flux2VAEConfig
from unitunner.models.flux2.loading import save_flux2_vae
from unitunner.models.flux2.vae import Flux2VAE
from unitunner.models.moss_image.config.model import MossImageConfig
from unitunner.models.moss_image.loading import save_moss_image
from unitunner.models.moss_image.text.encoder import load_moss_image_text
from unitunner.models.moss_image.transformer import MossImageTransformer

Image = pytest.importorskip("PIL.Image")
transformers = pytest.importorskip("transformers")
pytest.importorskip("cv2")


def run_command(arguments):
    """Run the real CLI on the explicitly selected application test device."""
    return main_with_local_tokenizers([*arguments, "--device", os.environ.get("UNITUNNER_TEST_DEVICE", "cpu")])


@pytest.fixture
def moss_assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(91)
    text_path = tmp_path / "text"
    text_path.mkdir()
    alphabet = sorted(import_module("tokenizers.pre_tokenizers").ByteLevel.alphabet())
    vocabulary = {value: index for index, value in enumerate(alphabet)}
    vocabulary.update({"<|endoftext|>": 256, "<|im_start|>": 257, "<|im_end|>": 258})
    tokenizer = transformers.Qwen2Tokenizer(vocab=vocabulary, merges=[], pad_token="<|endoftext|>")
    tokenizer.chat_template = "{{ '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n<|im_start|>assistant\n' }}{% if enable_thinking %}{{ '<think>\n' }}{% endif %}"
    tokenizer.save_pretrained(text_path)
    encoder = transformers.Qwen3_5TextModel(
        transformers.Qwen3_5TextConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=259,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 0],
            },
        )
    )
    encoder.save_pretrained(text_path)
    vae = Flux2VAE(
        Flux2VAEConfig(
            base_channels=8, channel_multipliers=(1, 1, 1, 1), layers_per_block=1, latent_channels=2, norm_groups=4
        )
    )
    save_flux2_vae(vae, tmp_path / "vae.safetensors")
    model = MossImageTransformer(
        MossImageConfig(dim=32, ffn_dim=48, num_heads=2, num_layers=2, text_dim=16, freq_dim=8, latent_channels=8)
    )
    save_moss_image(model, tmp_path / "model.safetensors")
    source = tmp_path / "images"
    source.mkdir()
    with Image.fromarray(np.arange(16 * 32 * 4, dtype=np.uint8).reshape(16, 32, 4)) as image:
        image.save(source / "a.png")
        image.save(source / "b.png")
    (source / "a.txt").write_text("first image")
    (source / "b.txt").write_text("a much longer second image caption")
    yield tmp_path
    torch.set_num_threads(previous)


@pytest.mark.parametrize("single_file", [False, True])
def test_moss_qwen35_unpadded_final_layer_local_reload(moss_assets, single_file):
    encoder = load_moss_image_text(
        moss_assets / "text/model.safetensors" if single_file else moss_assets / "text",
        text_config=local_text_config(moss_assets / "text"),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    prompts = ["short", "a longer prompt", ""]
    before = torch.get_rng_state().clone()
    batch = encoder.encode(prompts)
    assert batch.mask is not None
    for index, prompt in enumerate(prompts):
        formatted = encoder.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        assert isinstance(formatted, str)
        inputs = encoder.tokenizer(formatted, return_tensors="pt")
        with torch.no_grad():
            expected = encoder.model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        torch.testing.assert_close(
            batch.features[index, batch.mask[index]], expected.last_hidden_state[0], atol=0, rtol=0
        )
        assert not torch.equal(expected.hidden_states[-2], expected.last_hidden_state)
        torch.testing.assert_close(encoder.encode_prompt(prompt).features, expected.last_hidden_state, atol=0, rtol=0)
    assert not batch.features.requires_grad and not encoder.model.training
    assert all(not value.is_meta for value in encoder.model.buffers())
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)


def write_configs(root, algorithm, variant="base"):
    model = f'''[model]
family = "moss-image"
variant = "{variant}"
weights = "{root}/model.safetensors"
vae = "{root}/vae.safetensors"
text_encoder = "{root}/text"
'''
    adapter = (
        ""
        if algorithm == "full"
        else f'''[train.adapter]
algorithm = "{algorithm}"
rank = 2
alpha = 3.0
'''
    )
    training = root / "train.toml"
    training.write_text(
        model
        + f'''[dataset.groups.main]
source = "{root}/images"
{f'reference_sources = ["{root}/images"]' if variant == "edit" else ""}
cache_directory = "{root}/cache"
resolution = [32,16]
batch_size = 2
[train]
parameter_mode = "{"full" if algorithm == "full" else "adapter"}"
max_updates = 3
accumulation_steps = 2
lr_schedule = "cosine"
warmup_updates = 1
'''
        + adapter
        + f'''[execution]
recompute = "block"
[output]
max_resume_checkpoints = 0
directory = "{root}/run"
save_every_updates = 1
'''
    )
    generation = root / "generate.toml"
    generation.write_text(
        f'output = "{root}/sample.png"\n'
        + model
        + f"""[sampling]
{f'reference_images = ["{root}/images/a.png"]' if variant == "edit" else ""}
prompt = "first image"
width = 32
height = 16
steps = 3
guidance = 3.0
uncond_skip_layers = [0]
slg_layers = [1]
slg_scale = 0.5
slg_start = 0.25
slg_stop = 0.75
"""
    )
    return training, generation


@pytest.mark.parametrize("variant", ["base", "edit"])
@pytest.mark.parametrize("algorithm", ["full", "lora", "loha", "lokr"])
def test_moss_real_cache_train_resume_generate_and_preview_isolation(moss_assets, algorithm, variant):
    from unitunner.data.cache.storage import load_latent_manifest, read_latent_record

    root = moss_assets
    training, generation = write_configs(root, algorithm, variant)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert run_command(["cache", str(training)]) == 0
    manifest = load_latent_manifest(root / "cache")
    assert manifest.family == "moss-image"
    captions = [
        read_latent_record(root / "cache", entry, device=torch.device("cpu")).caption for entry in manifest.entries
    ]
    assert captions[0].shape[0] != captions[1].shape[0]
    assert run_command(["cache", str(training)]) == 0
    assert load_latent_manifest(root / "cache") == manifest
    assert run_command(["train", str(training)]) == 0
    expected = {index: (root / f"run/resume/model.state-{index:08d}.safetensors").read_bytes() for index in range(1, 4)}
    exported = (root / "run/weights/model.safetensors").read_bytes()
    if algorithm == "full":
        generation.write_text(
            generation.read_text().replace(str(root / "model.safetensors"), str(root / "run/weights/model.safetensors"))
        )
    else:
        generation.write_text(
            generation.read_text()
            + f'''[adapter]
weights = "{root}/run/weights/model.safetensors"
[adapter.config]
algorithm = "{algorithm}"
rank = 2
alpha = 1.0
'''
        )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert run_command(["generate", str(generation)]) == 0
    png = (root / "sample.png").read_bytes()
    assert run_command(["generate", str(generation)]) == 0
    assert (root / "sample.png").read_bytes() == png
    with Image.open(root / "sample.png") as image:
        assert image.size == (32, 16)
    plain = training.read_text()
    training.write_text(
        plain
        + f"""[preview]
every_updates = 1
at_start = true
[[preview.samples]]
{f'reference_images = ["{root}/images/a.png"]' if variant == "edit" else ""}
prompt = "preview"
width = 32
height = 16
steps = 2
uncond_skip_layers = []
"""
    )
    assert run_command(["train", str(training)]) == 0
    for index in range(1, 4):
        assert (root / f"run/resume/model.state-{index:08d}.safetensors").read_bytes() == expected[index]
    assert (root / "run/weights/model.safetensors").read_bytes() == exported
    assert len(list((root / "run/previews").glob("*.png"))) == 4
    preview = (root / "run/previews/step-00000003-00.png").read_bytes()
    assert (
        run_command(["train", str(training), "--resume", str(root / "run/resume/model.state-00000001.safetensors")])
        == 0
    )
    for index in (2, 3):
        assert (root / f"run/resume/model.state-{index:08d}.safetensors").read_bytes() == expected[index]
    assert (root / "run/previews/step-00000003-00.png").read_bytes() == preview
    (root / "images/a.txt").write_text("changed caption")
    assert run_command(["cache", str(training)]) == 0
    changed = load_latent_manifest(root / "cache")
    assert changed.entries[0] != manifest.entries[0] and changed.entries[1] == manifest.entries[1]
    assert (
        run_command(["train", str(training), "--resume", str(root / "run/resume/model.state-00000001.safetensors")])
        == 2
    )


def test_moss_multimodal_language_only_loader_and_strict_preflight(moss_assets):
    from unitunner.artifacts.tensors import iter_tensors, save_tensors

    root = moss_assets
    text = load_moss_image_text(
        root / "text", text_config=local_text_config(root / "text"), device=torch.device("cpu"), dtype=torch.float32
    )
    directory = root / "multimodal"
    directory.mkdir()
    text.tokenizer.save_pretrained(directory)
    transformers.Qwen3_5Config(text_config=text.model.config.to_dict()).save_pretrained(directory)
    state = {"model.language_model." + name: value for name, value in iter_tensors(root / "text/model.safetensors")}
    state["model.visual.unused"] = torch.ones(1)
    path = directory / "model.safetensors"
    save_tensors(state, path)
    loaded = load_moss_image_text(
        directory, text_config=local_text_config(directory), device=torch.device("cpu"), dtype=torch.float32
    )
    torch.testing.assert_close(
        loaded.encode_prompt("same").features, text.encode_prompt("same").features, atol=0, rtol=0
    )
    assert all(not name.startswith("visual") for name in loaded.model.state_dict())
    for prefix in ("language_model.", "model.", ""):
        save_tensors({prefix + k: v for k, v in text.model.state_dict().items()}, path)
        alternate = load_moss_image_text(
            directory, text_config=local_text_config(directory), device=torch.device("cpu"), dtype=torch.float32
        )
        torch.testing.assert_close(
            alternate.encode_prompt("same").features, text.encode_prompt("same").features, atol=0, rtol=0
        )
    for invalid in (
        {name: value for name, value in state.items() if name != "model.language_model.embed_tokens.weight"},
        {**state, "unrecognized.weight": torch.ones(1)},
        {**state, "model.language_model.norm.weight": torch.full((16,), float("nan"))},
    ):
        save_tensors(invalid, path)
        with pytest.raises(ValueError, match="MOSS text"):
            load_moss_image_text(
                directory, text_config=local_text_config(directory), device=torch.device("cpu"), dtype=torch.float32
            )
