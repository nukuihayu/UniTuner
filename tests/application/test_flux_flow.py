"""Exercise real FLUX variant cache/train/resume with native local model assets."""

import os

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file
from tokenizers.pre_tokenizers import ByteLevel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5Tokenizer

from tests.support.assertions import require_value
from tests.support.text_assets import main_with_local_tokenizers
from unitunner.artifacts.tensors import TensorReader
from unitunner.data.cache.storage import load_latent_manifest, read_latent_record
from unitunner.models.flux.config.model import FluxConfig
from unitunner.models.flux.config.vae import FluxVAEConfig
from unitunner.models.flux.loading import save_flux, save_flux_vae
from unitunner.models.flux.transformer import FluxTransformer
from unitunner.models.flux.vae import FluxVAE


def run(arguments):
    """Execute the real CLI on the caller-selected application device."""
    return main_with_local_tokenizers([*arguments, "--device", os.environ.get("UNITUNNER_TEST_DEVICE", "cpu")])


@pytest.fixture
def assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(191)
    t5_path, clip_path = tmp_path / "t5", tmp_path / "text_encoder"
    vocab = [("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -1.0)] + [
        (c, -2.0) for c in "abcdefghijklmnopqrstuvwxyz"
    ]
    T5Tokenizer(vocab=vocab, extra_ids=0).save_pretrained(t5_path)
    T5EncoderModel(
        T5Config(
            vocab_size=len(vocab),
            d_model=16,
            d_kv=8,
            d_ff=32,
            num_layers=1,
            num_heads=2,
            feed_forward_proj="gated-gelu",
        )
    ).save_pretrained(t5_path)
    alphabet = sorted(ByteLevel.alphabet())
    vocabulary = {c: i for i, c in enumerate(alphabet)}
    vocabulary.update({c + "</w>": i + 256 for i, c in enumerate(alphabet)})
    vocabulary.update({"<|startoftext|>": 512, "<|endoftext|>": 513})
    CLIPTokenizer(vocab=vocabulary, merges=[]).save_pretrained(clip_path)
    CLIPTextModel(
        CLIPTextConfig(
            vocab_size=514,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=77,
            bos_token_id=512,
            eos_token_id=513,
            pad_token_id=513,
            hidden_act="quick_gelu",
        )
    ).save_pretrained(clip_path)
    vae = FluxVAE(FluxVAEConfig(base_channels=8, channel_multipliers=(1, 1, 1, 1), norm_groups=4, layers_per_block=1))
    save_flux_vae(vae, tmp_path / "vae.safetensors")
    for directory in ("images", "conditions"):
        (tmp_path / directory).mkdir()
    rgb = np.arange(16 * 32 * 3, dtype=np.uint8).reshape(16, 32, 3)
    alpha = (np.indices((16, 32))[1] >= 16).astype(np.uint8) * 255
    for name in ("a", "b"):
        with Image.fromarray(rgb if name == "a" else 255 - rgb) as image:
            image.save(tmp_path / f"images/{name}.png")
        with Image.fromarray(np.concatenate((255 - rgb, alpha[..., None]), axis=-1)) as image:
            image.save(tmp_path / f"conditions/{name}.png")
    (tmp_path / "images/a.txt").write_text("first image")
    (tmp_path / "images/b.txt").write_text("a much longer second caption")
    yield tmp_path
    torch.set_num_threads(previous)


def write_training(root, variant, algorithm, *, residency="resident"):
    model = FluxTransformer(
        FluxConfig(
            variant=variant,
            in_channels=384 if variant == "fill" else 64,
            out_channels=64,
            vec_in_dim=16,
            context_in_dim=16,
            hidden_size=32,
            num_heads=2,
            depth=1,
            depth_single_blocks=1,
            axes_dim=(4, 6, 6),
            mlp_ratio=2.0,
        )
    )
    save_flux(model, root / "model.safetensors")
    text = f'''[model]
family = "flux"
assets = "{root.as_posix()}"
variant = "{variant}"
weights = "{root.as_posix()}/model.safetensors"
vae = "{root.as_posix()}/vae.safetensors"
text_encoder = "{root.as_posix()}/t5"
[model.text]
max_length = {512 if variant == "dev" else 64}
[dataset.groups.main]
source = "{root.as_posix()}/images"
cache_directory = "{root.as_posix()}/cache"
resolution = [32,16]
batch_size = 2
'''
    if variant != "dev":
        text += f'reference_sources = ["{root.as_posix()}/conditions"]\n'
    text += f'''[train]
max_updates = 3
seed = 73
accumulation_steps = 2
lr_schedule = "cosine"
warmup_updates = 1
[train.adapter]
algorithm = "{algorithm}"
rank = 2
alpha = 3.0
[method]
guidance = 2.5
[execution]
residency = "{residency}"
recompute = "block"
[output]
max_resume_checkpoints = 0
directory = "{root.as_posix()}/run"
save_every_updates = 1
'''
    if algorithm == "full":
        text = text.replace('[train.adapter]\nalgorithm = "full"\nrank = 2\nalpha = 3.0\n', 'parameter_mode = "full"\n')
    path = root / "train.toml"
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    ("variant", "algorithm"),
    [
        ("dev", "lora"),
        ("kontext", "loha"),
        ("fill", "lokr"),
        ("dev", "full"),
        ("kontext", "full"),
        ("fill", "full"),
    ],
)
def test_flux_real_variant_cache_training_resume_and_content_identity(assets, variant, algorithm, residency="resident"):
    root = assets
    training = write_training(root, variant, algorithm, residency=residency)
    rng = torch.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert run(["cache", str(training)]) == 0
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    manifest = load_latent_manifest(root / "cache")
    assert manifest.family == "flux" and manifest.variant == variant
    records = [read_latent_record(root / "cache", entry, device=torch.device("cpu")) for entry in manifest.entries]
    assert all(
        record.caption.shape == (512 if variant == "dev" else 64, 16) and require_value(record.pooled).shape == (16,)
        for record in records
    )
    if variant == "dev":
        assert all(record.control_latents is None for record in records)
    else:
        assert all(record.control_latents is not None for record in records)
    if variant == "fill":
        assert all(require_value(record.fill_mask).shape == (64, 2, 4) for record in records)
    assert run(["cache", str(training)]) == 0
    assert load_latent_manifest(root / "cache") == manifest
    independent = root / "independent.toml"
    independent.write_text(training.read_text().replace((root / "cache").as_posix(), (root / "cache-copy").as_posix()))
    assert run(["cache", str(independent)]) == 0
    assert load_latent_manifest(root / "cache-copy") == manifest
    for entry in manifest.entries:
        assert (root / "cache" / entry.file).read_bytes() == (root / "cache-copy" / entry.file).read_bytes()
    assert run(["train", str(training)]) == 0
    expected = (root / "run/resume/model.state-00000003.safetensors").read_bytes()
    exported = (root / "run/weights/model.safetensors").read_bytes()
    saved = root / "run/resume/model.state-00000001.safetensors"
    assert run(["train", str(training), "--resume", str(saved)]) == 0
    assert (root / "run/resume/model.state-00000003.safetensors").read_bytes() == expected
    assert (root / "run/weights/model.safetensors").read_bytes() == exported
    with TensorReader(root / "run/weights/model.safetensors") as reader:
        if algorithm != "full":
            assert reader.metadata()["base_model_version"] == f"flux_{variant}"
    assert load_file(root / "run/weights/model.safetensors")
    # Reload actual published training adapters through the public inference path.
    sampling = """[sampling]
prompt = "preview"
width = 32
height = 16
steps = 2
seed = 791
"""
    if variant != "dev":
        sampling += f'conditioning_image = "{root.as_posix()}/conditions/a.png"\n'
    if variant == "kontext":
        sampling += 'conditioning_resize = "original"\n'
    generation = root / "generate.toml"
    model_section = training.read_text().split("[dataset.groups.main]")[0]
    generation.write_text(
        f'output = "{root.as_posix()}/sample.png"\n'
        + model_section
        + sampling
        + f'''[adapter]
weights = "{root.as_posix()}/run/weights/model.safetensors"
[adapter.config]
algorithm = "{algorithm}"
rank = 2
alpha = 1.0
'''
    )
    if algorithm == "full":
        generation.write_text(
            generation.read_text()
            .split("[adapter]")[0]
            .replace(
                f'weights = "{root.as_posix()}/model.safetensors"',
                f'weights = "{root.as_posix()}/run/weights/model.safetensors"',
            )
        )
    before_generation = torch.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert run(["generate", str(generation)]) == 0
    torch.testing.assert_close(torch.get_rng_state(), before_generation, atol=0, rtol=0)
    png = (root / "sample.png").read_bytes()
    assert run(["generate", str(generation)]) == 0
    assert (root / "sample.png").read_bytes() == png
    with Image.open(root / "sample.png") as image:
        assert image.size == (32, 16)
    plain = training.read_text()
    expected_states = {
        index: (root / f"run/resume/model.state-{index:08d}.safetensors").read_bytes() for index in range(1, 4)
    }
    training.write_text(
        plain
        + "[preview]\nevery_updates = 1\nat_start = true\n"
        + sampling.replace("[sampling]", "[[preview.samples]]")
    )
    assert run(["train", str(training)]) == 0
    for index in range(1, 4):
        assert (root / f"run/resume/model.state-{index:08d}.safetensors").read_bytes() == expected_states[index]
    assert (root / "run/weights/model.safetensors").read_bytes() == exported
    assert len(list((root / "run/previews").glob("*.png"))) == 4
    assert (root / "run/previews/step-00000003-00.png").read_bytes() == png
    assert run(["train", str(training), "--resume", str(saved)]) == 0
    for index in (2, 3):
        assert (root / f"run/resume/model.state-{index:08d}.safetensors").read_bytes() == expected_states[index]
    assert (root / "run/previews/step-00000003-00.png").read_bytes() == png
    (root / "images/a.txt").write_text("changed caption")
    assert run(["cache", str(training)]) == 0
    changed = load_latent_manifest(root / "cache")
    assert changed.entries[0] != manifest.entries[0] and changed.entries[1] == manifest.entries[1]
    changed_record = read_latent_record(root / "cache", changed.entries[0], device=torch.device("cpu"))
    torch.testing.assert_close(changed_record.latents, records[0].latents, atol=0, rtol=0)
    assert run(["train", str(training), "--resume", str(saved)]) == 2
    if variant != "dev":
        with Image.open(root / "conditions/a.png") as image:
            values = np.array(image)
        values[0, 0, 0] ^= 127
        with Image.fromarray(values) as image:
            image.save(root / "conditions/a.png")
        assert run(["cache", str(training)]) == 0
        condition_changed = load_latent_manifest(root / "cache")
        assert condition_changed.entries[0] != changed.entries[0]
        assert condition_changed.entries[1] == changed.entries[1]
        (root / "conditions/extra.png").write_bytes((root / "conditions/a.png").read_bytes())
        manifest_bytes = (root / "cache/manifest.json").read_bytes()
        assert run(["cache", str(training)]) == 2
        assert (root / "cache/manifest.json").read_bytes() == manifest_bytes
