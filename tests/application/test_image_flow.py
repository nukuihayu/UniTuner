"""Run cache/train/generate commands using real small native artifacts and local Qwen assets."""

import json
from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
import torch

from tests.support.text_assets import local_text_config, main_with_local_tokenizers
from unitunner.artifacts.tensors import iter_tensors, save_tensors
from unitunner.data.cache.storage import load_latent_manifest
from unitunner.models.features.convnext import DinoConvNeXt
from unitunner.models.z_image.config.model import ZImageConfig
from unitunner.models.z_image.config.vae import ZImageVAEConfig
from unitunner.models.z_image.loading import load_z_image, load_z_image_vae, save_z_image, save_z_image_vae
from unitunner.models.z_image.text.encoder import load_z_image_text
from unitunner.models.z_image.transformer import ZImageTransformer
from unitunner.models.z_image.vae import ZImageVAE
from unitunner.training.methods.restoration import RestorationMethod

Image = pytest.importorskip("PIL.Image")
transformers = pytest.importorskip("transformers")
cv2 = pytest.importorskip("cv2")


@pytest.fixture
def image_assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(91)
    text_path = tmp_path / "text"
    text_path.mkdir()
    alphabet = sorted(import_module("tokenizers.pre_tokenizers").ByteLevel.alphabet())
    vocabulary = {value: index for index, value in enumerate(alphabet)}
    vocabulary.update({"<|endoftext|>": 256, "<|im_start|>": 257, "<|im_end|>": 258})
    (text_path / "vocab.json").write_text(json.dumps(vocabulary))
    (text_path / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = transformers.Qwen2Tokenizer(vocab=vocabulary, merges=[], pad_token="<|endoftext|>")
    tokenizer.chat_template = (
        "{{ '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n<|im_start|>assistant\n' }}"
    )
    tokenizer.save_pretrained(text_path)
    encoder = transformers.Qwen3Model(
        transformers.Qwen3Config(
            vocab_size=259,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
    )
    encoder.save_pretrained(text_path)
    vae = ZImageVAE(ZImageVAEConfig(block_channels=(8, 8, 8, 8), latent_channels=2, norm_groups=4, layers_per_block=1))
    save_z_image_vae(vae, tmp_path / "vae.safetensors")
    model = ZImageTransformer(
        ZImageConfig(
            in_channels=2,
            dim=24,
            num_layers=1,
            num_refiner_layers=1,
            num_heads=4,
            num_kv_heads=2,
            caption_dim=8,
            timestep_hidden_dim=32,
            axes_dims=(2, 2, 2),
            axes_lengths=(128, 32, 32),
        )
    )
    save_z_image(model, tmp_path / "model.safetensors")
    source = tmp_path / "images"
    source.mkdir()
    with Image.fromarray(np.arange(16 * 32 * 4, dtype=np.uint8).reshape(16, 32, 4)) as image:
        image.save(source / "a.png")
        image.save(source / "b.png")
    (source / "a.txt").write_text("first image")
    (source / "b.txt").write_text("second image")
    model_section = f'''[model]
family = "z-image"
variant = "base"
weights = "{tmp_path.as_posix()}/model.safetensors"
vae = "{tmp_path.as_posix()}/vae.safetensors"
text_encoder = "{text_path.as_posix()}"
'''
    training = tmp_path / "train.toml"
    training.write_text(
        model_section
        + f'''[dataset.groups.main]
source = "{source.as_posix()}"
cache_directory = "{tmp_path.as_posix()}/cache"
resolution = [32,16]
batch_size = 2
[train]
parameter_mode = "full"
max_updates = 2
accumulation_steps = 2
[output]
max_resume_checkpoints = 0
directory = "{tmp_path.as_posix()}/run"
'''
    )
    generation = tmp_path / "generate.toml"
    generation.write_text(
        f'output = "{tmp_path.as_posix()}/sample.png"\n'
        + model_section
        + """[sampling]
prompt = "first image"
negative_prompt = ""
width = 32
height = 16
steps = 3
guidance = 4.0
"""
    )
    yield training, generation
    torch.set_num_threads(previous)


def test_multiscale_cache_and_training_use_every_scale(image_assets):
    training, _ = image_assets
    root = training.parent
    training.write_text(training.read_text().replace("resolution = [32,16]", "resolution = [[64,32],[32,16]]"))
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    manifests = []
    for width, height in ((64, 32), (32, 16)):
        directory = root / "cache" / f"{width}x{height}"
        manifest = load_latent_manifest(directory)
        assert len(manifest.entries) == 2
        values = dict(iter_tensors(directory / manifest.entries[0].file))
        assert values["latents"].shape[-2:] == (height // 8, width // 8)
        manifests.append(manifest)
    assert manifests[0] != manifests[1]
    assert main_with_local_tokenizers(["train", str(training)]) == 0
    assert (root / "run/resume/model.state-00000002.safetensors").is_file()


def test_mid_timestep_command_needs_no_cache_text_or_denoiser_and_preserves_rng(image_assets, capsys):
    """Run actual sampled VAE calibration twice with unavailable training-only assets."""
    training, _ = image_assets
    root = training.parent
    document = (
        training.read_text()
        .replace(
            f'weights = "{root.as_posix()}/model.safetensors"',
            f'weights = "{root.as_posix()}/absent-denoiser.safetensors"',
        )
        .replace(f'text_encoder = "{root.as_posix()}/text"', f'text_encoder = "{root.as_posix()}/absent-text"')
    )
    document = document.replace(
        "[dataset.groups.main]",
        f'[dataset.groups.main]\nkind = "restoration"\nlow_quality_source = "{root.as_posix()}/images"',
    )
    document += '\n[method]\nkind = "restoration"\n[method.restoration]\nposterior = "sample"\n[losses]\ndv3d = 0.0\ngan = 0.0\nfdl = 0.0\n'
    training.write_text(document)
    assert not (root / "cache").exists()

    before = torch.get_rng_state()
    outputs = []
    for _ in range(2):
        assert main_with_local_tokenizers(["mid-timestep", str(training)]) == 0
        outputs.append(json.loads(capsys.readouterr().out))
    assert outputs[0] == outputs[1]
    result = outputs[0]
    assert result["pairs"] == 2 and result["posterior"] == "sample"
    assert 1 <= result["midtimestep"] <= 998 and 0 < result["sigma"] < 1
    assert np.isfinite(result["mean_snr_error"])
    assert torch.equal(before, torch.get_rng_state())
    assert not (root / "cache").exists()


def test_minimal_vae_calibration_cli_needs_no_training_or_cache_config(image_assets):
    training, _ = image_assets
    root = training.parent
    groups = {
        "main": {"source": str(root / "images"), "low_quality_source": str(root / "images"), "resolution": [32, 16]}
    }
    assert (
        main_with_local_tokenizers(
            [
                "mid-timestep",
                "--model.family",
                "z-image",
                "--model.vae",
                str(root / "vae.safetensors"),
                "--dataset.groups",
                json.dumps(groups),
                "--calibration-pairs",
                "1",
            ]
        )
        == 0
    )


@pytest.mark.parametrize("single_file", [False, True])
def test_real_image_cache_training_generation_and_content_invalidation(image_assets, single_file, capsys):
    training, generation = image_assets
    root = training.parent
    if single_file:
        for document in (training, generation):
            content = document.read_text().replace(
                f'text_encoder = "{root.as_posix()}/text"',
                f'text_encoder = "{root.as_posix()}/text/model.safetensors"',
            )
            content += f'\n[model.text.tokenizer]\nsource = "{root.as_posix()}/text"\noffline = true\n'
            document.write_text(content)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert main_with_local_tokenizers(["cache", str(training)]) == 0
    manifest = load_latent_manifest(root / "cache")
    assert len(manifest.entries) == 2
    assert manifest.entries[0].file != manifest.entries[1].file
    # Identical source pixels with different captions must never share one cache identity.
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    assert load_latent_manifest(root / "cache") == manifest
    capsys.readouterr()
    assert (
        main_with_local_tokenizers(["train", str(training), "--parameter-mode", "full", "--include-layers", "layers.*"])
        == 0
    )
    assert (root / "run/resume/model.state-00000002.safetensors").is_file()
    captured = capsys.readouterr()
    assert "[train] update=1/2" in captured.err
    assert "accumulation:" not in captured.err
    assert "loss=" in captured.err and "lr=" in captured.err
    assert "[train] update=2/2" in captured.err
    assert [json.loads(line)["update"] for line in captured.out.splitlines()] == [1, 2]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert main_with_local_tokenizers(["generate", str(generation)]) == 0
    before = (root / "sample.png").read_bytes()
    assert main_with_local_tokenizers(["generate", str(generation)]) == 0
    assert (root / "sample.png").read_bytes() == before
    with Image.open(root / "sample.png") as image:
        assert image.size == (32, 16) and image.mode == "RGB"
    (root / "images/a.txt").write_text("updated caption")
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    changed = load_latent_manifest(root / "cache")
    assert changed.entries[0] != manifest.entries[0]
    assert changed.entries[1] == manifest.entries[1]
    last_manifest = (root / "cache/manifest.json").read_bytes()
    (root / "images/z.png").write_bytes(b"broken image")
    assert main_with_local_tokenizers(["cache", str(training)]) == 2
    assert (root / "cache/manifest.json").read_bytes() == last_manifest


def test_cache_command_reencodes_corrupt_existing_cache_with_visible_progress(image_assets, monkeypatch, capsys):
    training, _ = image_assets
    root = training.parent
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    manifest = load_latent_manifest(root / "cache")
    (root / "cache" / manifest.entries[0].file).write_bytes(b"corrupt old record")
    (root / "cache/manifest.json").write_text("invalid old manifest")
    stale = root / "cache" / ("f" * 64 + ".safetensors")
    stale.write_bytes(b"obsolete")
    (root / "cache/notes.txt").write_text("keep")
    from unitunner.models.z_image.text.encoder import ZImageTextEncoder

    calls = []
    original = ZImageTextEncoder.encode

    def encode(self, prompts):
        calls.extend(prompts)
        return original(self, prompts)

    monkeypatch.setattr(ZImageTextEncoder, "encode", encode)
    capsys.readouterr()
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"cached": 2}
    assert "mode=rebuild" in captured.err
    assert "Loading z image text" in captured.err
    assert "[cache] Complete" in captured.err
    assert "100%" not in captured.err
    assert "Published fresh cache" in captured.err
    assert len(calls) == 2
    assert load_latent_manifest(root / "cache") == manifest
    assert not stale.exists()
    assert (root / "cache/notes.txt").read_text() == "keep"

    before = {path.name: path.read_bytes() for path in (root / "cache").iterdir()}
    calls.clear()

    def fail_second(self, prompts):
        if calls:
            raise ValueError("second image encoding failed")
        return encode(self, prompts)

    monkeypatch.setattr(ZImageTextEncoder, "encode", fail_second)
    assert main_with_local_tokenizers(["cache", str(training)]) == 2
    assert {path.name: path.read_bytes() for path in (root / "cache").iterdir()} == before


def test_global_prompts_without_caption_files_cache_train_and_validate_restoration(image_assets):
    """Run independent cache CLI, two prompt groups and real training without TXT files."""
    from unitunner.config.loading import load_training_config
    from unitunner.data.datasets.restoration import index_restoration_pairs
    from unitunner.data.media.image import discover_images

    training, _ = image_assets
    root = training.parent
    for caption in (root / "images").glob("*.txt"):
        caption.unlink()
    training.write_text(
        training.read_text()
        + f'''
[dataset.defaults]
prompt = "shared portrait"
caption_extension = ".txt"
[dataset.groups.empty]
source = "{(root / "images").as_posix()}"
cache_directory = "{(root / "empty-cache").as_posix()}"
resolution = [32, 16]
prompt = ""
'''
    )
    config = load_training_config(training)
    model = config.model.model_dump(mode="json", exclude={"weights"})
    dataset = config.dataset.model_dump(mode="json")
    flags = ["--model", json.dumps(model), "--dataset", json.dumps(dataset)]
    assert main_with_local_tokenizers(["cache", *flags]) == 0
    first = load_latent_manifest(root / "cache")
    empty = load_latent_manifest(root / "empty-cache")
    assert len(first.entries) == len(empty.entries) == 2
    assert first.entries != empty.entries
    (root / "images/a.txt").write_text("ignored caption")
    assert main_with_local_tokenizers(["cache", *flags]) == 0
    assert load_latent_manifest(root / "cache") == first
    images = discover_images(root / "images")
    pairs, _ = index_restoration_pairs(images, images, root / "cache", prompt="shared portrait")
    assert len(pairs) == 2
    with pytest.raises(ValueError, match="rebuild cache"):
        index_restoration_pairs(images, images, root / "cache", prompt="changed prompt")
    assert main_with_local_tokenizers(["train", str(training)]) == 0
    assert main_with_local_tokenizers(["cache", *flags, "--set", "dataset.groups.main.prompt=changed prompt"]) == 0
    assert load_latent_manifest(root / "cache").entries != first.entries
    assert load_latent_manifest(root / "empty-cache") == empty


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
def test_training_adapter_export_can_generate(image_assets, algorithm):
    training, generation = image_assets
    root = training.parent
    training.write_text(
        training.read_text().replace('parameter_mode = "full"', 'parameter_mode = "adapter"')
        + f'''
[train.adapter]
algorithm = "{algorithm}"
rank = 2
alpha = 3.0
'''
    )
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    assert (
        main_with_local_tokenizers(
            [
                "train",
                str(training),
                "--parameter-mode",
                "adapter",
                "--include-layers",
                "layers.*",
                "--exclude-layers",
                "layers.0.feed_forward.w3",
            ]
        )
        == 0
    )
    generation.write_text(
        generation.read_text()
        + f'''
[adapter]
weights = "{root.as_posix()}/run/weights/model.safetensors"
[adapter.config]
algorithm = "{algorithm}"
rank = 2
alpha = 1.0
'''
    )
    assert main_with_local_tokenizers(["generate", str(generation)]) == 0
    before = (root / "sample.png").read_bytes()
    assert main_with_local_tokenizers(["generate", str(generation)]) == 0
    assert (root / "sample.png").read_bytes() == before


@pytest.mark.parametrize("parameter_mode", ["full", "adapter"])
@pytest.mark.parametrize("calibrate", [False, True])
@pytest.mark.parametrize("synthetic", [False, True])
def test_image_restoration_encoder_update_and_exact_resume(
    image_assets, parameter_mode, calibrate, synthetic, monkeypatch
):
    training, _ = image_assets
    root = training.parent
    low = root / "low"
    low.mkdir()
    for name in ("a.png", "b.png"):
        with Image.open(root / "images" / name) as image:
            pixels = np.array(image.convert("RGB"))
        with Image.fromarray(np.roll(pixels, 3, axis=1)) as image:
            image.save(low / name)
    content = training.read_text().replace(
        "source = ", 'kind = "restoration"\nlow_quality_source = "' + low.as_posix() + '"\nsource = ', 1
    )
    content = content.replace('parameter_mode = "full"', f'parameter_mode = "{parameter_mode}"').replace(
        "max_updates = 2", "max_updates = 3"
    )
    content = content.replace("[output]", '[output]\nsave_every_updates = 1\nsave_precision = "bf16"')
    content += """
[method]
kind = "restoration"
[method.restoration]
encoder_rank = 2
[losses]
lrr = 1.0
l1 = 0.5
histogram = 0.1
dv3d = 0.0
gan = 0.0
fdl = 0.0
"""
    if parameter_mode == "adapter":
        content += '\n[train.adapter]\nalgorithm = "lora"\nrank = 2\nalpha = 3.0\n'
    if synthetic:
        content = content.replace(f'low_quality_source = "{low.as_posix()}"\n', "").replace(
            "resolution = [32,16]", "resolution = [32,32]"
        )
        content += "\n[dataset.groups.main.degradation]\nenabled = true\n"
    if calibrate:
        content = content.replace("encoder_rank = 2", "encoder_rank = 2\ncalibrate = true\ncalibration_pairs = 1")
    training.write_text(content)
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    if calibrate:
        original_calibrate = RestorationMethod.calibrate

        def checked_calibration(method, config, dataset):
            before = torch.get_rng_state()
            original_calibrate(method, config, dataset)
            sigma = method.sigma
            assert method.encoder is not None
            with torch.no_grad():
                for branch in method.encoder.layers:
                    branch.up.fill_(0.5)
                try:
                    original_calibrate(method, config, dataset)
                    assert method.sigma == sigma
                finally:
                    for branch in method.encoder.layers:
                        branch.up.zero_()
            assert torch.equal(before, torch.get_rng_state())

        monkeypatch.setattr(RestorationMethod, "calibrate", checked_calibration)
    assert main_with_local_tokenizers(["train", str(training)]) == 0
    if calibrate:

        def unexpected_calibration(*args):
            raise AssertionError("Resume must restore sigma without recalibrating")

        monkeypatch.setattr(
            "unitunner.training.methods.restoration.RestorationMethod.calibrate", unexpected_calibration
        )
    checkpoint = root / "run/resume/model.state-00000001.safetensors"
    training.write_text(content.replace((root / "run").as_posix(), (root / "resumed").as_posix()))
    assert main_with_local_tokenizers(["train", str(training), "--resume", str(checkpoint)]) == 0
    for artifact in (
        "weights/model.safetensors",
        "weights/model.encoder.safetensors",
        "resume/model.state-00000003.safetensors",
    ):
        expected = dict(iter_tensors(root / "run" / artifact))
        actual = dict(iter_tensors(root / "resumed" / artifact))
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    for artifact in ("weights/model.safetensors", "weights/model.encoder.safetensors"):
        assert all(value.dtype == torch.bfloat16 for _, value in iter_tensors(root / "run" / artifact))
    encoder = dict(iter_tensors(root / "run/weights/model.encoder.safetensors"))
    assert any(value.count_nonzero() for key, value in encoder.items() if key.endswith(".up"))
    # Restore with the actual trained full/adapter and nonzero encoder artifacts.
    restore = root / "restore.toml"
    model_path = root / "run/weights/model.safetensors" if parameter_mode == "full" else root / "model.safetensors"
    restore_text = f'''input = "{low.as_posix()}/a.png"
output = "{root.as_posix()}/restored.png"
encoder_weights = "{root.as_posix()}/run/weights/model.encoder.safetensors"
prompt = "first image"
[model]
family = "z-image"
variant = "base"
weights = "{model_path.as_posix()}"
vae = "{root.as_posix()}/vae.safetensors"
text_encoder = "{root.as_posix()}/text"
'''
    if parameter_mode == "adapter":
        restore_text += f'''[adapter]
weights = "{root.as_posix()}/run/weights/model.safetensors"
[adapter.config]
algorithm = "lora"
rank = 2
alpha = 3.0
'''
    restore.write_text(restore_text)
    assert main_with_local_tokenizers(["restore", str(restore)]) == 0
    first = (root / "restored.png").read_bytes()
    assert main_with_local_tokenizers(["restore", str(restore)]) == 0
    assert (root / "restored.png").read_bytes() == first
    with Image.open(root / "restored.png") as image:
        assert image.size == (32, 16)
    restore.write_text("sigma = 0.999\n" + restore_text)
    assert main_with_local_tokenizers(["restore", str(restore)]) == 2
    assert (root / "restored.png").read_bytes() == first
    restore.write_text(
        restore_text.replace(f'output = "{root.as_posix()}/restored.png"', f'output = "{low.as_posix()}/a.png"')
    )
    original = (low / "a.png").read_bytes()
    assert main_with_local_tokenizers(["restore", str(restore)]) == 2
    assert (low / "a.png").read_bytes() == original
    if synthetic:
        training.write_text(content + "scales = [2]\n")
        assert main_with_local_tokenizers(["train", str(training), "--resume", str(checkpoint)]) == 2
        training.write_text(content.replace((root / "run").as_posix(), (root / "resumed").as_posix()))
    (root / "images/a.txt").write_text("changed after cache")
    assert main_with_local_tokenizers(["train", str(training), "--resume", str(checkpoint)]) == 2


def test_image_restoration_real_adversarial_training_and_resume(image_assets, monkeypatch):
    from unitunner.training.workflows.common import images as workflow

    optimizers = []
    create = workflow.create_adamw

    def capture_optimizer(parameters, **kwargs):
        optimizer = create(parameters, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(workflow, "create_adamw", capture_optimizer)
    training, _ = image_assets
    root = training.parent
    low = root / "low"
    low.mkdir()
    for name in ("a.png", "b.png"):
        (low / name).write_bytes((root / "images" / name).read_bytes())
    backbone = DinoConvNeXt((4, 8, 12, 16), (1, 1, 1, 1))
    save_tensors(backbone.state_dict(), root / "dino.safetensors")
    content = training.read_text().replace(
        "source = ", f'kind = "restoration"\nlow_quality_source = "{low.as_posix()}"\nsource = ', 1
    )
    content = content.replace("resolution = [32,16]", "resolution = [512,512]").replace(
        "batch_size = 2", "batch_size = 1"
    )
    content = content.replace("accumulation_steps = 2", "accumulation_steps = 1").replace(
        "[output]", "[output]\nsave_every_updates = 1"
    )
    content = content.replace("[train]", "[train]\nweight_decay = 0.0003", 1)
    content += f'''
[method]
kind = "restoration"
[method.restoration]
encoder_rank = 2
dino_weights = "{root.as_posix()}/dino.safetensors"
dino_widths = [4,8,12,16]
dino_depths = [1,1,1,1]
[losses]
lrr = 1.0
l1 = 0.5
histogram = 0.0
dv3d = 0.1
gan = 0.1
fdl = 0.0
'''
    training.write_text(content)
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    assert main_with_local_tokenizers(["train", str(training)]) == 0
    assert len(optimizers) == 2
    for optimizer in optimizers:
        assert all(group["weight_decay"] == 0.0003 for group in optimizer.param_groups)
    training.write_text(content.replace((root / "run").as_posix(), (root / "resumed").as_posix()))
    assert (
        main_with_local_tokenizers(
            ["train", str(training), "--resume", (root / "run/resume/model.state-00000001.safetensors").as_posix()]
        )
        == 0
    )
    expected = dict(iter_tensors(root / "run/resume/model.state-00000002.safetensors"))
    actual = dict(iter_tensors(root / "resumed/resume/model.state-00000002.safetensors"))
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)

    training.write_text(
        content.replace((root / "run").as_posix(), (root / "gan-preview").as_posix())
        + "\n[preview]\nevery_updates = 1\nat_start = true\n"
    )
    assert main_with_local_tokenizers(["train", str(training)]) == 0
    for update in (1, 2):
        name = f"resume/model.state-{update:08d}.safetensors"
        assert (root / "run" / name).read_bytes() == (root / "gan-preview" / name).read_bytes()
    assert (root / "gan-preview/previews/step-00000002-00.png").is_file()


def test_restoration_inference_matches_direct_native_equation_and_preserves_odd_geometry(image_assets):
    training, _ = image_assets
    root = training.parent
    pixels = np.arange(13 * 19 * 3, dtype=np.uint8).reshape(13, 19, 3)
    with Image.fromarray(pixels) as image:
        image.save(root / "odd.png")
    config = root / "restore.toml"
    config.write_text(
        f'''input = "{root.as_posix()}/odd.png"
output = "{root.as_posix()}/restored.png"
sigma = 0.37
prompt = "first image"
[model]
family = "z-image"
variant = "base"
weights = "{root.as_posix()}/model.safetensors"
vae = "{root.as_posix()}/vae.safetensors"
text_encoder = "{root.as_posix()}/text"
'''
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert main_with_local_tokenizers(["restore", str(config)]) == 0
    vae = load_z_image_vae(root / "vae.safetensors").eval()
    model = load_z_image(root / "model.safetensors").eval()
    text = load_z_image_text(
        root / "text", text_config=local_text_config(root / "text"), device=torch.device("cpu"), dtype=torch.float32
    )
    with torch.no_grad():
        caption = text.encode_prompt("first image")
        padded = np.pad(pixels, ((0, 3), (0, 13), (0, 0)), mode="reflect")
        inputs = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
        low_latents = vae.normalize(vae.encode(inputs).mean)
        prediction = model(low_latents.unsqueeze(2), torch.tensor([1 - 0.37]), caption.features, caption.mask).squeeze(
            2
        )
        decoded = vae.decode(vae.denormalize(low_latents.float() + 0.37 * prediction.float()))[0, :, :13, :19]
        expected = ((decoded + 1) * 127.5).clamp(0, 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    with Image.open(root / "restored.png") as image:
        assert image.size == (19, 13)
        np.testing.assert_array_equal(np.array(image), expected)


@pytest.mark.parametrize("parameter_mode", ["full", "adapter"])
def test_mixed_restoration_sources_keep_cache_roles_repeats_and_exact_resume(image_assets, parameter_mode, monkeypatch):
    training, _ = image_assets
    root = training.parent
    low = root / "low"
    second = root / "second"
    low.mkdir()
    second.mkdir()
    for name in ("a.png", "b.png"):
        (low / name).write_bytes((root / "images" / name).read_bytes())
    pixels = np.random.default_rng(17).integers(0, 256, (40, 48, 3), dtype=np.uint8)
    with Image.fromarray(pixels) as image:
        image.save(second / "a.png")
    (second / "a.txt").write_text("distinct synthetic source prompt")
    content = training.read_text()
    start, end = content.index("[dataset.groups.main]"), content.index("[train]")
    content = (
        content[:start]
        + f'''[dataset.defaults]
resolution = [32,32]
batch_size = 3
[dataset.groups.aligned]
kind = "restoration"
horizontal_flip_probability = 0.0
source = "{root.as_posix()}/images"
low_quality_source = "{low.as_posix()}"
cache_directory = "{root.as_posix()}/cache-aligned"
repeats = 2
[dataset.groups.synthetic]
kind = "restoration"
horizontal_flip_probability = 0.0
source = "{second.as_posix()}"
cache_directory = "{root.as_posix()}/cache-synthetic"
[dataset.groups.synthetic.degradation]
enabled = true
'''
        + content[end:]
    )
    content = (
        content.replace('parameter_mode = "full"', f'parameter_mode = "{parameter_mode}"')
        .replace("accumulation_steps = 2", "accumulation_steps = 1")
        .replace("[output]", "[output]\nsave_every_updates = 1")
    )
    content += """
[method]
kind = "restoration"
[method.restoration]
encoder_rank = 2
calibrate = true
calibration_pairs = 3
[losses]
lrr = 1.0
l1 = 0.5
histogram = 0.0
dv3d = 0.0
gan = 0.0
fdl = 0.0
"""
    if parameter_mode == "adapter":
        content += '\n[train.adapter]\nalgorithm = "lora"\nrank = 2\nalpha = 3.0\n'
    training.write_text(content)
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    original = import_module("unitunner.training.run").train

    def checked_train(config, *, restoration_data, **kwargs):
        pairs = restoration_data.pairs
        assert [p.high.name for p in pairs] == ["a.png", "b.png", "a.png", "b.png", "a.png"]
        assert [p.cache_directory.name for p in pairs] == ["cache-aligned"] * 4 + ["cache-synthetic"]
        for index, pair in enumerate(pairs):
            low_pixels, high_pixels = restoration_data.read(pair, (32, 32), 7 + index, 0.0)
            if index < 4:
                torch.testing.assert_close(low_pixels, high_pixels, atol=0, rtol=0)
            else:
                assert not torch.equal(low_pixels, high_pixels)
        return original(config, restoration_data=restoration_data, **kwargs)

    original_calibration = RestorationMethod.calibrate

    def checked_calibration(method, config, dataset):
        observed = []

        def read(pair, resolution, seed, flip_probability):
            observed.append((pair.cache_directory.name, pair.high.name, seed))
            return dataset.read(pair, resolution, seed, flip_probability)

        original_calibration(method, config, replace(dataset, read=read))
        assert observed == [
            ("cache-aligned", "a.png", config.train.seed),
            ("cache-aligned", "b.png", config.train.seed + 1),
            ("cache-synthetic", "a.png", config.train.seed + 4),
        ]

    monkeypatch.setattr(RestorationMethod, "calibrate", checked_calibration)
    monkeypatch.setattr("unitunner.training.run.train", checked_train)
    assert main_with_local_tokenizers(["train", str(training)]) == 0
    checkpoint = root / "run/resume/model.state-00000001.safetensors"
    training.write_text(content.replace((root / "run").as_posix(), (root / "resumed").as_posix()))
    assert main_with_local_tokenizers(["train", str(training), "--resume", str(checkpoint)]) == 0
    for artifact in (
        "weights/model.safetensors",
        "weights/model.encoder.safetensors",
        "resume/model.state-00000002.safetensors",
    ):
        expected, actual = (dict(iter_tensors(root / directory / artifact)) for directory in ("run", "resumed"))
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
    (second / "a.txt").write_text("stale second source")
    assert main_with_local_tokenizers(["train", str(training), "--resume", str(checkpoint)]) == 2


@pytest.mark.parametrize("parameter_mode", ["full", "adapter"])
@pytest.mark.parametrize("restoration", [False, True])
@pytest.mark.parametrize("seed", [42, 99])
def test_training_previews_preserve_complete_updates_and_resume(image_assets, parameter_mode, restoration, seed):
    training, _ = image_assets
    root = training.parent
    content = training.read_text().replace('parameter_mode = "full"', f'parameter_mode = "{parameter_mode}"')
    content = content.replace("[train]", f"[train]\nseed = {seed}", 1)
    content = content.replace("max_updates = 2", "max_updates = 3").replace(
        "[output]", "[output]\nsave_every_updates = 1"
    )
    if parameter_mode == "adapter":
        content += '\n[train.adapter]\nalgorithm = "lora"\nrank = 2\nalpha = 3.0\n'
    if restoration:
        # Distinct input pixels make selection observable in the published PNG.
        Image.new("RGB", (32, 16), (220, 30, 80)).save(root / "images/b.png")
        content = content.replace(
            "[dataset.groups.main]",
            f'[dataset.groups.main]\nkind = "restoration"\nlow_quality_source = "{root.as_posix()}/images"',
        )
        content += """
[method]
kind = "restoration"
[method.restoration]
encoder_rank = 2
[losses]
lrr = 1.0
l1 = 0.5
dv3d = 0.0
fdl = 0.0
gan = 0.0
"""
    training.write_text(content)
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    assert main_with_local_tokenizers(["train", str(training)]) == 0
    preview = """
[preview]
every_updates = 1
at_start = true
"""
    if not restoration:
        preview += """
[[preview.samples]]
prompt = "fixed training preview"
negative_prompt = ""
width = 32
height = 16
steps = 2
guidance = 3.0
seed = 1234
"""
    training.write_text(content.replace((root / "run").as_posix(), (root / "with-previews").as_posix()) + preview)
    assert main_with_local_tokenizers(["train", str(training)]) == 0
    for update in range(4):
        path = root / f"with-previews/previews/step-{update:08d}-00.png"
        assert path.is_file()
        with Image.open(path) as image:
            assert image.size == ((104, 20) if restoration else (32, 16))
    if restoration:

        def low_pixels(update):
            with Image.open(root / f"with-previews/previews/step-{update:08d}-00.png") as image:
                return image.crop((2, 2, 34, 18)).tobytes()

        assert low_pixels(0) != low_pixels(1)
        assert low_pixels(0) == low_pixels(2)
        assert low_pixels(1) == low_pixels(3)
        # Seeds 42 and 99 produce opposite orders for these two input pairs.
        with Image.open(root / "images/b.png") as image:
            assert low_pixels(0 if seed == 42 else 1) == image.tobytes()
    # Compare every serialized module, optimizer, RNG stream and scalar cursor.
    for update in range(1, 4):
        filename = f"resume/model.state-{update:08d}.safetensors"
        assert (root / "run" / filename).read_bytes() == (root / "with-previews" / filename).read_bytes()
    # Enable previews while resuming a checkpoint produced with previews disabled.
    training.write_text(content.replace((root / "run").as_posix(), (root / "resumed-preview").as_posix()) + preview)
    assert (
        main_with_local_tokenizers(
            ["train", str(training), "--resume", (root / "run/resume/model.state-00000001.safetensors").as_posix()]
        )
        == 0
    )
    assert not (root / "resumed-preview/previews/step-00000000-00.png").exists()
    for update in (2, 3):
        filename = f"resume/model.state-{update:08d}.safetensors"
        assert (root / "run" / filename).read_bytes() == (root / "resumed-preview" / filename).read_bytes()
        assert (root / f"with-previews/previews/step-{update:08d}-00.png").read_bytes() == (
            root / f"resumed-preview/previews/step-{update:08d}-00.png"
        ).read_bytes()


def test_real_repa_cache_training_objectives_and_compact_checkpoint_resume(image_assets):
    """Run public cache/train/resume with real frozen ViT features and all feature objectives."""
    from dataclasses import asdict

    from unitunner.artifacts.hashing import file_sha256
    from unitunner.artifacts.tensors import TensorReader
    from unitunner.models.features.vit import DinoViT, DinoViTConfig

    training, _generation = image_assets
    root = training.parent
    config = load_z_image(root / "model.safetensors").config.model_copy(update={"num_layers": 2})
    save_z_image(ZImageTransformer(config), root / "model.safetensors")
    encoder = DinoViT(DinoViTConfig(width=16, depth=2, heads=2, patch=4, image_size=16, hidden=24))
    weights = root / "features.safetensors"
    save_tensors(encoder.state_dict(), weights, metadata={"unitunner.dinov3": json.dumps(asdict(encoder.config))})
    document = training.read_text().replace('parameter_mode = "full"', 'parameter_mode = "adapter"')
    document = document.replace("max_updates = 2", "max_updates = 3")
    document = document.replace("[output]", '[output]\nsave_every_updates=1\ncheckpoint_storage="frozen-reference"')
    document += f'''\n[train.adapter]
algorithm="lora"
rank=2
[execution]
recompute="block"
[method.tread]
end_layer=2
[method.layersync]
teacher_layer=1
weight=0.2
[method.repa]
weight=0.3
encoder_dim=16
encoder_weights="{weights.as_posix()}"
encoder_sha256="{file_sha256(weights)}"
cache_directory="{(root / "repa").as_posix()}"
'''
    training.write_text(document)
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    feature_manifest = (root / "repa/manifest.json").read_bytes()
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    assert (root / "repa/manifest.json").read_bytes() == feature_manifest
    assert main_with_local_tokenizers(["train", str(training)]) == 0
    checkpoint = root / "run/resume/model.state-00000001.safetensors"
    with TensorReader(checkpoint) as reader:
        manifest = json.loads(reader.metadata()["unitunner.checkpoint"])
        assert manifest["schema_version"] == 3
        assert "repa" in manifest["frozen_parameters"]
    training.write_text(document.replace((root / "run").as_posix(), (root / "resumed").as_posix()))
    assert main_with_local_tokenizers(["train", str(training), "--resume", str(checkpoint)]) == 0
    for name in ("weights/model.safetensors", "resume/model.state-00000003.safetensors"):
        assert (root / "run" / name).read_bytes() == (root / "resumed" / name).read_bytes()
