"""Exercise real native Wan video generation, adapter import and training preview isolation."""

from contextlib import closing
from fractions import Fraction
from importlib import import_module
from itertools import cycle

import av
import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import T5Tokenizer

from tests.support.text_assets import main_with_local_tokenizers
from unituner.adapters.config import AdapterConfig
from unituner.artifacts.formats.adapters import (
    AdapterTargetKeys,
    adapter_key_map,
    adapter_prefixes,
    convert_adapter_keys,
)
from unituner.artifacts.tensors import TensorReader, save_tensors
from unituner.data.media.video_output import save_video
from unituner.models.qwen_image.config.vae import QwenImageVAEConfig
from unituner.models.qwen_image.loading import save_qwen_image_vae
from unituner.models.qwen_image.vae import QwenImageVAE
from unituner.models.wan.config.clip import WanClipConfig
from unituner.models.wan.config.model import WanConfig
from unituner.models.wan.config.t5 import WanT5Config
from unituner.models.wan.loading import save_wan
from unituner.models.wan.projections import adapter_projections
from unituner.models.wan.text.loading import save_wan_t5
from unituner.models.wan.text.model import WanT5Encoder
from unituner.models.wan.transformer import WanTransformer
from unituner.models.wan.vision.loading import save_wan_clip
from unituner.models.wan.vision.model import WanClipVision
from unituner.training.parameters import ParameterInstallation


@pytest.fixture
def video_assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(563)
    vae = QwenImageVAE(QwenImageVAEConfig(base_channels=4, layers_per_block=2)).eval().requires_grad_(False)
    save_qwen_image_vae(vae, tmp_path / "vae.safetensors")
    vocabulary = [("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -1.0)] + [
        (c, -2.0) for c in "abcdefghijklmnopqrstuvwxyz"
    ]
    T5Tokenizer(vocab=vocabulary, extra_ids=0).save_pretrained(tmp_path / "tokenizer")
    text = WanT5Encoder(
        WanT5Config(vocab_size=len(vocabulary), hidden_size=24, attention_dim=24, ffn_dim=40, num_heads=4, num_layers=2)
    )
    save_wan_t5(text, tmp_path / "text.safetensors")
    Image.fromarray(np.random.default_rng(5).integers(0, 256, (20, 20, 3), dtype=np.uint8)).save(tmp_path / "image.png")
    yield tmp_path, vae
    torch.set_num_threads(previous)


def _publish_adapter(model, path, algorithm):
    targets = adapter_projections(model)
    with closing(
        ParameterInstallation(model, adapter=AdapterConfig(algorithm=algorithm, rank=2, alpha=7.0), targets=targets)
    ) as owner:
        with torch.no_grad():
            for parameter in owner.branches.parameters():
                parameter.normal_(std=0.03)
        prefixes = adapter_prefixes(targets, algorithm=algorithm, namespace="diffusion_model")
        mapping = adapter_key_map(
            [AdapterTargetKeys(str(index), prefixes[name]) for index, name in enumerate(targets)], algorithm=algorithm
        )
        save_tensors(convert_adapter_keys(owner.branches.state_dict(), mapping), path)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "version,variant,algorithm,solver,dual,control",
    [
        ("2.1", "t2v", "lora", "euler", False, False),
        ("2.1", "i2v", "loha", "dpm++", False, False),
        ("2.2", "i2v", "lokr", "unipc", True, True),
    ],
)
def test_public_video_assets_adapters_and_decoded_output(
    video_assets, version, variant, algorithm, solver, dual, control, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    root, _vae = video_assets
    config = WanConfig(
        version=version,
        variant=variant,
        hidden_size=48,
        num_heads=4,
        num_layers=2,
        ffn_dim=80,
        timestep_dim=16,
        text_dim=24,
        text_length=7,
        in_channels=48 if control else 36 if variant == "i2v" else 16,
    )
    for index in range(2 if dual else 1):
        model = WanTransformer(config)
        with torch.no_grad():
            model.head.head.weight.normal_(std=0.03)
        save_wan(model, root / f"model{index}.safetensors")
        _publish_adapter(model, root / f"adapter{index}.safetensors", algorithm)
    if config.image_context:
        image = WanClipVision(WanClipConfig(hidden_size=1280, output_dim=16, num_heads=16, num_layers=2, mlp_ratio=1))
        save_wan_clip(image, root / "image_encoder/model.safetensors")
    if control:
        save_video(torch.rand(3, 5, 16, 16) * 2 - 1, root / "control.mkv", fps=12)
    document = f'''output="{root / "out.mkv"}"
[model]
assets = "{root}"
family="wan"
version="{version}"
variant="{variant}"
weights="{root / "model0.safetensors"}"
vae="{root / "vae.safetensors"}"
text_encoder="{root / "text.safetensors"}"
'''
    document += f'''[sampling]
prompt="cat"
negative_prompt="bad"
width=16
height=16
frames=5
fps=12
steps=3
solver="{solver}"
seed=17
'''
    if variant == "i2v":
        document += f'conditioning_image="{root / "image.png"}"\n'
    if control:
        document += f'control_video="{root / "control.mkv"}"\n'
    document += f'''[adapter]
weights="{root / "adapter0.safetensors"}"
[adapter.config]
algorithm="{algorithm}"
rank=2
alpha=7.0
'''
    if dual:
        document += f'''[high_noise]
weights="{root / "model1.safetensors"}"
boundary=0.875
guidance=4.0
[high_noise.adapter]
weights="{root / "adapter1.safetensors"}"
[high_noise.adapter.config]
algorithm="{algorithm}"
rank=2
alpha=7.0
'''
    path = root / "generate.toml"
    path.write_text(document)
    rng = torch.get_rng_state().clone()
    with sdpa_kernel(SDPBackend.MATH):
        assert main_with_local_tokenizers(["generate-video", str(path), "--device", device]) == 0
    assert torch.equal(rng, torch.get_rng_state())
    with av.open(str(root / "out.mkv")) as container:
        assert container.streams.video[0].average_rate == Fraction(12, 1)
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(frames) == 5 and all(frame.shape == (16, 16, 3) for frame in frames)
    with sdpa_kernel(SDPBackend.MATH):
        assert main_with_local_tokenizers(["generate-video", str(path), "--device", device]) == 0
    with av.open(str(root / "out.mkv")) as container:
        repeated = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert all(np.array_equal(first, second) for first, second in zip(frames, repeated, strict=True))
    bad = root / "bad.toml"
    bad.write_text(document.replace("frames=5", "frames=4"))
    before = (root / "out.mkv").read_bytes()
    assert main_with_local_tokenizers(["generate-video", str(bad)]) == 2
    assert (root / "out.mkv").read_bytes() == before
    if control:
        save_video(torch.zeros(3, 1, 16, 16), root / "control.mkv", fps=12)
        assert main_with_local_tokenizers(["generate-video", str(path), "--device", device]) == 2
        assert (root / "out.mkv").read_bytes() == before


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "residency,parameter_mode", [("resident", "adapter"), ("frozen-blocks", "adapter"), ("resident", "full")]
)
def test_public_training_preview_and_resume_leave_final_artifacts_exact(
    video_assets, device, residency, parameter_mode
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if residency == "frozen-blocks" and device != "cuda":
        pytest.skip("H2D requires CUDA")
    root, _vae = video_assets
    model = WanTransformer(
        WanConfig(hidden_size=48, num_heads=4, num_layers=2, ffn_dim=80, timestep_dim=16, text_dim=24, text_length=7)
    )
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.03)
    save_wan(model, root / "train-model.safetensors")
    source = root / "source"
    source.mkdir()
    with Image.open(root / "image.png") as image:
        image.save(source / "frame.png")
    (source / "frame.txt").write_text("cat")
    document = f'''[model]
assets = "{root}"
family="wan"
version="2.1"
variant="t2v"
weights="{root / "train-model.safetensors"}"
vae="{root / "vae.safetensors"}"
text_encoder="{root / "text.safetensors"}"
[dataset.groups.main]
kind="image"
source="{source}"
cache_directory="{root / "cache"}"
resolution=[16,16]
[train]
max_updates=3
accumulation_steps=2
learning_rate=0.001
[train.adapter]
algorithm="lora"
rank=2
alpha=7.0
[execution]
recompute="block"
[output]
max_resume_checkpoints = 0
directory="{root / "continuous"}"
name="wan"
save_every_updates=1
'''
    if parameter_mode == "full":
        document = document.replace("[train]\n", '[train]\nparameter_mode="full"\n').replace(
            '[train.adapter]\nalgorithm="lora"\nrank=2\nalpha=7.0\n', ""
        )
    if residency == "frozen-blocks":
        document = document.replace(
            'recompute="block"', 'recompute="block"\nresidency="frozen-blocks"\nuse_pinned_memory_for_block_swap=true'
        )
        document = document.replace("[output]", '[output]\ncheckpoint_storage="frozen-reference"')
    path = root / "train.toml"
    path.write_text(document)
    with sdpa_kernel(SDPBackend.MATH):
        assert main_with_local_tokenizers(["cache", str(path), "--device", "cpu"]) == 0
        assert main_with_local_tokenizers(["train", str(path), "--device", device]) == 0
    preview_document = (
        document.replace(str(root / "continuous"), str(root / "preview"))
        + """[preview]
every_updates=1
at_start=true
[[preview.videos]]
prompt="cat"
negative_prompt="bad"
width=16
height=16
frames=5
fps=12
steps=2
seed=29
"""
    )
    path.write_text(preview_document)
    with sdpa_kernel(SDPBackend.MATH):
        assert main_with_local_tokenizers(["train", str(path), "--device", device]) == 0
    assert len(list((root / "preview/previews").glob("*.mp4"))) == 4
    for filename in ["weights/wan.safetensors", "resume/wan.state-00000003.safetensors"]:
        assert (root / "continuous" / filename).read_bytes() == (root / "preview" / filename).read_bytes()
    path.write_text(preview_document.replace(str(root / "preview"), str(root / "resumed")))
    with sdpa_kernel(SDPBackend.MATH):
        assert (
            main_with_local_tokenizers(
                [
                    "train",
                    str(path),
                    "--device",
                    device,
                    "--resume",
                    str(root / "preview/resume/wan.state-00000001.safetensors"),
                ]
            )
            == 0
        )
    for filename in ["weights/wan.safetensors", "resume/wan.state-00000003.safetensors"]:
        assert (root / "continuous" / filename).read_bytes() == (root / "resumed" / filename).read_bytes()
    for media in (root / "preview/previews").glob("*.mp4"):
        with av.open(str(media)) as container:
            assert len(list(container.decode(video=0))) == 5


@pytest.mark.parametrize(
    ("algorithm", "sigmas", "routing"),
    [
        ("full", (0.2, 0.9), "batch"),
        ("lora", (0.2, 0.9), "batch"),
        ("full", (0.2, 0.9), "batch-real"),
    ],
)
def test_wan22_trains_both_experts_and_resumes_exactly(video_assets, monkeypatch, algorithm, sigmas, routing):
    root, _ = video_assets
    topology = WanConfig(
        version="2.2",
        hidden_size=48,
        num_heads=4,
        num_layers=2,
        ffn_dim=80,
        timestep_dim=16,
        text_dim=24,
        text_length=7,
    )
    low = WanTransformer(topology)
    high = WanTransformer(topology)
    with torch.no_grad():
        low.head.head.weight.normal_(std=0.03)
        high.head.head.weight.normal_(std=0.06)
    save_wan(low, root / "low.safetensors")
    save_wan(high, root / "high.safetensors")
    source = root / "source"
    source.mkdir()
    with Image.open(root / "image.png") as image:
        for stem in ("a", "b"):
            image.save(source / f"{stem}.png")
            (source / f"{stem}.txt").write_text("cat", encoding="utf-8")
    document = f'''[model]
family = "wan"
version = "2.2"
variant = "t2v"
assets = "{root}"
weights = "{root / "low.safetensors"}"
vae = "{root / "vae.safetensors"}"
text_encoder = "{root / "text.safetensors"}"
[wan_experts]
high_noise_weights = "{root / "high.safetensors"}"
boundary = 0.5
[dataset.groups.main]
kind = "image"
source = "{source}"
cache_directory = "{root / "cache"}"
resolution = [16, 16]
batch_size = 2
[train]
max_updates = 2
learning_rate = 0.001
seed = 29
[execution]
recompute = "block"
[output]
directory = "{root / "continuous"}"
name = "wan"
save_every_updates = 1
max_resume_checkpoints = 0
max_weight_checkpoints = 1
'''
    if algorithm == "lora":
        document = document.replace(
            "[execution]", '[train.adapter]\nalgorithm = "lora"\nrank = 2\nalpha = 2.0\n[execution]'
        )
    else:
        document = document.replace("[train]\n", '[train]\nparameter_mode = "full"\n')
    if algorithm == "full" and sigmas == (0.2, 0.9):
        document = document.replace(
            "[execution]",
            "[method.layersync]\nweight = 0.1\nstudent_layer = 0\nteacher_layer = 1\n"
            "[method.tread]\ndrop_fraction = 0.25\nstart_layer = 0\nend_layer = 2\n[execution]",
        )
    training = import_module("unituner.training.workflows.wan.run")
    original_sample = training.sample_sigmas
    original_step = training.WanBinding.cached_step
    choices = cycle(sigmas)
    forwards = []
    if routing != "batch-real":
        monkeypatch.setattr(training, "_choose_high_noise", lambda **kwargs: next(choices) == 0.9)

    def cached_step(binding, records, noise, sigma, timesteps, *, latents=None):
        forwards.append((binding.model, len(records)))
        return original_step(binding, records, noise, sigma, timesteps, latents=latents)

    monkeypatch.setattr(training.WanBinding, "cached_step", cached_step)

    def sample(size, *, device, grid):
        if size == 2:
            value = 0.9 if bool(torch.all(grid >= 0.5)) else 0.2
            return torch.full((size,), value, device=device)
        return original_sample(size, device=device, grid=grid)

    if routing != "batch-real":
        monkeypatch.setattr(training, "sample_sigmas", sample)
    path = root / "train.toml"
    path.write_text(document, encoding="utf-8")
    with sdpa_kernel(SDPBackend.MATH):
        assert main_with_local_tokenizers(["cache", str(path), "--device", "cpu"]) == 0
        assert main_with_local_tokenizers(["train", str(path), "--device", "cpu"]) == 0
    assert len(forwards) == 2 and all(size == 2 for _, size in forwards)
    weights = root / "continuous/weights"
    assert (weights / "wan.safetensors").is_file()
    assert (weights / "wan.high_noise.safetensors").is_file()
    assert not (weights / "wan-00000001.safetensors").exists()
    assert not (weights / "wan-00000001.high_noise.safetensors").exists()
    assert (weights / "wan-00000002.high_noise.safetensors").is_file()
    if algorithm == "full":
        with TensorReader(weights / "wan.safetensors") as reader:
            low_changed = any(not torch.equal(reader.read(name), value) for name, value in low.state_dict().items())
        with TensorReader(weights / "wan.high_noise.safetensors") as reader:
            high_changed = any(not torch.equal(reader.read(name), value) for name, value in high.state_dict().items())
        if routing == "batch-real":
            assert low_changed or high_changed
        else:
            assert low_changed and high_changed
    preview = (
        document.replace(str(root / "continuous"), str(root / "preview"))
        + """[preview]
every_updates = 1
at_start = true
[[preview.videos]]
prompt = "cat"
negative_prompt = "bad"
width = 16
height = 16
frames = 5
fps = 12
steps = 2
seed = 29
"""
    )
    path.write_text(preview, encoding="utf-8")
    with sdpa_kernel(SDPBackend.MATH):
        assert main_with_local_tokenizers(["train", str(path), "--device", "cpu"]) == 0
    assert len(list((root / "preview/previews").glob("*.mp4"))) == 3
    path.write_text(preview.replace(str(root / "preview"), str(root / "resumed")), encoding="utf-8")
    choices = cycle((sigmas[1], sigmas[0]))
    with sdpa_kernel(SDPBackend.MATH):
        assert (
            main_with_local_tokenizers(
                [
                    "train",
                    str(path),
                    "--device",
                    "cpu",
                    "--resume",
                    str(root / "preview/resume/wan.state-00000001.safetensors"),
                ]
            )
            == 0
        )
    for name in (
        "weights/wan.safetensors",
        "weights/wan.high_noise.safetensors",
        "resume/wan.state-00000002.safetensors",
    ):
        expected = (root / "continuous" / name).read_bytes()
        assert (root / "preview" / name).read_bytes() == expected
        assert (root / "resumed" / name).read_bytes() == expected
    low_base = weights / "wan.safetensors" if algorithm == "full" else root / "low.safetensors"
    high_base = weights / "wan.high_noise.safetensors" if algorithm == "full" else root / "high.safetensors"
    generation = f'''output = "{root / "paired.mkv"}"
[model]
family = "wan"
version = "2.2"
variant = "t2v"
assets = "{root}"
weights = "{low_base}"
vae = "{root / "vae.safetensors"}"
text_encoder = "{root / "text.safetensors"}"
[sampling]
prompt = "cat"
negative_prompt = "bad"
width = 16
height = 16
frames = 5
fps = 12
steps = 2
seed = 29
'''
    if algorithm == "lora":
        generation += f'''[adapter]
weights = "{weights / "wan.safetensors"}"
[adapter.config]
algorithm = "lora"
rank = 2
alpha = 2.0
'''
    generation += f'''[high_noise]
weights = "{high_base}"
boundary = 0.5
'''
    if algorithm == "lora":
        generation += f'''[high_noise.adapter]
weights = "{weights / "wan.high_noise.safetensors"}"
[high_noise.adapter.config]
algorithm = "lora"
rank = 2
alpha = 2.0
'''
    generate = root / "generate-paired.toml"
    generate.write_text(generation, encoding="utf-8")
    with sdpa_kernel(SDPBackend.MATH):
        assert main_with_local_tokenizers(["generate-video", str(generate), "--device", "cpu"]) == 0
    with av.open(str(root / "paired.mkv")) as container:
        assert len(list(container.decode(video=0))) == 5


def test_lossless_video_quantization_and_rejected_input_keep_previous_output(tmp_path):
    values = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]).repeat(2)
    pixels = values[None, None, None, :].expand(3, 2, 4, 14).clone()
    path = tmp_path / "pixels.mkv"
    save_video(pixels, path, fps=13)
    expected = np.array([0, 0, 64, 128, 191, 255, 255] * 2, dtype=np.uint8)
    with av.open(str(path)) as container:
        assert container.streams.video[0].average_rate == Fraction(13, 1)
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(frames) == 2
    assert all(np.array_equal(frame, np.broadcast_to(expected[None, :, None], (4, 14, 3))) for frame in frames)
    before = path.read_bytes()
    pixels[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        save_video(pixels, path, fps=13)
    assert path.read_bytes() == before
