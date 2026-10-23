"""Exercise public Wan video/image caching, exact conditions and immutable content reuse."""

from dataclasses import replace
from fractions import Fraction

import av
import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import T5Tokenizer

from tests.support.text_assets import main_with_local_tokenizers
from unitunner.artifacts.tensors import TensorReader
from unitunner.config.datasets import VideoDatasetConfig
from unitunner.config.loading import ConfigError, load_training_config
from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.media.video import read_video, resample_frame_indices, select_video_windows
from unitunner.models.qwen_image.config.vae import QwenImageVAEConfig
from unitunner.models.qwen_image.loading import save_qwen_image_vae
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.models.wan.conditioning import encode_latent_conditions
from unitunner.models.wan.config.clip import WanClipConfig
from unitunner.models.wan.config.t5 import WanT5Config
from unitunner.models.wan.data.cache.storage import load_wan_manifest, read_wan_record, save_wan_record
from unitunner.models.wan.single_frame import encode_single_frame_conditions
from unitunner.models.wan.text.loading import save_wan_t5
from unitunner.models.wan.text.model import WanT5Encoder
from unitunner.models.wan.vision.encoder import WanImageEncoder
from unitunner.models.wan.vision.loading import save_wan_clip
from unitunner.models.wan.vision.model import WanClipVision


def write_video(path, pixels, *, fps=24):
    """Write real lossless FFV1 RGB frames without changing test pixel values."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width, stream.height = pixels.shape[2], pixels.shape[1]
        stream.pix_fmt = "bgr0"
        for index, values in enumerate(pixels):
            frame = av.VideoFrame.from_ndarray(values, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    vae = QwenImageVAE(QwenImageVAEConfig(base_channels=4, layers_per_block=2)).eval().requires_grad_(False)
    save_qwen_image_vae(vae, tmp_path / "vae.safetensors")
    vocabulary = [("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -1.0)] + [
        (char, -2.0) for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ&"
    ]
    T5Tokenizer(vocab=vocabulary, extra_ids=0).save_pretrained(tmp_path / "tokenizer")
    text = WanT5Encoder(
        WanT5Config(vocab_size=len(vocabulary), hidden_size=24, attention_dim=24, ffn_dim=40, num_heads=4, num_layers=2)
    )
    save_wan_t5(text, tmp_path / "text.safetensors")
    image = (
        WanClipVision(WanClipConfig(hidden_size=24, output_dim=16, num_heads=4, num_layers=2))
        .eval()
        .requires_grad_(False)
    )
    save_wan_clip(image, tmp_path / "image_encoder/model.safetensors")
    yield tmp_path, vae, image
    torch.set_num_threads(previous)


@pytest.mark.parametrize("kind", ["image", "video"])
def test_standalone_cache_global_prompt_without_sidecars(assets, kind):
    """Both Wan media paths honor group captions without any training config."""
    import json

    root, _, _ = assets
    source = root / "source"
    source.mkdir()
    pixels = np.zeros((5, 16, 16, 3), dtype=np.uint8)
    if kind == "image":
        Image.fromarray(pixels[0]).save(source / "clip.png")
    else:
        write_video(source / "clip.mkv", pixels)
    model = {
        "family": "wan",
        "version": "2.1",
        "variant": "t2v",
        "assets": str(root),
        "vae": str(root / "vae.safetensors"),
        "text_encoder": str(root / "text.safetensors"),
    }
    group = {
        "kind": kind,
        "source": str(source),
        "cache_directory": str(root / "cache"),
        "resolution": [16, 16],
        "caption_extension": ".txt",
        "prompt": "shared",
    }
    if kind == "video":
        group["video"] = {"target_frames": [5]}
    flags = ["--model", json.dumps(model), "--dataset.groups", json.dumps({"main": group})]
    assert main_with_local_tokenizers(["cache", *flags]) == 0
    original = load_wan_manifest(root / "cache")
    (source / "clip.txt").write_text("ignored")
    assert main_with_local_tokenizers(["cache", *flags]) == 0
    assert load_wan_manifest(root / "cache") == original
    assert main_with_local_tokenizers(["cache", *flags, "--set", 'dataset.groups.main.prompt=""']) == 0
    assert load_wan_manifest(root / "cache").entries != original.entries


@pytest.mark.parametrize(
    "version,variant,kind", [("2.1", "i2v", "image"), ("2.2", "i2v", "video"), ("2.1", "t2v", "video")]
)
def test_public_cache_values_reuse_caption_invalidation_and_record_integrity(assets, version, variant, kind):
    root, vae, image = assets
    source, controls, destination = root / "source", root / "control", root / "cache"
    source.mkdir()
    controls.mkdir()
    count = 1 if kind == "image" else 9
    pixels = np.random.default_rng(17).integers(0, 256, (count, 16, 16, 3), dtype=np.uint8)
    control = pixels[:1].copy()
    if kind == "image":
        Image.fromarray(pixels[0]).save(source / "clip.png")
        Image.fromarray(control[0]).save(controls / "clip.png")
    else:
        (source / "cache").mkdir()
        write_video(source / "clip.mkv", pixels)
        write_video(controls / "clip.mkv", control)
    caption = source / "clip.txt"
    caption.write_text(" HELLO &amp;amp; caption ")
    document = f'''[model]
assets = "{root}"
family = "wan"
version = "{version}"
variant = "{variant}"
weights = "unused-transformer.safetensors"
vae = "{root / "vae.safetensors"}"
text_encoder = "{root / "text.safetensors"}"
'''
    document += f'''[dataset.groups.main]
kind = "{kind}"
source = "{source}"
cache_directory = "{destination}"
resolution = [16,16]
control_source = "{controls}"
[train]
parameter_mode = "adapter"
[train.adapter]
algorithm = "lora"
rank = 2
'''
    if kind == "video":
        document += "[dataset.groups.main.video]\ntarget_frames = [9]\n"
    config_path = root / "cache.toml"
    config_path.write_text(document)
    rng = torch.get_rng_state().clone()
    assert main_with_local_tokenizers(["cache", str(config_path), "--device", "cpu"]) == 0
    assert torch.equal(rng, torch.get_rng_state())
    manifest = load_wan_manifest(destination)
    assert len(manifest.entries) == 1 and manifest.version == version and manifest.variant == variant
    record = read_wan_record(destination, manifest.entries[0], manifest=manifest, device=torch.device("cpu"))
    values = torch.from_numpy(pixels).permute(3, 0, 1, 2).contiguous().float().unsqueeze(0) / 127.5 - 1
    control_values = (
        torch.from_numpy(np.repeat(control, count, axis=0)).permute(3, 0, 1, 2).contiguous().float().unsqueeze(0)
        / 127.5
        - 1
    )
    with torch.no_grad():
        expected = encode_latent_conditions(
            vae, values, image_conditioning=variant == "i2v", control_pixels=control_values
        )
    assert expected.control_latents is not None and record.control_latents is not None
    assert torch.equal(record.latents, expected.latents[0]) and torch.equal(
        record.control_latents, expected.control_latents[0]
    )
    if variant == "i2v":
        assert expected.image_condition is not None and record.image_condition is not None
        assert torch.equal(record.image_condition, expected.image_condition[0])
    if manifest.image_dim is not None:
        assert record.image_features is not None
        assert torch.equal(record.image_features, WanImageEncoder(image).encode((values[0, :, :1],))[0])
    previous = (destination / "manifest.json").read_bytes()
    assert main_with_local_tokenizers(["cache", str(config_path), "--device", "cpu"]) == 0
    assert previous == (destination / "manifest.json").read_bytes()
    entry = manifest.entries[0]
    if manifest.image_dim is not None:
        with sdpa_kernel(SDPBackend.MATH):
            assert main_with_local_tokenizers(["cache", str(config_path), "--device", "cpu"]) == 0
        manifest = load_wan_manifest(destination)
        assert manifest.entries[0].file != entry.file
        assert not (destination / entry.file).exists()
        entry = manifest.entries[0]
        record = read_wan_record(destination, entry, manifest=manifest, device=torch.device("cpu"))
    with TensorReader(destination / entry.file) as reader:
        identity = CacheIdentity.model_validate_json(reader.metadata()["unitunner.cache.identity"])
    before = (destination / entry.file).read_bytes()
    with pytest.raises(ValueError, match="finite FP32"):
        save_wan_record(
            destination,
            identity,
            replace(record, caption=torch.full_like(record.caption, float("nan"))),
            manifest=manifest,
        )
    with pytest.raises(ValueError, match="collision"):
        save_wan_record(destination, identity, replace(record, caption=record.caption + 1), manifest=manifest)
    assert (destination / entry.file).read_bytes() == before
    caption.write_text("changed caption")
    assert main_with_local_tokenizers(["cache", str(config_path), "--device", "cpu"]) == 0
    assert load_wan_manifest(destination).entries[0].file != entry.file
    (destination / load_wan_manifest(destination).entries[0].file).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        load_wan_manifest(destination)
    with pytest.raises(ConfigError, match=r"1\+4k"):
        load_training_config(
            config_path, overrides={"dataset": {"groups": {"main": {"kind": "video", "video": {"target_frames": [8]}}}}}
        )


def test_video_windows_timestamps_and_lossless_frame_grid(tmp_path):
    pixels = np.random.default_rng(21).integers(0, 256, (13, 16, 32, 3), dtype=np.uint8)
    path = tmp_path / "clip.mkv"
    write_video(path, pixels)
    policy = VideoDatasetConfig(target_frames=(5, 9), extraction="chunk")
    decoded = read_video(path, (32, 16), policy)
    assert np.array_equal(decoded.pixels, pixels) and decoded.source_indices == tuple(range(13))
    assert [(window.start, window.count) for window in select_video_windows(13, policy)] == [(0, 5), (5, 5), (0, 9)]
    assert resample_frame_indices((0.0, 0.1, 0.2), source_frame_duration=0.1, target_fps=20) == (0, 0, 1, 1, 2, 2)
    with pytest.raises(ValueError, match="nondecreasing"):
        resample_frame_indices((0.0, 1.0, 0.1), source_frame_duration=0.1, target_fps=20)
    frames = read_video(path, (32, 16), policy.model_copy(update={"source_fps": 24.0}), end_frame=5)
    assert frames.source_indices == (0, 2, 3, 5, 7)


@pytest.mark.parametrize("version,count", [("2.1", 1), ("2.1", 2), ("2.2", 2)])
def test_public_indexed_single_frame_cache_preserves_binding_and_native_storage(assets, version, count):
    root, vae, image = assets
    source = root / "targets"
    source.mkdir()
    target = np.full((16, 16, 3), 97, dtype=np.uint8)
    Image.fromarray(target).save(source / "target.png")
    sources = []
    clean = []
    for index in range(count):
        directory = root / f"clean{index}"
        directory.mkdir()
        sources.append(directory)
        pixels = np.full_like(target, 32 + index * 160)
        clean.append(pixels)
        Image.fromarray(pixels).save(directory / "target.png")
    clean_indices = (24,) if count == 1 else (24, 0)
    config = root / "single.toml"
    document = f'''[model]
assets = "{root}"
family="wan"
version="{version}"
variant="i2v"
weights="unused.safetensors"
vae="{root / "vae.safetensors"}"
text_encoder="{root / "text.safetensors"}"
'''
    document += f'''[dataset.groups.main]
kind="image"
source="{source}"
cache_directory="{root / "single-cache"}"
resolution=[16,16]
[dataset.groups.main.wan.single_frame]
clean_sources=[{",".join('"' + str(path) + '"' for path in sources)}]
clean_indices={list(clean_indices)}
target_index=12
[train.adapter]
rank=2
'''
    config.write_text(document)
    rng = torch.get_rng_state().clone()
    assert main_with_local_tokenizers(["cache", str(config), "--device", "cpu"]) == 0
    manifest = load_wan_manifest(root / "single-cache")
    assert manifest.frame_layout == "single-frame" and manifest.clean_count == count
    record = read_wan_record(root / "single-cache", manifest.entries[0], manifest=manifest, device=torch.device("cpu"))
    target_pixels = torch.from_numpy(target).permute(2, 0, 1).float()[None, :, None] / 127.5 - 1
    clean_pixels = torch.from_numpy(np.stack(clean)).permute(3, 0, 1, 2).float()[None] / 127.5 - 1
    with torch.no_grad():
        expected = encode_single_frame_conditions(
            vae, target_pixels, clean_pixels, clean_indices=clean_indices, target_index=12
        )
    assert (
        record.frame_indices == expected.layout.frame_indices
        and record.target_position == expected.layout.target_position
    )
    assert record.image_condition is not None
    assert torch.equal(record.latents, expected.latents[0]) and torch.equal(
        record.image_condition, expected.image_condition[0]
    )
    if version == "2.1":
        features = WanImageEncoder(image).encode((clean_pixels[0, :, expected.layout.clean_order],))
        assert record.image_features is not None
        assert record.image_features.shape == (count, 257, 24) and torch.equal(record.image_features, features)
    else:
        assert record.image_features is None
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ConfigError, match="distinct"):
        load_training_config(
            config, overrides={"dataset": {"groups": {"main": {"wan": {"single_frame": {"target_index": 24}}}}}}
        )
