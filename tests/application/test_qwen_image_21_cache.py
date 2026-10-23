"""Exercise native Qwen2.1 public cache publication, reference identity and exact reuse."""

import hashlib
import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from tests.support.qwen_image_21 import write_text_assets
from tests.support.text_assets import local_text_config, main_with_local_tokenizers
from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.cache.storage import load_latent_manifest
from unitunner.data.media.image import resize_pixels
from unitunner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unitunner.models.qwen_image_2_1.data.cache.storage import read_qwen_image_21_record, save_qwen_image_21_record
from unitunner.models.qwen_image_2_1.data.encoding import encode_images
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21_vae, save_qwen_image_21_vae
from unitunner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text
from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE


@pytest.mark.parametrize("count", [0, 2, 1])
def test_public_cache_mean_text_metadata_reuse_and_invalidation(tmp_path, count):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(943)
        write_text_assets(tmp_path / "text")
        vae = QwenImage21VAE(
            QwenImage21VAEConfig(
                base_channels=2, decoder_base_channels=4, channel_multipliers=(1, 2, 4, 4, 4), layers_per_block=1
            )
        )
        save_qwen_image_21_vae(vae, tmp_path / "vae.safetensors")
        for directory in ["images", "ref0", "ref1"]:
            (tmp_path / directory).mkdir()
            for index, name in enumerate(["a", "b"]):
                height, width = (35, 49) if directory == "images" else (32, 64) if directory == "ref0" else (64, 32)
                channels = 3 if index else 4
                pixels = (
                    np.arange(height * width * channels, dtype=np.uint32).reshape(height, width, channels) * 17
                    + index * 11
                ).astype(np.uint8)
                Image.fromarray(pixels).save(tmp_path / directory / f"{name}.png")
        (tmp_path / "images/a.txt").write_text("short")
        (tmp_path / "images/b.txt").write_text("longer caption about textures")
        config = tmp_path / "cache.toml"
        config.write_text(f'''[model]
family = "qwen-image-2-1"
variant = "base"
weights = "{tmp_path}/not-loaded.safetensors"
vae = "{tmp_path}/vae.safetensors"
text_encoder = "{tmp_path}/text"
[dataset.groups.main]
source = "{tmp_path}/images"
cache_directory = "{tmp_path}/cache"
resolution = [64,32]
reference_sources = {json.dumps([str(tmp_path / f"ref{i}") for i in range(count)])}
[train]
parameter_mode = "full"
''')
        device = torch.device(os.environ.get("UNITUNNER_TEST_DEVICE", "cpu"))
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None
        assert main_with_local_tokenizers(["cache", str(config), "--device", str(device)]) == 0
        destination = tmp_path / "cache"
        manifest = load_latent_manifest(destination)
        assert manifest.family == "qwen-image-2-1" and manifest.variant is None and len(manifest.entries) == 2
        exact = (
            (destination / "manifest.json").read_bytes(),
            [(destination / entry.file).read_bytes() for entry in manifest.entries],
        )
        assert main_with_local_tokenizers(["cache", str(config), "--device", str(device)]) == 0
        assert exact == (
            (destination / "manifest.json").read_bytes(),
            [(destination / entry.file).read_bytes() for entry in manifest.entries],
        )
        frozen = load_qwen_image_21_vae(tmp_path / "vae.safetensors", device=device).eval().requires_grad_(False)
        text = load_qwen_image_21_text(
            tmp_path / "text", text_config=local_text_config(tmp_path / "text"), device=device, dtype=torch.float32
        )
        for name, entry in zip(["a", "b"], manifest.entries, strict=True):
            record = read_qwen_image_21_record(destination, entry, device=device)
            with Image.open(tmp_path / "images" / f"{name}.png") as image:
                target = resize_pixels(np.array(image.convert("RGBA"), copy=True), (64, 32))
            controls = []
            for i in range(count):
                with Image.open(tmp_path / f"ref{i}" / f"{name}.png") as image:
                    rgba = np.array(image.convert("RGBA"), copy=True)
                controls.append(resize_pixels(rgba, (64, 32)))
            expected = encode_images(frozen, [target, *controls])
            assert torch.equal(record.latents, expected[0])
            assert all(torch.equal(a, b) for a, b in zip(record.references, expected[1:], strict=True))
            prompt = (tmp_path / "images" / f"{name}.txt").read_text()
            encoded = text.encode([prompt], references=[controls])[0]
            assert torch.equal(record.caption, encoded.features)
            assert record.image_slots.tolist() == list(encoded.image_slots)
            assert record.reference_grids.tolist() == [list(grid) for grid in encoded.image_shapes]
            hashes = [
                list(hashlib.sha256(f"{image.shape[1]}x{image.shape[0]}:".encode() + image.tobytes()).digest())
                for image in controls
            ]
            assert record.reference_hashes.shape == (count, 32) and record.reference_hashes.tolist() == hashes
            assert (
                record.image_slots.dtype == record.reference_grids.dtype == torch.int64
                and record.reference_hashes.dtype == torch.uint8
            )
            identity = CacheIdentity(
                source_sha256="a" * 64,
                encoder_sha256="b" * 64,
                transform_sha256="c" * 64,
                layout=f"qwen-image-2-1/C1HW+LD+R{count}",
            )
            other = tmp_path / f"copy-{name}"
            saved = save_qwen_image_21_record(other, identity, record)
            before = (other / saved.file).read_bytes()
            with pytest.raises(ValueError, match="collision"):
                save_qwen_image_21_record(other, identity, replace(record, caption=record.caption + 1))
            assert (other / saved.file).read_bytes() == before
            if count:
                with pytest.raises(ValueError, match="grids differ"):
                    save_qwen_image_21_record(
                        other, identity, replace(record, reference_grids=record.reference_grids + 1)
                    )
                with pytest.raises(ValueError, match="int64"):
                    save_qwen_image_21_record(other, identity, replace(record, image_slots=record.image_slots.float()))
        changed = tmp_path / "ref0/a.png" if count else tmp_path / "images/a.txt"
        if count:
            Image.new("RGBA", (64, 32), (211, 31, 73, 47)).save(changed)
        else:
            changed.write_text("changed prompt")
        assert main_with_local_tokenizers(["cache", str(config), "--device", str(device)]) == 0
        updated = load_latent_manifest(destination)
        assert updated.entries[0] != manifest.entries[0] and updated.entries[1] == manifest.entries[1]
        assert torch.equal(rng, torch.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    finally:
        torch.set_num_threads(previous)
