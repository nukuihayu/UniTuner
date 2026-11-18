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
from unituner.data.cache.schema import CacheIdentity
from unituner.data.cache.storage import load_latent_manifest
from unituner.data.media.buckets import ImageBuckets
from unituner.data.media.image import resize_pixels
from unituner.models.qwen_image.data.media import reference_resolution
from unituner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unituner.models.qwen_image_2_1.data.cache.storage import read_qwen_image_21_record, save_qwen_image_21_record
from unituner.models.qwen_image_2_1.data.encoding import encode_images
from unituner.models.qwen_image_2_1.loading import load_qwen_image_21_vae, save_qwen_image_21_vae
from unituner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text
from unituner.models.qwen_image_2_1.vae import QwenImage21VAE


def test_public_cache_default_mismatched_references(tmp_path):
    test_public_cache_mean_text_metadata_reuse_and_invalidation(tmp_path, 2, False, True, different=True)


@pytest.mark.parametrize(
    "count,independent,bucketed",
    [(0, False, False), (2, False, False), (2, False, True), (1, False, False), (2, True, False), (1, False, True)],
)
def test_public_cache_mean_text_metadata_reuse_and_invalidation(
    tmp_path, count, independent, bucketed, *, different=False
):
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
                height, width = (
                    ((35, 49) if index == 0 else (95, 113))
                    if directory == "images" or (not independent and not different and count > 0)
                    else (32, 64)
                    if directory == "ref0"
                    else (64, 32)
                )
                channels = 3 if index else 4
                pixels = (
                    np.arange(height * width * channels, dtype=np.uint32).reshape(height, width, channels) * 17
                    + index * 11
                ).astype(np.uint8)
                if count == 2 and not independent and not different:
                    # A region landmark shared by target, original and grayscale
                    # mask detects crop offsets and channel-conversion differences.
                    pixels[..., 0] = 0
                    pixels[height // 4 : height * 3 // 4, width // 3 : width * 2 // 3, 0] = 255
                    if channels == 4:
                        pixels[..., 3] = 255
                Image.fromarray(
                    pixels[..., 0] if directory == "ref1" and not independent and not different else pixels
                ).save(tmp_path / directory / f"{name}.png")
        (tmp_path / "images/a.txt").write_text("short")
        (tmp_path / "images/b.txt").write_text("longer caption about textures")
        config = tmp_path / "cache.toml"
        config.write_text(f'''[model]
family = "qwen-image-2-1"
variant = "base"
weights = "{tmp_path.as_posix()}/not-loaded.safetensors"
vae = "{tmp_path.as_posix()}/vae.safetensors"
text_encoder = "{tmp_path.as_posix()}/text"
[dataset.groups.main]
source = "{tmp_path.as_posix()}/images"
cache_directory = "{tmp_path.as_posix()}/cache"
resolution = [64,32]
enable_bucket = {str(bucketed).lower()}
reference_sources = {json.dumps([str(tmp_path / f"ref{i}") for i in range(count)])}
{"reference_resolution = [128,64]" if independent else ""}
[train]
parameter_mode = "full"
''')
        device = torch.device(os.environ.get("UNITUNER_TEST_DEVICE", "cpu"))
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
            resolution = ImageBuckets((64, 32), alignment=32, enabled=bucketed).for_image(
                tmp_path / "images" / f"{name}.png"
            )
            with Image.open(tmp_path / "images" / f"{name}.png") as image:
                target = resize_pixels(np.array(image.convert("RGBA"), copy=True), resolution)
            controls = []
            for i in range(count):
                with Image.open(tmp_path / f"ref{i}" / f"{name}.png") as image:
                    rgba = np.array(image.convert("RGBA"), copy=True)
                size = (
                    reference_resolution(rgba.shape[1], rgba.shape[0], area=128 * 64, alignment=32)
                    if independent
                    else (64, 32)
                    if different and i == 0
                    else (32, 64)
                    if different
                    else resolution
                )
                controls.append(resize_pixels(rgba, size))
            expected = encode_images(frozen, [target, *controls])
            if count == 2 and not independent and not different:
                assert target.shape == controls[0].shape == controls[1].shape
                assert np.array_equal(target, controls[0])
                assert np.array_equal(target[..., 0], controls[1][..., 0])
            assert torch.equal(record.latents, expected[0])
            assert all(torch.equal(a, b) for a, b in zip(record.references, expected[1:], strict=True))
            if count == 1 and not different:
                # Identical target/control pixels must stay exactly aligned for
                # both upsampling and downsampling, including the posterior mean.
                assert torch.equal(record.latents, record.references[0])
            prompt = (tmp_path / "images" / f"{name}.txt").read_text()
            encoded = text.encode([prompt], references=[controls])[0]
            assert torch.equal(record.caption, encoded.features)
            assert record.image_slots.tolist() == list(encoded.image_slots)
            assert record.reference_grids.tolist() == [list(grid) for grid in encoded.image_shapes]
            if count:
                assert record.references[0].shape[-2:] == (
                    (4, 8) if independent else (2, 4) if different else record.latents.shape[-2:]
                )
            if count == 2:
                assert record.references[1].shape[-2:] == (
                    (8, 4) if independent else (4, 2) if different else record.latents.shape[-2:]
                )
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
        if independent:
            # Changing the area budget must rebuild references.
            config.write_text(config.read_text().replace("reference_resolution = [128,64]", ""))
            assert main_with_local_tokenizers(["cache", str(config), "--device", str(device)]) == 0
            paired = load_latent_manifest(destination)
            assert all(a.file != b.file for a, b in zip(paired.entries, updated.entries, strict=True))
            for entry in paired.entries:
                refs = read_qwen_image_21_record(destination, entry, device=device).references
                assert refs[0].shape[-2:] == (2, 4)
                assert refs[1].shape[-2:] == (2, 4)
        assert torch.equal(rng, torch.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    finally:
        torch.set_num_threads(previous)
