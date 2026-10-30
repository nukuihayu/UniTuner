"""Exercise real public Wan bucketed adapter training, export and exact checkpoint recovery."""

import shutil

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.support.text_assets import main_with_local_tokenizers
from unituner.artifacts.tensors import TensorReader
from unituner.data.cache.schema import CacheIdentity
from unituner.models.wan.config.model import WanConfig
from unituner.models.wan.data.batching import build_wan_batches
from unituner.models.wan.data.cache.storage import (
    WanCacheFormat,
    WanCacheManifest,
    WanCacheRecord,
    save_wan_manifest,
    save_wan_record,
)
from unituner.models.wan.loading import save_wan
from unituner.models.wan.transformer import WanTransformer


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "algorithm,version,variant,control",
    [("lora", "2.1", "i2v", False), ("loha", "2.2", "i2v", True), ("lokr", "2.1", "t2v", False)],
)
def test_public_bucketed_training_export_and_resume(tmp_path, algorithm, version, variant, control, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(529)
        topology = WanConfig(
            version=version,
            variant=variant,
            hidden_size=48,
            num_heads=4,
            num_layers=2,
            ffn_dim=80,
            timestep_dim=16,
            text_dim=32,
            text_length=7,
            in_channels=48 if control else 36 if variant == "i2v" else 16,
        )
        model = WanTransformer(topology)
        with torch.no_grad():
            model.head.head.weight.normal_(std=0.03)
        weights = tmp_path / "source.safetensors"
        save_wan(model, weights)
        cache = tmp_path / "cache"
        geometry = WanCacheFormat(
            version=version,
            variant=variant,
            text_dim=32,
            image_dim=1280 if topology.image_context else None,
            control=control,
        )
        entries = []
        for index, frames in enumerate([1, 2, 1, 2, 1]):
            latents = torch.randn(16, frames, 4, 4)
            condition = None
            if variant == "i2v":
                visibility = torch.cat((torch.ones(4, 1, 4, 4), torch.zeros(4, frames - 1, 4, 4)), dim=1)
                condition = torch.cat((visibility, torch.randn_like(latents)))
            record = WanCacheRecord(
                latents,
                torch.randn(3 + index % 2, 32),
                condition,
                torch.randn(257, 1280) if topology.image_context else None,
                torch.randn_like(latents) if control else None,
            )
            entries.append(
                save_wan_record(
                    cache,
                    CacheIdentity(
                        source_sha256=str(index) * 64,
                        encoder_sha256="b" * 64,
                        transform_sha256="c" * 64,
                        layout=geometry.layout,
                    ),
                    record,
                    manifest=geometry,
                )
            )
        manifest = WanCacheManifest(**geometry.model_dump(), entries=tuple(entries))
        save_wan_manifest(manifest, cache)
        plan = build_wan_batches(cache, manifest, batch_size=2, repeats=1)
        assert sorted(len(plan.batch(i)) for i in range(plan.num_batches)) == [1, 2, 2]
        assert plan.samples_before(6) == 10
        document = f'''[model]
family="wan"
version="{version}"
variant="{variant}"
weights="{weights.as_posix()}"
[dataset.groups.main]
kind="video"
source="unused"
cache_directory="{cache.as_posix()}"
resolution=[32,32]
batch_size=2
[train]
max_updates=3
learning_rate=0.001
accumulation_steps=2
lr_schedule="cosine"
warmup_updates=1
[train.adapter]
algorithm="{algorithm}"
rank=2
alpha=7.0
[execution]
recompute="block"
[output]
max_resume_checkpoints = 0
directory="{(tmp_path / "continuous").as_posix()}"
name="wan"
save_every_updates=1
save_precision="bf16"
'''
        second = tmp_path / "second-cache"
        shutil.copytree(cache, second)
        document += f'''[dataset.groups.second]
kind="video"
source="second-unused"
cache_directory="{second.as_posix()}"
resolution=[32,32]
batch_size=1
repeats=2
'''
        path = tmp_path / "train.toml"
        path.write_text(document)
        with sdpa_kernel(SDPBackend.MATH):
            assert main_with_local_tokenizers(["train", str(path), "--device", device]) == 0
        resumed = tmp_path / "resume.toml"
        resumed.write_text(document.replace((tmp_path / "continuous").as_posix(), (tmp_path / "resumed").as_posix()))
        with sdpa_kernel(SDPBackend.MATH):
            assert (
                main_with_local_tokenizers(
                    [
                        "train",
                        str(resumed),
                        "--device",
                        device,
                        "--resume",
                        str(tmp_path / "continuous/resume/wan.state-00000001.safetensors"),
                    ]
                )
                == 0
            )
        for filename in ["weights/wan.safetensors", "resume/wan.state-00000003.safetensors"]:
            assert (tmp_path / "continuous" / filename).read_bytes() == (tmp_path / "resumed" / filename).read_bytes()
        with TensorReader(tmp_path / "resumed/weights/wan.safetensors") as reader:
            prefix = "diffusion_model."
            assert all(key.startswith(prefix) for key in reader.keys())
            assert all(reader.read(key).dtype == torch.bfloat16 for key in reader.keys())
            assert any(key.endswith(".alpha") for key in reader.keys())
            assert reader.metadata()["network_alpha"] == "7.0"
            assert not any(key.startswith("ss_") for key in reader.metadata())
            assert reader.metadata()["unituner.adapter.layout"] == "wan-comfyui"
    finally:
        torch.set_num_threads(previous)
