"""Exercise actual cached native training, artifact export and interrupted-run restoration."""

from pathlib import Path

import pytest
import torch
from pydantic import JsonValue

from unitunner.artifacts.tensors import TensorReader
from unitunner.config.loading import load_training_config
from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.cache.storage import (
    LatentManifest,
    load_latent_manifest,
    save_latent_manifest,
    save_latent_record,
)
from unitunner.models.z_image.config.model import ZImageConfig
from unitunner.models.z_image.loading import save_z_image
from unitunner.models.z_image.transformer import ZImageTransformer
from unitunner.training.workflows.common.images import train


@pytest.mark.parametrize("retention", [0, 1, 2])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("algorithm", [None, "lora", "loha", "lokr"])
@pytest.mark.parametrize("save_interval", [1, 2])
def test_cached_training_resume_and_export(tmp_path, monkeypatch, algorithm, grouped, save_interval, retention):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
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
        weights = tmp_path / "source.safetensors"
        save_z_image(model, weights)
        cache = tmp_path / "cache"
        entries = tuple(
            save_latent_record(
                cache,
                CacheIdentity(
                    source_sha256=str(index) * 64,
                    encoder_sha256="b" * 64,
                    transform_sha256="c" * 64,
                    layout="z-image/CHW+LD",
                ),
                torch.randn(2, 4, 6),
                torch.randn(3 + index, 8),
            )
            for index in range(3)
        )
        save_latent_manifest(LatentManifest(entries=entries), cache)
        groups: dict[str, JsonValue] = {"main": {"cache_directory": str(cache), "batch_size": 2}}
        if grouped:
            second = tmp_path / "second-cache"
            second_entries = tuple(
                save_latent_record(
                    second,
                    CacheIdentity(
                        source_sha256=str(index) * 64,
                        encoder_sha256="b" * 64,
                        transform_sha256="c" * 64,
                        layout="z-image/CHW+LD",
                    ),
                    torch.randn(2, 6, 4),
                    torch.randn(4, 8),
                )
                for index in (4, 5)
            )
            save_latent_manifest(LatentManifest(entries=second_entries), second)
            groups["second"] = {
                "source": "second-images",
                "cache_directory": str(second),
                "batch_size": 1,
                "repeats": 2,
            }
        config = load_training_config(
            Path(__file__).resolve().parents[2] / "docs/examples/full.toml",
            overrides={
                "model": {"family": "z-image", "variant": "base", "weights": str(weights)},
                "dataset": {"groups": groups},
                "train": {
                    "include_layers": ["layers.0"],
                    "exclude_layers": ["layers.0.feed_forward.w3"],
                    "max_updates": 3,
                    "lr_schedule": "cosine",
                    "warmup_updates": 1,
                    "accumulation_steps": 2,
                    "parameter_mode": "full" if algorithm is None else "adapter",
                    "adapter": None if algorithm is None else {"algorithm": algorithm, "rank": 2, "alpha": 3.0},
                },
                "execution": {"precision": "fp32"},
                "output": {
                    "max_resume_checkpoints": retention,
                    "name": "model",
                    "directory": str(tmp_path / "continuous"),
                    "save_every_updates": save_interval,
                },
            },
        )
        state = train(config, device=torch.device("cpu"))
        assert (state.updates, state.microbatches, state.samples) == (3, 6, 7 if grouped else 9)
        output = tmp_path / "continuous"
        assert not list(output.glob("*.safetensors"))
        saved_updates = [update for update in (1, 2, 3) if update % save_interval == 0 or update == 3]
        retained_updates = saved_updates[-retention:] if retention else saved_updates
        for update in (1, 2, 3):
            if update % save_interval and update != 3:
                assert not (output / "weights" / f"model-{update:08d}.safetensors").exists()
                assert not (output / "resume" / f"model.state-{update:08d}.safetensors").exists()
                continue
            with TensorReader(output / "weights" / f"model-{update:08d}.safetensors") as reader:
                assert "unitunner.checkpoint" not in reader.metadata()
            state_path = output / "resume" / f"model.state-{update:08d}.safetensors"
            if update in retained_updates:
                with TensorReader(state_path) as reader:
                    assert "unitunner.checkpoint" in reader.metadata()
            else:
                assert not state_path.exists()
        assert (output / "weights/model-00000003.safetensors").read_bytes() == (
            output / "weights/model.safetensors"
        ).read_bytes()
        resumed = config.model_copy(
            update={
                "train": config.train.model_copy(update={"skip_dataset_validation": True}),
                "output": config.output.model_copy(
                    update={"directory": str(tmp_path / "resumed"), "save_precision": "bf16"}
                )
            }
        )
        restored = train(
            resumed,
            device=torch.device("cpu"),
            resume=tmp_path / "continuous/resume" / f"model.state-{retained_updates[0]:08d}.safetensors",
        )
        assert state == restored
        filenames = ["weights/model.safetensors"]
        if retained_updates[0] < 3:
            filenames.append("resume/model.state-00000003.safetensors")
        for filename in filenames:
            with (
                TensorReader(tmp_path / "continuous" / filename) as expected,
                TensorReader(tmp_path / "resumed" / filename) as actual,
            ):
                assert actual.keys() == expected.keys()
                for key in actual.keys():
                    reference = expected.read(key)
                    if filename == "weights/model.safetensors" and reference.is_floating_point():
                        reference = reference.to(torch.bfloat16)
                    torch.testing.assert_close(actual.read(key), reference, atol=0, rtol=0)
        if algorithm is not None:
            with TensorReader(tmp_path / "resumed/weights/model.safetensors") as reader:
                assert not any("feed_forward.w3" in key or "feed_forward_w3" in key for key in reader.keys())
                assert all(
                    key.startswith("transformer.layers.0." if algorithm == "lora" else "lycoris_layers_0_")
                    for key in reader.keys()
                )
        if algorithm is None and not grouped and save_interval == 1 and retention == 1:
            from unitunner.training import checkpointing

            save = checkpointing.save_checkpoint

            def fail_second_save(path, *args, **kwargs):
                if path.name.endswith("00000002.safetensors"):
                    raise OSError("Simulated disk failure")
                return save(path, *args, **kwargs)

            failed_output = tmp_path / "failed"
            failed_config = config.model_copy(
                update={
                    "output": config.output.model_copy(
                        update={
                            "directory": str(failed_output),
                            "max_weight_checkpoints": 1,
                        }
                    )
                }
            )
            with monkeypatch.context() as patch:
                patch.setattr(checkpointing, "save_checkpoint", fail_second_save)
                with pytest.raises(OSError, match="Simulated disk failure"):
                    train(failed_config, device=torch.device("cpu"))
            assert (failed_output / "resume/model.state-00000001.safetensors").is_file()
            assert (failed_output / "weights/model-00000001.safetensors").is_file()
            assert not (failed_output / "resume/model.state-00000002.safetensors").exists()
    finally:
        torch.set_num_threads(previous)


def test_corrupted_cache_rejected_before_training(tmp_path):
    identity = CacheIdentity(
        source_sha256="a" * 64, encoder_sha256="b" * 64, transform_sha256="c" * 64, layout="z-image/CHW+LD"
    )
    entry = save_latent_record(tmp_path, identity, torch.randn(2, 4, 6), torch.randn(3, 8))
    save_latent_manifest(LatentManifest(entries=(entry,)), tmp_path)
    with (tmp_path / entry.file).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="digest"):
        load_latent_manifest(tmp_path)


def test_cache_identity_collision_keeps_existing_record(tmp_path):
    identity = CacheIdentity(
        source_sha256="a" * 64, encoder_sha256="b" * 64, transform_sha256="c" * 64, layout="z-image/CHW+LD"
    )
    latents, caption = torch.randn(2, 4, 6), torch.randn(3, 8)
    entry = save_latent_record(tmp_path, identity, latents, caption)
    before = (tmp_path / entry.file).read_bytes()
    assert save_latent_record(tmp_path, identity, latents, caption) == entry
    with pytest.raises(ValueError, match="identity collision"):
        save_latent_record(tmp_path, identity, latents + 1, caption)
    assert (tmp_path / entry.file).read_bytes() == before
