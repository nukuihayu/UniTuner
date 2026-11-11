"""Exercise H3-Image caching, CLI commands and native training/checkpoint wiring."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from unituner.artifacts.tensors import save_tensors
from unituner.config.cache import CacheConfig
from unituner.config.generation import GenerationConfig
from unituner.config.model import EncoderModelConfig
from unituner.config.restoration_inference import RestorationInferenceConfig
from unituner.config.training import TrainingConfig
from unituner.data.cache.representation import RepaManifest, save_repa_manifest, save_repa_target
from unituner.data.cache.schema import CacheIdentity
from unituner.data.production.minimax_h3_image import image_source_digest, produce_minimax_h3_image
from unituner.inference.generation import run as generate_command
from unituner.inference.restoration.dispatch import run as restore_command
from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.image import MiniMaxH3ImageTransformer, save_image_transformer
from unituner.models.minimax_h3.image_cache import (
    MiniMaxH3ImageManifest,
    MiniMaxH3ImageRecord,
    load_image_manifest,
    read_image_record,
    save_image_manifest,
    save_image_record,
)
from unituner.models.minimax_h3.image_encoders import load_image_encoders
from unituner.training.integrations.minimax_h3.image_previews import MiniMaxH3ImagePreviews
from unituner.training.output import resume_path
from unituner.training.workflows.minimax_h3.image import train_minimax_h3_image


def _topology(*, layers=1):
    return MiniMaxH3Config(
        hidden_size=24,
        num_layers=layers,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=12,
        ffn_hidden_size=32,
        text_dim=5120,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        rope_inv_freq_len=2,
    )


def _workspace(tmp_path, *, restoration=False, layers=1):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "one.png"
    Image.fromarray(np.zeros((32, 64, 3), dtype=np.uint8)).save(image)
    (source / "one.txt").write_text("portrait", encoding="utf-8")
    cache = tmp_path / "cache"
    record = MiniMaxH3ImageRecord(
        torch.randn(24, 1, 2, 4),
        torch.randn(2, 5120),
        torch.tensor([1, 0]),
        () if restoration else (torch.randn(24, 1, 2, 4),),
    )
    identity = CacheIdentity(
        source_sha256=image_source_digest(image, "portrait"),
        encoder_sha256="a" * 64,
        transform_sha256="b" * 64,
        layout=f"minimax-h3-image/1/C1HW+LD+tags+R{len(record.references)}",
    )
    entry = save_image_record(cache, identity, record)
    save_image_manifest(cache, MiniMaxH3ImageManifest(entries=(entry,)))
    weights = tmp_path / "model.safetensors"
    schedule = tmp_path / "pdd_config.json"
    schedule.write_text(json.dumps({"num_inference_steps": 1, "sigmas": [1.0, 0.0]}), encoding="utf-8")
    save_image_transformer(MiniMaxH3ImageTransformer(_topology(layers=layers), dtype=torch.float32), weights)
    data = {
        "model": {
            "family": "minimax-h3-image",
            "variant": "base",
            "weights": str(weights),
            "pdd_config": str(schedule),
            "image_topology": _topology(layers=layers).model_dump(),
        },
        "dataset": {
            "groups": {
                "main": {"source": str(source), "cache_directory": str(cache), "resolution": [64, 32], "batch_size": 1}
            }
        },
        "train": {"parameter_mode": "full", "max_updates": 1},
        "execution": {"precision": "fp32"},
        "output": {"directory": str(tmp_path / "run")},
    }
    if restoration:
        low = tmp_path / "low"
        low.mkdir()
        Image.fromarray(np.zeros((32, 64, 3), dtype=np.uint8)).save(low / "one.png")
        data["dataset"]["groups"]["main"].update(kind="restoration", low_quality_source=str(low))
        data["method"] = {"kind": "restoration", "restoration": {"sigma": 0.5, "encoder_rank": 0}}
        data["losses"] = {"lrr": 0, "dv3d": 0, "gan": 0, "fdl": 0, "l1": 1}
        data["model"]["vae"] = str(tmp_path / "vae.safetensors")
        save_tensors({"weight": torch.ones(1)}, tmp_path / "vae.safetensors")
    return data, entry


@pytest.mark.parametrize(("configured", "expected"), [(None, "cuda"), ("cpu", "cpu")])
def test_image_encoder_devices_share_one_policy(monkeypatch, configured, expected):
    model = EncoderModelConfig.model_validate_json(
        json.dumps(
            {
                "family": "minimax-h3-image",
                "variant": "base",
                "vae": "vae",
                "text_encoder": "text",
                "minimax_h3": {"text_config": "config", "text_device": configured},
            }
        )
    )
    vae = torch.nn.Linear(1, 1)
    devices = []
    monkeypatch.setattr("unituner.models.minimax_h3.image_encoders.load_image_vae", lambda *args, **kwargs: vae)

    def text_encoder(*args, **kwargs):
        devices.append((kwargs["device"], kwargs["processor_family"]))
        return object()

    monkeypatch.setattr("unituner.models.minimax_h3.image_encoders.load_text_encoder", text_encoder)
    loaded, _ = load_image_encoders(model, device=torch.device("cuda"))
    assert loaded is vae and not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert devices == [(torch.device(expected), "minimax-h3-image")]


def test_image_cache_roundtrip_and_tampering(tmp_path):
    data, entry = _workspace(tmp_path)
    cache_model = {key: value for key, value in data["model"].items() if key != "weights"}
    assert (
        CacheConfig.model_validate_json(
            json.dumps(
                {
                    "model": {
                        **cache_model,
                        "vae": "vae",
                        "text_encoder": "text",
                        "minimax_h3": {"text_config": "config"},
                    },
                    "dataset": data["dataset"],
                }
            )
        ).model.family.value
        == "minimax-h3-image"
    )
    record = read_image_record(tmp_path / "cache", entry, device=torch.device("cpu"))
    assert record.token_tags.tolist() == [1, 0] and record.references[0].shape == (24, 1, 2, 4)
    assert load_image_manifest(tmp_path / "cache").entries == (entry,)
    (tmp_path / "cache" / entry.file).write_bytes(b"bad")
    with pytest.raises(ValueError, match="digest"):
        load_image_manifest(tmp_path / "cache")


def test_image_cache_production_preserves_reference_text_tags(tmp_path, monkeypatch):
    from unituner.models.minimax_h3 import image_inference

    monkeypatch.setattr(
        image_inference, "_resize_reference", lambda *args: torch.zeros(1, 32, 64, 3, dtype=torch.uint8)
    )
    data, _ = _workspace(tmp_path)
    text_path, vae_path = tmp_path / "text.safetensors", tmp_path / "vae.safetensors"
    save_tensors({"weight": torch.ones(1)}, text_path)
    save_tensors({"weight": torch.ones(1)}, vae_path)
    text_config = tmp_path / "text_config"
    text_config.mkdir()
    (text_config / "config.json").write_text("{}", encoding="utf-8")
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "processor_config.json").write_text("{}", encoding="utf-8")
    references = tmp_path / "refs"
    references.mkdir()
    Image.fromarray(np.zeros((32, 64, 3), dtype=np.uint8)).save(references / "one.png")
    cache_model = {key: value for key, value in data["model"].items() if key != "weights"}
    config = CacheConfig.model_validate_json(
        json.dumps(
            {
                "model": {
                    **cache_model,
                    "vae": str(vae_path),
                    "text_encoder": str(text_path),
                    "text": {"tokenizer": {"source": str(processor), "offline": True}},
                    "minimax_h3": {"text_config": str(text_config)},
                },
                "dataset": {
                    "groups": {"main": {**data["dataset"]["groups"]["main"], "reference_sources": [str(references)]}}
                },
            }
        )
    )

    class SmallVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def encode(self, pixels):
            return torch.zeros(1, 24, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)

    class SmallText:
        def encode(self, presentation):
            assert presentation.processor_text.startswith("Picture 1: ")
            return SimpleNamespace(hidden_states=torch.ones(2, 5120), token_tags=torch.tensor([0, 1]))

    class TargetWriter:
        def publish_normalized(self, entry, pixels):
            assert pixels.shape == (3, 1, 32, 64)

    def text_encoder(*args, **kwargs):
        assert kwargs["processor_family"] == "minimax-h3-image"
        return SmallText()

    monkeypatch.setattr("unituner.models.minimax_h3.image_encoders.load_image_vae", lambda *args, **kwargs: SmallVAE())
    monkeypatch.setattr("unituner.models.minimax_h3.image_encoders.load_text_encoder", text_encoder)
    assert produce_minimax_h3_image(config, device="cpu", repa=TargetWriter(), tokenizer_digests={}) == 1
    manifest = load_image_manifest(tmp_path / "cache")
    record = read_image_record(tmp_path / "cache", manifest.entries[0], device=torch.device("cpu"))
    assert record.token_tags.tolist() == [0, 1] and record.references[0].shape == (24, 1, 2, 4)


def test_image_flow_trains_and_restores_checkpoint(tmp_path):
    data, _ = _workspace(tmp_path)
    state = train_minimax_h3_image(TrainingConfig.model_validate_json(json.dumps(data)), device=torch.device("cpu"))
    assert (state.updates, state.samples) == (1, 1)
    assert (tmp_path / "run" / "weights").is_dir()


def test_image_flow_repa_uses_verified_target_cache(tmp_path):
    data, entry = _workspace(tmp_path)
    target_dir = tmp_path / "repa"
    encoder_hash = "a" * 64
    target = save_repa_target(target_dir, entry, torch.ones(1, 2, 4, 8), encoder_sha256=encoder_hash)
    save_repa_manifest(RepaManifest(encoder_sha256=encoder_hash, encoder_dim=8, entries=(target,)), target_dir)
    data["method"] = {
        "repa": {
            "layer": 0,
            "weight": 0.2,
            "encoder_dim": 8,
            "encoder_sha256": encoder_hash,
            "cache_directory": str(target_dir),
        }
    }
    data["output"]["save_every_updates"] = 1
    config = TrainingConfig.model_validate_json(json.dumps(data))
    state = train_minimax_h3_image(config, device=torch.device("cpu"))
    assert state.updates == 1
    assert train_minimax_h3_image(config, device=torch.device("cpu"), resume=resume_path(config, 1)) == state


def test_image_flow_layersync_tread_checkpoint_resume(tmp_path):
    data, _ = _workspace(tmp_path, layers=2)
    data["method"] = {
        "layersync": {"student_layer": 0, "teacher_layer": 1},
        "tread": {"start_layer": 0, "end_layer": 1, "drop_fraction": 0.5},
    }
    data["output"]["save_every_updates"] = 1
    config = TrainingConfig.model_validate_json(json.dumps(data))
    state = train_minimax_h3_image(config, device=torch.device("cpu"))
    assert state.updates == 1
    assert train_minimax_h3_image(config, device=torch.device("cpu"), resume=resume_path(config, 1)) == state


def test_image_adapter_rejects_unverified_diffusers_export(tmp_path):
    data, _ = _workspace(tmp_path)
    data["train"].update(parameter_mode="adapter", adapter={"algorithm": "lora", "rank": 2})
    data["output"]["adapter_format"] = "diffusers"
    with pytest.raises(ValueError, match="Diffusers LoRA projection mapping"):
        TrainingConfig.model_validate_json(json.dumps(data))


def test_image_flow_preview_preserves_automatic_reference_dimensions(tmp_path, monkeypatch):
    data, _ = _workspace(tmp_path)
    data["model"].update(vae="vae", text_encoder="text", minimax_h3={"text_config": "config"})
    data["preview"] = {
        "samples": [
            {"prompt": "edit", "reference_images": ["source.png"]},
            {"prompt": "edit", "width": 64, "reference_images": ["source.png"]},
        ]
    }
    config = TrainingConfig.model_validate_json(json.dumps(data))
    observed = []
    def encoders(*args, **kwargs):
        assert kwargs["device"] == torch.device("cpu")
        return torch.nn.Module(), object()

    monkeypatch.setattr("unituner.training.integrations.minimax_h3.image_previews.load_image_encoders", encoders)

    def generate(*args, **kwargs):
        observed.append((kwargs["width"], kwargs["height"]))
        return torch.zeros(1, 3, 1, 32, 64)

    monkeypatch.setattr("unituner.training.integrations.minimax_h3.image_previews.generate_image", generate)
    MiniMaxH3ImagePreviews(config)(0, MiniMaxH3ImageTransformer(_topology(), dtype=torch.float32), None, None)
    assert observed == [(None, None), (64, None)]


@pytest.mark.parametrize(("rank", "lrr"), [(0, 0), (1, 1)])
def test_image_restoration_trains_with_encoder_adapter(tmp_path, monkeypatch, rank, lrr):
    data, _ = _workspace(tmp_path, restoration=True)
    data["method"]["restoration"]["encoder_rank"] = rank
    data["losses"]["lrr"] = lrr

    class SmallVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(3, 3, 1)

        def encode(self, pixels):
            values = self.encoder(pixels[:, :, 0]).unsqueeze(2)
            return values[:, :1].repeat(1, 24, 1, 1, 1)[:, :, :, ::16, ::16]

        def decode(self, latents):
            return latents[:, :3].repeat_interleave(16, -2).repeat_interleave(16, -1)

    monkeypatch.setattr(
        "unituner.training.integrations.minimax_h3.restoration.load_image_vae", lambda *args, **kwargs: SmallVAE()
    )
    monkeypatch.setattr(
        "unituner.training.integrations.minimax_h3.restoration.encode_restoration_image",
        lambda vae, pixels, *, posterior: vae.encode(pixels.unsqueeze(2)),
    )
    state = train_minimax_h3_image(TrainingConfig.model_validate_json(json.dumps(data)), device=torch.device("cpu"))
    assert state.updates == 1
    assert bool(list((tmp_path / "run" / "weights").glob("*.encoder.safetensors"))) == bool(rank)


def test_image_generation_and_restoration_commands(tmp_path, monkeypatch):
    data, _ = _workspace(tmp_path)

    class SmallVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def encode(self, pixels):
            return torch.zeros(1, 24, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)

        def decode(self, latents):
            return torch.zeros(1, 3, 1, latents.shape[-2] * 16, latents.shape[-1] * 16)

    class SmallText:
        def encode(self, presentation):
            return SimpleNamespace(hidden_states=torch.zeros(1, 5120), token_tags=torch.ones(1, dtype=torch.long))

    monkeypatch.setattr(
        "unituner.models.minimax_h3.image_application.load_image_encoders",
        lambda *args, **kwargs: (SmallVAE(), SmallText()),
    )

    def restore_encoding(vae, pixels, *, posterior):
        assert posterior == "mean"
        return vae.encode(pixels.unsqueeze(2))

    monkeypatch.setattr("unituner.models.minimax_h3.image_application.encode_restoration_image", restore_encoding)
    data["model"].update(vae="vae", text_encoder="text", minimax_h3={"text_config": "config"})
    generation = GenerationConfig.model_validate_json(
        json.dumps(
            {
                "model": data["model"],
                "sampling": {"prompt": "portrait", "width": 64, "height": 32},
                "output": str(tmp_path / "generated.png"),
            }
        )
    )
    assert generate_command(generation, device="cpu") == 0
    with Image.open(tmp_path / "generated.png") as image:
        assert image.size == (64, 32)
    restoration = RestorationInferenceConfig.model_validate_json(
        json.dumps(
            {
                "model": data["model"],
                "input": str(tmp_path / "source" / "one.png"),
                "output": str(tmp_path / "restored.png"),
                "sigma": 0.5,
            }
        )
    )
    assert restore_command(restoration, device="cpu") == 0
    with Image.open(tmp_path / "restored.png") as image:
        assert image.size == (64, 32)
