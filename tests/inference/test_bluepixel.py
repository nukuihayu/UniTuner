"""Check BluePixel's editing contract without downloading the released weights."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from unituner.artifacts.tensors import save_tensors
from unituner.config.generation import GenerationConfig
from unituner.config.loading import load_training_config
from unituner.config.restoration_inference import RestorationInferenceConfig
from unituner.data.cache.images import cache_images
from unituner.data.cache.storage import read_latent_record
from unituner.data.datasets.restoration import RestorationDataset, index_restoration_pairs
from unituner.data.media.restoration import read_restoration_pair
from unituner.inference.restoration import run as restoration
from unituner.models.bluepixel.inference.sampling import flow_schedule, sample_latents
from unituner.models.bluepixel.loading import load_bluepixel, save_bluepixel
from unituner.models.bluepixel.transformer import BluePixelTransformer
from unituner.models.bluepixel.vae import BluePixelVAE
from unituner.models.catalog import ModelFamily
from unituner.models.components.conditioning import Caption
from unituner.models.flux2.config.model import Flux2Config
from unituner.models.flux2.text.encoder import Flux2TextEncoder
from unituner.training.integrations.flux2.binding import Flux2Binding, Flux2FlowRequest
from unituner.training.workflows.common.images import train


def test_generation_contract_requires_one_aligned_reference():
    model = {"family": ModelFamily.BLUEPIXEL, "variant": "turbo", "weights": "transformer"}
    valid = {"model": model, "sampling": {"reference_images": ("input.png",), "width": 128, "height": 64},
             "output": "edited.png"}
    assert GenerationConfig.model_validate(valid).sampling.steps is None
    for changes in ({"reference_images": ()}, {"width": 96}, {"shift": 2.0},
                    {"negative_prompt": "bad"}, {"conditioning_image": "input.png"}):
        with pytest.raises(ValueError):
            GenerationConfig.model_validate({**valid, "sampling": {**valid["sampling"], **changes}})


def test_empirical_schedule_matches_published_grid():
    from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

    steps, count = 4, 256
    mu = compute_empirical_mu(image_seq_len=count, num_steps=steps)
    raw = torch.linspace(1, 1 / steps, steps)
    expected = (raw * torch.exp(torch.tensor(mu)) / (1 + (torch.exp(torch.tensor(mu)) - 1) * raw)).tolist()
    assert flow_schedule(steps, count) == pytest.approx((*expected, 0.0), abs=1e-6)


def test_reference_ids_and_cfg_are_kept_through_flow_steps():
    class Model:
        config = type("Config", (), {"in_channels": 2, "context_in_dim": 3})()

        def __init__(self):
            self.inputs = []

        def __call__(self, tokens, ids, time, context, text_ids, guidance):
            self.inputs.append((tokens.clone(), ids.clone(), context.clone()))
            return torch.full_like(tokens, context[0, 0, 0])

    model = Model()
    noise = torch.ones(1, 2, 2, 2)
    reference = torch.full_like(noise, 5)
    result = sample_latents(model, noise, reference, torch.ones(2, 3), torch.zeros(2, 3), steps=2)
    assert torch.allclose(result, torch.full_like(noise, -2))
    assert len(model.inputs) == 4
    for tokens, ids, _ in model.inputs:
        assert (ids[0, 4:, 0] == 10).all()
        assert (tokens[0, 4:] == 5).all()


def test_dcae_scales_both_directions():
    class VAE(torch.nn.Module):
        config = type("Config", (), {"latent_channels": 128, "encoder_block_out_channels": (1,) * 7,
                                     "scaling_factor": 0.4538})()

        def encode(self, pixels, *, return_dict):
            return (torch.ones(pixels.shape[0], 128, 1, 1),)

        def decode(self, latents, *, return_dict):
            return (latents,)

    vae = BluePixelVAE(VAE())
    latent = vae.encode(torch.zeros(1, 3, 64, 64))
    assert torch.allclose(latent, torch.full_like(latent, 0.4538))
    assert torch.allclose(vae.decode(latent), torch.ones_like(latent))


def test_transformer_loads_strict_native_keys(tmp_path):
    topology = Flux2Config(version="klein-9b", variant="base", in_channels=2, context_in_dim=12,
                           hidden_size=16, num_heads=2, axes_dim=(2, 2, 2, 2), depth=1,
                           depth_single_blocks=1)
    source = BluePixelTransformer(topology)
    weights = tmp_path / "transformer.safetensors"
    save_tensors(source.state_dict(), weights)
    loaded = load_bluepixel(weights, config=topology, dtype=torch.float32)
    assert loaded.config == topology
    assert loaded.state_dict().keys() == source.state_dict().keys()
    save_tensors({"unknown": torch.zeros(1)}, weights)
    with pytest.raises(ValueError, match="artifact keys differ"):
        load_bluepixel(weights, config=topology)


def test_restore_preserves_unaligned_image_size(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.png", tmp_path / "restored.png"
    Image.new("RGB", (35, 17), color=(70, 80, 90)).save(source)
    config = RestorationInferenceConfig(
        model={"family": ModelFamily.BLUEPIXEL, "variant": "turbo", "weights": "unused",
               "vae": "unused", "text_encoder": "unused"},
        input=str(source), output=str(destination), prompt="repair", sigma=0.35,
    )
    class Backend(torch.nn.Module):
        config = type("Config", (), {"latent_channels": 128, "encoder_block_out_channels": (1,) * 7,
                                     "scaling_factor": 0.4538})()

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(3, 128, 1)

        def encode(self, pixels, *, return_dict):
            return (torch.nn.functional.avg_pool2d(self.encoder(pixels), 64),)

        def decode(self, latents, *, return_dict):
            return (torch.nn.functional.interpolate(latents[:, :3], scale_factor=64),)

    topology = Flux2Config(version="klein-9b", variant="base", in_channels=128, context_in_dim=12,
                           hidden_size=16, num_heads=2, axes_dim=(2, 2, 2, 2), depth=1,
                           depth_single_blocks=1)
    captured = []
    from unituner.models.flux2.restoration import restore_latents

    def capture(model, latents, caption, *, sigma):
        captured.append((latents.shape, sigma))
        return restore_latents(model, latents, caption, sigma=sigma)

    monkeypatch.setattr(restoration, "_restoration_sigma", lambda value: value.sigma)
    monkeypatch.setattr(restoration, "encode_caption", lambda *args, **kwargs: Caption(torch.ones(1, 2, 12)))
    monkeypatch.setattr(restoration, "load_image_vae", lambda *args, **kwargs: BluePixelVAE(Backend()).eval())
    monkeypatch.setattr(restoration, "load_image_model", lambda *args, **kwargs: BluePixelTransformer(topology).eval())
    monkeypatch.setattr(restoration, "restore_flux2", capture)
    assert restoration.run(config, device="cpu") == 0
    assert captured == [(torch.Size((1, 128, 1, 1)), 0.35)]
    with Image.open(destination) as output:
        assert output.size == (35, 17)


def test_bluepixel_dcae_cache_and_native_flow(tmp_path, monkeypatch):
    class Backend(torch.nn.Module):
        config = type("Config", (), {"latent_channels": 128, "encoder_block_out_channels": (1,) * 7,
                                     "scaling_factor": 0.4538})()

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(3, 128, 1)

        def encode(self, pixels, *, return_dict):
            return (torch.nn.functional.avg_pool2d(self.encoder(pixels), 64),)

        def decode(self, latents, *, return_dict):
            return (torch.nn.functional.interpolate(latents[:, :3], scale_factor=64),)

    class Text(Flux2TextEncoder):
        def __init__(self):
            self.version = "klein-9b"
            self.model = torch.nn.Linear(1, 1).eval().requires_grad_(False)

        def encode_prompt(self, prompt):
            return Caption(torch.ones(1, 2, 12))

    source, references, cache = tmp_path / "source", tmp_path / "references", tmp_path / "cache"
    source.mkdir()
    references.mkdir()
    Image.new("RGB", (64, 64), "white").save(source / "image.png")
    Image.new("RGB", (64, 64), "black").save(references / "image.png")
    from unituner.config.dataset_inputs import DatasetInputConfig

    manifest = cache_images(
        source, cache, BluePixelVAE(Backend()).eval(), Text(), resolution=(64, 64),
        encoder_sha256="f" * 64, variant="turbo",
        data=DatasetInputConfig(reference_sources=(str(references),)),
    )
    assert manifest.family == "bluepixel" and manifest.variant == "turbo"
    record = read_latent_record(cache, manifest.entries[0], device=torch.device("cpu"))
    assert record.latents.shape == (128, 1, 1)
    assert len(record.references) == 1 and record.references[0].shape == record.latents.shape
    assert not torch.equal(record.references[0], record.latents)
    topology = Flux2Config(version="klein-9b", variant="base", in_channels=128, context_in_dim=12,
                           hidden_size=16, num_heads=2, axes_dim=(2, 2, 2, 2), depth=1,
                           depth_single_blocks=1)
    binding = Flux2Binding(BluePixelTransformer(topology))
    clean = record.latents.unsqueeze(0)
    result = binding.step(Flux2FlowRequest(
        clean, torch.zeros_like(clean), torch.tensor([0.5]), (record.caption,), ((record.references[0],),)
    ))
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert binding.model.img_in.weight.grad is not None
    restored = binding.restore(clean, Caption(record.caption.unsqueeze(0)), sigma=0.244)
    assert restored.shape == clean.shape and torch.isfinite(restored).all()

    weights = tmp_path / "bluepixel.safetensors"
    save_bluepixel(binding.model, weights)
    recipe = Path(__file__).resolve().parents[2] / "docs/examples/bluepixel-flow.toml"
    config = load_training_config(recipe, overrides={
        "model": {"weights": str(weights), "vae": str(tmp_path), "text_encoder": str(tmp_path),
                  "assets": str(tmp_path)},
        "dataset": {"groups": {"main": {"source": str(source), "reference_sources": [str(references)],
                                         "cache_directory": str(cache), "resolution": [64, 64]}}},
        "train": {"max_updates": 1},
        "execution": {"precision": "fp32", "compile": False},
        "output": {"directory": str(tmp_path / "run")},
    })
    assert train(config, device=torch.device("cpu")).updates == 1

    restore_cache = tmp_path / "restoration-cache"
    restored_manifest = cache_images(
        source, restore_cache, BluePixelVAE(Backend()).eval(), Text(), resolution=(64, 64),
        encoder_sha256="f" * 64, variant="turbo", kind="restoration",
    )
    assert restored_manifest.family == "bluepixel"
    pairs, digest = index_restoration_pairs(
        (source / "image.png",), (references / "image.png",), restore_cache, family="bluepixel"
    )
    restoration_data = RestorationDataset(pairs, digest, read_restoration_pair)
    monkeypatch.setattr(
        "unituner.training.methods.restoration.load_image_vae",
        lambda *args, **kwargs: BluePixelVAE(Backend()).eval().requires_grad_(False),
    )
    restoration_recipe = Path(__file__).resolve().parents[2] / "docs/examples/bluepixel-restoration-train.toml"
    restoration_config = load_training_config(restoration_recipe, overrides={
        "model": {"weights": str(weights), "vae": str(weights), "text_encoder": str(tmp_path),
                  "assets": str(tmp_path)},
        "dataset": {"groups": {"main": {"source": str(source), "low_quality_source": str(references),
                                         "cache_directory": str(restore_cache), "resolution": [64, 64]}}},
        "train": {"max_updates": 1},
        "execution": {"precision": "fp32", "compile": False},
        "output": {"directory": str(tmp_path / "restoration-run")},
    })
    assert train(restoration_config, device=torch.device("cpu"), restoration_data=restoration_data).updates == 1
