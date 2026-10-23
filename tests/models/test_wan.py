"""Exercise strict Wan artifact ownership and explicit native geometry boundaries."""

import pytest
import torch

from unitunner.artifacts.formats.qwen_image import convert_vae_tensors, vae_key_map
from unitunner.artifacts.tensors import TensorReader, save_tensors
from unitunner.cli.main import main
from unitunner.models.qwen_image.config.vae import QwenImageVAEConfig
from unitunner.models.qwen_image.loading import load_qwen_image_vae
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.models.wan.conditioning import encode_latent_conditions
from unitunner.models.wan.config.model import WanConfig
from unitunner.models.wan.loading import load_wan, save_wan
from unitunner.models.wan.transformer import WanTransformer


@pytest.fixture
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(721)
    yield WanTransformer(
        WanConfig(hidden_size=48, num_heads=4, num_layers=2, ffn_dim=80, timestep_dim=16, text_dim=32, text_length=7)
    )
    torch.set_num_threads(previous)


def test_public_import_exact_state_and_rng(tmp_path, model):
    source, destination = tmp_path / "external.safetensors", tmp_path / "native.safetensors"
    topology = tmp_path / "topology.json"
    topology.write_text(model.config.model_dump_json())
    save_tensors(model.state_dict(), source)
    rng = torch.get_rng_state().clone()
    assert (
        main(
            [
                "weights",
                "import",
                str(source),
                str(destination),
                "--family",
                "wan",
                "--kind",
                "transformer",
                "--topology",
                str(topology),
            ]
        )
        == 0
    )
    loaded = load_wan(destination)
    assert loaded.config == model.config
    assert torch.equal(torch.get_rng_state(), rng)
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    save_wan(loaded, destination, metadata={"note": "native"})
    with TensorReader(destination) as reader:
        assert reader.metadata()["note"] == "native"
        assert not any(key.startswith("ss_") for key in reader.metadata())
    with pytest.raises(ValueError, match="topology differs"):
        load_wan(destination, config=model.config.model_copy(update={"version": "2.2"}))
    low_precision = load_wan(destination, dtype=torch.bfloat16)
    assert low_precision.patch_embedding.weight.dtype == torch.bfloat16
    assert next(low_precision.time_embedding.parameters()).dtype == torch.float32


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "nonfinite"])
def test_reject_invalid_artifact_before_publication(tmp_path, model, damage):
    state = dict(model.state_dict())
    key = next(iter(state))
    if damage == "missing":
        state.pop(key)
    elif damage == "extra":
        state["unexpected"] = torch.ones(1)
    elif damage == "shape":
        state[key] = state[key].flatten()
    else:
        state[key] = torch.full_like(state[key], float("nan"))
    source, destination, topology = (
        tmp_path / "invalid.safetensors",
        tmp_path / "result.safetensors",
        tmp_path / "topology.json",
    )
    save_tensors(state, source)
    destination.write_bytes(b"existing artifact")
    topology.write_text(model.config.model_dump_json())
    with pytest.raises(ValueError, match=r"keys differ|invalid tensor"):
        load_wan(source, config=model.config)
    assert (
        main(
            [
                "weights",
                "import",
                str(source),
                str(destination),
                "--family",
                "wan",
                "--kind",
                "transformer",
                "--topology",
                str(topology),
            ]
        )
        == 2
    )
    assert destination.read_bytes() == b"existing artifact"


def test_geometry_rejects_silent_truncation_and_invalid_frames(model):
    clip, caption, times = torch.randn(16, 1, 4, 4), torch.randn(3, 32), torch.tensor([244.0])
    result = model((clip,), times, (caption,), sequence_length=8, frame_indices=((8,),))
    assert result[0].shape == clip.shape
    assert result[0].dtype == torch.float32
    with pytest.raises(ValueError, match="multiples"):
        model((clip[:, :, :, :3],), times, (caption,), sequence_length=8)
    with pytest.raises(ValueError, match="sequence_length"):
        model((clip,), times, (caption,), sequence_length=3)
    with pytest.raises(ValueError, match="frame_indices"):
        model((clip,), times, (caption,), sequence_length=8, frame_indices=((0.5,),))
    with pytest.raises(ValueError, match="timesteps"):
        model((clip,), times[:, None], (caption,), sequence_length=8)


def test_unsupported_conditioning_is_explicit(model):
    clip, caption, times = torch.randn(16, 1, 4, 4), torch.randn(3, 32), torch.tensor([244.0])
    with pytest.raises(ValueError, match="only apply"):
        model((clip,), times, (caption,), sequence_length=4, image_features=torch.randn(1, 257, 1280))
    with pytest.raises(ValueError, match="channels"):
        model((clip,), times, (caption,), sequence_length=4, conditioning=(torch.randn(20, 1, 4, 4),))
    with pytest.raises(ValueError, match="variant"):
        WanConfig.model_validate({"variant": "unsupported"})


def test_shared_video_vae_import_and_conditions(tmp_path, model):
    config = QwenImageVAEConfig(base_channels=4, layers_per_block=2)
    vae = QwenImageVAE(config).eval()
    keys = vae_key_map(tuple(vae.state_dict()), num_blocks=4, layers_per_block=2)
    source, destination, topology = (
        tmp_path / "wan.safetensors",
        tmp_path / "native.safetensors",
        tmp_path / "config.json",
    )
    save_tensors(
        {"vae." + name: value for name, value in convert_vae_tensors(vae.state_dict(), keys, to_native=False).items()},
        source,
    )
    topology.write_text(config.model_dump_json())
    rng = torch.get_rng_state().clone()
    assert (
        main(
            [
                "weights",
                "import",
                str(source),
                str(destination),
                "--family",
                "wan",
                "--kind",
                "vae",
                "--topology",
                str(topology),
                "--source-format",
                "external",
                "--prefix",
                "vae.",
            ]
        )
        == 0
    )
    loaded = load_qwen_image_vae(destination)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in vae.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)
    pixels, control = torch.randn(2, 3, 9, 8, 16), torch.randn(2, 3, 1, 8, 16)
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        result = encode_latent_conditions(loaded, pixels, image_conditioning=True, control_pixels=control)
        repeated = encode_latent_conditions(loaded, pixels, image_conditioning=True, control_pixels=control)
    assert torch.equal(result.latents, repeated.latents)
    assert result.latents.shape == (2, 16, 3, 1, 2)
    assert result.image_condition is not None
    assert result.image_condition.shape == (2, 20, 3, 1, 2)
    assert result.image_condition is not None
    assert torch.equal(result.image_condition[:, :4, 0], torch.ones(2, 4, 1, 2))
    assert result.image_condition is not None
    assert torch.count_nonzero(result.image_condition[:, :4, 1:]) == 0
    assert result.control_latents is not None
    assert result.control_latents.shape == (2, 16, 1, 1, 2)
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="1\\+4k"):
        encode_latent_conditions(loaded, pixels[:, :, :8])
    pixels.requires_grad_()
    conditioned = encode_latent_conditions(loaded, pixels, image_conditioning=True)
    assert conditioned.image_condition is not None
    gradient = torch.autograd.grad(conditioned.image_condition[:, 4:].square().mean(), pixels)[0]
    assert torch.count_nonzero(gradient[:, :, :1]) > 0
    assert torch.count_nonzero(gradient[:, :, 1:]) == 0
    topology.write_text(config.model_copy(update={"input_channels": 4}).model_dump_json())
    before = destination.read_bytes()
    assert (
        main(
            [
                "weights",
                "import",
                str(source),
                str(destination),
                "--family",
                "wan",
                "--kind",
                "vae",
                "--topology",
                str(topology),
                "--source-format",
                "external",
                "--prefix",
                "vae.",
            ]
        )
        == 2
    )
    assert destination.read_bytes() == before
