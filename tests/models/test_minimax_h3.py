"""Validate native AV artifacts, input ownership and AdaLN topology boundaries."""

from contextlib import closing

import pytest
import torch
from pydantic import ValidationError
from torch.nn.attention import SDPBackend, sdpa_kernel

from unituner.adapters.config import AdapterConfig
from unituner.adapters.loading import LoadedAdapter
from unituner.artifacts.formats.adapters import (
    AdapterTargetKeys,
    adapter_key_map,
    adapter_prefixes,
    convert_adapter_keys,
)
from unituner.artifacts.tensors import TensorReader, save_tensors
from unituner.cli.main import main
from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.layout import build_layout
from unituner.models.minimax_h3.loading import load_minimax_h3, save_minimax_h3
from unituner.models.minimax_h3.projections import adapter_projections
from unituner.models.minimax_h3.transformer import MiniMaxH3Transformer
from unituner.training.parameters import ParameterInstallation


def _model(variant):
    config = MiniMaxH3Config(
        hidden_size=24,
        num_layers=2,
        token_refiner_num_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        ffn_hidden_size=40,
        text_dim=12,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        rope_inv_freq_len=2,
        adaln_curve_grid=1025 if variant == "curve" else None,
        adaln_rank=3 if variant == "basis" else None,
    )
    model = MiniMaxH3Transformer(config)
    with torch.no_grad():
        for buffer in model.buffers():
            buffer.copy_(torch.linspace(0.01, 0.3, buffer.numel()).reshape(buffer.shape))
    return model


@pytest.mark.parametrize("variant", ["full", "curve", "basis"])
def test_av_artifact_round_trip_preserves_topology_weights_and_rng(tmp_path, variant):
    torch.manual_seed(607)
    model = _model(variant)
    source = tmp_path / "source.safetensors"
    save_tensors(model.state_dict(), source)
    before = source.read_bytes()
    rng = torch.get_rng_state().clone()
    loaded = load_minimax_h3(source, config=model.config)
    assert torch.equal(rng, torch.get_rng_state()) and source.read_bytes() == before
    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in model.state_dict().items())
    destination = tmp_path / "native.safetensors"
    save_minimax_h3(loaded, destination, metadata={"note": "native"})
    with TensorReader(destination) as reader:
        assert not any(key.startswith("ss_") for key in reader.metadata())
    restored = load_minimax_h3(destination)
    assert restored.config == model.config
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    with pytest.raises(ValueError, match="topology differs"):
        load_minimax_h3(destination, config=model.config.model_copy(update={"num_layers": 3}))
    broken = dict(model.state_dict())
    broken["rope.inv_freq"] = torch.full_like(broken["rope.inv_freq"], float("nan"))
    save_tensors(broken, tmp_path / "broken.safetensors")
    with pytest.raises(ValueError, match="invalid tensor"):
        load_minimax_h3(tmp_path / "broken.safetensors", config=model.config)


def test_native_av_prediction_and_backward_preserve_inputs_parameters_and_rng():
    torch.manual_seed(613)
    model = _model("full")
    layout = build_layout(task="t2va", text_length=3, target_video=(2, 2, 2), target_audio_frames=8)
    video = torch.randn(1, 24, 2, 2, 2, requires_grad=True)
    audio = torch.randn(1, 32, 2, 8, requires_grad=True)
    text = torch.randn(1, 3, 12, requires_grad=True)
    originals = tuple(value.detach().clone() for value in (video, audio, text))
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with sdpa_kernel(SDPBackend.MATH):
        result = model(
            video_latents=video,
            audio_latents=audio,
            text_hidden_states=text,
            text_token_tags=torch.tensor([[0, 1, 0]]),
            layout=layout,
            model_t_video=0.25,
            model_t_audio=0.5,
        )
        (result.video.square().mean() + result.audio.square().mean()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in (video, audio, text))
    assert all(torch.equal(value, original) for value, original in zip((video, audio, text), originals, strict=True))
    assert all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state()) and model.training
    with pytest.raises(ValueError, match="batch_size=1"):
        model(
            video_latents=video.expand(2, -1, -1, -1, -1),
            audio_latents=audio.expand(2, -1, -1, -1),
            text_hidden_states=text.expand(2, -1, -1),
            text_token_tags=torch.tensor([[0, 1, 0], [0, 1, 0]]),
            layout=layout,
            model_t_video=0.25,
            model_t_audio=0.5,
        )


def test_native_av_config_rejects_invalid_rotary_norm_and_adaln_topologies():
    for fields in (
        {"rope_inv_freq_len": 22},
        {"norm_eps": float("nan")},
        {"norm_eps": 0.0},
        {"hidden_size": True},
        {"adaln_curve_grid": 1025, "time_embed_dim": 8, "adaln_rank": 3},
        {"adaln_curve_grid": 1024},
        {"timestep_input_dim": 7},
        {"patch_size": (1, 1, 2)},
    ):
        with pytest.raises(ValidationError):
            MiniMaxH3Config.model_validate(fields)


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
def test_native_adapters_close_reload_and_export_without_changing_base(tmp_path, algorithm):
    torch.manual_seed(619)
    model = _model("full")
    targets = adapter_projections(model)
    assert len(targets) == model.config.num_layers * 4
    flags = tuple(parameter.requires_grad for parameter in model.parameters())
    identities = tuple(map(id, model.parameters()))
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    config = AdapterConfig(algorithm=algorithm, rank=2, alpha=7.0)
    prefixes = adapter_prefixes(targets, algorithm=algorithm)
    projection = next(iter(targets.values()))
    inputs = torch.randn(1, 3, projection.in_features)
    plain = projection(inputs).detach().clone()
    with closing(ParameterInstallation(model, adapter=config, targets=targets)) as owner:
        with torch.no_grad():
            for parameter in owner.branches.parameters():
                parameter.normal_(std=0.1)
        adapted = projection(inputs).detach().clone()
        assert not torch.equal(plain, adapted)
        mapping = adapter_key_map(
            [AdapterTargetKeys(str(index), prefixes[name]) for index, name in enumerate(targets)], algorithm=algorithm
        )
        published = convert_adapter_keys(owner.branches.state_dict(), mapping)
        artifact = tmp_path / "adapter.safetensors"
        save_tensors(
            published, artifact, metadata={"unituner.adapter.layout": "minimax-h3-native", "network_alpha": "7.0"}
        )
    assert tuple(parameter.requires_grad for parameter in model.parameters()) == flags
    assert tuple(map(id, model.parameters())) == identities
    assert torch.equal(projection(inputs), plain)
    rng = torch.get_rng_state().clone()
    with closing(LoadedAdapter(artifact, config, targets, external_prefixes=prefixes)):
        assert torch.equal(projection(inputs), adapted)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())
    topology = tmp_path / "topology.json"
    topology.write_text(model.config.model_dump_json())
    destination = tmp_path / "consumer.safetensors"
    source_bytes = artifact.read_bytes()
    assert (
        main(
            [
                "adapters",
                "export-comfyui",
                str(artifact),
                str(destination),
                "--topology",
                str(topology),
                "--algorithm",
                algorithm,
                "--family",
                "minimax-h3",
            ]
        )
        == 0
    )
    with TensorReader(destination) as reader:
        assert reader.metadata()["unituner.adapter.layout"] == "minimax-h3-comfyui"
        assert not any(key.startswith("ss_") for key in reader.metadata())
        consumer_map = adapter_key_map(
            [AdapterTargetKeys(str(index), "diffusion_model." + name) for index, name in enumerate(targets)],
            algorithm=algorithm,
        )
        for internal, external in consumer_map.items():
            assert torch.equal(reader.read(external), published[mapping[internal]])
    assert artifact.read_bytes() == source_bytes
    before = destination.read_bytes()
    incomplete = dict(published)
    incomplete.pop(next(iter(incomplete)))
    save_tensors(incomplete, tmp_path / "incomplete.safetensors")
    assert (
        main(
            [
                "adapters",
                "export-comfyui",
                str(tmp_path / "incomplete.safetensors"),
                str(destination),
                "--topology",
                str(topology),
                "--algorithm",
                algorithm,
                "--family",
                "minimax-h3",
            ]
        )
        == 2
    )
    assert destination.read_bytes() == before
