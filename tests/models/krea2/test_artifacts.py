"""Exercise strict native Krea2 import and nonzero published adapter reuse."""

from contextlib import closing

import pytest
import torch
from torch import nn

from tests.support.assertions import require_type
from unitunner.adapters.config import AdapterConfig
from unitunner.adapters.loading import LoadedAdapter
from unitunner.artifacts.formats.adapters import (
    AdapterTargetKeys,
    adapter_key_map,
    adapter_metadata,
    convert_adapter_keys,
)
from unitunner.artifacts.formats.krea2_adapters import krea2_adapter_prefixes
from unitunner.artifacts.tensors import TensorReader, save_tensors
from unitunner.cli.main import main
from unitunner.models.krea2.config.model import Krea2Config
from unitunner.models.krea2.loading import load_krea2, save_krea2
from unitunner.models.krea2.projections import adapter_projections
from unitunner.models.krea2.transformer import Krea2Transformer
from unitunner.models.qwen_image.config.vae import QwenImageVAEConfig
from unitunner.models.qwen_image.loading import load_qwen_image_vae
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.training.parameters import ParameterInstallation


@pytest.fixture
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(219)
    yield Krea2Transformer(
        Krea2Config(
            hidden_size=64,
            timestep_dim=16,
            text_dim=32,
            num_heads=4,
            num_kv_heads=2,
            mlp_multiplier=2,
            num_layers=1,
            text_layers=3,
            text_heads=4,
            text_kv_heads=2,
        )
    )
    torch.set_num_threads(previous)


def test_public_import_strict_reload_and_rng(tmp_path, model):
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
                "krea2",
                "--kind",
                "transformer",
                "--topology",
                str(topology),
            ]
        )
        == 0
    )
    loaded = load_krea2(destination)
    assert loaded.config == model.config
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="topology differs"):
        load_krea2(destination, config=model.config.model_copy(update={"variant": "turbo"}))
    state = dict(model.state_dict())
    state.pop(next(iter(state)))
    save_tensors(state, source)
    with pytest.raises(ValueError, match="keys differ"):
        load_krea2(source, config=model.config)
    state = dict(model.state_dict())
    state[next(iter(state))] = torch.full_like(state[next(iter(state))], float("inf"))
    save_tensors(state, source)
    with pytest.raises(ValueError, match="invalid tensor"):
        load_krea2(source, config=model.config)
    save_krea2(model, destination, metadata={"note": "calibrated"})
    with TensorReader(destination) as reader:
        assert reader.metadata()["note"] == "calibrated"
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
def test_all_linear_nonzero_adapter_publication_and_gradients(tmp_path, model, algorithm):
    targets = adapter_projections(model)
    assert set(targets) == {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    assert any(name.startswith("txtfusion.projector") for name in targets)
    prefixes = krea2_adapter_prefixes(targets, algorithm=algorithm)
    assert all(value.startswith("transformer." if algorithm == "lora" else "lycoris_") for value in prefixes.values())
    config = AdapterConfig(algorithm=algorithm, rank=2, alpha=7.0)
    identities = tuple(id(value) for value in model.parameters())
    state = {name: value.clone() for name, value in model.state_dict().items()}
    inputs = {name: torch.randn(2, 3, layer.in_features, requires_grad=True) for name, layer in targets.items()}
    base = {name: layer(inputs[name]) for name, layer in targets.items()}
    with closing(ParameterInstallation(model, adapter=config, targets=targets)) as installation:
        with torch.no_grad():
            for value in installation.branches.parameters():
                value.normal_(0, 0.13)
        expected = {name: layer(inputs[name]) for name, layer in targets.items()}
        assert all(not torch.equal(expected[name], base[name]) for name in targets)
        gradients = torch.autograd.grad(
            require_type(sum(value.square().mean() for value in expected.values()), torch.Tensor),
            tuple(inputs.values()),
        )
        mapping = adapter_key_map(
            [AdapterTargetKeys(str(index), prefixes[name]) for index, name in enumerate(targets)], algorithm=algorithm
        )
        published = convert_adapter_keys(installation.branches.state_dict(), mapping)
        metadata = adapter_metadata(
            algorithm=algorithm, base_model="krea2", rank=2, alpha=7.0, lora_module_identifier="networks.lora_krea2"
        )
        assert not any(key.startswith("ss_") for key in metadata)
        path = tmp_path / "adapter.safetensors"
        save_tensors(published, path, metadata=metadata)
    rng = torch.get_rng_state().clone()
    with closing(LoadedAdapter(path, config.model_copy(update={"alpha": 1.0}), targets, external_prefixes=prefixes)):
        actual = {name: layer(inputs[name]) for name, layer in targets.items()}
        observed = torch.autograd.grad(
            require_type(sum(value.square().mean() for value in actual.values()), torch.Tensor), tuple(inputs.values())
        )
        for name in targets:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        for value, reference in zip(observed, gradients, strict=True):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert tuple(id(value) for value in model.parameters()) == identities
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)


def test_shared_rgb_vae_import_preserves_posterior_and_normalization(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vae = QwenImageVAE(QwenImageVAEConfig(base_channels=2, layers_per_block=1))
        source, destination = tmp_path / "qwen.safetensors", tmp_path / "krea.safetensors"
        topology = tmp_path / "vae.json"
        topology.write_text(vae.config.model_dump_json())
        save_tensors(vae.state_dict(), source)
        pixels = torch.randn(1, 3, 1, 16, 16)
        rng = torch.get_rng_state().clone()
        assert (
            main(
                [
                    "weights",
                    "import",
                    str(source),
                    str(destination),
                    "--family",
                    "krea2",
                    "--kind",
                    "vae",
                    "--topology",
                    str(topology),
                ]
            )
            == 0
        )
        restored = load_qwen_image_vae(destination)
        actual, expected = restored.posterior(pixels), vae.posterior(pixels)
        torch.testing.assert_close(actual.mean, expected.mean, atol=0, rtol=0)
        torch.testing.assert_close(actual.log_variance, expected.log_variance, atol=0, rtol=0)
        torch.testing.assert_close(restored.encode(pixels), restored.normalize(actual.mean), atol=0, rtol=0)
        assert torch.equal(rng, torch.get_rng_state())
        topology.write_text(vae.config.model_copy(update={"input_channels": 4}).model_dump_json())
        before = destination.read_bytes()
        assert (
            main(
                [
                    "weights",
                    "import",
                    str(source),
                    str(destination),
                    "--family",
                    "krea2",
                    "--kind",
                    "vae",
                    "--topology",
                    str(topology),
                ]
            )
            == 2
        )
        assert destination.read_bytes() == before
    finally:
        torch.set_num_threads(previous)
