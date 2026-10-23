"""Check exact, reversible external adapter naming and algorithm-specific labels."""

import json

import pytest
import torch

from unitunner.artifacts.formats.adapters import (
    AdapterTargetKeys,
    adapter_key_map,
    adapter_metadata,
    adapter_prefixes,
    convert_adapter_keys,
)


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr", "lokr-full"])
def test_bijective_roundtrip_preserves_tensor_values(algorithm):
    mapping = adapter_key_map([AdapterTargetKeys("blocks.0.proj", "lora_unet_blocks_0_proj")], algorithm=algorithm)
    tensors = {name: torch.randn(2, 3) for name in mapping}
    published = convert_adapter_keys(tensors, mapping)
    restored = convert_adapter_keys(published, {v: k for k, v in mapping.items()})
    for key, value in tensors.items():
        assert restored[key] is value
    with pytest.raises(ValueError, match="missing"):
        convert_adapter_keys({}, mapping)


def test_colliding_external_prefixes_fail():
    with pytest.raises(ValueError, match="bijective"):
        adapter_key_map(
            [AdapterTargetKeys("a.b", "lora_unet_a_b"), AdapterTargetKeys("a_b", "lora_unet_a_b")], algorithm="lora"
        )


def test_lora_is_not_mislabeled_lycoris():
    lora = adapter_metadata(
        algorithm="lora", base_model="z_image", rank=4, alpha=4.0, lora_module_identifier="networks.lora_zimage"
    )
    loha = adapter_metadata(algorithm="loha", base_model="z_image", rank=4, alpha=4.0)
    assert not any(key.startswith("ss_") for record in (lora, loha) for key in record)
    assert lora["network_module"] == "networks.lora_zimage"
    assert loha["network_module"] == "lycoris.kohya"
    assert json.loads(loha["network_args"])["algo"] == "loha"


def test_published_lora_roundtrip_restores_scale_and_prediction(tmp_path):
    from unitunner.adapters.lora import LoRA
    from unitunner.artifacts.tensors import iter_tensors, save_tensors

    torch.manual_seed(84)
    branch = LoRA(3, 4, rank=2, alpha=7.0, dtype=torch.float64)
    with torch.no_grad():
        branch.up.normal_()
    targets = [AdapterTargetKeys("projection", "lora_unet_projection")]
    mapping = adapter_key_map(targets, algorithm="lora")
    state = {f"projection.{key}": value for key, value in branch.state_dict().items()}
    path = tmp_path / "adapter.safetensors"
    save_tensors(convert_adapter_keys(state, mapping), path)
    restored = LoRA(3, 4, rank=2, alpha=1.0, dtype=torch.float64)
    internal = convert_adapter_keys(dict(iter_tensors(path)), {value: key for key, value in mapping.items()})
    restored.load_state_dict({key.removeprefix("projection."): value for key, value in internal.items()}, strict=True)
    inputs = torch.randn(5, 3, dtype=torch.float64)
    torch.testing.assert_close(restored(inputs), branch(inputs), rtol=0, atol=0)


def test_namespaces_preserve_dotted_paths_and_reject_lossy_lycoris_names():
    names = ("a.b", "a_b")
    assert adapter_prefixes(names, algorithm="lora") == {"a.b": "transformer.a.b", "a_b": "transformer.a_b"}
    for algorithm in ("loha", "lokr"):
        with pytest.raises(ValueError, match="bijective"):
            adapter_prefixes(names, algorithm=algorithm)
        assert adapter_prefixes(("layers.0.to_q",), algorithm=algorithm) == {"layers.0.to_q": "lycoris_layers_0_to_q"}


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
def test_native_consumer_namespace_preserves_paths_and_factor_values(tmp_path, algorithm):
    from contextlib import closing

    from torch import nn

    from unitunner.adapters.config import AdapterConfig
    from unitunner.adapters.loading import LoadedAdapter
    from unitunner.artifacts.tensors import save_tensors
    from unitunner.training.parameters import ParameterInstallation

    torch.manual_seed(47)
    model = nn.Linear(8, 8)
    targets = {"blocks.0.proj": model}
    config = AdapterConfig(algorithm=algorithm, rank=2, alpha=7.0)
    prefixes = adapter_prefixes(targets, algorithm=algorithm, namespace="diffusion_model")
    inputs = torch.randn(2, 3, 8)
    with closing(ParameterInstallation(model, adapter=config, targets=targets)) as owner:
        with torch.no_grad():
            for parameter in owner.branches.parameters():
                parameter.normal_(std=0.1)
        expected = model(inputs).detach().clone()
        mapping = adapter_key_map([AdapterTargetKeys("0", prefixes["blocks.0.proj"])], algorithm=algorithm)
        published = convert_adapter_keys(owner.branches.state_dict(), mapping)
        assert all(key.startswith("diffusion_model.blocks.0.proj.") for key in published)
        path = tmp_path / "adapter.safetensors"
        save_tensors(published, path)
    with closing(LoadedAdapter(path, config, targets, external_prefixes=prefixes)):
        torch.testing.assert_close(model(inputs), expected, atol=0, rtol=0)
