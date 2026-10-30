"""Protect shared publication ordering, caching and adapter serialization contracts."""

from contextlib import closing

import pytest
import torch
from torch import nn

from unituner.adapters.config import AdapterConfig
from unituner.artifacts.formats.adapters import (
    AdapterTargetKeys,
    adapter_key_map,
    adapter_metadata,
    adapter_prefixes,
    convert_adapter_keys,
)
from unituner.artifacts.tensors import TensorReader, iter_tensors
from unituner.training.parameters import ParameterInstallation
from unituner.training.previews.publication import CaptionCache, publish_images
from unituner.training.publishing import publish_adapter


def test_caption_cache_retries_failure_and_is_per_instance():
    cache = CaptionCache[tuple[int, ...]]()
    calls = []

    def load():
        assert not torch.is_autocast_enabled("cpu")
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("encoding failed")
        return (42,)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="encoding failed"):
            cache.cached_captions(load)
        assert cache.captions is None
        assert cache.cached_captions(load) == (42,)
        assert cache.cached_captions(load) == (42,)
        assert torch.is_autocast_enabled("cpu")
    assert len(calls) == 2
    assert CaptionCache().captions is None


def test_preview_publication_preserves_order_layers_rounding_and_failure(tmp_path):
    seen = []

    def render(sample):
        seen.append(("render", sample))
        if sample == 2:
            raise ValueError("decode failed")
        return [torch.full((3, 2, 2), float(sample)) for _ in range(2)]

    def save(pixels, path, *, rounding):
        seen.append((path.name, float(pixels.mean()), rounding))

    with pytest.raises(ValueError, match="decode failed"):
        publish_images(range(4), render, directory=tmp_path, update=7, rounding="truncate", save=save)
    assert seen == [
        ("render", 0),
        ("step-00000007-00.png", 0.0, "truncate"),
        ("step-00000007-00.layer-001.png", 0.0, "truncate"),
        ("render", 1),
        ("step-00000007-01.png", 1.0, "truncate"),
        ("step-00000007-01.layer-001.png", 1.0, "truncate"),
        ("render", 2),
    ]


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
@pytest.mark.parametrize("layout", ["default", "custom", "sliced"])
def test_adapter_publication_matches_explicit_assembly(tmp_path, algorithm, layout):
    model = nn.Linear(8, 8)
    config = AdapterConfig(algorithm=algorithm, rank=2, alpha=7.0)
    names = ("blocks.0.proj",)
    prefixes = (
        {names[0]: "custom_projection"}
        if layout == "custom"
        else adapter_prefixes(
            names,
            algorithm=algorithm,
            namespace="diffusion_model",
        )
    )
    with closing(ParameterInstallation(model, adapter=config, targets={names[0]: model})) as owner:
        mapping = adapter_key_map([AdapterTargetKeys("0", prefixes[names[0]])], algorithm=algorithm)
        expected = convert_adapter_keys(owner.branches.state_dict(), mapping)
        before = {name: value.clone() for name, value in owner.branches.state_dict().items()}
        path = tmp_path / "adapter.safetensors"
        publish_adapter(
            path,
            branches=owner.branches,
            target_names=names,
            adapter=config,
            base_model="test",
            module_identifier="networks.test",
            dtype=torch.bfloat16,
            namespace="diffusion_model",
            metadata={"layout": layout, "modelspec.title": "override"},
            prefixes=prefixes if layout == "custom" else None,
            published=expected if layout == "sliced" else None,
        )
        actual = dict(iter_tensors(path))
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value.to(torch.bfloat16), atol=0, rtol=0)
        for name, value in owner.branches.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        info = adapter_metadata(
            algorithm=algorithm,
            base_model="test",
            rank=2,
            alpha=7.0,
            lora_module_identifier="networks.test",
        )
        info.update({"layout": layout, "modelspec.title": "override", "unituner.adapter.format": "comfyui"})
        with TensorReader(path) as reader:
            assert reader.metadata() == info
