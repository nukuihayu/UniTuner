"""Verify native image publishers retain caching, output layers and rounding policies."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from unitunner.config.execution import ExecutionConfig
from unitunner.config.sampling import SamplingConfig
from unitunner.models.components.conditioning import Caption


@pytest.mark.parametrize(
    "family, cached, rounding",
    [
        ("z_image", True, "nearest"),
        ("flux", True, "nearest"),
        ("flux2", True, "truncate"),
        ("flux3", True, "truncate"),
        ("moss_image", True, "nearest"),
        ("krea2", False, "truncate"),
        ("qwen_image", False, "truncate"),
        ("qwen_image_2_1", False, "truncate"),
    ],
)
def test_family_preview_publication_policy(tmp_path, monkeypatch, family, cached, rounding):
    # Keep external text dependencies outside this orchestration contract test.
    events = []
    caption = Caption(torch.ones(1, 2))

    def load_text(*args, **kwargs):
        events.append("text")
        return SimpleNamespace(encode_prompt=lambda *a, **k: caption, encode=lambda *a, **k: caption)

    provider = ModuleType(f"unitunner.models.{family}.text.encoder")
    loader_name = "load_qwen_image_21_text" if family == "qwen_image_2_1" else f"load_{family}_text"
    setattr(provider, loader_name, load_text)
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    if family in ("krea2", "qwen_image_2_1"):
        conditioning = ModuleType(f"unitunner.models.{family}.inference.conditioning")
        if family == "krea2":
            conditioning.encode_krea2_prompts = lambda *a, **k: (caption, None)
        else:
            conditioning.build_qwen_image_21_conditions = lambda *a, **k: SimpleNamespace(
                positive=caption,
                negative=None,
                references=(),
                mask=None,
            )
        monkeypatch.setitem(sys.modules, conditioning.__name__, conditioning)
    path = Path(__file__).parents[2] / f"src/unitunner/training/integrations/{family}/previews.py"
    spec = importlib.util.spec_from_file_location(f"preview_contract_{family}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = nn.Linear(1, 1)
    model.config = SimpleNamespace(
        variant="base",
        version="dev",
        in_channels=16,
        out_channels=16,
        image_channels=16,
        latent_channels=16,
    )
    channels = 16 if family in ("krea2", "qwen_image", "qwen_image_2_1") else 4
    if family == "z_image":
        model.config.in_channels = 4
    vae = nn.Module()
    vae.config = SimpleNamespace(
        latent_channels=channels,
        input_channels=3,
        spatial_factor=8,
        block_channels=(4, 4, 4, 4),
    )

    def load_vae(*args, **kwargs):
        events.append("vae")
        return vae

    for name in vars(module).copy():
        if name.startswith("load_") and name.endswith("_vae"):
            monkeypatch.setattr(module, name, load_vae)
    if family == "flux":
        monkeypatch.setattr(module, "build_flux_condition", lambda *a, **k: None)
    elif family in ("flux2", "flux3"):
        monkeypatch.setattr(module, f"build_{family}_references", lambda *a, **k: ())
    elif family == "qwen_image":
        monkeypatch.setattr(
            module, "build_qwen_image_references", lambda *a, **k: SimpleNamespace(images=(), latents=())
        )
    generated, saved = [], []

    def generate(model, positive, sample, **kwargs):
        generated.append(sample.prompt)
        return torch.zeros(1, channels, 2, 2)

    monkeypatch.setattr(module, "generate_latents", generate)
    layers = 2 if family == "qwen_image" else 1
    monkeypatch.setattr(module, "decode_on_device", lambda *a, **k: torch.zeros(layers, 3, 2, 2))
    monkeypatch.setattr(module, "save_image", lambda pixels, path, **k: saved.append((path.name, k["rounding"])))
    config = SimpleNamespace(
        model=SimpleNamespace(
            family=family.replace("_", "-"),
            variant="base",
            version="dev",
            vae="vae",
            text_encoder="text",
            asset_path=lambda *a: Path("clip"),
            text_source_format=None,
            text=SimpleNamespace(resolved_max_length=lambda *a: 8),
        ),
        preview=SimpleNamespace(samples=tuple(SamplingConfig(prompt=p, guidance=1.0) for p in ("first", "second"))),
        output=SimpleNamespace(directory=str(tmp_path)),
        execution=ExecutionConfig(),
    )
    publisher = module.ImagePreviews(config, model)
    assert events == []
    publisher(0)
    publisher(3)
    assert generated == ["first", "second", "first", "second"]
    assert events == (["text", "vae", "vae"] if cached else ["vae", "text", "vae", "text"])
    expected = []
    for update in (0, 3):
        for index in range(2):
            for layer in range(layers):
                suffix = "" if layer == 0 else f".layer-{layer:03d}"
                expected.append((f"step-{update:08d}-{index:02d}{suffix}.png", rounding))
    assert saved == expected
