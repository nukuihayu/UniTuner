"""Verify cache assembly preserves native options, identities and preflight ordering."""

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from unitunner.config.cache import CacheOptions
from unitunner.config.datasets import CacheDatasetConfig
from unitunner.config.model import EncoderModelConfig
from unitunner.config.text import TextEncodingConfig, TokenizerSourceConfig
from unitunner.models.catalog import ModelFamily
from unitunner.models.components.text_artifacts import text_encoding_identity

CASES = (
    ("z_image", "z-image", "turbo", None),
    ("flux", "flux", "dev", None),
    ("flux2", "flux2", "distilled", "dev"),
    ("flux3", "flux3", "distilled", None),
    ("moss_image", "moss-image", "base", None),
    ("krea2", "krea2", "turbo", None),
    ("qwen_image", "qwen-image", "original", None),
    ("qwen_image_2_1", "qwen-image-2-1", "base", None),
)


def fake_module(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)


def model_config(family, variant, version):
    return EncoderModelConfig(
        family=ModelFamily(family),
        variant=variant,
        version=version,
        vae="vae.safetensors",
        text_encoder="text",
        assets="assets",
    )


def test_entry_and_image_assembly_do_not_import_optional_providers():
    code = (
        "import sys; import unitunner.data.build; import unitunner.data.production.images; "
        "import unitunner.inference.restoration.run; "
        "assert 'transformers' not in sys.modules; assert 'av' not in sys.modules; "
        "assert 'unitunner.data.production.wan' not in sys.modules; "
        "assert 'unitunner.data.production.minimax_h3' not in sys.modules"
    )
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")}
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module,family,variant,version",
    (
        *CASES,
        ("qwen_image", "qwen-image", "layered", None),
        ("qwen_image", "qwen-image", "edit-2511", None),
        ("flux2", "flux2", "base", "klein-4b"),
    ),
)
def test_encoder_loading_keeps_native_arguments(monkeypatch, module, family, variant, version):
    from unitunner.data.production import images as image_encoders

    calls = []
    vae = torch.nn.Linear(1, 1)
    text = object()

    def load_vae(path, **options):
        calls.append(("vae", path, options))
        return vae

    def load_text(*paths, **options):
        calls.append(("text", paths, options))
        return text

    name = "load_qwen_image_21_text" if module == "qwen_image_2_1" else f"load_{module}_text"
    fake_module(monkeypatch, f"unitunner.models.{module}.text.encoder", **{name: load_text})
    from unitunner.models.components import image_encoders as native

    vae_family = (
        "flux2" if module in ("flux2", "flux3", "moss_image") else "qwen_image" if module == "krea2" else module
    )
    name = "load_qwen_image_21_vae" if vae_family == "qwen_image_2_1" else f"load_{vae_family}_vae"
    monkeypatch.setattr(native, name, load_vae)
    model = model_config(family, variant, version)
    assert image_encoders.load_image_encoders(model, device=torch.device("cpu")) == (vae, text)
    assert not vae.training and not any(p.requires_grad for p in vae.parameters())
    assert [call[0] for call in calls] == ["vae", "text"]
    options = calls[1][2]
    assert options["dtype"] == torch.float32 and options["device"] == torch.device("cpu")
    assert options.get("text_config", options.get("config")) is model.text
    if module in ("krea2", "qwen_image_2_1", "qwen_image"):
        assert options["source_format"] == "auto"
    if module == "qwen_image":
        assert calls[0][2]["input_channels"] == (4 if variant == "layered" else 3)
        assert options["variant"] == variant
    if module == "flux":
        assert calls[1][1][1] == model.asset_path("clip_text")
    if module == "flux2":
        assert options["version"] == version


@pytest.mark.parametrize("module,family,variant,version", CASES)
def test_group_loop_preserves_identity_options_and_order(monkeypatch, tmp_path, module, family, variant, version):
    from unitunner.data.production import images

    model = model_config(family, variant, version)
    groups = {
        name: CacheDatasetConfig(
            source=str(tmp_path / name),
            cache_directory=str(tmp_path / (name + "_cache")),
            resolution=(32, 32),
            prompt="",
            bucket_no_upscale=True,
        )
        for name in ("first", "second")
    }
    config = SimpleNamespace(
        model=model,
        dataset=SimpleNamespace(groups=groups),
        runtime=SimpleNamespace(model_dump=lambda **_: {}),
        seed=123,
        cache=CacheOptions(batch_size=16 if family == "z-image" else 1),
    )
    vae, text, repa = object(), object(), object()
    seen = []
    monkeypatch.setattr(images, "discover_images", lambda _: ())
    monkeypatch.setattr(images, "weight_artifact_sha256", lambda _: "vae-hash")
    monkeypatch.setattr(images, "text_artifact_sha256", lambda *_, **__: "text-hash")
    monkeypatch.setattr(images, "load_image_encoders", lambda *_, **__: (vae, text))

    def encode(source, destination, actual_vae, actual_text, **options):
        assert actual_vae is vae and actual_text is text
        seen.append((source, destination, options))
        return SimpleNamespace(entries=[1, 2])

    providers = {
        "krea2": ("unitunner.models.krea2.data.cache", "cache_krea2"),
        "qwen_image": ("unitunner.models.qwen_image.data.cache", "cache_qwen_image"),
        "qwen_image_2_1": ("unitunner.models.qwen_image_2_1.data.cache.encoding", "cache_qwen_image_21"),
    }
    provider, name = providers.get(module, ("unitunner.data.cache.images", "cache_images"))
    fake_module(monkeypatch, provider, **{name: encode})
    assert images.produce_images(config, device="cpu", repa=repa, tokenizer_digests={"text": "tokenizer"}) == 4
    for (source, destination, options), data in zip(seen, groups.values(), strict=True):
        assert source == Path(data.source) and destination == Path(data.cache_directory)
        assert options["repa"] is repa
        buckets = options["buckets"]
        assert buckets.prompt == "" and buckets.no_upscale and buckets.enabled
        assert buckets.alignment == (32 if module == "qwen_image_2_1" else 16)
        if module != "krea2":
            assert options["data"] is data
        if module not in providers:
            assert options["batch_size"] == config.cache.batch_size
            assert options["seed"] == 123
            assert options["variant"] == (variant if module in ("flux", "flux2", "flux3") else None)
            assert options["version"] == version


def test_missing_condition_fails_before_hashing_or_loading(monkeypatch, tmp_path):
    from unitunner.data.production import images

    data = CacheDatasetConfig(
        source=str(tmp_path),
        cache_directory=str(tmp_path / "cache"),
        resolution=(32, 32),
        reference_sources=(str(tmp_path / "missing"),),
    )
    config = SimpleNamespace(
        model=model_config("flux2", "distilled", "dev"), dataset=SimpleNamespace(groups={"data": data})
    )
    monkeypatch.setattr(images, "discover_images", lambda _: (tmp_path / "a.png",))

    def forbidden(*args, **kwargs):
        pytest.fail("Artifacts must not be hashed or loaded before input validation")

    monkeypatch.setattr(images, "load_image_encoders", forbidden)
    monkeypatch.setattr(images, "weight_artifact_sha256", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        images.produce_images(config, device="cpu", repa=None, tokenizer_digests={})


def test_text_policy_identity_preserves_default_and_tokenizer_only_sections():
    assert text_encoding_identity(TextEncodingConfig()) == {}
    tokenizer_only = TextEncodingConfig(tokenizer=TokenizerSourceConfig(source="local"))
    assert text_encoding_identity(tokenizer_only) == {"text_encoding": {}}
    assert text_encoding_identity(TextEncodingConfig(precision="bf16")) == {"text_encoding": {"precision": "bf16"}}


@pytest.mark.parametrize(
    "family,module,name",
    [
        ("wan", "wan", "produce_wan"),
        ("minimax-h3", "minimax_h3", "produce_minimax_h3"),
        ("z-image", "images", "produce_images"),
    ],
)
def test_dispatch_preserves_shared_owners(monkeypatch, family, module, name):
    from unitunner.data.build import _run

    config = SimpleNamespace(model=SimpleNamespace(family=ModelFamily(family)))
    repa, digests = object(), {"text": "digest"}

    def produce(actual_config, **options):
        assert actual_config is config
        assert options == {"device": "cuda:1", "repa": repa, "tokenizer_digests": digests}
        return 13

    fake_module(monkeypatch, f"unitunner.data.production.{module}", **{name: produce})
    assert _run(config, device="cuda:1", repa=repa, tokenizer_digests=digests) == 13


@pytest.mark.parametrize("family,variant,version", [("qwen-image", "original", None), ("flux2", "distilled", "dev")])
def test_invalid_encoder_topology_fails_before_loading_weights(monkeypatch, family, variant, version):
    from unitunner.data.production import images

    model = model_config(family, variant, version)
    model = model.model_copy(update={"variant": "unknown"} if family == "qwen-image" else {"version": None})

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid configuration loaded encoder weights")

    monkeypatch.setattr(images, "load_image_vae", forbidden)
    monkeypatch.setattr(images, "load_image_text", forbidden)
    with pytest.raises(ValueError):
        images.load_image_encoders(model, device=torch.device("cpu"))
