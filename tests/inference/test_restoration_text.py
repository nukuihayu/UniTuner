"""Preserve native restoration caption layouts without unrelated optional providers."""

from types import SimpleNamespace

import pytest
import torch

from tests.data.test_cache_production import CASES, fake_module, model_config
from unitunner.inference.restoration.text import encode_caption
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux.conditioning import FluxTextFeatures
from unitunner.models.krea2.conditioning import Krea2TextFeatures


@pytest.mark.parametrize("module,family,variant,version", CASES)
@pytest.mark.parametrize("source_format", [None, "transformers"])
def test_restoration_caption_keeps_native_layout_and_loader_policy(
    monkeypatch,
    module,
    family,
    variant,
    version,
    source_format,
):
    model = model_config(family, variant, version).model_copy(update={"text_source_format": source_format})
    config = SimpleNamespace(model=model, prompt="test prompt")
    features = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = Caption(features[None])
    calls = []

    class Encoder:
        def encode_prompt(self, prompt):
            assert prompt == config.prompt
            return expected

        def encode(self, prompts):
            assert prompts == [config.prompt]
            if module == "flux":
                return FluxTextFeatures(features[None], torch.ones(1, 2))
            if module == "krea2":
                return (features[:, None],)
            return (SimpleNamespace(features=features),)

    def load(*paths, **options):
        calls.append((paths, options))
        assert not torch.is_autocast_enabled("cpu")
        return Encoder()

    name = "load_qwen_image_21_text" if module == "qwen_image_2_1" else f"load_{module}_text"
    fake_module(monkeypatch, f"unitunner.models.{module}.text.encoder", **{name: load})
    with torch.autocast("cpu", dtype=torch.bfloat16):
        caption = encode_caption(config, device=torch.device("cpu"))
    assert len(calls) == 1
    options = calls[0][1]
    assert options["device"] == torch.device("cpu")
    assert options.get("text_config", options.get("config")) is model.text
    if module != "flux":
        assert options["dtype"] == torch.float32
    if module in ("krea2", "qwen_image", "qwen_image_2_1"):
        assert options["source_format"] == (source_format or ("qwen" if module == "qwen_image" else "transformers"))
    if module == "flux":
        assert isinstance(caption, FluxTextFeatures)
        torch.testing.assert_close(caption.captions, features[None])
        torch.testing.assert_close(caption.pooled, torch.ones(1, 2))
    elif module == "krea2":
        assert isinstance(caption, Krea2TextFeatures) and caption.lengths == (3,)
        torch.testing.assert_close(caption.features, features[None, :, None])
    else:
        assert isinstance(caption, Caption)
        torch.testing.assert_close(caption.features, features[None])


def test_flux2_invalid_version_does_not_load_provider():
    config = SimpleNamespace(
        model=model_config("flux2", "distilled", "dev").model_copy(update={"version": None}), prompt=""
    )
    with pytest.raises(ValueError, match=r"requires model\.version"):
        encode_caption(config, device=torch.device("cpu"))
