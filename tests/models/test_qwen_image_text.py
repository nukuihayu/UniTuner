"""Verify local Qwen VL loading, reference ownership and feature-only execution."""

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

from tests.support.assertions import invalid_input
from tests.support.qwen_image import write_text_assets
from tests.support.text_assets import local_text_config
from unituner.models.qwen_image.text.encoder import load_qwen_image_text


@pytest.fixture
def text_assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    write_text_assets(tmp_path)
    yield tmp_path
    torch.set_num_threads(previous)


@pytest.mark.parametrize("source_format", ["auto", "qwen"])
def test_qwen_comfy_visual_prefix_matches_legacy_and_rejects_duplicates(text_assets, source_format):
    config = local_text_config(text_assets)
    original = load_qwen_image_text(
        text_assets,
        source_format=source_format,
        variant="edit-2511",
        device=torch.device("cpu"),
        dtype=torch.float32,
        text_config=config,
    )
    expected = original.encode(["hello"]).features
    path = text_assets / "model.safetensors"
    legacy = load_file(path)
    values = {("model." + key if key.startswith("visual.") else key): value for key, value in legacy.items()}
    save_file(values, path)
    restored = load_qwen_image_text(
        path,
        source_format=source_format,
        variant="edit-2511",
        device=torch.device("cpu"),
        dtype=torch.float32,
        text_config=config,
    )
    torch.testing.assert_close(restored.encode(["hello"]).features, expected, atol=0, rtol=0)
    vision_key = next(key for key in legacy if key.startswith("visual."))
    save_file({**values, vision_key: legacy[vision_key].clone()}, path)
    with pytest.raises(ValueError, match="Duplicate"):
        load_qwen_image_text(
            path,
            source_format=source_format,
            variant="edit-2511",
            device=torch.device("cpu"),
            dtype=torch.float32,
            text_config=config,
        )


def test_qwen_hub_tokenizer_and_processor_are_separate_from_native_weights(text_assets, monkeypatch):
    import shutil

    from unituner.config.text import TextEncodingConfig, TokenizerSourceConfig
    from unituner.models import text_assets as resolver

    snapshot = text_assets / "hub"
    for folder in ("tokenizer", "processor"):
        destination = snapshot / folder
        destination.mkdir(parents=True)
        for filename in resolver.TOKENIZER_FILES:
            source = text_assets / filename
            if source.is_file():
                shutil.copyfile(source, destination / filename)
    calls = []

    def download(**kwargs):
        calls.append((kwargs["repo_id"], kwargs["local_files_only"]))
        return str(snapshot)

    monkeypatch.setattr(resolver, "snapshot_download", download)
    config = TextEncodingConfig(tokenizer=TokenizerSourceConfig(offline=True))
    encoder = load_qwen_image_text(
        text_assets,
        text_config=config,
        device=torch.device("cpu"),
        variant="edit-2511",
        dtype=torch.float32,
        source_format="qwen",
    )
    assert calls == [("Qwen/Qwen-Image-Edit", True), ("Qwen/Qwen-Image", True)]
    tokenizer = encoder.processor.tokenizer
    assert encoder.processor.image_token_id == tokenizer.convert_tokens_to_ids(encoder.processor.image_token)
    assert encoder.encode_prompt("change color", references=[np.zeros((12, 12, 3), dtype=np.uint8)]).features.numel()


@pytest.mark.parametrize("single_file", [False, True])
def test_qwen_text_reference_ownership_and_lm_head_bypass(text_assets, single_file):
    rng = torch.get_rng_state().clone()
    encoder = load_qwen_image_text(
        (text_assets / "model.safetensors" if single_file else text_assets),
        text_config=local_text_config(text_assets),
        device=torch.device("cpu"),
        variant="edit-2511",
        dtype=torch.float32,
        source_format="qwen",
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert not encoder.model.training and all(not p.requires_grad for p in encoder.model.parameters())
    assert all(not b.is_meta for b in encoder.model.buffers())
    image = np.full((12, 12, 4), 128, dtype=np.uint8)
    image[..., 3] = 0
    original = image.copy()
    references = [[image], []]

    def reject_logits(module, args):
        raise AssertionError("Feature extraction must bypass vocabulary logits")

    hook = encoder.model.lm_head.register_forward_pre_hook(reject_logits)
    try:
        caption = encoder.encode(["", "change color"], references=references)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            repeated = encoder.encode(["", "change color"], references=references)
        assert torch.equal(caption.features, repeated.features)
        assert np.array_equal(image, original) and references[0][0] is image and references[1] == []
        assert caption.mask is not None
        assert caption.mask.dtype == torch.bool and not caption.features.requires_grad
        assert caption.mask is not None
        assert torch.count_nonzero(caption.features[~caption.mask]) == 0
        one = encoder.encode_prompt("", references=[image])
        assert caption.mask is not None
        torch.testing.assert_close(one.features[0], caption.features[0, caption.mask[0]], atol=1e-5, rtol=1e-4)
        tensor_image = torch.from_numpy(image.copy()).permute(2, 0, 1)
        assert torch.equal(encoder.encode_prompt("", references=[tensor_image]).features, one.features)
        torch.testing.assert_close(
            encoder.encode_prompt("", references=[tensor_image.float() / 255]).features,
            one.features,
            atol=1e-5,
            rtol=1e-4,
        )
        encoder.processor.tokenizer.padding_side = "left"
        left = encoder.encode(["", "change color"], references=references)
        torch.testing.assert_close(left.features, caption.features, atol=1e-5, rtol=1e-4)
        assert caption.mask is not None and left.mask is not None
        assert torch.equal(left.mask, caption.mask)
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        hook.remove()
    with pytest.raises(ValueError, match="batch"):
        encoder.encode([])
    with pytest.raises(ValueError, match="groups"):
        encoder.encode(["a", "b"], references=[[image]])


def test_qwen_text_explicit_weight_formats_and_invalid_artifacts(text_assets):
    encoder = load_qwen_image_text(
        text_assets,
        text_config=local_text_config(text_assets),
        device=torch.device("cpu"),
        variant="original",
        dtype=torch.float32,
        source_format="qwen",
    )
    expected = encoder.encode_prompt("caption").features
    path = text_assets / "model.safetensors"
    values = encoder.model.state_dict()
    save_file(values, path)
    restored = load_qwen_image_text(
        text_assets,
        text_config=local_text_config(text_assets),
        device=torch.device("cpu"),
        variant="original",
        dtype=torch.float32,
    )
    assert torch.equal(restored.encode_prompt("caption").features, expected)
    native = {key.removeprefix("model."): value for key, value in values.items()}
    save_file(native, path)
    restored = load_qwen_image_text(
        path,
        text_config=local_text_config(text_assets),
        device=torch.device("cpu"),
        variant="original",
        dtype=torch.float32,
    )
    assert torch.equal(restored.encode_prompt("caption").features, expected)
    key = "model.language_model.embed_tokens.weight"
    for bad in (
        {k: v for k, v in values.items() if k != key},
        {**values, key: torch.full_like(values[key], float("nan"))},
        {**values, key: values[key][:-1]},
        {**values, "unrecognized.weight": torch.zeros(1)},
    ):
        save_file(bad, path)
        with pytest.raises(ValueError, match="Qwen text weight"):
            load_qwen_image_text(
                text_assets,
                text_config=local_text_config(text_assets),
                device=torch.device("cpu"),
                variant="original",
                dtype=torch.float32,
            )
    save_file(values, path)
    with pytest.raises(ValueError, match="does not consume"):
        restored.encode(["caption"], references=[[]])
    edit = load_qwen_image_text(
        text_assets,
        text_config=local_text_config(text_assets),
        device=torch.device("cpu"),
        variant="edit",
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="requires a reference"):
        edit.encode(["caption"], references=[[]])
    assert load_file(path).keys() == values.keys()


def test_qwen_text_tied_embedding_artifact_has_one_owned_parameter(text_assets):
    config = Qwen2_5_VLConfig.from_pretrained(text_assets, local_files_only=True)
    config.tie_word_embeddings = True
    assert config.text_config is not None
    assert isinstance(config.text_config, Qwen2_5_VLTextConfig)
    config.text_config.tie_word_embeddings = True
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.save_pretrained(text_assets)
    encoder = load_qwen_image_text(
        text_assets,
        text_config=local_text_config(text_assets),
        device=torch.device("cpu"),
        variant="original",
        dtype=torch.float32,
        source_format="qwen",
    )
    assert encoder.model.lm_head.weight is encoder.model.model.language_model.embed_tokens.weight
    assert torch.isfinite(encoder.encode_prompt("caption").features).all()


def test_qwen_caption_generation_uses_local_policy_and_restores_rng_on_failure(text_assets):
    import json

    path = text_assets / "generation_config.json"
    config = json.loads(path.read_text())
    config.update(do_sample=True, top_k=7, temperature=0.8, top_p=0.9)
    path.write_text(json.dumps(config))
    encoder = load_qwen_image_text(
        text_assets,
        text_config=local_text_config(text_assets),
        device=torch.device("cpu"),
        variant="layered",
        dtype=torch.float32,
        source_format="qwen",
    )
    assert encoder.model.generation_config.do_sample and encoder.model.generation_config.top_k == 7
    image = np.full((12, 16, 4), 128, dtype=np.uint8)
    image[..., 3] = 0
    original = image.copy()
    features = encoder.encode_prompt("unchanged context").features
    rng = torch.get_rng_state().clone()
    text = encoder.describe_image(image, language="en", seed=787)
    assert isinstance(text, str) and text == text.strip()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert encoder.describe_image(image, language="en", seed=787) == text
    assert encoder.describe_image(image, language="en", seed=788) != text
    assert torch.equal(rng, torch.get_rng_state()) and np.array_equal(image, original)
    assert not encoder.model.training and all(not p.requires_grad for p in encoder.model.parameters())
    assert torch.equal(features, encoder.encode_prompt("unchanged context").features)

    def fail(module, args):
        raise RuntimeError("caption interrupted")

    hook = encoder.model.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="caption interrupted"):
            encoder.describe_image(image, seed=789)
    finally:
        hook.remove()
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="language"):
        encoder.describe_image(image, language=invalid_input("invalid"))
