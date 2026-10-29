"""Verify actual local Qwen3-VL pre-norm text, media geometry and ownership."""

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from tests.support.assertions import invalid_input
from tests.support.qwen_image_21 import write_text_assets
from tests.support.text_assets import local_text_config
from unitunner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text


@pytest.fixture
def text_assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    write_text_assets(tmp_path)
    yield tmp_path
    torch.set_num_threads(previous)


@pytest.mark.parametrize("single_file", [False, True])
def test_pre_norm_features_slots_alpha_and_scoped_capture(text_assets, single_file):
    rng = torch.get_rng_state().clone()
    encoder = load_qwen_image_21_text(
        (text_assets / "model.safetensors" if single_file else text_assets),
        text_config=local_text_config(text_assets),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert not encoder.model.training and all(not p.requires_grad for p in encoder.model.parameters())
    assert all(not b.is_meta for b in encoder.model.buffers())
    rgba = np.full((32, 64, 4), 73, dtype=np.uint8)
    rgba[..., 3] = 0
    rgb = Image.new("RGB", (32, 64), (21, 173, 49))
    original = rgba.copy(), rgb.tobytes()
    groups = [[rgba, rgb], [], [rgb]]
    features = encoder.encode(["", "a longer prompt", "change color"], references=groups)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        repeated = encoder.encode(["", "a longer prompt", "change color"], references=groups)
    for a, b in zip(features, repeated, strict=True):
        assert (
            torch.equal(a.features, b.features) and a.image_slots == b.image_slots and a.image_shapes == b.image_shapes
        )
        assert a.features.ndim == 2 and not a.features.requires_grad and torch.isfinite(a.features).all()
    assert features[0].image_shapes == ((2, 4), (4, 2)) and len(features[0].image_slots) == 2
    assert features[0].image_slots[0] < features[0].image_slots[1]
    assert features[1].image_slots == features[1].image_shapes == ()
    white = encoder.encode([""], references=[[Image.new("RGB", (64, 32), "white"), rgb]])[0]
    torch.testing.assert_close(white.features, features[0].features, atol=1e-5, rtol=1e-4)
    assert white.image_slots == features[0].image_slots
    assert np.array_equal(rgba, original[0]) and rgb.tobytes() == original[1] and groups[0][0] is rgba
    before = features[1].features
    with torch.no_grad():
        encoder.model.language_model.norm.weight.fill_(4.75)
    after = encoder.encode(["a longer prompt"])[0].features
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-4)
    assert torch.equal(rng, torch.get_rng_state()) and encoder.encode([]) == []
    norm = encoder.model.language_model.norm
    hooks = len(norm._forward_pre_hooks)

    def fail(module, args):
        raise RuntimeError("deliberate final norm failure")

    handle = norm.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="deliberate"):
            encoder.encode(["failure"])
        assert len(norm._forward_pre_hooks) == hooks + 1
    finally:
        handle.remove()
    assert len(norm._forward_pre_hooks) == hooks and torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="groups"):
        encoder.encode(["x"], references=[])
    with pytest.raises(ValueError, match="multiples"):
        encoder.encode(["x"], references=[[Image.new("RGB", (33, 32))]])
    with pytest.raises(ValueError, match="uint8"):
        encoder.encode(["x"], references=invalid_input([[rgba.astype(np.float32)]]))
    with pytest.raises(ValueError, match="spans"):
        encoder.encode(["<|image_pad|>"])


@pytest.mark.parametrize("automatic", [False, True])
def test_strict_explicit_text_formats_and_invalid_weights(text_assets, automatic):
    model = load_qwen_image_21_text(
        text_assets, text_config=local_text_config(text_assets), device=torch.device("cpu"), dtype=torch.float32
    )
    expected = model.encode(["hello"])[0].features
    path = text_assets / "model.safetensors"
    native = model.model.state_dict()
    for source_format, values in [
        ("native", native),
        (
            "qwen",
            {
                ("model." + k.removeprefix("language_model.") if k.startswith("language_model.") else k): v
                for k, v in native.items()
            },
        ),
    ]:
        assert source_format in ("native", "transformers", "qwen")
        save_file(values, path)
        rng = torch.get_rng_state().clone()
        restored = load_qwen_image_21_text(
            text_assets,
            text_config=local_text_config(text_assets),
            device=torch.device("cpu"),
            dtype=torch.float32,
            source_format="auto" if automatic else source_format,
        )
        assert torch.equal(restored.encode(["hello"])[0].features, expected) and torch.equal(rng, torch.get_rng_state())
    key = "language_model.embed_tokens.weight"
    for bad in [
        {k: v for k, v in native.items() if k != key},
        {**native, key: torch.full_like(native[key], float("nan"))},
        {**native, key: native[key][:-1]},
        {**native, "unknown.weight": torch.zeros(1)},
    ]:
        save_file(bad, path)
        with pytest.raises(ValueError, match="text weight"):
            load_qwen_image_21_text(
                text_assets,
                text_config=local_text_config(text_assets),
                device=torch.device("cpu"),
                dtype=torch.float32,
                source_format="native",
            )
    assert "unknown.weight" in load_file(path)


@pytest.mark.parametrize("source_format", ["auto", "qwen"])
@pytest.mark.parametrize("visual_prefix", ["visual.", "model.visual."])
def test_qwen_single_file_visual_namespaces_and_duplicate_validation(text_assets, visual_prefix, source_format):
    path = text_assets / "model.safetensors"
    canonical = load_file(path)
    config = local_text_config(text_assets)
    original = load_qwen_image_21_text(path, text_config=config, device=torch.device("cpu"), dtype=torch.float32)
    reference = Image.new("RGB", (32, 32), (21, 173, 49))
    expected = original.encode(["hello"], references=[[reference]])[0]
    values = {}
    for key, value in canonical.items():
        if key.startswith("model.language_model."):
            key = "model." + key.removeprefix("model.language_model.")
        elif key.startswith("model.visual."):
            key = visual_prefix + key.removeprefix("model.visual.")
        values[key] = value
    save_file(values, path)
    rng = torch.get_rng_state().clone()
    restored = load_qwen_image_21_text(
        path, text_config=config, device=torch.device("cpu"), dtype=torch.float32, source_format=source_format
    )
    actual = restored.encode(["hello"], references=[[reference]])[0]
    torch.testing.assert_close(actual.features, expected.features, atol=0, rtol=0)
    assert actual.image_slots == expected.image_slots and torch.equal(rng, torch.get_rng_state())

    vision_key = next(k for k in values if k.startswith(visual_prefix))
    alias_prefix = "visual." if visual_prefix == "model.visual." else "model.visual."
    alias = alias_prefix + vision_key.removeprefix(visual_prefix)
    save_file({**values, alias: values[vision_key].clone()}, path)
    with pytest.raises(ValueError, match="Duplicate"):
        load_qwen_image_21_text(path, text_config=config, device=torch.device("cpu"), source_format=source_format)

    for bad in (torch.full_like(values[vision_key], float("nan")), values[vision_key].flatten()[:1]):
        save_file({**values, vision_key: bad}, path)
        with pytest.raises(ValueError, match="text weight"):
            load_qwen_image_21_text(path, text_config=config, device=torch.device("cpu"), source_format=source_format)
