"""Validate actual local language-only Krea2 conditioning, formats and ownership."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.support.qwen_image_21 import write_text_assets, write_text_features
from tests.support.text_assets import local_text_config
from unitunner.config.text import TextEncodingConfig
from unitunner.models.krea2.text.encoder import load_krea2_text


@pytest.fixture
def assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(951)
    write_text_assets(tmp_path, feature_layers=(2, 0, 1))
    yield tmp_path
    torch.set_num_threads(previous)


@pytest.mark.parametrize("single_file", [False, True])
def test_selected_layer_order_language_only_and_autocast(assets, single_file):
    rng = torch.get_rng_state().clone()
    encoder = load_krea2_text(
        (assets / "model.safetensors" if single_file else assets),
        device=torch.device("cpu"),
        dtype=torch.float32,
        config=local_text_config(assets, base=TextEncodingConfig(max_length=256)),
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert not encoder.model.training and all(not value.requires_grad for value in encoder.model.parameters())
    assert all(not value.is_meta for value in encoder.model.buffers())
    assert all(not key.startswith(("visual", "lm_head")) for key in encoder.model.state_dict())
    canonical = load_file(assets / "model.safetensors")
    for key, value in encoder.model.state_dict().items():
        assert torch.equal(value, canonical["model.language_model." + key])
    prompts = ["", "a longer prompt about red flowers", "x " * 300]
    features = encoder.encode(prompts)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        repeated = encoder.encode(prompts)
    assert len({row.shape[0] for row in features}) == 3
    for expected, actual in zip(features, repeated, strict=True):
        assert expected.shape[1:] == (3, 16)
        assert torch.equal(expected, actual) and not actual.requires_grad and torch.isfinite(actual).all()
    with torch.no_grad():
        encoder.model.norm.weight.mul_(3)
    after = encoder.encode(prompts)
    for before, current in zip(features, after, strict=True):
        torch.testing.assert_close(current, before, atol=0, rtol=0)
    assert encoder.encode([]) == () and torch.equal(rng, torch.get_rng_state())
    assert not encoder.model.training and not any(value.requires_grad for value in encoder.model.parameters())
    modules = [*encoder.model.layers, encoder.model.norm]
    original_hooks = [len(module._forward_pre_hooks) for module in modules]

    def fail(module, arguments):
        raise RuntimeError("deliberate language failure")

    handle = encoder.model.norm.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="deliberate"):
            encoder.encode(["failure"])
    finally:
        handle.remove()
    assert [len(module._forward_pre_hooks) for module in modules] == original_hooks
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("qwen_visual_prefix", ["visual.", "model.visual."])
def test_strict_full_artifact_validation_and_explicit_formats(assets, qwen_visual_prefix, automatic):
    write_text_features(assets, (0, 1, 2))
    config = TextEncodingConfig(max_length=256)
    owner = load_krea2_text(
        assets, device=torch.device("cpu"), dtype=torch.float32, config=local_text_config(assets, base=config)
    )
    expected = owner.encode(["hello"])[0]
    path = assets / "model.safetensors"
    canonical = load_file(path)
    native = {key.removeprefix("model."): value for key, value in canonical.items() if key != "lm_head.weight"}
    for source_format, values in [
        ("native", native),
        (
            "qwen",
            {
                (
                    "model." + key.removeprefix("language_model.")
                    if key.startswith("language_model.")
                    else qwen_visual_prefix + key.removeprefix("visual.")
                ): value
                for key, value in native.items()
            },
        ),
    ]:
        assert source_format in ("native", "transformers", "qwen")
        save_file(values, path)
        rng = torch.get_rng_state().clone()
        loaded = load_krea2_text(
            assets,
            device=torch.device("cpu"),
            dtype=torch.float32,
            config=local_text_config(assets, base=config),
            source_format="auto" if automatic else source_format,
        )
        torch.testing.assert_close(loaded.encode(["hello"])[0], expected, atol=0, rtol=0)
        assert torch.equal(rng, torch.get_rng_state())
    visual = next(key for key in canonical if key.startswith("model.visual."))
    head = "lm_head.weight"
    for values in [
        {key: value for key, value in canonical.items() if key != visual},
        {**canonical, visual: torch.full_like(canonical[visual], float("nan"))},
        {**canonical, head: torch.full_like(canonical[head], float("inf"))},
        {**canonical, visual: canonical[visual].flatten()[:1]},
    ]:
        save_file(values, path)
        with pytest.raises(ValueError, match="text weight"):
            load_krea2_text(
                assets, device=torch.device("cpu"), dtype=torch.float32, config=local_text_config(assets, base=config)
            )
    save_file(canonical, path)
    write_text_features(assets, (3,))
    with pytest.raises(ValueError, match="exceeds"):
        load_krea2_text(assets, config=local_text_config(assets), device=torch.device("cpu"))
