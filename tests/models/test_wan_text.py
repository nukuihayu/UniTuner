"""Exercise Wan UMT5 strict artifacts, native prompt boundaries and frozen ownership."""

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, T5Tokenizer

from unitunner.models.wan.config.t5 import WanT5Config
from unitunner.models.wan.text.encoder import WanTextEncoder, clean_text, load_wan_text
from unitunner.models.wan.text.loading import load_wan_t5, save_wan_t5
from unitunner.models.wan.text.model import WanT5Encoder


@pytest.fixture
def assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    vocabulary = [("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -1.0)] + [
        (char, -2.0) for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZé&"
    ]
    T5Tokenizer(vocab=vocabulary, extra_ids=0).save_pretrained(tmp_path / "tokenizer")
    config = WanT5Config(
        vocab_size=len(vocabulary), hidden_size=32, attention_dim=24, ffn_dim=56, num_heads=4, num_layers=2
    )
    model = WanT5Encoder(config)
    save_wan_t5(model, tmp_path / "weights.safetensors")
    yield tmp_path, model
    torch.set_num_threads(previous)


def test_local_text_loading_exact_prefixes_frozen_state_and_rng(assets):
    root, model = assets
    rng = torch.get_rng_state().clone()
    encoder = load_wan_text(root / "weights.safetensors", root / "tokenizer", max_length=32)
    assert not encoder.model.training and not any(parameter.requires_grad for parameter in encoder.model.parameters())
    assert encoder.model.config == model.config
    for name, value in encoder.model.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    assert clean_text("  HELLO &amp;amp; World\n\t café  ") == "HELLO & World café"
    prompts = ["", "  HELLO &amp;amp; World\n\t café  ", "a caption " * 40]
    tokens = encoder.tokenize(prompts)
    assert tokens.lengths[0] == 1 and tokens.lengths[-1] == 32
    identities = tuple(map(id, encoder.model.parameters()))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        rows = encoder.encode(prompts)
    repeated = encoder.encode(prompts)
    assert all(
        row.shape == (length, 32) and row.dtype == torch.float32 and row.device.type == "cpu"
        for row, length in zip(rows, tokens.lengths, strict=True)
    )
    assert all(torch.equal(row, other) for row, other in zip(rows, repeated, strict=True))
    assert torch.equal(rng, torch.get_rng_state())
    assert identities == tuple(map(id, encoder.model.parameters()))
    external = root / "external.safetensors"
    save_file({"encoder." + name: value for name, value in model.state_dict().items()}, external)
    loaded = load_wan_text(external, root / "tokenizer", config=model.config, max_length=32)
    assert all(torch.equal(left, right) for left, right in zip(rows, loaded.encode(prompts), strict=True))


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "nonfinite", "mixed-prefix"])
def test_strict_t5_artifact_rejection(assets, damage):
    root, model = assets
    tensors = load_file(root / "weights.safetensors")
    name = next(iter(tensors))
    prefix = ""
    if damage == "missing":
        tensors.pop(name)
    elif damage == "extra":
        tensors["unexpected.weight"] = torch.ones(1)
    elif damage == "shape":
        tensors[name] = tensors[name].flatten()
    elif damage == "nonfinite":
        tensors[name] = torch.full_like(tensors[name], float("nan"))
    else:
        tensors = {"encoder." + name: value for name, value in tensors.items()}
        tensors["token_embedding.weight"] = tensors.pop("encoder.token_embedding.weight")
        prefix = "encoder."
    save_file(tensors, root / "broken.safetensors")
    with pytest.raises(ValueError, match=r"keys differ|invalid tensor|anchored prefix"):
        load_wan_t5(root / "broken.safetensors", config=model.config, prefix=prefix)
    with pytest.raises(ValueError, match="topology differs"):
        load_wan_t5(root / "weights.safetensors", config=model.config.model_copy(update={"shared_position_bias": True}))


def test_invalid_tokenizer_and_model_mode_fail_before_encoding(assets):
    root, model = assets
    with pytest.raises(ValueError, match="max_length"):
        load_wan_text(root / "missing.safetensors", root / "missing-tokenizer", max_length=0)
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    tokenizer.padding_side = "left"
    with pytest.raises(ValueError, match="right padding"):
        WanTextEncoder(model, tokenizer)
    tokenizer.padding_side = "right"
    encoder = WanTextEncoder(model, tokenizer, max_length=16)
    with pytest.raises(ValueError, match="frozen eval"):
        encoder.encode(["hello"])
    model.eval().requires_grad_(False)
    with pytest.raises(ValueError, match="nonempty"):
        encoder.encode([])
    tokenizer.add_tokens(["new_asset_token"])
    with pytest.raises(ValueError, match="encoder vocabulary"):
        encoder.encode(["new_asset_token"])
    tokenizer.padding_side = "left"
    with pytest.raises(ValueError, match="right padding"):
        encoder.encode(["hello"])
    tokenizer.init_kwargs["padding_side"] = "left"
    tokenizer.save_pretrained(root / "left-tokenizer")
    with pytest.raises(ValueError, match="right padding"):
        load_wan_text(root / "missing.safetensors", root / "left-tokenizer")


def test_native_t5_visibility_and_precision_boundary(assets):
    _, model = assets
    ids = torch.ones(2, 7, dtype=torch.int64)
    with pytest.raises(ValueError, match="visibility"):
        model(ids, torch.ones(2, 6, dtype=torch.int64))
    with pytest.raises(ValueError, match="int64 IDs"):
        model(ids.float())
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast disabled"):
        model(ids)
