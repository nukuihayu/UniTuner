"""Verify local FLUX text semantics, strict language assets and deterministic encoding."""

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers.pre_tokenizers import ByteLevel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5Tokenizer

from tests.support.text_assets import local_text_config
from unituner.models.flux.text.encoder import clean_clip_text, load_flux_text


@pytest.fixture
def assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(191)
    t5_path, clip_path = tmp_path / "t5", tmp_path / "clip"
    vocab = [("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -1.0)] + [
        (c, -2.0) for c in "abcdefghijklmnopqrstuvwxyz"
    ]
    T5Tokenizer(vocab=vocab, extra_ids=0).save_pretrained(t5_path)
    T5EncoderModel(
        T5Config(
            vocab_size=len(vocab),
            d_model=16,
            d_kv=8,
            d_ff=32,
            num_layers=2,
            num_heads=2,
            dropout_rate=0.1,
            feed_forward_proj="gated-gelu",
        )
    ).save_pretrained(t5_path)
    alphabet = sorted(ByteLevel.alphabet())
    clip_vocab = {c: i for i, c in enumerate(alphabet)}
    clip_vocab.update({c + "</w>": i + 256 for i, c in enumerate(alphabet)})
    clip_vocab.update({"<|startoftext|>": 512, "<|endoftext|>": 513})
    CLIPTokenizer(vocab=clip_vocab, merges=[]).save_pretrained(clip_path)
    CLIPTextModel(
        CLIPTextConfig(
            vocab_size=514,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=77,
            bos_token_id=512,
            eos_token_id=513,
            pad_token_id=513,
            hidden_act="quick_gelu",
        )
    ).save_pretrained(clip_path)
    yield t5_path, clip_path
    torch.set_num_threads(previous)


@pytest.mark.parametrize("single_file", [False, True])
def test_flux_local_text_native_unmasked_final_and_clip_pooled_semantics(assets, single_file):
    t5_path, clip_path = assets
    before = torch.get_rng_state().clone()
    encoder = load_flux_text(
        t5_path / "model.safetensors" if single_file else t5_path,
        clip_path / "model.safetensors" if single_file else clip_path,
        text_config=local_text_config(t5_path, clip=clip_path),
        device=torch.device("cpu"),
    )
    assert encoder.t5.shared.weight is encoder.t5.encoder.embed_tokens.weight
    assert all(not p.requires_grad for model in (encoder.t5, encoder.clip) for p in model.parameters())
    assert all(not value.is_meta for model in (encoder.t5, encoder.clip) for value in model.buffers())
    prompts = ["", "hello there", "  cafÃ©\n\t  picture  ", "long caption " * 100]
    actual = encoder.encode(prompts)
    assert actual.captions.shape == (4, 512, 16) and actual.pooled.shape == (4, 16)
    t5 = encoder.t5_tokenizer(prompts, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    clip = encoder.clip_tokenizer(
        [clean_clip_text(p) for p in prompts], max_length=77, padding="max_length", truncation=True, return_tensors="pt"
    )
    assert clean_clip_text(prompts[2]) == "café picture"
    with torch.no_grad():
        expected_t5 = encoder.t5(t5["input_ids"], attention_mask=None).last_hidden_state
        masked_t5 = encoder.t5(**t5).last_hidden_state
        expected_clip = encoder.clip(clip["input_ids"]).pooler_output
    torch.testing.assert_close(actual.captions, expected_t5, atol=0, rtol=0)
    torch.testing.assert_close(actual.pooled, expected_clip, atol=0, rtol=0)
    assert not torch.equal(expected_t5, masked_t5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        repeated = encoder.encode(prompts)
    torch.testing.assert_close(repeated.captions, actual.captions, atol=0, rtol=0)
    torch.testing.assert_close(repeated.pooled, actual.pooled, atol=0, rtol=0)
    assert not actual.captions.requires_grad and not actual.pooled.requires_grad
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
    with pytest.raises(ValueError, match="nonempty"):
        encoder.encode([])
    encoder.t5.encoder.block[0].train()
    with pytest.raises(ValueError, match="eval"):
        encoder.encode(["test"])


def test_flux_text_strict_tied_embeddings_historical_clip_and_incomplete_weights(assets):
    t5_path, clip_path = assets
    initial = load_flux_text(
        t5_path,
        clip_path,
        text_config=local_text_config(t5_path, clip=clip_path),
        device=torch.device("cpu"),
        max_length=12,
    ).encode(["hello"])
    t5_file, clip_file = t5_path / "model.safetensors", clip_path / "model.safetensors"
    t5_weights, clip_weights = load_file(t5_file), load_file(clip_file)
    save_file({"text_model." + key: value for key, value in clip_weights.items()}, clip_file)
    reloaded = load_flux_text(
        t5_path,
        clip_path,
        text_config=local_text_config(t5_path, clip=clip_path),
        device=torch.device("cpu"),
        max_length=12,
    ).encode(["hello"])
    torch.testing.assert_close(initial.pooled, reloaded.pooled, atol=0, rtol=0)
    duplicate = dict(t5_weights)
    duplicate["encoder.embed_tokens.weight"] = duplicate["shared.weight"] + 1
    save_file(duplicate, t5_file)
    with pytest.raises(ValueError, match="embedding values conflict"):
        load_flux_text(
            t5_path, clip_path, text_config=local_text_config(t5_path, clip=clip_path), device=torch.device("cpu")
        )
    save_file(t5_weights, t5_file)
    malformed = dict(clip_weights)
    malformed["encoder.layers.0.self_attn.q_proj.weight"] = torch.full_like(
        malformed["encoder.layers.0.self_attn.q_proj.weight"], float("nan")
    )
    save_file(malformed, clip_file)
    with pytest.raises(ValueError, match="values invalid"):
        load_flux_text(
            t5_path, clip_path, text_config=local_text_config(t5_path, clip=clip_path), device=torch.device("cpu")
        )
    incomplete = dict(clip_weights)
    incomplete.pop("final_layer_norm.bias")
    save_file(incomplete, clip_file)
    with pytest.raises(ValueError, match="keys differ"):
        load_flux_text(
            t5_path, clip_path, text_config=local_text_config(t5_path, clip=clip_path), device=torch.device("cpu")
        )


def test_flux_text_standard_shards_preserve_features_and_mixed_clip_layout_fails(assets):
    t5_path, clip_path = assets
    encoder = load_flux_text(
        t5_path,
        clip_path,
        text_config=local_text_config(t5_path, clip=clip_path),
        device=torch.device("cpu"),
        max_length=12,
    )
    baseline = encoder.encode(["hello"])
    sharded = t5_path / "sharded"
    encoder.t5.save_pretrained(sharded, max_shard_size="2KB")
    encoder.t5_tokenizer.save_pretrained(sharded)
    assert len(list(sharded.glob("model-*-of-*.safetensors"))) > 1
    actual = load_flux_text(
        sharded,
        clip_path,
        text_config=local_text_config(sharded, clip=clip_path),
        device=torch.device("cpu"),
        max_length=12,
    ).encode(["hello"])
    torch.testing.assert_close(actual.captions, baseline.captions, atol=0, rtol=0)
    torch.testing.assert_close(actual.pooled, baseline.pooled, atol=0, rtol=0)
    weights = load_file(clip_path / "model.safetensors")
    first = next(iter(weights))
    weights["text_model." + first] = weights.pop(first)
    save_file(weights, clip_path / "model.safetensors")
    with pytest.raises(ValueError, match="mixes native"):
        load_flux_text(
            t5_path, clip_path, text_config=local_text_config(t5_path, clip=clip_path), device=torch.device("cpu")
        )
