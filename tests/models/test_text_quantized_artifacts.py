"""Regress packed tied aliases and unused quantized language heads at public loaders."""

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import T5Config, T5EncoderModel

from tests.support.qwen_image_21 import write_text_assets
from tests.support.text_assets import local_text_config
from unituner.artifacts.quantization import quantize_weight, serialize_weight
from unituner.artifacts.tensors import save_tensors
from unituner.models.components.text_loading import resolve_tied_text_weights
from unituner.models.flux.text.encoder import _load_model_weights
from unituner.models.krea2.text.encoder import load_krea2_text
from unituner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text


@pytest.fixture(autouse=True)
def kitchen():
    module = pytest.importorskip("comfy_kitchen")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with module.use_backend("eager"):
        yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("alias", ("shared.weight", "encoder.embed_tokens.weight"))
@pytest.mark.parametrize("precision", ("int8_convrot", "int8_rowwise", "float8_e4m3fn"))
def test_flux_t5_single_packed_alias(tmp_path, alias, precision):
    config = T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2)
    original = T5EncoderModel(config)
    state = dict(original.state_dict())
    values = state.pop("shared.weight")
    state.pop("encoder.embed_tokens.weight")
    packed = quantize_weight(values, precision, rotation_size=4)
    state.update(serialize_weight(alias, packed, precision))
    save_tensors(state, tmp_path / "model.safetensors")
    with torch.device("meta"):
        restored = T5EncoderModel(config)
    _load_model_weights(restored, tmp_path, device=torch.device("cpu"), dtype=torch.bfloat16)
    assert restored.shared.weight is restored.encoder.embed_tokens.weight
    torch.testing.assert_close(restored.shared.weight, packed.dequantize().bfloat16(), atol=0, rtol=0)
    assert torch.isfinite(restored(input_ids=torch.tensor([[1, 2, 3]])).last_hidden_state).all()


def test_tied_codes_with_different_scales_are_rejected_without_mutation():
    packed = quantize_weight(torch.randn(32, 16), "int8_convrot", rotation_size=4)
    state = serialize_weight("a.weight", packed, "int8_convrot")
    state.update(serialize_weight("b.weight", packed, "int8_convrot"))
    state["b.weight_scale"] = state["b.weight_scale"] * 2
    original = dict(state)
    with pytest.raises(ValueError, match="tied embedding values conflict"):
        resolve_tied_text_weights(state, ("a.weight", "b.weight"), label="test")
    assert state.keys() == original.keys()
    assert all(state[key] is value for key, value in original.items())


@pytest.mark.parametrize("family", ("qwen21", "krea2"))
@pytest.mark.parametrize("source_format", ("auto", "transformers", "qwen"))
def test_unused_packed_head_sidecars_follow_namespace(tmp_path, family, source_format):
    write_text_assets(tmp_path, language_layers=3, feature_layers=(0, 1, 2))
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    state.update(
        serialize_weight(
            "lm_head.weight",
            quantize_weight(state.pop("lm_head.weight"), "int8_convrot", rotation_size=4),
            "int8_convrot",
        )
    )
    if source_format == "qwen":
        state = {
            (
                "model." + key.removeprefix("model.language_model.") if key.startswith("model.language_model.") else key
            ): value
            for key, value in state.items()
        }
    save_file(state, path)
    options = local_text_config(tmp_path)
    if family == "qwen21":
        owner = load_qwen_image_21_text(
            tmp_path, device=torch.device("cpu"), text_config=options, source_format=source_format
        )
        features = owner.encode(["hello"])[0].features
    else:
        owner = load_krea2_text(tmp_path, device=torch.device("cpu"), config=options, source_format=source_format)
        features = owner.encode(["hello"])[0]
    assert torch.isfinite(features).all()
    assert not any(key.startswith("lm_head.") for key in owner.model.state_dict())


def test_z_image_causal_artifact_accepts_packed_unused_head(tmp_path):
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3ForCausalLM

    from unituner.models.z_image.text.encoder import load_z_image_text

    vocabulary = {char: index for index, char in enumerate(sorted(ByteLevel.alphabet()))}
    vocabulary["<|endoftext|>"] = 256
    tokenizer = Qwen2Tokenizer(vocab=vocabulary, merges=[], pad_token="<|endoftext|>")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(tmp_path)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=257,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    )
    model.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    for key in ("lm_head.weight", "model.embed_tokens.weight"):
        state.update(
            serialize_weight(key, quantize_weight(state.pop(key), "int8_convrot", rotation_size=4), "int8_convrot")
        )
    save_file(state, path)
    owner = load_z_image_text(
        tmp_path, device=torch.device("cpu"), dtype=torch.bfloat16, text_config=local_text_config(tmp_path)
    )
    assert torch.isfinite(owner.encode(["hello"]).features).all()
    assert owner.model.embed_tokens.weight.dtype == torch.bfloat16
