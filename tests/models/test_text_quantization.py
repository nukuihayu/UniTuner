"""Exercise actual text networks with packed weights, strict reload, and local prompt assets."""

import copy
from typing import Any, Literal

import pytest
import torch
from torch import nn
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    MistralConfig,
    MistralModel,
    Qwen2Tokenizer,
    Qwen3Config,
    Qwen3Model,
    T5Config,
    T5EncoderModel,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from tests.support.text_assets import local_text_config
from unituner.artifacts.conversion import convert_weights
from unituner.artifacts.formats.quantization import FORMATS
from unituner.artifacts.tensors import TensorReader, save_tensors
from unituner.config.text import TextEncodingConfig
from unituner.models.components.quantized import QuantizedLinear
from unituner.models.components.text_loading import (
    load_text_state,
    prepare_text_model,
    text_execution,
    validate_cache_text,
)
from unituner.models.wan.config.t5 import WanT5Config
from unituner.models.wan.text.model import WanT5Encoder
from unituner.models.z_image.text.encoder import load_z_image_text

FORMATS_TO_TEST = tuple(name for name, spec in FORMATS.items() if not spec.calibrated)
DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def text_model(family):
    options: dict[str, Any] = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "vocab_size": 64,
    }
    if family == "qwen3":
        return Qwen3Model(Qwen3Config(**options))
    if family == "mistral":
        return MistralModel(MistralConfig(**options))
    if family in ("qwen2-5-vl", "qwen3-vl"):
        options["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000, "mrope_section": [2, 2, 4]}
        return (
            Qwen2_5_VLTextModel(Qwen2_5_VLTextConfig(**options))
            if family == "qwen2-5-vl"
            else Qwen3VLTextModel(Qwen3VLTextConfig(**options))
        )
    if family == "qwen3-5":
        options.update(
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            layer_types=["full_attention", "linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
            },
        )
        return Qwen3_5TextModel(Qwen3_5TextConfig(**options))
    if family == "clip":
        return CLIPTextModel(
            CLIPTextConfig(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=3,
                num_attention_heads=4,
                max_position_embeddings=16,
            )
        )
    if family == "t5":
        return T5EncoderModel(
            T5Config(vocab_size=64, d_model=64, d_kv=16, d_ff=128, num_layers=3, num_heads=4, dropout_rate=0)
        )
    return WanT5Encoder(
        WanT5Config(vocab_size=64, hidden_size=64, attention_dim=64, ffn_dim=128, num_heads=4, num_layers=3, dropout=0)
    )


def forward(model):
    ids = torch.tensor([[3, 4, 5, 6]], device=next(model.parameters()).device)
    if isinstance(model, WanT5Encoder):
        return model(ids, torch.ones_like(ids))
    return model(input_ids=ids, return_dict=True).last_hidden_state


@pytest.mark.parametrize("family", ("qwen3", "qwen2-5-vl", "qwen3-vl", "qwen3-5", "mistral", "t5", "umt5", "clip"))
@pytest.mark.parametrize("precision", FORMATS_TO_TEST)
@torch.no_grad()
def test_text_networks_all_layouts_match_reconstructed_floating_model(tmp_path, family, precision):
    kitchen = pytest.importorskip("comfy_kitchen")
    torch.manual_seed(47)
    model = text_model(family).eval().requires_grad_(False)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    config = TextEncodingConfig(quantization=precision, quantization_rotation_size=16)
    with kitchen.use_backend("eager"):
        prepare_text_model(model, config, family=family, device=torch.device("cpu"))
    validate_cache_text(model)
    quantized = [name for name, layer in model.named_modules() if isinstance(layer, QuantizedLinear)]
    assert quantized
    assert all(any(f"{prefix}.1." in name for prefix in ("layers", "blocks", "block")) for name in quantized)
    for key, tensor in original.items():
        if key in model.state_dict() and not any(key.startswith(name + ".") for name in quantized):
            torch.testing.assert_close(model.state_dict()[key], tensor, atol=0, rtol=0)
    dense = copy.deepcopy(model)
    for name, layer in tuple(dense.named_modules()):
        if isinstance(layer, QuantizedLinear):
            replacement = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
            replacement.weight = nn.Parameter(layer.dequantized(torch.float32), requires_grad=False)
            if layer.bias is not None:
                replacement.bias = nn.Parameter(layer.bias.detach(), requires_grad=False)
            dense.set_submodule(name, replacement.eval())
    actual = forward(model)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, forward(dense), atol=0, rtol=0)
    source, packed = tmp_path / "source.safetensors", tmp_path / "packed.safetensors"
    save_tensors(original, source)
    with kitchen.use_backend("eager"):
        convert_weights(source, packed, precision=precision, family=family, rotation_size=16)
    restored = text_model(family)
    with TensorReader(packed) as reader:
        load_text_state(restored, {k: reader.read(k) for k in reader.keys()}, dtype=torch.float32, label="test text")
    restored.eval().requires_grad_(False)
    torch.testing.assert_close(forward(restored), actual, atol=0, rtol=0)
    assert text_execution(restored) == text_execution(model)


@pytest.mark.parametrize("precision", FORMATS_TO_TEST)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@torch.no_grad()
def test_public_qwen3_prompt_loader_online_and_offline_quantization(tmp_path, precision, device, dtype):
    from tokenizers.pre_tokenizers import ByteLevel

    kitchen = pytest.importorskip("comfy_kitchen")
    vocabulary = {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}
    vocabulary["<|endoftext|>"] = 256
    tokenizer = Qwen2Tokenizer(vocab=vocabulary, merges=[], pad_token="<|endoftext|>")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(tmp_path)
    model = Qwen3Model(
        Qwen3Config(
            vocab_size=257,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
        )
    )
    model.to(dtype).save_pretrained(tmp_path)
    precisions: dict[torch.dtype, Literal["fp32", "fp16", "bf16"]] = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    options = TextEncodingConfig(
        quantization=precision,
        quantization_rotation_size=16,
        precision=precisions[dtype],
    )
    with kitchen.use_backend("eager"):
        online = load_z_image_text(
            tmp_path, device=torch.device(device), max_length=16, text_config=local_text_config(tmp_path, base=options)
        )
    expected = online.encode(["test"]).features
    with torch.autocast(device, dtype=torch.bfloat16):
        torch.testing.assert_close(online.encode(["test"]).features, expected, atol=0, rtol=0)
    assert expected.dtype == dtype and torch.isfinite(expected).all()
    source, output = tmp_path / "model.safetensors", tmp_path / "packed.safetensors"
    with kitchen.use_backend("eager"):
        convert_weights(source, output, precision=precision, family="qwen3", rotation_size=16, device=device)
    output.replace(source)
    offline = load_z_image_text(
        tmp_path, text_config=local_text_config(tmp_path), device=torch.device(device), dtype=dtype, max_length=16
    )
    torch.testing.assert_close(offline.encode(["test"]).features, expected, atol=0, rtol=0)
    assert all(not value.requires_grad for value in offline.model.parameters())


def test_text_quantization_config_rejects_calibrated_and_invalid_options():
    from pydantic import ValidationError

    for options in (
        {"quantization": "awq_w4a16"},
        {"quantization": "svdquant_w4a4"},
        {"quantization_rotation_size": 8},
        {"quantization_group_size": 6},
    ):
        with pytest.raises(ValidationError):
            TextEncodingConfig.model_validate(options)


@pytest.mark.parametrize("family", ("qwen3", "qwen2-5-vl", "qwen3-vl", "qwen3-5", "mistral", "t5", "umt5", "clip"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@torch.no_grad()
def test_text_floating_precision_and_fp8_storage_keep_floating_activations(family, dtype):
    kitchen = pytest.importorskip("comfy_kitchen")
    model = text_model(family).to(dtype).eval().requires_grad_(False)
    baseline = forward(model)
    assert baseline.dtype == dtype and torch.isfinite(baseline).all()
    with kitchen.use_backend("eager"):
        prepare_text_model(
            model, TextEncodingConfig(quantization="float8_e4m3fn"), family=family, device=torch.device("cpu")
        )
    actual = forward(model)
    assert actual.dtype == dtype and torch.isfinite(actual).all()


@pytest.mark.parametrize("gated", (False, True))
@torch.no_grad()
def test_t5_feed_forward_uses_logical_dtype_for_packed_output_projection(gated):
    kitchen = pytest.importorskip("comfy_kitchen")
    config = T5Config(
        vocab_size=64,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=3,
        num_heads=4,
        dropout_rate=0,
        feed_forward_proj="gated-gelu" if gated else "relu",
    )
    model = T5EncoderModel(config).bfloat16().eval().requires_grad_(False)
    with kitchen.use_backend("eager"):
        prepare_text_model(model, TextEncodingConfig(quantization="nvfp4"), family="t5", device=torch.device("cpu"))
    assert forward(model).dtype == torch.bfloat16


@pytest.mark.parametrize("index", (0, 2))
@pytest.mark.parametrize("precision", ("float8_e4m3fn", "int8_convrot"))
@torch.no_grad()
def test_prepacked_boundary_text_layer_preserves_storage_and_outputs(index, precision):
    from unituner.artifacts.quantization import quantize_weight, serialize_weight

    kitchen = pytest.importorskip("comfy_kitchen")
    model = text_model("qwen3").eval().requires_grad_(False)
    name = f"layers.{index}.self_attn.q_proj"
    projection = model.get_submodule(name)
    assert isinstance(projection, nn.Linear)
    with kitchen.use_backend("eager"):
        packed = quantize_weight(projection.weight, precision, rotation_size=16)
    model.set_submodule(
        name,
        QuantizedLinear(serialize_weight("weight", packed, precision), dtype=torch.float32),
    )
    state = {key: value.clone() for key, value in model.state_dict().items()}
    with kitchen.use_backend("eager"):
        expected = forward(model)
        restored = text_model("qwen3")
        load_text_state(restored, state, dtype=torch.float32, label="Qwen3")
        restored.train()
        prepare_text_model(restored, TextEncodingConfig(), family="qwen3", device=torch.device("cpu"))
        prepare_text_model(model, TextEncodingConfig(), family="qwen3", device=torch.device("cpu"))
        validate_cache_text(restored)
        assert isinstance(restored.get_submodule(name), QuantizedLinear)
        for key, value in state.items():
            torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
        torch.testing.assert_close(forward(restored), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("format", ("int8_convrot", "asym_w4a8_int8", "w6a8_int8"))
def test_explicit_text_quantization_rejects_existing_parameter_mismatches(format):
    kitchen = pytest.importorskip("comfy_kitchen")
    model = text_model("qwen3").eval().requires_grad_(False)
    options = TextEncodingConfig(quantization=format, quantization_rotation_size=16)
    with kitchen.use_backend("eager"):
        prepare_text_model(model, options, family="qwen3", device=torch.device("cpu"))
        prepare_text_model(model, options, family="qwen3", device=torch.device("cpu"))
        with pytest.raises(ValueError, match="rotation size differs"):
            prepare_text_model(
                model,
                options.model_copy(update={"quantization_rotation_size": 64}),
                family="qwen3",
                device=torch.device("cpu"),
            )
        if format != "int8_convrot":
            with pytest.raises(ValueError, match="group size differs"):
                prepare_text_model(
                    model,
                    options.model_copy(update={"quantization_group_size": 8}),
                    family="qwen3",
                    device=torch.device("cpu"),
                )


def test_text_quantization_ignores_ambient_autocast():
    kitchen = pytest.importorskip("comfy_kitchen")
    model = text_model("qwen3").eval().requires_grad_(False)
    reference = copy.deepcopy(model)
    options = TextEncodingConfig(quantization="int8_convrot", quantization_rotation_size=16)
    with kitchen.use_backend("eager"):
        prepare_text_model(reference, options, family="qwen3", device=torch.device("cpu"))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prepare_text_model(model, options, family="qwen3", device=torch.device("cpu"))
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
