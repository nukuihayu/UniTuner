"""Read ComfyUI Qwen 2.1 topology and fused weights without private metadata."""

import json

import pytest
import torch

from unitunner.adapters.config import AdapterConfig
from unitunner.artifacts.formats.transformer import convert_transformer
from unitunner.artifacts.quantization import json_tensor, quantize_weight, serialize_weight
from unitunner.artifacts.tensors import save_tensors
from unitunner.models.components.quantized import QuantizedLinear
from unitunner.models.qwen_image_2_1.config.model import QwenImage21Config
from unitunner.models.qwen_image_2_1.layers import QwenImage21Block
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21
from unitunner.models.qwen_image_2_1.projections import adapter_projections
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unitunner.training.parameters import ParameterInstallation


@pytest.fixture
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(72)
    yield QwenImage21Transformer(
        QwenImage21Config(
            in_channels=8, out_channels=8, context_in_dim=12, num_layers=2, num_attention_heads=1, mlp_ratio=2
        )
    )
    torch.set_num_threads(previous)


def fused_state(model):
    state = dict(model.state_dict())
    for index in range(model.config.num_layers):
        root = f"transformer_blocks.{index}.img_mlp."
        state[root + "gate_up.weight"] = torch.cat(
            (state.pop(root + "gate_layer.weight"), state.pop(root + "proj.weight"))
        )
    return state


@pytest.mark.parametrize("prefix", ("", "transformer.", "diffusion_model.", "model.diffusion_model."))
@pytest.mark.parametrize("fused", (False, True))
@pytest.mark.parametrize("sharded", (False, True))
def test_metadata_free_topology_and_exact_weights(tmp_path, model, prefix, fused, sharded):
    state = fused_state(model) if fused else model.state_dict()
    state = {prefix + key: value for key, value in state.items()}
    path = tmp_path / "comfy.safetensors"
    if sharded:
        mapping = {}
        for index in range(2):
            shard = {key: value for n, (key, value) in enumerate(state.items()) if n % 2 == index}
            name = f"model-{index}.safetensors"
            save_tensors(shard, tmp_path / name)
            mapping.update(dict.fromkeys(shard, name))
        path = tmp_path / "model.safetensors.index.json"
        path.write_text(json.dumps({"weight_map": mapping}))
    else:
        save_tensors(state, path)
    rng = torch.get_rng_state().clone()
    loaded = load_qwen_image_21(path)
    assert torch.equal(torch.get_rng_state(), rng)
    assert loaded.config == model.config
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    target, context, time = torch.randn(1, 4, 8), torch.randn(1, 3, 12), torch.tensor([0.25274774])
    expected = model(target, context, time, (((1, 2, 2),),))
    actual = loaded(target, context, time, (((1, 2, 2),),))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("precision", ("int8_rowwise", "int8_convrot"))
def test_fused_quantization_preserves_other_packed_layers(tmp_path, model, precision):
    pytest.importorskip("comfy_kitchen")
    state = fused_state(model)
    dense = dict(state)
    for key in tuple(state):
        if key.endswith((".gate_up.weight", ".attn.to_q.weight", ".img_mlp.out.weight")):
            weight = quantize_weight(state.pop(key), precision, rotation_size=64)
            dense[key] = weight.dequantize()
            state.update(serialize_weight(key, weight, precision))
    path = tmp_path / "quantized.safetensors"
    save_tensors(state, path)
    loaded = load_qwen_image_21(path)
    assert loaded.config == model.config
    for index in range(model.config.num_layers):
        block = loaded.transformer_blocks[index]
        assert isinstance(block, QwenImage21Block)
        assert isinstance(block.attn.to_q, QuantizedLinear)
        assert isinstance(block.img_mlp.out, QuantizedLinear)
        assert type(block.img_mlp.gate_layer) is torch.nn.Linear
        assert type(block.img_mlp.proj) is torch.nn.Linear
    loaded_state = loaded.state_dict()
    for key, value in state.items():
        if ".gate_up." not in key:
            torch.testing.assert_close(loaded_state[key], value, atol=0, rtol=0)
    model.load_state_dict(convert_transformer("qwen_image_2_1", dense, model.state_dict()))
    target, context, time = torch.randn(1, 4, 8), torch.randn(1, 3, 12), torch.tensor([0.25274774])
    actual = loaded(target, context, time, (((1, 2, 2),),))
    expected = model(target, context, time, (((1, 2, 2),),))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    assert loaded.img_in.weight.grad is not None and torch.isfinite(loaded.img_in.weight.grad).all()
    installation = ParameterInstallation(
        loaded, adapter=AdapterConfig(rank=2, alpha=2.0), targets=adapter_projections(loaded)
    )
    try:
        loaded(target, context, time, (((1, 2, 2),),)).square().mean().backward()
        gradients = [parameter.grad for parameter in installation.branches.parameters()]
        assert all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert any(value is not None and value.count_nonzero() for value in gradients)
    finally:
        installation.close()


@pytest.mark.parametrize("malformed", (False, True))
def test_comfy_attention_backend_metadata_is_not_a_model_parameter(tmp_path, model, malformed):
    state = fused_state(model)
    state["transformer_blocks.1.attn.comfy_attention.config"] = (
        torch.ones(35) if malformed else json_tensor({"attention": "comfy_kitchen_int8"})
    )
    path = tmp_path / "turbo.safetensors"
    save_tensors(state, path)
    if malformed:
        with pytest.raises(ValueError, match="attention configuration"):
            load_qwen_image_21(path)
    else:
        loaded = load_qwen_image_21(path)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("invalid", ("duplicate", "odd_rows", "wrong_width", "missing_block", "sidecar"))
def test_rejects_ambiguous_or_invalid_architecture(tmp_path, model, invalid):
    state = fused_state(model)
    key = "transformer_blocks.0.img_mlp.gate_up.weight"
    if invalid == "duplicate":
        state["transformer_blocks.0.img_mlp.proj.weight"] = model.transformer_blocks[0].img_mlp.proj.weight
    elif invalid == "odd_rows":
        state[key] = state[key][:-1]
    elif invalid == "wrong_width":
        state[key] = state[key][:, :-1]
    elif invalid == "missing_block":
        state = {key.replace("transformer_blocks.1.", "transformer_blocks.2."): value for key, value in state.items()}
    else:
        (tmp_path / "config.json").write_text(json.dumps({"num_layers": 9}))
    path = tmp_path / "invalid.safetensors"
    save_tensors(state, path)
    with pytest.raises(ValueError):
        load_qwen_image_21(path)
