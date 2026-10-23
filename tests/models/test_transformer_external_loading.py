"""Exercise metadata-free checkpoints and consumer projection ordering."""

import json
from importlib import import_module

import pytest
import torch

from tests.models.test_z_image import build_model, request
from unitunner.artifacts.conversion import convert_weights
from unitunner.artifacts.formats.transformer import convert_transformer
from unitunner.artifacts.quantization import tensor_json
from unitunner.artifacts.tensors import TensorReader, resolve_shards, save_tensors
from unitunner.cli.main import main
from unitunner.models.components.transformer_artifacts import published_config
from unitunner.models.flux2.config.model import Flux2Config
from unitunner.models.qwen_image.config.model import QwenImageConfig
from unitunner.models.qwen_image.loading import load_qwen_image
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.models.wan.config.model import WanConfig
from unitunner.models.wan.loading import load_wan, save_wan
from unitunner.models.wan.transformer import WanTransformer
from unitunner.models.z_image.config.model import ZImageConfig
from unitunner.models.z_image.loading import load_z_image, save_z_image
from unitunner.training.integrations.z_image.binding import ZImageBinding

_CASES = [
    (
        "z_image",
        "ZImage",
        {
            "in_channels": 2,
            "dim": 24,
            "num_layers": 1,
            "num_refiner_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 2,
            "caption_dim": 8,
            "timestep_hidden_dim": 32,
            "axes_dims": (2, 2, 2),
        },
    ),
    (
        "flux",
        "Flux",
        {
            "in_channels": 8,
            "out_channels": 8,
            "vec_in_dim": 6,
            "context_in_dim": 12,
            "hidden_size": 32,
            "num_heads": 2,
            "depth": 1,
            "depth_single_blocks": 1,
            "axes_dim": (4, 6, 6),
        },
    ),
    (
        "flux2",
        "Flux2",
        {
            "version": "klein-4b",
            "variant": "base",
            "in_channels": 8,
            "context_in_dim": 12,
            "hidden_size": 32,
            "num_heads": 2,
            "depth": 1,
            "depth_single_blocks": 1,
            "axes_dim": (4, 4, 4, 4),
        },
    ),
    (
        "flux3",
        "Flux3",
        {
            "hidden_size": 32,
            "num_heads": 2,
            "depth": 1,
            "depth_single": 1,
            "axes_dim": (4, 4, 4, 4),
            "image_channels": 8,
            "text_channels": 12,
        },
    ),
    (
        "moss_image",
        "MossImage",
        {
            "dim": 32,
            "ffn_dim": 48,
            "num_heads": 2,
            "num_layers": 1,
            "text_dim": 12,
            "freq_dim": 8,
            "latent_channels": 8,
        },
    ),
    (
        "qwen_image",
        "QwenImage",
        {
            "in_channels": 8,
            "out_channels": 2,
            "num_layers": 1,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 12,
            "axes_dims_rope": (4, 6, 6),
        },
    ),
    (
        "qwen_image_2_1",
        "QwenImage21",
        {
            "in_channels": 8,
            "out_channels": 8,
            "num_layers": 1,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "context_in_dim": 12,
            "axes_dims_rope": (4, 6, 6),
        },
    ),
    (
        "krea2",
        "Krea2",
        {
            "hidden_size": 64,
            "timestep_dim": 16,
            "text_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "mlp_multiplier": 2,
            "num_layers": 1,
            "text_layers": 3,
            "text_heads": 4,
            "text_kv_heads": 2,
        },
    ),
    (
        "wan",
        "Wan",
        {
            "hidden_size": 48,
            "num_heads": 4,
            "num_layers": 1,
            "ffn_dim": 80,
            "timestep_dim": 16,
            "text_dim": 32,
            "text_length": 7,
        },
    ),
    (
        "minimax_h3",
        "MiniMaxH3",
        {
            "hidden_size": 24,
            "num_layers": 1,
            "token_refiner_num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "ffn_hidden_size": 40,
            "text_dim": 12,
            "timestep_input_dim": 8,
            "time_embed_hidden_size": 16,
            "time_embed_dim": 8,
            "rope_inv_freq_len": 2,
        },
    ),
]


@pytest.mark.parametrize("module,stem,options", _CASES)
def test_all_families_accept_metadata_free_prefixed_weights_and_local_config(tmp_path, module, stem, options):
    config = getattr(import_module(f"unitunner.models.{module}.config.model"), stem + "Config")(**options)
    model = getattr(import_module(f"unitunner.models.{module}.transformer"), stem + "Transformer")(config)
    state = model.state_dict()
    # H3 owns checkpoint rotary values rather than initializing them on construction.
    if "rope.inv_freq" in state:
        state["rope.inv_freq"].fill_(0.25)
    component = tmp_path / "transformer"
    component.mkdir()
    path = component / "diffusion_pytorch_model.safetensors"
    save_tensors({"model.diffusion_model." + k: v for k, v in state.items()}, path)
    (component / "config.json").write_text(config.model_dump_json(), encoding="utf-8")
    suffix = "qwen_image_21" if module == "qwen_image_2_1" else module
    loader = getattr(import_module(f"unitunner.models.{module}.loading"), "load_" + suffix)
    before = torch.get_rng_state().clone()
    loaded = loader(tmp_path)
    assert torch.equal(torch.get_rng_state(), before)
    assert loaded.config == config
    assert loaded.state_dict().keys() == state.keys()
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    # Native metadata is authoritative even beside an invalid external sidecar.
    native_path = tmp_path / "native.safetensors"
    saver = getattr(import_module(f"unitunner.models.{module}.loading"), "save_" + suffix)
    saver(model, native_path)
    native_path.with_suffix(".json").write_text("invalid sidecar", encoding="utf-8")
    assert loader(native_path).config == config
    assert loader(native_path, config=config).config == config
    dimension = next(name for name, value in options.items() if type(value) is int)
    conflicting = config.model_copy(update={dimension: options[dimension] * 2})
    with pytest.raises(ValueError, match="topology differs"):
        loader(native_path, config=conflicting)
    # Prefix normalization must not hide a foreign component or unexpected key.
    save_tensors({**{"model.diffusion_model." + k: v for k, v in state.items()}, "foreign.weight": torch.ones(1)}, path)
    with pytest.raises(ValueError):
        loader(tmp_path)


@pytest.mark.parametrize("indexed", [False, True])
def test_z_image_comfy_export_reloads_without_private_metadata_from_shards(tmp_path, indexed):
    model = build_model()
    source, output = tmp_path / "native.safetensors", tmp_path / "comfy.safetensors"
    save_z_image(model, source)
    # Exercise the user-facing default, not just the format helper.
    assert main(["weights", "convert", str(source), str(output), "--precision", "bf16"]) == 0
    with TensorReader(output) as reader:
        state = {key: reader.read(key) for key in reader.keys()}
    assert "x_embedder.weight" in state and "final_layer.linear.weight" in state
    qkv = state["layers.0.attention.qkv.weight"]
    native = model.state_dict()
    torch.testing.assert_close(
        qkv,
        torch.cat([native[f"layers.0.attention.to_{p}.weight"] for p in ("q", "k", "v")]).bfloat16(),
        rtol=0,
        atol=0,
    )
    assert "layers.0.attention.out.weight" in state
    assert "layers.0.attention.q_norm.weight" in state
    output.unlink()
    (tmp_path / "config.json").write_text(model.config.model_dump_json(), encoding="utf-8")
    items = list(state.items())
    index = {}
    for i, pairs in enumerate((items[::2], items[1::2]), 1):
        filename = f"custom-{i}.safetensors" if indexed else f"model-{i:05d}-of-00002.safetensors"
        save_tensors(dict(pairs), tmp_path / filename)
        index.update(dict.fromkeys(dict(pairs), filename))
    if indexed:
        path = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
        path.write_text(json.dumps({"weight_map": index}), encoding="utf-8")
    else:
        path = tmp_path / "model-00002-of-00002.safetensors"
    loaded = load_z_image(path, variant="turbo")
    assert loaded.config.variant == "turbo"
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, native[key].bfloat16(), rtol=0, atol=0)
    # External layout participates in the same differentiable training graph.
    prediction = ZImageBinding(loaded.float()).predict(request()).prediction
    prediction.square().mean().backward()
    assert loaded.layers[0].attention.to_q.weight.grad is not None
    if indexed:
        index[items[0][0]] = "../outside.safetensors"
        path.write_text(json.dumps({"weight_map": index}), encoding="utf-8")
        with pytest.raises(ValueError, match="sibling"):
            resolve_shards(path)


def test_published_size_selection_and_diffusers_config_names(tmp_path):
    path = tmp_path / "model.safetensors"
    save_tensors({"txt_in.weight": torch.empty(1, 7680)}, path)
    klein = published_config(path, Flux2Config, variant="base")
    assert (klein.version, klein.hidden_size, klein.depth, klein.depth_single_blocks) == ("klein-4b", 3072, 5, 20)
    with pytest.raises(ValueError, match="dimensions"):
        published_config(path, Flux2Config, version="klein-9b")
    save_tensors({"patch_embedding.weight": torch.empty(1536, 16, 1, 2, 2)}, path)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_attention_heads": 12,
                "attention_head_dim": 128,
                "num_layers": 30,
                "ffn_dim": 8960,
                "qk_norm": "rms_norm_across_heads",
                "cross_attn_norm": True,
            }
        ),
        encoding="utf-8",
    )
    wan = published_config(path, WanConfig)
    assert wan.hidden_size == 1536 and wan.num_heads == 12 and wan.ffn_dim == 8960 and wan.qk_norm
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"dim": 24, "n_heads": 4, "n_kv_heads": 2, "n_layers": 1, "cap_feat_dim": 8, "axes_dims": [2, 2, 2]}
        ),
        encoding="utf-8",
    )
    z = published_config(path, ZImageConfig, variant="turbo")
    assert z.dim == 24 and z.num_heads == 4 and z.num_kv_heads == 2 and z.variant == "turbo"


def test_transformer_import_needs_no_topology_and_exports_comfyui(tmp_path):
    model = build_model()
    source, destination = tmp_path / "external.safetensors", tmp_path / "comfy.safetensors"
    save_tensors(model.state_dict(), source)
    source.with_suffix(".json").write_text(model.config.model_dump_json(), encoding="utf-8")
    assert (
        main(["weights", "import", str(source), str(destination), "--family", "z-image", "--kind", "transformer"]) == 0
    )
    with TensorReader(destination) as reader:
        assert "layers.0.attention.qkv.weight" in reader.keys()
        assert "x_embedder.weight" in reader.keys()
    loaded = load_z_image(destination)
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_variant_selection_survives_conversion_with_embedded_topology(tmp_path):
    path = tmp_path / "model.safetensors"
    model = build_model()
    save_z_image(model, path)
    turbo = load_z_image(path, variant="turbo")
    assert turbo.config.variant == "turbo" and turbo.config.dim == model.config.dim
    wan = WanTransformer(WanConfig(hidden_size=48, num_heads=4, num_layers=1, ffn_dim=80, timestep_dim=16, text_dim=32))
    save_wan(wan, path)
    expert = load_wan(path, version="2.2", variant="t2v")
    assert expert.config.version == "2.2"
    for key, value in expert.state_dict().items():
        torch.testing.assert_close(value, wan.state_dict()[key], rtol=0, atol=0)


def test_diffusers_qwen2511_config_adds_comfy_reference_marker(tmp_path):
    model = QwenImageTransformer(
        QwenImageConfig(
            variant="edit-2511",
            in_channels=8,
            out_channels=2,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=12,
            axes_dims_rope=(4, 6, 6),
        )
    )
    source, output = tmp_path / "diffusers.safetensors", tmp_path / "comfy.safetensors"
    save_tensors(model.state_dict(), source)
    config = model.config.model_dump()
    config.pop("variant")
    config["zero_cond_t"] = True
    config["_class_name"] = "QwenImageTransformer2DModel"
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    convert_weights(source, output, family="qwen-image", precision="fp32", output_format="comfyui")
    with TensorReader(output) as reader:
        assert reader.shape("__index_timestep_zero__") == (0,)
    loaded = load_qwen_image(output)
    assert loaded.config.variant == "edit-2511"
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("family", ["flux", "flux2"])
def test_flux_qkv_and_output_modulation_order(family):
    native = {
        "double_blocks.0.img_attn.qkv.weight": torch.empty(6, 2),
        "final_layer.adaLN_modulation.1.weight": torch.empty(4, 2),
    }
    source = {
        "x_embedder.weight": torch.ones(2, 2),
        "transformer_blocks.0.attn.to_q.weight": torch.full((2, 2), 1.0),
        "transformer_blocks.0.attn.to_k.weight": torch.full((2, 2), 2.0),
        "transformer_blocks.0.attn.to_v.weight": torch.full((2, 2), 3.0),
        "norm_out.linear.weight": torch.tensor([[10.0, 10.0], [10.0, 10.0], [20.0, 20.0], [20.0, 20.0]]),
    }
    native["img_in.weight"] = torch.empty(2, 2)
    actual = convert_transformer(family, source, native)
    assert actual.keys() == native.keys()
    assert actual["double_blocks.0.img_attn.qkv.weight"][:, 0].tolist() == [1, 1, 2, 2, 3, 3]
    assert actual["final_layer.adaLN_modulation.1.weight"][:, 0].tolist() == [20, 20, 10, 10]


def test_minimax_diffusers_swiglu_and_derived_rope():
    source = {
        "context_embedder.weight": torch.ones(2, 2),
        "transformer_blocks.0.ff.net.0.proj.weight": torch.tensor([[2.0, 2.0], [2.0, 2.0], [3.0, 3.0], [3.0, 3.0]]),
    }
    expected = {
        "condition_proj.weight": torch.empty(2, 2),
        "blocks.0.mlp.fc1.weight": torch.empty(4, 2),
        "rope.inv_freq": torch.empty(2),
    }
    converted = convert_transformer("minimax_h3", source, expected)
    assert converted.keys() == expected.keys()
    assert converted["blocks.0.mlp.fc1.weight"][:, 0].tolist() == [3, 3, 2, 2]
    torch.testing.assert_close(converted["rope.inv_freq"], torch.tensor([1.0, 0.01]))


def test_qwen21_fused_gate_and_krea_modulation():
    gate = torch.ones(3, 2)
    up = torch.full((3, 2), 2.0)
    actual = convert_transformer(
        "qwen_image_2_1", {"transformer_blocks.0.img_mlp.gate_up.weight": torch.cat((gate, up))}, {}
    )
    torch.testing.assert_close(actual["transformer_blocks.0.img_mlp.gate_layer.weight"], gate)
    torch.testing.assert_close(actual["transformer_blocks.0.img_mlp.proj.weight"], up)
    actual = convert_transformer(
        "krea2",
        {"img_in.weight": torch.ones(2, 2), "transformer_blocks.0.scale_shift_table": torch.arange(12.0).reshape(6, 2)},
        {"first.weight": torch.empty(2, 2), "blocks.0.mod.lin": torch.empty(12)},
    )
    assert actual["blocks.0.mod.lin"].tolist() == list(range(12))


def test_z_image_quantized_comfy_export_can_train(tmp_path):
    pytest.importorskip("comfy_kitchen")
    model = build_model()
    source, packed = tmp_path / "native.safetensors", tmp_path / "packed.safetensors"
    save_z_image(model, source)
    report = convert_weights(
        source, packed, precision="int8_rowwise", family="z-image", output_format="comfyui", backend="eager"
    )
    assert report["converted"] > 0
    with TensorReader(packed) as reader:
        assert "layers.0.attention.qkv.comfy_quant" in reader.keys()
    loaded = load_z_image(packed)
    # Attention layout conversion decodes fused quantized rows before splitting.
    q = loaded.state_dict()["layers.0.attention.to_q.weight"]
    torch.testing.assert_close(q, model.state_dict()["layers.0.attention.to_q.weight"], atol=0.002, rtol=0.02)
    # Older ComfyUI files store exactly these descriptions in the file header.
    with TensorReader(packed) as reader:
        tensors = {key: reader.read(key) for key in reader.keys()}
        metadata = reader.metadata()
    layers = {
        key.removesuffix(".comfy_quant"): tensor_json(tensors.pop(key))
        for key in tuple(tensors)
        if key.endswith(".comfy_quant")
    }
    metadata["_quantization_metadata"] = json.dumps({"layers": layers})
    legacy = tmp_path / "legacy.safetensors"
    save_tensors(tensors, legacy, metadata=metadata)
    restored = load_z_image(legacy)
    torch.testing.assert_close(restored.state_dict()["layers.0.attention.to_q.weight"], q)
    dense = tmp_path / "dense.safetensors"
    convert_weights(legacy, dense, precision="fp32")
    restored = load_z_image(dense)
    torch.testing.assert_close(restored.state_dict()["layers.0.attention.to_q.weight"], q)
