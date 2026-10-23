"""Pin model quantization boundaries and protection precedence."""

import pytest
import torch

from unitunner.artifacts.conversion import convert_weights
from unitunner.artifacts.formats.precision_policy import POLICIES, protection_reason, resolve_family
from unitunner.artifacts.tensors import TensorReader, save_tensors


@pytest.mark.parametrize(
    "family,key",
    [
        ("z-image", "layers.0.attention.to_q.weight"),
        ("flux", "double_blocks.0.img_attn.qkv.weight"),
        ("flux2", "single_blocks.0.linear1.weight"),
        ("flux3", "content_mode_blocks.image.0.q_proj.weight"),
        ("moss-image", "layers.0.proj.ffn_up_gate.weight"),
        ("qwen-image", "transformer_blocks.0.attn.to_q.weight"),
        ("qwen-image-2-1", "transformer_blocks.0.img_mlp.gate_layer.weight"),
        ("krea2", "blocks.0.attn.qkv.weight"),
        ("wan", "blocks.0.self_attn.q.weight"),
        ("minimax-h3", "double_blocks.0.video.adaln_proj.linear.weight"),
    ],
)
def test_all_model_policies_keep_conditioning_and_allow_core_projections(family, key):
    assert family in POLICIES
    for prefix in ("", "model.diffusion_model.", "diffusion_model."):
        assert protection_reason(family, prefix + key) is None
        assert protection_reason(family, prefix + "time_in.in_layer.weight")
        assert protection_reason(family, prefix + "final_layer.linear.weight")
    assert protection_reason(family, key.replace("weight", "norm.weight"))


def test_metadata_detection_is_explicit_for_ambiguous_layouts():
    assert resolve_family("auto", {"unitunner.qwen_image_2_1.config": "{}"}, ()) == "qwen-image-2-1"
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_family("auto", {}, ("transformer_blocks.0.attn.to_q.weight",))
    with pytest.raises(ValueError, match="differs"):
        resolve_family("wan", {"unitunner.flux.config": "{}"}, ())


def test_sensitive_layers_cannot_be_forced_by_include(tmp_path):
    pytest.importorskip("comfy_kitchen")
    source, output = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    values = {
        key: torch.randn(16, 16)
        for key in (
            "model.diffusion_model.blocks.0.self_attn.q.weight",
            "model.diffusion_model.blocks.0.modulation.weight",
            "model.diffusion_model.time_embedding.0.weight",
        )
    }
    save_tensors(values, source)
    report = convert_weights(
        source, output, precision="int8_convrot", family="wan", include=("*",), rotation_size=4, backend="eager"
    )
    assert report["converted"] == 1 and len(report["skipped"]) == 2
    with TensorReader(output) as reader:
        for key in report["skipped"]:
            torch.testing.assert_close(reader.read(key), values[key], atol=0, rtol=0)
            assert key.removesuffix("weight") + "comfy_quant" not in reader.keys()


@pytest.mark.parametrize("family", ("qwen2-5-vl", "qwen3", "qwen3-vl", "qwen3-5", "mistral", "t5", "umt5", "clip"))
@pytest.mark.parametrize(
    "stack",
    (
        "model.layers.",
        "language_model.layers.",
        "model.language_model.layers.",
        "encoder.block.",
        "blocks.",
        "text_model.encoder.layers.",
    ),
)
def test_text_encoder_policies_preserve_first_last_and_special_branches(tmp_path, family, stack):
    from unitunner.artifacts.formats.precision_policy import language_boundaries

    keys = tuple(f"{stack}{i}.self_attn.q_proj.weight" for i in (0, 1, 2, 10))
    boundaries = language_boundaries(keys)
    assert protection_reason(family, keys[0], boundaries)
    assert protection_reason(family, keys[-1], boundaries)
    assert protection_reason(family, keys[1], boundaries) is None
    for suffix in ("input_layernorm.weight", "in_proj_a.weight", "in_proj_b.weight"):
        assert protection_reason(family, f"{stack}1.{suffix}", boundaries)
    for name in (
        "embed_tokens.weight",
        "lm_head.weight",
        "shared.weight",
        "visual.layers.1.attn.qkv.weight",
        "vision.layers.1.attn.qkv.weight",
        "mtp.layers.1.attn.q_proj.weight",
        "decoder.block.1.attn.q.weight",
    ):
        assert protection_reason(family, name, boundaries)


def test_text_conversion_retains_boundaries_and_all_original_keys(tmp_path):
    source, output = tmp_path / "qwen.safetensors", tmp_path / "qwen-fp8.safetensors"
    values = {f"model.layers.{i}.self_attn.q_proj.weight": torch.randn(16, 16) for i in (0, 1, 2, 10)}
    values.update(
        {
            "lm_head.weight": torch.randn(16, 16),
            "model.embed_tokens.weight": torch.randn(16, 16),
            "visual.layers.1.attn.qkv.weight": torch.randn(16, 16),
        }
    )
    save_tensors(values, source)
    report = convert_weights(source, output, precision="float8_e4m3fn", family="qwen3-vl", backend="eager")
    assert report["converted"] == 2
    with TensorReader(output) as reader:
        assert set(values) <= set(reader.keys())
        for key in report["skipped"]:
            torch.testing.assert_close(reader.read(key), values[key], atol=0, rtol=0)


def test_z_image_refiners_and_qwen_modulation_remain_dense():
    for name in ("context_refiner.0.attention.to_q.weight", "noise_refiner.0.feed_forward.w1.weight"):
        assert protection_reason("z-image", name)
    assert protection_reason("qwen-image", "transformer_blocks.0.img_mod.1.weight")


# Keys come from convert_to_quant 5d2f539, independently of UniTunner's glob
# definitions. Exercise them inside eligible blocks, where an allowlist alone
# would accidentally quantize auxiliary projections.
@pytest.mark.parametrize(
    "family,block",
    (
        ("z-image", "layers.1"),
        ("flux", "double_blocks.1"),
        ("flux2", "double_blocks.1"),
        ("flux3", "single_blocks.1"),
        ("moss-image", "layers.1"),
        ("qwen-image", "transformer_blocks.1"),
        ("qwen-image-2-1", "transformer_blocks.1"),
        ("krea2", "blocks.1"),
        ("wan", "blocks.1"),
        ("minimax-h3", "double_blocks.1"),
        ("qwen3", "model.layers.1"),
        ("qwen3-vl", "model.language_model.layers.1"),
        ("qwen3-5", "model.layers.1"),
        ("qwen2-5-vl", "model.language_model.layers.1"),
        ("mistral", "model.layers.1"),
        ("t5", "encoder.block.1"),
        ("umt5", "blocks.1"),
        ("clip", "text_model.encoder.layers.1"),
        ("generic", "layers.1"),
    ),
)
def test_reference_common_exclusions_cover_nested_auxiliary_projections(family, block):
    reference_paths = (
        "norm",
        "bias",
        "embed_tokens",
        "lm_head",
        "shared",
        "patch_embedding",
        "audio_model.patch_embedding",
        "ref_conv",
        "control_adapter",
        "motion_encoder.enc.net_app",
        "face_encoder.conv",
        "pose_patch_embedding",
        "motion_encoder.enc.fc",
        "img_emb.proj",
        "k_norm",
        "q_norm",
        "motion_encoder.dec",
        "head.modulation",
        "casual_audio_encoder",
        "cond_encoder",
        "frame_packer",
        "norm_k",
        "norm_q",
        "tekken_model",
        "multi_modal_projector",
        "patch_conv",
        "ln_pre",
        "input_layernorm",
        "attention_norm",
        "post_attention_layernorm",
        "mm_input_projection_weight",
    )
    for path in reference_paths:
        assert protection_reason(family, f"{block}.{path}.attn.qkv_proj.weight"), (family, path)


@pytest.mark.parametrize(
    "family,block,paths",
    (
        ("flux", "double_blocks.1", ("stream_modulation", "guidance_in", "time_in", "final_layer", "img_in", "txt_in")),
        (
            "flux2",
            "double_blocks.1",
            ("stream_modulation", "guidance_in", "time_in", "final_layer", "img_in", "txt_in"),
        ),
        ("qwen-image", "transformer_blocks.1", ("time_text_embed", "img_in", "norm_out", "proj_out", "txt_in")),
        ("qwen-image-2-1", "transformer_blocks.1", ("time_text_embed", "img_in", "norm_out", "proj_out", "txt_in")),
        (
            "z-image",
            "layers.1",
            (
                "x_embedder",
                "clip_text_pooled_proj",
                "final_layer",
                "cap_embedder.1",
                "adaLN_modulation",
                "t_embedder",
                "time_text_embed",
                "context_refiner",
                "noise_refiner",
            ),
        ),
        (
            "krea2",
            "blocks.1",
            ("first", "last.linear", "tmlp.0", "tmlp.2", "tproj.1", "txtfusion.projector", "txtmlp.1", "txtmlp.3"),
        ),
        (
            "wan",
            "blocks.1",
            (
                "text_embedding",
                "time_embedding",
                "audio_model.text_embedding",
                "trainable_cond_mask",
                "time_projection",
                "video_model.time_projection",
                "head.head",
                "face_encoder.out_proj",
                "face_adapter",
                "audio_injector",
            ),
        ),
        (
            "minimax-h3",
            "double_blocks.1",
            ("audio_patch_proj", "condition_proj", "final_layer", "time_embedder", "token_refiner", "video_patch_proj"),
        ),
    ),
)
def test_reference_family_exclusions_cannot_be_quantized_inside_core_blocks(family, block, paths):
    for path in paths:
        assert protection_reason(family, f"{block}.{path}.attn.qkv_proj.weight"), (family, path)


def test_conversion_and_training_preparation_share_nested_protections(tmp_path):
    from torch import nn

    from unitunner.models.components.quantized import QuantizedLinear
    from unitunner.models.components.weight_quantization import quantize_model

    kitchen = pytest.importorskip("comfy_kitchen")
    model = nn.Module()
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.q = nn.Linear(16, 16, bias=False)
    for name in ("audio_injector", "face_adapter", "control_adapter", "time_projection", "ln_pre", "cond_encoder"):
        block.add_module(name, nn.Linear(16, 16, bias=False))
    model.blocks = nn.ModuleList([block])
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    source, output = tmp_path / "source.safetensors", tmp_path / "packed.safetensors"
    save_tensors(original, source)
    with kitchen.use_backend("eager"):
        report = convert_weights(
            source, output, family="wan", precision="int8_convrot", rotation_size=4, include=("*",)
        )
        assert quantize_model(model, "int8_convrot", family="wan", rotation_size=4) == report["converted"] == 1
    assert isinstance(model.get_submodule("blocks.0.self_attn.q"), QuantizedLinear)
    with TensorReader(output) as reader:
        for key in report["skipped"]:
            torch.testing.assert_close(reader.read(key), original[key], atol=0, rtol=0)
            torch.testing.assert_close(model.state_dict()[key], original[key], atol=0, rtol=0)
            assert key.removesuffix("weight") + "comfy_quant" not in reader.keys()
