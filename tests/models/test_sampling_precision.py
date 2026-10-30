"""Exercise real sampler trajectories with AMP masters and low precision LoRA bases."""

from importlib import import_module

import pytest
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.models.test_minimax_h3_sampling import _model as h3_model
from tests.models.test_moss_image import small_config
from tests.models.test_z_image import build_model as z_model
from unituner.adapters.attachment import AdapterAttachment
from unituner.adapters.lora import LoRA
from unituner.models.components.conditioning import Caption
from unituner.models.flux.config.model import FluxConfig
from unituner.models.flux.transformer import FluxTransformer
from unituner.models.flux2.config.model import Flux2Config
from unituner.models.flux2.transformer import Flux2Transformer
from unituner.models.flux3.config.model import Flux3Config
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.models.krea2.conditioning import Krea2TextFeatures
from unituner.models.krea2.config.model import Krea2Config
from unituner.models.krea2.transformer import Krea2Transformer
from unituner.models.minimax_h3.layout import build_layout
from unituner.models.moss_image.transformer import MossImageTransformer
from unituner.models.qwen_image.config.model import QwenImageConfig
from unituner.models.qwen_image.transformer import QwenImageTransformer
from unituner.models.qwen_image_2_1.conditioning import QwenImage21TextFeatures
from unituner.models.qwen_image_2_1.config.model import QwenImage21Config
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.models.wan.config.model import WanConfig
from unituner.models.wan.transformer import WanTransformer
from unituner.precision import FLOAT_DTYPES, sampling_context, sampling_dtype, sampling_precision
from unituner.training.previews.context import preview_context


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_case(family, dtype):
    """Use tiny real models, CFG and independent references where supported."""
    noise = torch.randn(1, 8, 2, 4).to(dtype)
    text = torch.randn(1, 3, 12)
    kwargs = {"steps": 3}
    if family == "flux":
        model = FluxTransformer(
            FluxConfig(
                in_channels=8,
                out_channels=8,
                vec_in_dim=6,
                context_in_dim=12,
                hidden_size=32,
                num_heads=2,
                depth=1,
                depth_single_blocks=1,
                axes_dim=(4, 6, 6),
            )
        )
        args = (torch.randn(1, 2, 2, 4).to(dtype), text, torch.randn(1, 6))
    elif family == "flux2":
        model = Flux2Transformer(
            Flux2Config(
                version="klein-4b",
                variant="base",
                in_channels=8,
                context_in_dim=12,
                hidden_size=32,
                num_heads=2,
                depth=1,
                depth_single_blocks=1,
                axes_dim=(4, 4, 4, 4),
            )
        )
        args = (noise, (text[0],))
        kwargs.update(negative=(text[0] * 0.5,), references=((torch.randn(8, 1, 2),),))
    elif family == "flux3":
        model = Flux3Transformer(
            Flux3Config(
                variant="base",
                hidden_size=32,
                num_heads=2,
                depth=1,
                depth_single=1,
                axes_dim=(4, 4, 4, 4),
                image_channels=8,
                text_channels=12,
            )
        )
        args = (noise, (text[0],))
        kwargs.update(negative=(text[0] * 0.5,), references=((torch.randn(8, 1, 2),),))
    elif family == "qwen_image":
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
        args = (torch.randn(1, 2, 1, 2, 4).to(dtype), Caption(text))
        kwargs.update(negative=Caption(text * 0.5), references=(torch.randn(1, 2, 1, 4, 2),))
    elif family == "qwen_image_2_1":
        model = QwenImage21Transformer(
            QwenImage21Config(
                in_channels=4,
                out_channels=4,
                context_in_dim=12,
                num_layers=1,
                attention_head_dim=8,
                num_attention_heads=2,
                axes_dims_rope=(2, 2, 4),
            )
        )
        args = (torch.randn(1, 4, 1, 2, 4).to(dtype), QwenImage21TextFeatures(text[0], (1,), ((4, 2),)))
        kwargs.update(
            negative=QwenImage21TextFeatures(text[0] * 0.5, (0,), ((4, 2),)),
            references=(torch.randn(1, 4, 1, 4, 2),),
            guidance=2.0,
        )
    elif family == "krea2":
        model = Krea2Transformer(
            Krea2Config(
                hidden_size=64,
                timestep_dim=16,
                text_dim=16,
                num_heads=4,
                num_kv_heads=2,
                mlp_multiplier=2,
                num_layers=1,
                text_layers=2,
                text_heads=2,
                text_kv_heads=1,
            )
        )
        features = Krea2TextFeatures(torch.randn(1, 3, 2, 16), (3,))
        args = (torch.randn(1, 16, 2, 4).to(dtype), features)
        kwargs.update(negative=features)
    elif family == "moss_image":
        model = MossImageTransformer(small_config())
        args = (noise, text)
        kwargs.update(negative=text * 0.5, uncond_skip_layers=())
    elif family == "z_image":
        model = z_model()
        args = (torch.randn(1, 2, 4, 6).to(dtype), Caption(torch.randn(1, 3, 8)))
    elif family == "wan":
        model = WanTransformer(
            WanConfig(
                hidden_size=48, num_heads=4, num_layers=1, ffn_dim=80, timestep_dim=16, text_dim=32, text_length=7
            )
        )
        args = (torch.randn(1, 16, 2, 4, 4).to(dtype), (torch.randn(3, 32),))
        kwargs.update(guidance=None)
    else:
        model = h3_model()
        layout = build_layout(task="t2va", text_length=3, target_video=(2, 2, 4), target_audio_frames=8)
        args = (
            torch.randn(1, 24, 2, 2, 4).to(dtype),
            torch.randn(1, 32, 2, 8).to(dtype),
            text,
            torch.tensor([[1, 0, 1]]),
            layout,
        )
    return model, args, kwargs


def test_krea2_turbo_native_defaults_do_not_require_cfg():
    from unituner.models.krea2.inference.sampling import sample_latents

    model, args, _ = make_case("krea2", torch.float32)
    model.config = model.config.model_copy(update={"variant": "turbo"})
    default = sample_latents(model, *args)
    explicit = sample_latents(model, *args, steps=8, guidance=1, mu=1.15)
    assert torch.equal(default, explicit)


@pytest.mark.parametrize(
    "variant,guidance,negative", [("raw", None, True), ("turbo", None, False), ("turbo", 2.0, True)]
)
def test_krea2_default_conditioning_follows_variant(variant, guidance, negative):
    from unituner.config.sampling import SamplingConfig
    from unituner.models.krea2.inference.conditioning import encode_krea2_prompts

    calls = []

    class Text:
        def encode(self, prompts):
            calls.extend(prompts)
            return (torch.ones(3, 2, 16),)

    _, uncond = encode_krea2_prompts(Text(), SamplingConfig(prompt="test", guidance=guidance), variant=variant)
    assert calls == (["test", ""] if negative else ["test"])
    assert (uncond is not None) == negative


@pytest.mark.parametrize(
    "family",
    ["flux", "flux2", "flux3", "qwen_image", "qwen_image_2_1", "krea2", "moss_image", "z_image", "wan", "minimax_h3"],
)
@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("storage", ["masters", "adapter"])
def test_preview_sampling_supports_training_precision_and_preserves_owners(family, precision, storage):
    dtype = FLOAT_DTYPES[precision]
    model, args, kwargs = make_case(family, dtype)
    model.train()
    if storage == "adapter":
        # Follow native loading, including FP32 time/projection islands.
        if family == "minimax_h3":
            template = type(model)(model.config, dtype=dtype)
            dtypes = {name: p.dtype for name, p in template.named_parameters()}
            for name, parameter in model.named_parameters():
                parameter.data = parameter.data.to(dtypes[name])
        elif family == "wan":
            for name, parameter in model.named_parameters():
                protected = name.startswith(("time_embedding.", "time_projection.")) or name.endswith(".modulation")
                parameter.data = parameter.data.to(torch.float32 if protected else dtype)
        else:
            model.to(dtype)
        model.requires_grad_(False)
    projection = (
        model.condition_proj
        if family == "minimax_h3"
        else next(module for module in model.modules() if isinstance(module, nn.Linear))
    )
    branch = LoRA(projection.in_features, projection.out_features, rank=2, alpha=1.0)
    attachment = AdapterAttachment(projection, branch)
    # Include all parameters (with master adapters), flags and RNG in the audit.
    before = [(p, p.detach().clone(), p.dtype) for p in (*model.parameters(), *branch.parameters())]
    flags = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    observed = []
    handle = projection.register_forward_hook(lambda module, inputs, output: observed.append(output.dtype))
    try:
        sampler = import_module(f"unituner.models.{family}.inference.sampling").sample_latents
        with (
            preview_context(model, additional_models=(branch,)),
            sampling_precision(precision),
            sdpa_kernel(SDPBackend.MATH),
        ):
            result = sampler(model, *args, **kwargs)
        outputs = (result.video, result.audio) if family == "minimax_h3" else (result,)
        assert all(value.dtype == dtype and torch.isfinite(value).all() for value in outputs)
        assert observed and all(value == dtype for value in observed)
        assert all(p.dtype == expected_dtype and torch.equal(p, value) for p, value, expected_dtype in before)
        assert flags == [module.training for module in model.modules()]
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        handle.remove()
        attachment.close()


def test_sampling_precision_scope_restores_on_failure():
    model = nn.Linear(2, 2).bfloat16()
    assert sampling_dtype(model) == torch.bfloat16
    with sampling_precision("auto"):
        assert sampling_dtype(model) == torch.bfloat16
    with pytest.raises(RuntimeError), sampling_precision("fp16"):
        assert sampling_dtype(model) == torch.float16
        raise RuntimeError("preview failed")
    assert sampling_dtype(model) == torch.bfloat16
    with sampling_context(model, torch.device("cpu")):
        assert torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize(
    "family",
    ["flux", "flux2", "flux3", "qwen_image", "qwen_image_2_1", "krea2", "moss_image", "z_image", "wan", "minimax_h3"],
)
def test_explicit_low_precision_state_with_fp32_model_needs_no_preview_scope(family):
    model, args, kwargs = make_case(family, torch.bfloat16)
    model.eval()
    sampler = import_module(f"unituner.models.{family}.inference.sampling").sample_latents
    with sdpa_kernel(SDPBackend.MATH):
        result = sampler(model, *args, **kwargs)
    outputs = (result.video, result.audio) if family == "minimax_h3" else (result,)
    assert all(value.dtype == torch.bfloat16 and torch.isfinite(value).all() for value in outputs)


@pytest.mark.parametrize(
    "family",
    ["flux", "flux2", "flux3", "qwen_image", "qwen_image_2_1", "krea2", "moss_image", "z_image", "wan", "minimax_h3"],
)
def test_quantized_storage_is_not_used_as_sampling_dtype(family):
    pytest.importorskip("comfy_kitchen")
    from unituner.models.components.weight_quantization import quantize_model

    model, args, kwargs = make_case(family, torch.bfloat16)
    model.eval().requires_grad_(False)
    assert quantize_model(model, "float8_e4m3fn", family=family.replace("_", "-")) > 0
    dtypes = [p.dtype for p in model.parameters()]
    sampler = import_module(f"unituner.models.{family}.inference.sampling").sample_latents
    with sampling_precision("bf16"), sdpa_kernel(SDPBackend.MATH):
        result = sampler(model, *args, **kwargs)
    outputs = (result.video, result.audio) if family == "minimax_h3" else (result,)
    assert all(value.dtype == torch.bfloat16 and torch.isfinite(value).all() for value in outputs)
    assert dtypes == [p.dtype for p in model.parameters()]
