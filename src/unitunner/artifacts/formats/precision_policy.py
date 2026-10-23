"""Model-specific quantization boundaries, independent of Torch and optional kernels."""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase

# Shared exclusions follow convert_to_quant 5d2f539, extended by native
# model boundaries below. Match nested conditioning/adapters too: a block
# allowlist alone does not protect auxiliary projections inside that block.
COMMON_PROTECTED = tuple(
    f"*{name}*"
    for name in (
        "norm",
        "bias",
        "embed_tokens",
        "lm_head",
        "shared",
        "patch_embedding",
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
)
MODULATION_PROTECTED = ("*modulation*", "*_mod.*", "*.mod.*", "*adaLN*")


@dataclass(frozen=True)
class PrecisionPolicy:
    """Quantize core projections while retaining shared and family-specific safeguards."""

    blocks: tuple[str, ...]
    protected: tuple[str, ...] = ()


_FLUX_PROTECTED = tuple(
    f"*{name}*"
    for name in (
        "stream_modulation",
        "guidance_in",
        "time_in",
        "final_layer",
        "img_in",
        "txt_in",
    )
)
_QWEN_PROTECTED = tuple(
    f"*{name}*"
    for name in (
        "time_text_embed",
        "img_in",
        "norm_out",
        "proj_out",
        "transformer_blocks.0.img_mod.1",
        "txt_in",
    )
)
POLICIES = {
    "z-image": PrecisionPolicy(
        ("layers.*",),
        tuple(
            f"*{name}*"
            for name in (
                "x_embedder",
                "clip_text_pooled_proj",
                "final_layer",
                "cap_embedder",
                "t_embedder",
                "time_text_embed",
                "context_refiner",
                "noise_refiner",
                "cap_pad_token",
                "x_pad_token",
            )
        ),
    ),
    "flux": PrecisionPolicy(("double_blocks.*", "single_blocks.*"), _FLUX_PROTECTED),
    "flux2": PrecisionPolicy(("double_blocks.*", "single_blocks.*"), _FLUX_PROTECTED),
    "flux3": PrecisionPolicy(("content_mode_blocks.*", "txt_mode_blocks.*", "single_blocks.*")),
    "moss-image": PrecisionPolicy(("layers.*",), ("*caption_refiner*",)),
    "qwen-image": PrecisionPolicy(("transformer_blocks.*",), _QWEN_PROTECTED),
    "qwen-image-2-1": PrecisionPolicy(("transformer_blocks.*",), _QWEN_PROTECTED),
    "krea2": PrecisionPolicy(
        ("blocks.*",),
        tuple(
            f"*{name}*"
            for name in (
                "first.",
                "last.",
                "tmlp.",
                "txtfusion",
                "txtmlp.",
                "tproj.",
            )
        ),
    ),
    "wan": PrecisionPolicy(
        ("blocks.*",),
        tuple(
            f"*{name}*"
            for name in (
                "text_embedding",
                "time_embedding",
                "trainable_cond_mask",
                "time_projection",
                "head.head",
                "face_encoder.out_proj",
                "face_adapter",
                "audio_injector",
            )
        ),
    ),
    "minimax-h3": PrecisionPolicy(
        ("*attn.qkv_proj.*", "*attn.out_proj.*", "*mlp.fc1.*", "*mlp.fc2.*", "*adaln_proj.linear.*"),
        tuple(
            f"*{name}*"
            for name in (
                "audio_patch_proj",
                "condition_proj",
                "final_layer",
                "time_embedder",
                "token_refiner",
                "video_patch_proj",
            )
        ),
    ),
}
# Text-only projections: multimodal towers and auxiliary prediction heads stay dense.
for _family in ("qwen2-5-vl", "qwen3", "qwen3-vl", "qwen3-5", "mistral", "t5", "umt5", "clip"):
    POLICIES[_family] = PrecisionPolicy(
        (
            "*layers.*",
            "model.layers.*",
            "language_model.layers.*",
            "model.language_model.layers.*",
            "encoder.block.*",
            "encoder.blocks.*",
            "blocks.*",
            "text_model.encoder.layers.*",
        ),
        (
            "*norm*",
            "*embed*",
            "*lm_head*",
            "*shared*",
            "*relative_attention_bias*",
            "*visual*",
            "*vision*",
            "*audio*",
            "*mtp*",
            "*projector*",
            "*in_proj_a*",
            "*in_proj_b*",
        ),
    )
TEXT_FAMILIES = frozenset(("qwen2-5-vl", "qwen3", "qwen3-vl", "qwen3-5", "mistral", "t5", "umt5", "clip"))


def language_boundaries(keys: Iterable[str]) -> frozenset[str]:
    """Protect first and last blocks independently for each encoder stack."""
    stacks: dict[str, set[int]] = {}
    for key in keys:
        match = re.match(r"(.*?(?:layers|blocks|block)\.)(\d+)\.", key)
        if match:
            stacks.setdefault(match[1], set()).add(int(match[2]))
    return frozenset(
        f"{prefix}{index}." for prefix, indices in stacks.items() for index in (min(indices), max(indices))
    )


FAMILIES = ("auto", "generic", *POLICIES)
_METADATA = {
    "z_image": "z-image",
    "flux": "flux",
    "flux2": "flux2",
    "flux3": "flux3",
    "moss_image": "moss-image",
    "qwen_image": "qwen-image",
    "qwen_image_2_1": "qwen-image-2-1",
    "krea2": "krea2",
    "wan": "wan",
    "minimax_h3": "minimax-h3",
}


def unprefixed(key: str) -> str:
    """Remove only known outer diffusion prefixes for policy matching, retaining file keys."""
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def resolve_family(family: str, metadata: Mapping[str, str], keys: Iterable[str]) -> str:
    """Prefer authoritative native topology; structural guesses must be unambiguous."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown conversion family: {family}")
    embedded = {value for name, value in _METADATA.items() if f"unitunner.{name}.config" in metadata}
    if "unitunner.wan.t5_config" in metadata:
        embedded.add("umt5")
    if len(embedded) > 1:
        raise ValueError("Conflicting model families in artifact metadata")
    if family != "auto":
        if embedded and family != "generic" and family not in embedded:
            raise ValueError("Conversion family differs from artifact topology metadata")
        return family
    if embedded:
        return next(iter(embedded))
    names = tuple(unprefixed(key) for key in keys)
    signatures = {
        "flux3": ("content_mode_blocks.", "txt_mode_blocks."),
        "flux": ("vector_in.", "time_text_embed.text_embedder."),
        "flux2": ("double_stream_modulation_img.",),
        "z-image": ("noise_refiner.", "context_refiner."),
        "moss-image": ("text_embedder.blocks.",),
        "wan": ("patch_embedding.", "time_projection."),
        "minimax-h3": ("token_refiner.",),
        "krea2": ("txtfusion.projector.", "text_fusion.projector."),
        "qwen-image": ("txt_norm.",),
        "qwen-image-2-1": ("txt_in.text_norm.",),
    }
    detected = [name for name, prefixes in signatures.items() if any(key.startswith(prefixes) for key in names)]
    if len(detected) == 1:
        return detected[0]
    # FLUX vs FLUX.2 and Qwen vs Qwen 2.1 share block names: do not guess.
    if any(
        key.startswith(
            (
                "double_blocks.",
                "single_blocks.",
                "transformer_blocks.",
                "blocks.",
                "layers.",
                "model.layers.",
                "model.language_model.",
                "language_model.",
                "encoder.",
                "text_model.",
            )
        )
        for key in names
    ):
        raise ValueError("Ambiguous model layout: select --family explicitly for model-specific protection")
    return "generic"


def protection_reason(family: str, key: str, boundaries: Iterable[str] = ()) -> str | None:
    """Return a stable reason for preserving a matrix in its source precision."""
    name = unprefixed(key)
    if family in TEXT_FAMILIES and any(name.startswith(prefix) for prefix in boundaries):
        return f"{family}: first/last language block"
    policy = POLICIES.get(family)
    if policy is None:
        return (
            "sensitive normalization/embedding/conditioning"
            if any(fnmatchcase(name, pattern) for pattern in (*COMMON_PROTECTED, *MODULATION_PROTECTED))
            or any(token in name.lower() for token in ("embed", "time_in", "time_text"))
            else None
        )
    if not any(fnmatchcase(name, pattern) for pattern in policy.blocks):
        return f"{family}: outside transformer quantization targets"
    if any(fnmatchcase(name, pattern) for pattern in (*COMMON_PROTECTED, *MODULATION_PROTECTED, *policy.protected)):
        return f"{family}: protected normalization/modulation/conditioning or auxiliary path"
    return None
