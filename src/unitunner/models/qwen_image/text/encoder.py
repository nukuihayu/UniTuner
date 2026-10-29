# Copyright 2025 The Qwen-Image Team and The HuggingFace Team.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Extract optional local Qwen2.5-VL features with native prompt/reference semantics."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoProcessor,
    GenerationConfig,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLVisionRotaryEmbedding,
)

from unitunner.artifacts.tensors import iter_tensors
from unitunner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig
from unitunner.models.components.conditioning import Caption
from unitunner.models.components.text_artifacts import text_config_path
from unitunner.models.components.text_loading import (
    load_text_state,
    prepare_text_model,
    qwen_vl_base_key,
    qwen_vl_source_format,
    text_dtype,
    text_weight_path,
)
from unitunner.models.qwen_image.config.model import QwenImageConfig, QwenImageVariant
from unitunner.models.qwen_image.text.templates import CHINESE_CAPTION_TEMPLATE, ENGLISH_CAPTION_TEMPLATE
from unitunner.models.text_assets import tokenizer_directory

type QwenImageReference = Image.Image | NDArray[np.uint8] | Tensor

_TEXT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
_EDIT_TEMPLATE = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
_IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"


def _rgb_reference(image: QwenImageReference) -> QwenImageReference:
    """Read RGB from PIL, uint8 HWC arrays or CPU CHW tensors without mutation."""
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or min(image.shape) < 1 or image.shape[-1] not in (3, 4) or image.dtype != np.uint8:
            raise ValueError("Qwen references require nonempty uint8 HWC RGB/RGBA arrays")
        return image[..., :3]
    if (
        image.ndim != 3
        or min(image.shape) < 1
        or image.shape[0] not in (3, 4)
        or image.device.type != "cpu"
        or image.dtype not in (torch.uint8, torch.float32)
    ):
        raise ValueError("Qwen reference tensors require nonempty CPU CHW RGB/RGBA uint8 or FP32")
    if image.dtype == torch.float32 and (
        not torch.isfinite(image).all() or torch.any(image < 0) or torch.any(image > 1)
    ):
        raise ValueError("Qwen FP32 reference tensors require finite [0,1] pixels")
    # The native processor rescales byte-range pixels by 1/255. Preserve that
    # path for mixed image types while accepting explicitly unit-range tensors.
    return image[:3] * 255 if image.dtype == torch.float32 else image[:3]


def _caption(hidden: Tensor, mask: Tensor, *, drop: int) -> Caption:
    """Gather real tokens before dropping native prefixes, then right-pad with zeros."""
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("Qwen text requires BLD hidden states and BL masks")
    rows = [row[valid.bool()][drop:] for row, valid in zip(hidden, mask, strict=True)]
    if not rows or any(row.shape[0] == 0 for row in rows):
        raise ValueError("Qwen prompts must retain tokens after the native template prefix")
    maximum = max(row.shape[0] for row in rows)
    features = torch.stack([F.pad(row, (0, 0, 0, maximum - row.shape[0])) for row in rows])
    lengths = torch.tensor([row.shape[0] for row in rows], device=hidden.device)
    validity = torch.arange(maximum, device=hidden.device)[None] < lengths[:, None]
    return Caption(features.detach(), validity)


class QwenImageTextEncoder:
    """Own a local frozen Qwen2.5-VL model and processor for one Qwen Image variant.

    Original/layered text uses the native 34-token prefix drop and1058-token
    cap. Edit variants use64-token drop without text truncation; original edit
    uses only each sample's first reference, while2509/2511 preserve all ordered
    references, including empty groups. No mode/RNG/residency changes occur in
    encode. The LM head remains available for captioning but is bypassed when
    extracting features, avoiding unused vocabulary logits.
    """

    def __init__(
        self,
        processor: Qwen2VLProcessor,
        model: Qwen2_5_VLForConditionalGeneration,
        *,
        variant: QwenImageVariant,
    ) -> None:
        # Use the same strict variant schema as the transformer at construction.
        QwenImageConfig(variant=variant)
        if not isinstance(processor.tokenizer, PreTrainedTokenizerBase):
            raise TypeError("Qwen Image text requires a local tokenizer")
        if model.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Qwen Image text supports FP32/FP16/BF16 computation")
        self.processor = processor
        self.model = model
        self.variant = variant

    @torch.no_grad()
    def encode(
        self, prompts: list[str], *, references: Sequence[Sequence[QwenImageReference]] | None = None
    ) -> Caption:
        """Return detached native BLD features and boolean BL masks in model precision.

        Reference groups correspond one-to-one with prompts. RGBA inputs lose
        alpha without modifying caller arrays/lists. Empty strings are valid
        negative prompts. Padding is removed before native prefix trimming;
        output is right-padded only to this batch's longest remaining context.
        Ambient autocast is disabled; use_cache=False avoids KV allocation.
        """
        if not prompts:
            raise ValueError("Qwen text requires a nonempty prompt batch")
        parameter = next(self.model.parameters())
        if self.variant in ("original", "layered"):
            if references is not None:
                raise ValueError("Qwen original/layered text does not consume edit references")
            inputs = self.processor.tokenizer(
                [_TEXT_TEMPLATE.format(prompt) for prompt in prompts],
                max_length=1058,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            drop = 34
        else:
            groups = [[] for _ in prompts] if references is None else references
            if len(groups) != len(prompts):
                raise ValueError("Qwen reference groups must match the prompt batch")
            if self.variant == "edit" and any(not group for group in groups):
                raise ValueError("Original Qwen edit requires a reference for every prompt")
            images, texts = [], []
            for prompt, group in zip(prompts, groups, strict=True):
                selected = group[:1] if self.variant == "edit" else group
                prefix = (
                    _IMAGE_MARKER
                    if self.variant == "edit"
                    else "".join(f"Picture {index + 1}: {_IMAGE_MARKER}" for index in range(len(selected)))
                )
                images.extend(_rgb_reference(image) for image in selected)
                texts.append(_EDIT_TEMPLATE.format(prefix + prompt))
            inputs = cast(Callable[..., BatchFeature], self.processor)(
                text=texts, images=images or None, padding=True, return_tensors="pt"
            )
            drop = 64
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        if not isinstance(ids, Tensor) or not isinstance(mask, Tensor) or ids.ndim != 2 or ids.shape != mask.shape:
            raise TypeError("Qwen processor must return matching BL tensor IDs and masks")
        pixels, grids = inputs.get("pixel_values"), inputs.get("image_grid_thw")
        if pixels is not None and (not isinstance(pixels, Tensor) or not isinstance(grids, Tensor)):
            raise TypeError("Qwen image processor must return tensor pixels and THW grids")
        with torch.autocast(parameter.device.type, enabled=False):
            output = self.model.model(
                input_ids=ids.to(parameter.device),
                attention_mask=mask.to(parameter.device),
                pixel_values=None if pixels is None else pixels.to(device=parameter.device, dtype=parameter.dtype),
                image_grid_thw=None if grids is None else grids.to(parameter.device),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        if output.hidden_states is None:
            raise RuntimeError("Qwen2.5-VL did not return requested hidden states")
        return _caption(output.hidden_states[-1], mask.to(parameter.device), drop=drop)

    def encode_prompt(self, prompt: str, *, references: Sequence[QwenImageReference] | None = None) -> Caption:
        """Return one unpadded CPU1LD caption with optional ordered edit references."""
        caption = self.encode([prompt], references=None if references is None else [references])
        if caption.mask is None:
            raise RuntimeError("Qwen text encoder omitted its validity mask")
        return Caption(caption.features[0, caption.mask[0]].unsqueeze(0).cpu())

    @torch.no_grad()
    def describe_image(self, image: QwenImageReference, *, language: Literal["en", "zh"] = "en", seed: int = 42) -> str:
        """Generate a stripped native image caption with at most 512 new tokens.

        Use the verbatim English/Chinese annotation template and the local model's
        generation configuration, including greedy/sampling and stop choices.
        Seed CPU and the model's CUDA RNG inside an RNG fork, then restore both
        even on failure. Other CUDA RNG states, modes, weights, image storage and
        residency remain untouched. Require frozen eval weights; disable ambient
        autocast and use the loaded model precision. No remote assets or code.
        """
        if language not in ("en", "zh"):
            raise ValueError("Qwen caption language must be en or zh")
        if any(module.training for module in self.model.modules()) or any(
            p.requires_grad for p in self.model.parameters()
        ):
            raise ValueError("Qwen image captioning requires frozen eval weights")
        parameter = next(self.model.parameters())
        device = parameter.device
        inputs = cast(Callable[..., BatchFeature], self.processor)(
            text=ENGLISH_CAPTION_TEMPLATE if language == "en" else CHINESE_CAPTION_TEMPLATE,
            images=[_rgb_reference(image)],
            padding=True,
            return_tensors="pt",
        )
        ids, mask, pixels, grids = (
            inputs[name] for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        )
        if not all(isinstance(value, Tensor) for value in (ids, mask, pixels, grids)):
            raise TypeError("Qwen image captioning requires tensor IDs, masks, pixels and grids")
        if ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape:
            raise ValueError("Qwen image captioning requires one BL prompt and matching mask")
        token_types = inputs.get("mm_token_type_ids")
        if token_types is not None and (
            not isinstance(token_types, Tensor) or token_types.shape != ids.shape or token_types.dtype != ids.dtype
        ):
            raise ValueError("Qwen image captioning requires matching integer multimodal token types")
        devices = (device.index or 0,) if device.type == "cuda" else ()
        with torch.random.fork_rng(devices=list(devices)), torch.autocast(device.type, enabled=False):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            if device.type == "cuda":
                torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(seed).get_state(), device)
            # Transformers generate has inconsistent bound-method/protocol annotations.
            generated = cast(Callable[..., Tensor], self.model.generate)(  # pyright: ignore[reportAttributeAccessIssue] -- Transformers generation protocol omits concrete model attributes.
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                pixel_values=pixels.to(device=device, dtype=parameter.dtype),
                image_grid_thw=grids.to(device),
                # Generation uses these types to build image-aware rotary positions.
                mm_token_type_ids=token_types.to(device) if token_types is not None else None,
                max_new_tokens=512,
                return_dict_in_generate=False,
            )
        if (
            not isinstance(generated, Tensor)
            or generated.ndim != 2
            or generated.shape[0] != 1
            or generated.shape[1] < ids.shape[1]
        ):
            raise TypeError("Qwen image captioning must return one complete token sequence")
        return self.processor.batch_decode(
            generated[:, ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def load_qwen_image_text(
    directory: Path,
    *,
    device: torch.device,
    variant: QwenImageVariant,
    dtype: torch.dtype = torch.bfloat16,
    source_format: Literal["auto", "transformers", "qwen"] = "auto",
    text_config: TextEncodingConfig = DEFAULT_TEXT_CONFIG,
) -> QwenImageTextEncoder:
    """Load exact local processor/config/safetensors without downloads or remote code.

    Auto detects base/conditional/legacy/Comfy namespaces from all shard headers.
    Transformers keys use model.language_model/model.visual/lm_head. Explicit
    qwen format converts published model.* language and visual.* vision keys;
    duplicate destinations, unknown/missing keys, shape/nonfinite/overflow
    mismatches are rejected. Meta construction and FP32 rotary reconstruction
    consume no RNG. Execution policy does not enter the weight namespace.
    """
    dtype = text_dtype(text_config, dtype)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16) or source_format not in (
        "auto",
        "transformers",
        "qwen",
    ):
        raise ValueError("Qwen Image text requires FP32/FP16/BF16 and auto/transformers/qwen format")
    QwenImageConfig(variant=variant)
    config = AutoConfig.from_pretrained(
        str(text_config_path(directory, "qwen-image")), local_files_only=True, trust_remote_code=False
    )
    if not isinstance(config, Qwen2_5_VLConfig):
        raise ValueError("Qwen Image text requires a local Qwen2.5-VL configuration")
    if not isinstance(config.text_config, Qwen2_5_VLTextConfig) or not isinstance(
        config.vision_config, Qwen2_5_VLVisionConfig
    ):
        raise ValueError("Expected typed language/vision subconfigurations")
    processor = AutoProcessor.from_pretrained(
        str(tokenizer_directory(text_config, "qwen-image", directory, role="processor")),
        local_files_only=True,
        trust_remote_code=False,
    )
    if not isinstance(processor, Qwen2VLProcessor):
        raise ValueError("Qwen Image text requires a local Qwen2-VL processor")
    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(tokenizer_directory(text_config, "qwen-image", directory)),
        local_files_only=True,
    )
    processor = Qwen2VLProcessor(
        image_processor=processor.image_processor,
        video_processor=processor.video_processor,
        tokenizer=tokenizer,
        chat_template=processor.chat_template,
    )
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen2_5_VLForConditionalGeneration(config)
    if ((directory.parent if directory.is_file() else directory) / "generation_config.json").is_file():
        model.generation_config = GenerationConfig.from_pretrained(
            str(directory.parent if directory.is_file() else directory), local_files_only=True
        )
    tensors = {}
    path = text_weight_path(directory)
    automatic = source_format == "auto"
    detected = qwen_vl_source_format(path) if automatic else source_format
    native_base = detected == "native"
    if automatic:
        source_format = "qwen" if detected == "qwen" else "transformers"
    for original, value in iter_tensors(path):
        key = "model." + original if native_base and original.startswith(("language_model.", "visual.")) else original
        if source_format == "qwen":
            if original.startswith(("visual.", "model.")):
                key = "model." + qwen_vl_base_key(original)
        if key in tensors:
            raise ValueError(f"Duplicate Qwen text tensor: {original}")
        tensors[key] = value
    if config.tie_word_embeddings:
        embedding = tensors.get("model.language_model.embed_tokens.weight")
        head = tensors.get("lm_head.weight")
        if embedding is not None and head is None:
            tensors["lm_head.weight"] = embedding
        elif embedding is not None and head is not None and not torch.equal(embedding, head):
            raise ValueError("Qwen tied language head differs from token embeddings")
    load_text_state(model, tensors, dtype=dtype, label="Qwen text weight")
    if config.tie_word_embeddings:
        model.tie_weights()
    with torch.device("cpu"):
        model.model.language_model.rotary_emb = Qwen2_5_VLRotaryEmbedding(config.text_config)  # pyright: ignore[reportArgumentType] -- Upstream annotates the outer config instead of its language subconfig.
        model.model.visual.rotary_pos_emb = Qwen2_5_VLVisionRotaryEmbedding(config.vision_config)
    prepare_text_model(model, text_config, family="qwen2-5-vl", device=device)
    return QwenImageTextEncoder(processor, model, variant=variant)
