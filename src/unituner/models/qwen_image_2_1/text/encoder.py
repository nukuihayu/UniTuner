# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Extract local Qwen3-VL pre-normalization text and ordered reference insertion slots."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoProcessor,
    PreTrainedTokenizerBase,
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLProcessor,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionRotaryEmbedding,
)

from unituner.artifacts.tensors import iter_tensors
from unituner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig
from unituner.models.components.text_artifacts import text_config_path
from unituner.models.components.text_loading import (
    load_text_state,
    prepare_text_model,
    qwen_vl_base_key,
    qwen_vl_source_format,
    text_dtype,
    text_weight_path,
)
from unituner.models.qwen_image_2_1.conditioning import QwenImage21TextFeatures
from unituner.models.qwen_image_2_1.data.encoding import ImageInput
from unituner.models.text_assets import tokenizer_directory

_SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"


def _reference_rgb(image: ImageInput) -> Image.Image:
    """Composite RGBA over white for vision, retaining the input pixels untouched."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError("Qwen Image2.1 references require uint8 HWC RGB/RGBA")
        image = Image.fromarray(image)
    if min(image.size) < 32 or image.width % 32 or image.height % 32:
        raise ValueError("Qwen Image2.1 reference dimensions must be positive multiples of32")
    if image.mode == "RGBA":
        white = Image.new("RGB", image.size, "white")
        white.paste(image, mask=image.getchannel("A"))
        return white
    return image.convert("RGB")


class QwenImage21TextEncoder:
    """Own a frozen local Qwen3-VL base model and processor for feature extraction.

    References must already match the VAE image size; vision processing never
    resizes them. RGBA composites on white only for vision, unlike VAE RGBA.
    Encoding excludes system/padding/vision tokens, retains user delimiters and
    captures final language norm inputs. It preserves model modes, RNG and the
    caller's media/groups. No vocabulary head or generation cache is allocated.
    """

    def __init__(self, processor: Qwen3VLProcessor, model: Qwen3VLModel) -> None:
        if not isinstance(processor.tokenizer, PreTrainedTokenizerBase):
            raise TypeError("Qwen Image2.1 text requires a local tokenizer")
        if model.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Qwen Image2.1 text requires FP32/FP16/BF16")
        if not isinstance(model.config, Qwen3VLConfig):
            raise ValueError("Qwen Image2.1 text requires Qwen3-VL config")
        vision = model.config.vision_config
        if not isinstance(vision, Qwen3VLVisionConfig) or vision.patch_size != 16 or vision.spatial_merge_size != 2:
            raise ValueError("Qwen Image2.1 vision requires patch16 and spatial merge2")
        self.processor = processor
        self.model = model

    @torch.no_grad()
    def encode(
        self, prompts: Sequence[str], *, references: Sequence[Sequence[ImageInput]] | None = None
    ) -> list[QwenImage21TextFeatures]:
        """Return per-sample unpadded features and reference slots, without truncation.

        Prompts may be empty strings; an empty batch returns an empty list.
        Every prompt has its own ordered reference group, including empty groups.
        A scoped hook captures pre-RMSNorm hidden states and is removed on error.
        Ambient autocast is disabled; model dtype controls numerical execution.
        """
        groups = [[] for _ in prompts] if references is None else references
        if len(groups) != len(prompts):
            raise ValueError("Qwen Image2.1 reference groups must match the prompt batch")
        if not prompts:
            return []
        texts, images, shapes = [], [], []
        for prompt, group in zip(prompts, groups, strict=True):
            markers = " ".join(f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(len(group)))
            texts.append(
                _SYSTEM_PROMPT
                + "<|im_start|>user\n"
                + markers
                + (prompt or " ")
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            rgb = [_reference_rgb(image) for image in group]
            images.extend(rgb)
            shapes.append(tuple((image.height // 16, image.width // 16) for image in rgb))
        inputs = cast(Callable[..., BatchFeature], self.processor)(
            text=texts,
            images=images or None,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            do_resize=False,
        )
        if images:
            grids = inputs.get("image_grid_thw")
            expected = torch.tensor([shape for group in shapes for shape in group], dtype=torch.int64)
            if (
                not isinstance(grids, Tensor)
                or grids.shape != (len(images), 3)
                or not torch.equal(grids[:, 0], torch.ones(len(images), dtype=grids.dtype))
                or not torch.equal(grids[:, 1:], expected)
            ):
                raise ValueError("Qwen Image2.1 vision and vae reference grids must agree; resizing is forbidden")
        ids, mask = inputs.get("input_ids"), inputs.get("attention_mask")
        if not isinstance(ids, Tensor) or not isinstance(mask, Tensor) or ids.ndim != 2 or ids.shape != mask.shape:
            raise ValueError("Qwen Image2.1 processor must return BL token IDs and validity")
        tokenizer = self.processor.tokenizer
        start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if not isinstance(start_id, int) or not isinstance(image_id, int) or start_id == image_id:
            raise ValueError("Qwen Image2.1 tokenizer requires distinct scalar turn/image token IDs")
        if image_id != self.model.config.image_token_id:
            raise ValueError("Qwen Image2.1 processor/model image token IDs differ")
        device = next(self.model.parameters()).device
        captured: list[Tensor] = []

        def capture(module: nn.Module, arguments: tuple[Tensor, ...]) -> None:
            if len(arguments) != 1:
                raise RuntimeError("Qwen Image2.1 final norm must receive hidden states")
            captured.append(arguments[0])

        handle = self.model.language_model.norm.register_forward_pre_hook(capture)
        try:
            with torch.autocast(device.type, enabled=False):
                self.model(**inputs.to(device), output_hidden_states=False, return_dict=True, use_cache=False)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape[:2] != ids.shape:
            raise RuntimeError("Expected one Qwen Image2.1 pre-normalization language feature tensor")
        results = []
        for row, group in enumerate(groups):
            # Derive small token geometry on the host, without one device sync per token.
            row_ids = ids[row, mask[row].bool()].tolist()
            hidden = captured[0][row, mask[row].to(device).bool()]
            starts = [i for i, token in enumerate(row_ids) if token == start_id]
            if len(starts) < 2:
                raise ValueError("Qwen Image2.1 template requires system and user turns")
            row_ids, hidden = row_ids[starts[1] :], hidden[starts[1] :]
            slots, kept, previous_image = [], [], False
            for i, token in enumerate(row_ids):
                is_image = token == image_id
                if is_image and not previous_image:
                    slots.append(len(kept))
                if not is_image:
                    kept.append(i)
                previous_image = is_image
            if len(slots) != len(group):
                raise ValueError("Qwen Image2.1 vision token spans must match reference images")
            indices = torch.tensor(kept, device=device, dtype=torch.int64)
            results.append(
                QwenImage21TextFeatures(hidden.index_select(0, indices).contiguous(), tuple(slots), shapes[row])
            )
        return results


def load_qwen_image_21_text(
    directory: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    source_format: Literal["auto", "native", "transformers", "qwen"] = "auto",
    text_config: TextEncodingConfig = DEFAULT_TEXT_CONFIG,
) -> QwenImage21TextEncoder:
    """Load local config/processor and exact safetensors into a frozen Qwen3-VL base.

    Native keys are language_model/visual; transformers keys have model prefixes
    and an optional validated lm_head; qwen uses model.* language and either
    visual.* or Comfy's model.visual.* vision keys.
    Auto detects the language namespace from all shard headers. Explicit overrides remain available.
    Conversion rejects unknown/duplicate/missing/shape/nonfinite
    or dtype-overflow values. No weight downloads/remote code/RNG draws occur. Rotary
    buffers are reconstructed in FP32 after meta construction. The returned owner
    holds the materialized model on the requested device, without an LM head.
    """
    dtype = text_dtype(text_config, dtype)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16) or source_format not in (
        "auto",
        "native",
        "transformers",
        "qwen",
    ):
        raise ValueError("Qwen Image2.1 text requires FP32/FP16/BF16 and auto/native/transformers/qwen format")
    config = AutoConfig.from_pretrained(
        str(text_config_path(directory, "qwen-image-2-1")), local_files_only=True, trust_remote_code=False
    )
    if not isinstance(config, Qwen3VLConfig):
        raise ValueError("Qwen Image2.1 text requires local Qwen3-VL config")
    if not isinstance(config.text_config, Qwen3VLTextConfig) or not isinstance(
        config.vision_config, Qwen3VLVisionConfig
    ):
        raise ValueError("Expected typed language/vision subconfigurations")
    processor = AutoProcessor.from_pretrained(
        str(tokenizer_directory(text_config, "qwen-image-2-1", directory)),
        local_files_only=True,
        trust_remote_code=False,
    )
    if not isinstance(processor, Qwen3VLProcessor):
        raise ValueError("Qwen Image2.1 text requires local Qwen3-VL processor")
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3VLModel(config)
    owner = QwenImage21TextEncoder(processor, model)
    expected, tensors = dict(model.state_dict()), {}
    path = text_weight_path(directory)
    automatic = source_format == "auto"
    if automatic:
        source_format = qwen_vl_source_format(path)
    for original, value in iter_tensors(path):
        key = original
        if source_format == "transformers" and original != "lm_head.weight" and not original.startswith("model."):
            raise ValueError(f"Invalid Qwen Image2.1 text weight prefix: {original}")
        if source_format == "transformers" and original.startswith("model."):
            key = original.removeprefix("model.")
        elif source_format == "qwen":
            key = qwen_vl_base_key(original)
        is_head = (automatic or source_format != "native") and original == "lm_head.weight"
        if key in tensors:
            raise ValueError(f"Duplicate Qwen Image2.1 text tensor: {original}")
        tensors[key] = value
        if is_head:
            expected["lm_head.weight"] = torch.empty(
                config.text_config.vocab_size, config.text_config.hidden_size, device="meta"
            )
    load_text_state(model, tensors, dtype=dtype, label="Qwen Image2.1 text weight", expected=expected)
    with torch.device("cpu"):
        model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(config.text_config)
        model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.vision_config)
    prepare_text_model(model, text_config, family="qwen3-vl", device=device)
    return owner
