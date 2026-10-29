# Derived from Krea's Apache-2.0 encoder and pinned Musubi; see THIRD_PARTY_NOTICES.md.
"""Extract compact selected-layer Krea2 captions from a strictly loaded local language model."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase, Qwen3VLConfig, Qwen3VLModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding

from unitunner.artifacts.tensors import iter_tensors
from unitunner.config.text import TextEncodingConfig
from unitunner.models.components.text_artifacts import text_config_path
from unitunner.models.components.text_loading import (
    load_text_state,
    prepare_text_model,
    qwen_vl_base_key,
    text_dtype,
    text_weight_path,
)
from unitunner.models.krea2.config.features import Krea2TextFeatures
from unitunner.models.text_assets import tokenizer_directory

_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_PREFIX_TOKENS = 34
_SUFFIX_TOKENS = 5
_DEFAULT_CONFIG = TextEncodingConfig()


class Krea2TextEncoder:
    """Own a frozen language-only Qwen3-VL and local tokenizer for native Krea2 text.

    Selected indices follow the original Transformers4.57 hidden-state tuple:
    decoder inputs, including embeddings and the final pre-normalization state.
    Preserve native suffix-after-pad
    ordering, derive monotonic positions from validity and compact only at output.
    No vision model, vocabulary head, generation cache or mutable RoPE deltas are
    retained. The caller owns device residency for this encoder.
    """

    def __init__(self, model: Qwen3VLTextModel, tokenizer: PreTrainedTokenizerBase, config: Krea2TextFeatures) -> None:
        select_layers = config.select_layers
        if max(select_layers) > model.config.num_hidden_layers:
            raise ValueError("Krea2 selected hidden-state index exceeds language model depth")
        if model.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Krea2 text requires FP32/FP16/BF16")
        if tokenizer.pad_token_id is None:
            raise ValueError("Krea2 tokenizer requires a padding token")
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.select_layers = select_layers
        self.max_length = config.max_length

    @torch.no_grad()
    def encode(self, prompts: Sequence[str]) -> tuple[Tensor, ...]:
        """Return detached compact SLD rows in selected-layer order on model device.

        Apply native fixed prefix34/suffix5 budgeting, append suffix after padded
        prompt tokens and explicitly continue rotary positions across the gap.
        Ambient autocast is disabled; stored model dtype controls arithmetic.
        Empty input returns an empty tuple. RNG, tokenizer policy and model modes
        stay unchanged; truncation follows the local tokenizer's declared side.
        """
        if not prompts:
            return ()
        suffix = self.tokenizer([_SUFFIX] * len(prompts), return_tensors="pt")
        encoded = self.tokenizer(
            [_PREFIX + prompt for prompt in prompts],
            truncation=True,
            padding="max_length",
            max_length=self.max_length + _PREFIX_TOKENS - _SUFFIX_TOKENS,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_ids, prompt_mask = encoded.get("input_ids"), encoded.get("attention_mask")
        suffix_ids, suffix_mask = suffix.get("input_ids"), suffix.get("attention_mask")
        if (
            not isinstance(prompt_ids, Tensor)
            or not isinstance(prompt_mask, Tensor)
            or not isinstance(suffix_ids, Tensor)
            or not isinstance(suffix_mask, Tensor)
        ):
            raise TypeError("Krea2 token tensors are required")
        if any(value.ndim != 2 for value in (prompt_ids, prompt_mask, suffix_ids, suffix_mask)):
            raise ValueError("Krea2 tokenizer must return BL token IDs and validity")
        if prompt_ids.shape != prompt_mask.shape or suffix_ids.shape != suffix_mask.shape:
            raise ValueError("Krea2 token IDs and validity shapes must agree")
        ids = torch.cat((prompt_ids, suffix_ids), dim=1)
        mask = torch.cat((prompt_mask.bool(), suffix_mask.bool()), dim=1)
        positions = (mask.long().cumsum(-1) - 1).clamp(min=0)
        device = next(self.model.parameters()).device
        captured: dict[int, Tensor] = {}

        def capture(index: int) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
            def observe(module: nn.Module, arguments: tuple[Tensor, ...]) -> None:
                if len(arguments) != 1:
                    raise RuntimeError("Krea2 selected language boundary requires one hidden tensor")
                captured[index] = arguments[0]

            return observe

        handles = []
        try:
            for index in set(self.select_layers):
                module = self.model.norm if index == len(self.model.layers) else self.model.layers[index]
                handles.append(module.register_forward_pre_hook(capture(index)))
            with torch.autocast(device.type, enabled=False):
                self.model(
                    input_ids=ids.to(device),
                    attention_mask=mask.to(device),
                    position_ids=positions.to(device),
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in reversed(handles):
                handle.remove()
        if captured.keys() != set(self.select_layers):
            raise RuntimeError("Krea2 encoder did not return selected hidden states")
        features = torch.stack([captured[index] for index in self.select_layers], dim=2)
        features, validity = features[:, _PREFIX_TOKENS:], mask[:, _PREFIX_TOKENS:]
        if any(not row.any() for row in validity):
            raise ValueError("Krea2 tokenization produced an empty caption")
        return tuple(features[index, row.to(device)].contiguous() for index, row in enumerate(validity))


def load_krea2_text(
    directory: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    config: TextEncodingConfig = _DEFAULT_CONFIG,
    source_format: Literal["native", "transformers", "qwen"] = "transformers",
) -> Krea2TextEncoder:
    """Strictly validate complete local Qwen3-VL weights, materializing language only.

    Native full-base keys are language_model/visual; transformers adds model. and
    may include lm_head.weight; qwen uses model. for language and visual. or
    model.visual. for vision. Reject duplicate/missing/extra/invalid tensors, including unused vision
    and head tensors. Meta construction preserves RNG. Rebuild rotary buffers in
    FP32, freeze/evaluate the returned model and perform no network requests.
    """
    dtype = text_dtype(config, dtype)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16) or source_format not in (
        "native",
        "transformers",
        "qwen",
    ):
        raise ValueError("Krea2 text requires FP32/FP16/BF16 and native/transformers/qwen format")
    topology = AutoConfig.from_pretrained(
        str(text_config_path(directory, "krea2")), local_files_only=True, trust_remote_code=False
    )
    if not isinstance(topology, Qwen3VLConfig):
        raise ValueError("Krea2 text requires local Qwen3-VL config")
    if not isinstance(topology.text_config, Qwen3VLTextConfig):
        raise ValueError("Expected typed language/vision subconfigurations")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_directory(config, "krea2", directory)), local_files_only=True, trust_remote_code=False
    )
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("Krea2 text requires a local tokenizer")
    topology._attn_implementation = "sdpa"
    with torch.device("meta"):
        base = Qwen3VLModel(topology)
    features = Krea2TextFeatures.model_validate_json(
        json.dumps(topology.text_config.to_dict().get("unitunner_features", {}))
    )
    if config.max_length is not None:
        features = Krea2TextFeatures(max_length=config.max_length, select_layers=features.select_layers)
    owner = Krea2TextEncoder(base.language_model, tokenizer, features)
    expected, tensors = dict(base.state_dict()), {}
    path = text_weight_path(directory)
    for original, value in iter_tensors(path):
        key = original
        if source_format == "transformers" and original != "lm_head.weight":
            if not original.startswith("model."):
                raise ValueError(f"Invalid Krea2 text weight prefix: {original}")
            key = original.removeprefix("model.")
        elif source_format == "qwen":
            key = qwen_vl_base_key(original)
        is_head = source_format != "native" and original == "lm_head.weight"
        if key in tensors:
            raise ValueError(f"Duplicate Krea2 text tensor: {original}")
        tensors[key] = value
        if is_head:
            expected["lm_head.weight"] = torch.empty(
                topology.text_config.vocab_size, topology.text_config.hidden_size, device="meta"
            )
    # Validate the complete source while materializing only the selected language model.
    load_text_state(
        owner.model,
        tensors,
        dtype=dtype,
        label="Krea2 text weight",
        expected=expected,
        selected_prefix="language_model.",
    )
    with torch.device("cpu"):
        owner.model.rotary_emb = Qwen3VLTextRotaryEmbedding(topology.text_config)
    prepare_text_model(owner.model, config, family="qwen3-vl", device=device)
    return owner
