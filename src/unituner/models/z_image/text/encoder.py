"""Provide optional official Qwen3 text computation with Z-Image prompt semantics.

This module is the Transformers dependency boundary. Ordinary package/config
imports do not import it. Tokenizer/model files must be supplied locally.
"""

from pathlib import Path

import torch
from torch import Tensor
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

from unituner.artifacts.tensors import iter_tensors
from unituner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig
from unituner.models.components.conditioning import Caption
from unituner.models.components.text_artifacts import text_config_path
from unituner.models.components.text_loading import (
    language_weight_prefix,
    load_text_state,
    prepare_text_model,
    text_dtype,
    text_weight_path,
)
from unituner.models.text_assets import tokenizer_directory


class ZImageTextEncoder:
    """Own an official Qwen3 encoder and tokenizer with explicit inference lifetime.

    Right padding, max length 512, thinking-enabled user chat template and the
    penultimate hidden layer preserve the native Z-Image text contract. Weight storage may be quantized; no global autocast or training/preview resource policy occurs.
    """

    def __init__(self, tokenizer: Qwen2Tokenizer, model: Qwen3Model, *, max_length: int = 512) -> None:
        if max_length < 1 or tokenizer.padding_side != "right":
            raise ValueError("Z-Image text requires positive max_length and right padding")
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> Caption:
        """Encode nonempty prompt batches into detached BLD features and boolean BL masks.

        Empty strings are legal negative prompts. Truncation uses the supplied
        local tokenizer's pinned chat template. Returned tensors remain on the
        encoder device and in its dtype; the caller controls cache placement.
        """
        if not prompts:
            raise ValueError("Text encoding requires a nonempty prompt batch")
        formatted: list[str] = []
        for prompt in prompts:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            if not isinstance(text, str):
                raise TypeError("Qwen3 chat template must produce text")
            formatted.append(text)
        encoded = self.tokenizer(
            formatted, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
        if not isinstance(input_ids, Tensor) or not isinstance(attention_mask, Tensor):
            raise TypeError("Qwen3 tokenizer must return tensor IDs and attention masks")
        parameter = next(self.model.parameters())
        mask = attention_mask.to(device=parameter.device, dtype=torch.bool)
        with torch.autocast(parameter.device.type, enabled=False):
            outputs = self.model(
                input_ids=input_ids.to(parameter.device),
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        states = outputs.hidden_states
        if states is None or len(states) < 2:
            raise RuntimeError("Z-Image text requires a penultimate Qwen3 hidden state")
        features = states[-2]
        if not isinstance(features, Tensor) or features.shape[:2] != mask.shape:
            raise TypeError("Qwen3 hidden states must have the tokenized batch/length")
        return Caption(features.detach(), mask)

    def encode_prompt(self, prompt: str) -> Caption:
        """Return one trimmed CPU caption; the transformer owns learned inner padding.

        Trimming happens once at the text boundary, never per denoising step.
        A missing negative prompt stays distinct from an encoded empty string.
        """
        caption = self.encode([prompt])
        if caption.mask is None or not caption.mask.any():
            raise ValueError("Text encoder produced no valid prompt tokens")
        return Caption(caption.features[0, caption.mask[0]].unsqueeze(0).cpu())


def load_z_image_text(
    directory: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_length: int = 512,
    text_config: TextEncodingConfig = DEFAULT_TEXT_CONFIG,
) -> ZImageTextEncoder:
    """Load local Qwen3 weights and an independently resolved tokenizer without remote code.

    Accept a local model directory, single safetensors file or numbered shard.
    Bare weights use the released Qwen3-4B topology; local JSON overrides it.
    Artifact/tokenizer contents must be hashed by cache callers.
    """
    dtype = text_dtype(text_config, dtype)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Z-Image text loading supports FP32, FP16 and BF16")
    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(tokenizer_directory(text_config, "z-image", directory)), local_files_only=True
    )
    config = Qwen3Config.from_pretrained(str(text_config_path(directory, "z-image")), local_files_only=True)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3Model(config)
    path = text_weight_path(directory)
    prefix = language_weight_prefix(path, ("", "model."))
    tensors = dict(iter_tensors(path))
    expected = dict(model.state_dict())
    if prefix:
        if any(key != "lm_head.weight" and not key.startswith("model.") for key in tensors):
            raise ValueError("Z-Image causal text artifact requires uniform model prefixes")
        tensors = {key.removeprefix("model."): value for key, value in tensors.items()}
        if "lm_head.weight" in tensors:
            expected["lm_head.weight"] = torch.empty(config.vocab_size, config.hidden_size, device="meta")
    load_text_state(model, tensors, dtype=dtype, label="Z-Image text", expected=expected)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    with torch.device("cpu"):
        model.rotary_emb = Qwen3RotaryEmbedding(config)
    if not isinstance(tokenizer, Qwen2Tokenizer) or not isinstance(model, Qwen3Model):
        raise TypeError("Local text artifacts must contain Qwen2 tokenizer and Qwen3 model")
    prepare_text_model(model, text_config, family="qwen3", device=device)
    return ZImageTextEncoder(tokenizer, model, max_length=max_length)
