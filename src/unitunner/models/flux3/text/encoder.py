# Derived from Black Forest Labs FLUX3 Image Klein, Apache-2.0.
"""Provide optional local Qwen3-VL language features with official FLUX3 layer/padding rules."""

import math
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

from unitunner.artifacts.tensors import iter_tensors
from unitunner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig
from unitunner.models.components.conditioning import Caption
from unitunner.models.components.text_artifacts import text_config_path
from unitunner.models.components.text_loading import load_text_state, prepare_text_model, text_dtype, text_weight_path
from unitunner.models.flux3.conditioning import Flux3Tokens, text_tokens
from unitunner.models.text_assets import tokenizer_directory

_OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


class Flux3TextEncoder:
    """Own a frozen Qwen3-VL language model and its official processor template.

    Encode each prompt independently. Retain padding hidden states through the
    next multiple of 80, capped at 4096, and concatenate eight selected layer
    states along features. Official width is 8*2560=20480. No vision tower is
    needed for text-only conditioning; processor/tokenizer assets remain local.
    """

    def __init__(self, processor: Qwen3VLProcessor, model: Qwen3VLTextModel) -> None:
        if model.config.num_hidden_layers < _OUTPUT_LAYERS[-1]:
            raise ValueError("FLUX3 text requires at least 32 Qwen3-VL language layers")
        if not isinstance(processor.tokenizer, PreTrainedTokenizerBase) or processor.tokenizer.padding_side != "right":
            raise ValueError("FLUX3 text requires a right-padded tokenizer")
        if processor.chat_template is None:
            raise ValueError("FLUX3 text requires a local processor chat template")
        self.processor = processor
        self.model = model

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> tuple[Flux3Tokens, ...]:
        """Return detached LD BF16 features and L4 IDs per prompt, retaining all padding.

        Use the processor's user chat template with a generation prompt. Masked
        position values are one; four copies supply text/temporal/height/width
        coordinates. The output layer order and BF16 feature cast match official
        inference even when the caller explicitly loads FP32 computation weights.
        No RNG or model modes are changed; ambient autocast is disabled.
        """
        if not prompts:
            raise ValueError("FLUX3 text encoding requires a nonempty prompt batch")
        parameter = next(self.model.parameters())
        rows = []
        tokenizer = self.processor.tokenizer
        for prompt in prompts:
            formatted = self.processor.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            if not isinstance(formatted, str):
                raise TypeError("FLUX3 processor chat template must produce text")
            real = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096)["input_ids"]
            if not isinstance(real, Tensor) or real.ndim != 2 or real.shape[0] != 1 or real.shape[1] < 1:
                raise ValueError("FLUX3 tokenizer must produce nonempty 1L tensor IDs")
            length = min(math.ceil(real.shape[1] / 80) * 80, 4096)
            inputs = tokenizer(formatted, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
            ids, mask = inputs["input_ids"], inputs["attention_mask"]
            if (
                not isinstance(ids, Tensor)
                or not isinstance(mask, Tensor)
                or ids.shape != (1, length)
                or mask.shape != ids.shape
            ):
                raise ValueError("FLUX3 tokenizer must produce matching padded 1L IDs/masks")
            mask = mask.to(parameter.device)
            positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
            with torch.autocast(parameter.device.type, enabled=False):
                output = self.model(
                    input_ids=ids.to(parameter.device),
                    attention_mask=mask,
                    position_ids=positions[None].expand(4, -1, -1),
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            states = output.hidden_states
            if states is None or len(states) <= _OUTPUT_LAYERS[-1]:
                raise RuntimeError("FLUX3 Qwen3-VL encoder did not return all required hidden layers")
            stacked = torch.stack([states[index] for index in _OUTPUT_LAYERS], dim=1)
            features = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16)
            rows.append(text_tokens(features.detach()))
        return tuple(rows)

    def encode_prompt(self, prompt: str) -> Caption:
        """Return one CPU 1LD caption with every encoder padding token retained."""
        return Caption(self.encode([prompt])[0].values.unsqueeze(0).cpu())


def load_flux3_text(
    directory: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    text_config: TextEncodingConfig = DEFAULT_TEXT_CONFIG,
) -> Flux3TextEncoder:
    """Load owned local Qwen3-VL language weights and processor without a vision model.

    Accept official multimodal model.language_model.* weights or a text-only
    Qwen3VLTextModel directory with the same processor assets. Only the official
    vision and LM-head namespaces are excluded from a multimodal checkpoint;
    all language keys/shapes/finiteness/conversion ranges must match exactly.
    Support single/standard-sharded safetensors. No weight downloads or remote code are
    allowed. Derived rotary buffers are rebuilt in FP32 without consuming RNG.
    """
    dtype = text_dtype(text_config, dtype)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("FLUX3 text loading supports FP32, FP16 or BF16")
    source = AutoConfig.from_pretrained(
        str(text_config_path(directory, "flux3")), local_files_only=True, trust_remote_code=False
    )
    if isinstance(source, Qwen3VLConfig):
        config, prefix = source.text_config, "model.language_model."
    elif isinstance(source, Qwen3VLTextConfig):
        config, prefix = source, ""
    else:
        raise ValueError("FLUX3 text requires a local Qwen3-VL language configuration")
    if not isinstance(config, Qwen3VLTextConfig):
        raise ValueError("Expected a typed language configuration")
    if config.num_hidden_layers < _OUTPUT_LAYERS[-1]:
        raise ValueError("FLUX3 text requires at least 32 Qwen3-VL language layers")
    processor = AutoProcessor.from_pretrained(
        str(tokenizer_directory(text_config, "flux3", directory)), local_files_only=True, trust_remote_code=False
    )
    if not isinstance(processor, Qwen3VLProcessor) or not isinstance(processor.tokenizer, PreTrainedTokenizerBase):
        raise ValueError("FLUX3 text requires a local Qwen3-VL processor and tokenizer")
    if processor.chat_template is None:
        raise ValueError("FLUX3 text requires a local processor chat template")
    processor.tokenizer.padding_side = "right"
    config._attn_implementation = "sdpa"
    path = text_weight_path(directory)
    with torch.device("meta"):
        model = Qwen3VLTextModel(config)
    tensors: dict[str, Tensor] = {}
    for original, tensor in iter_tensors(path):
        if prefix and original.startswith(("model.visual.", "lm_head.")):
            continue
        if not original.startswith(prefix):
            raise ValueError(f"Unexpected FLUX3 text weight namespace: {original}")
        key = original[len(prefix) :]
        if key in tensors:
            raise ValueError(f"Duplicate FLUX3 text tensor: {original}")
        tensors[key] = tensor
    load_text_state(model, tensors, dtype=dtype, label="FLUX3 text tensor")
    with torch.device("cpu"):
        model.rotary_emb = Qwen3VLTextRotaryEmbedding(config)
    prepare_text_model(model, text_config, family="qwen3-vl", device=device)
    return Flux3TextEncoder(processor, model)
