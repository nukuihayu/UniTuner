"""Provide optional local Qwen3.5 language-only final-layer conditioning for MOSS."""

from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding

from unitunner.artifacts.tensors import iter_tensors
from unitunner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig
from unitunner.models.components.conditioning import Caption, batch_captions
from unitunner.models.components.text_artifacts import text_config_path
from unitunner.models.components.text_loading import load_text_state, prepare_text_model, text_dtype, text_weight_path
from unitunner.models.text_assets import tokenizer_directory


class MossImageTextEncoder:
    """Own a frozen language model/tokenizer; encode prompts separately without padding.

    Qwen3.5 recurrent/gated-delta layers must not ingest artificial batch padding.
    Apply the local thinking-enabled user template and return the final hidden
    layer, not Z-Image's penultimate layer. The caller owns residency and RNG.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, model: Qwen3_5TextModel, *, max_length: int = 4096) -> None:
        if type(max_length) is not int or not 1 <= max_length <= 4096:
            raise ValueError("MOSS text max_length must be between 1 and 4096")
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> Caption:
        """Return detached BLD final states and optional prefix mask on the model device.

        Forward each prompt without padding and with cache disabled, then pad only
        the returned feature tensors. Empty strings are valid negative prompts;
        empty batches are rejected. No network access or mode mutation occurs.
        """
        if not prompts:
            raise ValueError("MOSS text encoding requires a nonempty prompt batch")
        parameter = next(self.model.parameters())
        rows: list[Tensor] = []
        for prompt in prompts:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            if not isinstance(formatted, str):
                raise TypeError("MOSS chat template must produce text")
            encoded = self.tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=self.max_length, padding=False
            )
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
            if (
                not isinstance(ids, Tensor)
                or not isinstance(mask, Tensor)
                or ids.ndim != 2
                or ids.shape[0] != 1
                or ids.shape[1] < 1
            ):
                raise ValueError("MOSS tokenizer must produce nonempty 1L tensor IDs and masks")
            with torch.autocast(parameter.device.type, enabled=False):
                output = self.model(
                    input_ids=ids.to(parameter.device),
                    attention_mask=mask.to(parameter.device),
                    use_cache=False,
                    return_dict=True,
                )
            features = output.last_hidden_state
            if not isinstance(features, Tensor) or features.shape[:2] != ids.shape:
                raise TypeError("MOSS text features must match tokenized batch/length")
            rows.append(features[0].detach())
        return batch_captions(rows)

    def encode_prompt(self, prompt: str) -> Caption:
        """Return one unpadded CPU 1LD final-layer caption without encoder residency."""
        caption = self.encode([prompt])
        return Caption(caption.features.cpu())


def load_moss_image_text(
    directory: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_length: int = 4096,
    text_config: TextEncodingConfig = DEFAULT_TEXT_CONFIG,
) -> MossImageTextEncoder:
    """Load owned local Qwen3.5 text weights and tokenizer without a vision tower.

    Accept a text-only Qwen3.5TextModel directory or an official multimodal Qwen3.5
    directory whose language keys use model.language_model.*. Only declared vision,
    MTP and LM-head namespaces are excluded for that multimodal layout. Preflight
    all language keys, shapes and finite conversion values before strict assignment.
    Support model.safetensors or standard numbered shards; resolve tokenizer assets/chat templates independently.
    No weight downloads, remote code or weight fallback.
    """
    dtype = text_dtype(text_config, dtype)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("MOSS text loading supports FP32, FP16 and BF16")
    if type(max_length) is not int or not 1 <= max_length <= 4096:
        raise ValueError("MOSS text max_length must be between 1 and 4096")
    source = AutoConfig.from_pretrained(
        str(text_config_path(directory, "moss-image")), local_files_only=True, trust_remote_code=False
    )
    if isinstance(source, Qwen3_5Config):
        config, prefix = source.text_config, "model.language_model."
    elif isinstance(source, Qwen3_5TextConfig):
        config, prefix = source, ""
    else:
        raise ValueError("MOSS text requires a local Qwen3.5 language configuration")
    if not isinstance(config, Qwen3_5TextConfig):
        raise ValueError("Expected a typed language configuration")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_directory(text_config, "moss-image", directory)), local_files_only=True, trust_remote_code=False
    )
    if not isinstance(tokenizer, PreTrainedTokenizerBase) or tokenizer.chat_template is None:
        raise ValueError("MOSS text requires a local tokenizer with a chat template")
    config._attn_implementation = "sdpa"
    path = text_weight_path(directory)
    with torch.device("meta"):
        model = Qwen3_5TextModel(config)
    tensors: dict[str, Tensor] = {}
    for original, tensor in iter_tensors(path):
        if prefix and original.startswith(("model.visual.", "mtp.", "model.mtp.", "lm_head.")):
            continue
        if not original.startswith(prefix):
            raise ValueError(f"Unexpected MOSS text weight namespace: {original}")
        key = original[len(prefix) :]
        if key in tensors:
            raise ValueError(f"Duplicate MOSS text tensor: {original}")
        tensors[key] = tensor
    load_text_state(model, tensors, dtype=dtype, label="MOSS text")
    with torch.device("cpu"):
        model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    prepare_text_model(model, text_config, family="qwen3-5", device=device)
    return MossImageTextEncoder(tokenizer, model, max_length=max_length)
