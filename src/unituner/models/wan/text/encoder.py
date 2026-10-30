# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Own optional local Wan tokenization and unpadded UMT5 prompt features."""

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ftfy
import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from unituner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig, TokenizerSourceConfig
from unituner.models.text_assets import tokenizer_directory
from unituner.models.wan.config.t5 import WanT5Config
from unituner.models.wan.text.loading import load_wan_t5
from unituner.models.wan.text.model import WanT5Encoder

_WHITESPACE = re.compile(r"\s+")
_CPU = torch.device("cpu")


def clean_text(prompt: str) -> str:
    """Repair text, unescape HTML twice, collapse whitespace and strip; preserve case."""
    return _WHITESPACE.sub(" ", html.unescape(html.unescape(ftfy.fix_text(prompt))).strip()).strip()


@dataclass(frozen=True, slots=True)
class WanTextTokens:
    """Own CPU int64 BL IDs/mask and validated host prefix lengths."""

    ids: Tensor
    attention_mask: Tensor
    lengths: tuple[int, ...]


class WanTextEncoder:
    """Own frozen native UMT5 and a local tokenizer, with no hidden residency changes."""

    def __init__(self, model: WanT5Encoder, tokenizer: PreTrainedTokenizerBase, *, max_length: int = 512) -> None:
        if type(max_length) is not int or not 1 <= max_length <= 512:
            raise ValueError("Wan text max_length must be an integer in[1,512]")
        if tokenizer.padding_side != "right":
            raise ValueError("Wan text tokenizer requires right padding for valid prefix extraction")
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, prompts: list[str]) -> WanTextTokens:
        """Return validated host tokens using native whitespace cleaning and EOS policy.

        Empty strings remain legal. Reject non-prefix validity or out-of-range
        IDs before GPU transfer, including tokenizers with added unknown assets.
        """
        if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("Wan text requires a nonempty batch of strings")
        if self.tokenizer.padding_side != "right":
            raise ValueError("Wan text tokenizer requires right padding for valid prefix extraction")
        encoded = self.tokenizer(
            [clean_text(prompt) for prompt in prompts],
            return_tensors="pt",
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        if (
            not isinstance(ids, Tensor)
            or not isinstance(mask, Tensor)
            or ids.device.type != "cpu"
            or mask.device.type != "cpu"
            or ids.dtype != torch.int64
            or mask.dtype != torch.int64
            or ids.shape != (len(prompts), self.max_length)
            or mask.shape != ids.shape
        ):
            raise ValueError("Wan tokenizer must return fixed-length CPU int64 BL IDs and masks")
        lengths = tuple(mask.sum(dim=1).tolist())
        visibility = torch.arange(self.max_length)[None] < torch.tensor(lengths)[:, None]
        if (
            any(length < 1 for length in lengths)
            or not torch.equal(mask, visibility.long())
            or ids.min() < 0
            or ids.max() >= self.model.config.vocab_size
        ):
            raise ValueError("Wan tokenizer requires nonempty valid prefixes and IDs within encoder vocabulary")
        return WanTextTokens(ids, mask, lengths)

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> tuple[Tensor, ...]:
        """Return finite CPU SD final-normalized features, cropped by host valid lengths.

        Require a frozen eval encoder, disable ambient autocast, and preserve
        RNG, parameter identities, modes and residency. Padding remains inside
        the encoder's native attention formula until after the final norm.
        """
        if any(module.training for module in self.model.modules()) or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise ValueError("Wan text encoding requires a frozen eval encoder")
        tokens = self.tokenize(prompts)
        device = self.model.token_embedding.weight.device
        with torch.autocast(device.type, enabled=False):
            features = self.model(tokens.ids.to(device), tokens.attention_mask.to(device))
        if features.shape != (*tokens.ids.shape, self.model.config.hidden_size) or not torch.isfinite(features).all():
            raise ValueError("Wan text encoder must return finite matching BLD features")
        return tuple(row[:length].detach().cpu() for row, length in zip(features, tokens.lengths, strict=True))


def load_wan_text(
    weights: Path,
    tokenizer_path: Path | None = None,
    *,
    config: WanT5Config | None = None,
    prefix: Literal["", "encoder."] = "",
    max_length: int = 512,
    device: torch.device = _CPU,
    text_config: TextEncodingConfig = DEFAULT_TEXT_CONFIG,
) -> WanTextEncoder:
    """Load frozen native weights and a separately resolved tokenizer without RNG draws."""
    if type(max_length) is not int or not 1 <= max_length <= 512:
        raise ValueError("Wan text max_length must be an integer in[1,512]")
    if tokenizer_path is not None and text_config.tokenizer.source is None:
        text_config = text_config.model_copy(
            update={"tokenizer": TokenizerSourceConfig(source=str(tokenizer_path), offline=True)}
        )
    tokenizer_path = tokenizer_directory(text_config, "wan")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("Wan requires a Hugging Face text tokenizer")
    if tokenizer.padding_side != "right":
        raise ValueError("Wan text tokenizer requires right padding for valid prefix extraction")
    model = load_wan_t5(weights, config=config, prefix=prefix, device=device, text_config=text_config)
    return WanTextEncoder(model, tokenizer, max_length=max_length)
