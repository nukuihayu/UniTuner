"""Strictly load local H3 text artifacts, explicitly selecting native layer50 features."""

from pathlib import Path
from typing import Literal

import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionRotaryEmbedding,
)

from unitunner.artifacts.tensors import iter_tensors
from unitunner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig, TokenizerSourceConfig
from unitunner.models.components.text_loading import load_text_state, prepare_text_model, text_dtype
from unitunner.models.minimax_h3.text.encoder import TEXT_LAYERS, MiniMaxH3TextEncoder
from unitunner.models.minimax_h3.text.presentation import TEXT_WIDTH
from unitunner.models.minimax_h3.text.processor import load_text_processor
from unitunner.models.text_assets import tokenizer_directory


def load_text_encoder(
    weights: Path,
    *,
    config_directory: Path,
    processor_directory: Path | None = None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    source_format: Literal["native", "qwen", "transformers"] = "native",
    text_config: TextEncodingConfig = DEFAULT_TEXT_CONFIG,
) -> MiniMaxH3TextEncoder:
    """Load exact local safetensors into a frozen50-layer pre-norm Qwen3-VL owner.

    Native files contain language_model/visual keys for the truncated model with
    no final norm. Qwen files use model.* language keys and visual.* vision keys
    for that same truncated inventory. Transformers files contain model.* base
    keys plus lm_head.weight for the full config depth; validate the entire
    supplied inventory, including discarded layers/norm/head, before selecting
    layers0..49. Unknown, duplicate, missing, invalid-shape/nonfinite or dtype
    overflow tensors fail; no silent aliases or partial inventories are accepted.

    Construction uses meta storage without RNG draws. Nonpersistent rotary
    buffers are rebuilt in FP32. CPU tensors can retain file-backed storage;
    device/dtype conversion materializes tensors. No weight downloads or remote code,
    vocabulary head or residency hooks are installed. Optional weight quantization is applied before placement.
    """
    dtype = text_dtype(text_config, dtype)
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("MiniMaxH3 text loading requires FP32/BF16/FP16")
    if source_format not in ("native", "qwen", "transformers"):
        raise ValueError("MiniMaxH3 text source_format must be native/qwen/transformers")
    config = Qwen3VLConfig.from_pretrained(str(config_directory), local_files_only=True)
    if not isinstance(config.text_config, Qwen3VLTextConfig) or not isinstance(
        config.vision_config, Qwen3VLVisionConfig
    ):
        raise ValueError("MiniMaxH3 text loading requires Qwen3-VL language and vision configs")
    if config.text_config.hidden_size != TEXT_WIDTH or config.vision_config.out_hidden_size != TEXT_WIDTH:
        raise ValueError("MiniMaxH3 text and visual output widths must be5120")
    if config.text_config.num_hidden_layers < TEXT_LAYERS:
        raise ValueError("MiniMaxH3 text config must provide at least50 language layers")
    if (config.image_token_id, config.video_token_id, config.vision_start_token_id, config.vision_end_token_id) != (
        151655,
        151656,
        151652,
        151653,
    ):
        raise ValueError("MiniMaxH3 text config requires the native vision token IDs")
    if processor_directory is not None and text_config.tokenizer.source is None:
        text_config = text_config.model_copy(
            update={"tokenizer": TokenizerSourceConfig(source=str(processor_directory), offline=True)}
        )
    processor = load_text_processor(tokenizer_directory(text_config, "minimax-h3"))
    config._attn_implementation = "sdpa"
    config.text_config.use_cache = False
    if source_format != "transformers":
        config.text_config.num_hidden_layers = TEXT_LAYERS
    with torch.device("meta"):
        model = Qwen3VLModel(config)
    if source_format != "transformers":
        model.language_model.norm = nn.Identity()
    expected = dict(model.state_dict())
    if source_format == "transformers":
        expected["lm_head.weight"] = torch.empty(config.text_config.vocab_size, TEXT_WIDTH, device="meta", dtype=dtype)
    model.language_model.layers = model.language_model.layers[:TEXT_LAYERS]
    model.language_model.norm = nn.Identity()
    config.text_config.num_hidden_layers = TEXT_LAYERS
    tensors = {}
    for original, value in iter_tensors(weights):
        key = original
        if source_format == "transformers":
            if original == "model.lm_head.weight" or (
                original != "lm_head.weight" and not original.startswith("model.")
            ):
                raise ValueError(f"Invalid MiniMaxH3 transformers text key: {original}")
            key = original.removeprefix("model.")
        elif source_format == "qwen":
            if original.startswith("model."):
                key = "language_model." + original.removeprefix("model.")
            elif not original.startswith("visual."):
                raise ValueError(f"Invalid MiniMaxH3 qwen text key: {original}")
        if key in tensors:
            raise ValueError(f"Duplicate MiniMaxH3 text tensor: {original}")
        tensors[key] = value
    load_text_state(model, tensors, dtype=dtype, label="MiniMaxH3 text", expected=expected)
    with torch.device("cpu"):
        model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(config.text_config)
        model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.vision_config)
    prepare_text_model(model, text_config, family="qwen3-vl", device=device)
    return MiniMaxH3TextEncoder(processor, model)
