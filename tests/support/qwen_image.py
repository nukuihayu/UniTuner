"""Build small local Qwen2.5-VL processor/model assets for native application tests."""

from pathlib import Path

from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
)
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor


def write_text_assets(directory: Path) -> None:
    """Write synthetic one-layer language/vision weights and a byte tokenizer locally."""
    directory.mkdir(parents=True, exist_ok=True)
    vocabulary = {value: index for index, value in enumerate(sorted(ByteLevel.alphabet()))}
    vocabulary.update(
        {
            "<|endoftext|>": 256,
            "<|im_start|>": 257,
            "<|im_end|>": 258,
            "<|image_pad|>": 259,
            "<|video_pad|>": 260,
            "<|vision_start|>": 261,
            "<|vision_end|>": 262,
        }
    )
    tokenizer = Qwen2Tokenizer(
        vocab=vocabulary,
        merges=[],
        pad_token="<|endoftext|>",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        vision_start_token="<|vision_start|>",
        vision_end_token="<|vision_end|>",
    )
    Qwen2VLProcessor(
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(),
        image_processor=Qwen2VLImageProcessor(
            patch_size=2, temporal_patch_size=2, merge_size=2, min_pixels=16, max_pixels=256
        ),
    ).save_pretrained(directory)
    config = Qwen2_5_VLConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 263,
            "bos_token_id": 256,
            "eos_token_id": 258,
            "pad_token_id": 256,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000, "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "window_size": 16,
            "fullatt_block_indexes": [0],
        },
        image_token_id=259,
        video_token_id=260,
        vision_start_token_id=261,
        vision_end_token_id=262,
        tie_word_embeddings=False,
    )
    Qwen2_5_VLForConditionalGeneration(config).save_pretrained(directory)
