"""Build synthetic Qwen3-VL language/vision assets with native patch16 geometry."""

import json
from pathlib import Path

from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    Qwen2Tokenizer,
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor


def write_text_assets(
    directory: Path, *, language_layers: int = 2, feature_layers: tuple[int, ...] | None = None
) -> None:
    """Write small language/two-layer vision weights, including a deepstack branch."""
    directory.mkdir(parents=True, exist_ok=True)
    vocabulary = {value: index for index, value in enumerate(sorted(ByteLevel.alphabet()))}
    vocabulary.update(
        {
            token: i + 256
            for i, token in enumerate(
                [
                    "<|endoftext|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|image_pad|>",
                    "<|video_pad|>",
                    "<|vision_start|>",
                    "<|vision_end|>",
                ]
            )
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
    tokenizer.add_special_tokens({"additional_special_tokens": list(vocabulary)[256:]})
    Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2, min_pixels=1024, max_pixels=4096
        ),
        video_processor=Qwen3VLVideoProcessor(),
    ).save_pretrained(directory)
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": language_layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 263,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000, "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=259,
        video_token_id=260,
        vision_start_token_id=261,
        vision_end_token_id=262,
        tie_word_embeddings=False,
    )
    Qwen3VLForConditionalGeneration(config).save_pretrained(directory)

    if feature_layers is not None:
        write_text_features(directory, feature_layers)


def write_text_features(directory: Path, layers: tuple[int, ...]) -> None:
    """Declare explicit synthetic feature geometry alongside the actual encoder topology."""
    path = directory / "config.json"
    values = json.loads(path.read_text())
    values["text_config"]["unituner_features"] = {"select_layers": list(layers)}
    path.write_text(json.dumps(values))
