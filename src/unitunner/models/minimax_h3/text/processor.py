"""Load the local H3-specific Qwen3-VL processor and expand non-chat visual inputs."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

from torch import Tensor
from transformers import Qwen3VLProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.video_utils import VideoMetadata

from unitunner.models.minimax_h3.text.presentation import (
    IMAGE_TOKEN_ID,
    MAX_TEXT_ROWS,
    TEXT_VISUAL_FPS,
    VIDEO_PLACEHOLDER,
    VIDEO_TOKEN_ID,
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
    MiniMaxH3Presentation,
    build_token_tags,
)

_SPECIAL_TOKENS = {
    "<|vision_start|>": VISION_START_TOKEN_ID,
    "<|vision_end|>": VISION_END_TOKEN_ID,
    "<|image_pad|>": IMAGE_TOKEN_ID,
    "<|video_pad|>": VIDEO_TOKEN_ID,
    "<d>": 151669,
    "</d>": 151670,
    "<|cutoff|>": 151671,
    "<|lyrics_start|>": 151672,
    "<|lyrics_end|>": 151673,
    "<|caption_start|>": 151674,
    "<|caption_end|>": 151675,
}


def load_text_processor(directory: Path) -> Qwen3VLProcessor:
    """Load local processor files and reject generic Qwen tokenization.

    The directory contains the released H3 processor files, rather than the
    Diffusers repository root. No download or remote Python execution occurs.
    Every H3 delimiter must encode to its native single token ID.
    """
    processor = Qwen3VLProcessor.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    for token, token_id in _SPECIAL_TOKENS.items():
        if processor.tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"MiniMaxH3 processor requires atomic token {token!r} with ID{token_id}")
    return processor


def process_presentation(processor: Qwen3VLProcessor, presentation: MiniMaxH3Presentation) -> dict[str, Tensor]:
    """Expand one H3 prompt with native2Hz video metadata and retain multimodal types.

    Return CPU processor tensors without chat wrapping, padding or frame sampling.
    Original visual tensors are borrowed and never altered. Ordinary token type
    IDs are omitted as in the released encoder; multimodal token types survive
    for image/video rotary positions. Reject overflow before model execution.
    """
    # Qwen3-VL owns per-temporal-pair boundaries, unlike its image replacement.
    # Keeping an enclosing video boundary creates invalid nested vision spans.
    text = presentation.processor_text
    if presentation.videos:
        text = text.replace(VIDEO_PLACEHOLDER, "<|video_pad|>")
    processed = cast(Callable[..., BatchFeature], processor)(
        text=[text],
        return_tensors="pt",
        padding=False,
        images=[image.detach().cpu().numpy() for image in presentation.images] or None,
        videos=[video.detach().cpu().numpy() for video in presentation.videos] or None,
        video_metadata=[
            VideoMetadata(
                total_num_frames=len(video),
                fps=float(TEXT_VISUAL_FPS),
                duration=len(video) / TEXT_VISUAL_FPS,
                frames_indices=list(range(len(video))),
                height=int(video.shape[1]),
                width=int(video.shape[2]),
            )
            for video in presentation.videos
        ]
        or None,
        do_sample_frames=False,
    )
    tensors: dict[str, Tensor] = {}
    for key, value in processed.items():
        if key == "token_type_ids":
            continue
        if not isinstance(value, Tensor):
            raise ValueError(f"MiniMaxH3 processor field {key!r} must be a tensor")
        tensors[key] = value
    if "input_ids" not in tensors:
        raise ValueError("MiniMaxH3 processor returned no input_ids")
    tags = build_token_tags(tensors["input_ids"])
    if len(tags) > MAX_TEXT_ROWS:
        raise ValueError("MiniMaxH3 processed text exceeds32768 rows")
    return tensors
