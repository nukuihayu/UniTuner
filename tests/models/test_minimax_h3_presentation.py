"""Validate native H3 presentation ordering, modality labels and invalid boundaries."""

import pytest
import torch

from unituner.models.minimax_h3.text.presentation import (
    IMAGE_PLACEHOLDER,
    VIDEO_PLACEHOLDER,
    MiniMaxH3TextReference,
    MiniMaxH3TextVisual,
    build_presentation,
    build_token_tags,
    validate_text_rows,
)


def test_native_reference_counters_video_pairs_and_pixel_ownership():
    frames = torch.arange(3 * 32 * 32 * 3).remainder(256).to(torch.uint8).reshape(3, 32, 32, 3)
    before = frames.clone()
    rng = torch.get_rng_state().clone()
    video = MiniMaxH3TextVisual(frames, (0.0, 0.5, 1.0))
    image = MiniMaxH3TextVisual(frames[:1])
    result = build_presentation(
        "caption",
        "ref2va",
        references=(
            MiniMaxH3TextReference("audio"),
            MiniMaxH3TextReference("video", video, has_audio=True),
            MiniMaxH3TextReference("image", image),
            MiniMaxH3TextReference("video", video),
        ),
    )
    timestamps = f"<0.2 seconds>{VIDEO_PLACEHOLDER}<1.0 seconds>{VIDEO_PLACEHOLDER}"
    assert result.text == (
        f"<Audio 1>: <Audio 2>: <Video 1>: {timestamps}<Picture 1>: {IMAGE_PLACEHOLDER}<Video 2>: {timestamps}caption"
    )
    assert result.processor_text == (
        f"<Audio 1>: <Audio 2>: <Video 1>: {VIDEO_PLACEHOLDER}"
        f"<Picture 1>: {IMAGE_PLACEHOLDER}<Video 2>: {VIDEO_PLACEHOLDER}caption"
    )
    assert result.videos[0] is frames and result.videos[1] is frames
    assert result.images[0].data_ptr() == frames.data_ptr()
    assert torch.equal(frames, before) and torch.equal(torch.get_rng_state(), rng)
    assert build_presentation("caption", "t2va").processor_text == "caption"
    last = build_presentation("caption", "fl2va", anchors={"last": image})
    assert last.text == f"<Picture 1>: {IMAGE_PLACEHOLDER}caption"
    ordered = build_presentation("", "fl2va", anchors={"cond_001": image, "cond_000": image})
    assert ordered.text == f"<Picture 1>: {IMAGE_PLACEHOLDER}<Picture 2>: {IMAGE_PLACEHOLDER}"


def test_native_modality_labels_and_feature_geometry():
    ids = torch.tensor([[5, 151652, 151655, 151655, 151653, 6, 151652, 151656, 151653]])
    tags = build_token_tags(ids)
    assert tags.tolist() == [1, 0, 0, 0, 0, 1, 0, 0, 0]
    assert tags.device.type == "cpu" and tags.dtype == torch.int64
    validate_text_rows(torch.zeros(9, 5120), tags)
    with pytest.raises(ValueError, match="binary"):
        validate_text_rows(torch.zeros(9, 5120), tags + 1)
    with pytest.raises(ValueError, match="floating"):
        validate_text_rows(torch.zeros(9, 5120, dtype=torch.int64), tags)
    with pytest.raises(ValueError, match="32768"):
        validate_text_rows(torch.empty(0, 5120), torch.empty(0, dtype=torch.int64))
    for values in ([151653], [151652], [151652, 151652], [151652, 7, 151653]):
        with pytest.raises(ValueError, match="vision"):
            build_token_tags(torch.tensor(values))
    for invalid in (torch.tensor([1.0]), torch.empty(0, dtype=torch.int64), torch.ones(2, 1, dtype=torch.int64)):
        with pytest.raises(ValueError, match="integer"):
            build_token_tags(invalid)


def test_invalid_visuals_and_cross_task_conditions_fail_before_processing():
    frames = torch.zeros(1, 32, 32, 3, dtype=torch.uint8)
    visual = MiniMaxH3TextVisual(frames)
    for invalid in (torch.empty(1, 0, 32, 3), torch.empty(1, 32, 32, 2)):
        with pytest.raises(ValueError, match="THWC"):
            MiniMaxH3TextVisual(invalid)
    for times in ((float("nan"),), (-1.0,), (0.0, 1.0)):
        with pytest.raises(ValueError, match="timestamps"):
            MiniMaxH3TextVisual(frames, times)
    with pytest.raises(ValueError, match="ordered"):
        MiniMaxH3TextVisual(frames.expand(2, -1, -1, -1), (1.0, 0.0))
    for anchors in ({}, {"first": visual, "cond_000": visual}, {"cond_001": visual}):
        with pytest.raises(ValueError):
            build_presentation("", "fl2va", anchors=anchors)
    with pytest.raises(ValueError, match="anchors"):
        build_presentation("", "t2va", anchors={"first": visual})
    with pytest.raises(ValueError, match="references"):
        build_presentation("", "fl2va", references=(MiniMaxH3TextReference("audio"),))
    with pytest.raises(ValueError, match="one frame"):
        MiniMaxH3TextReference("image", MiniMaxH3TextVisual(frames.expand(2, -1, -1, -1)))
    with pytest.raises(ValueError, match="require visual"):
        MiniMaxH3TextReference("video")
