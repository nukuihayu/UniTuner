"""Exercise native H3 synchronization, actual timed AV encoding and atomic failure publication."""

from dataclasses import replace
from fractions import Fraction

import av
import numpy as np
import pytest
import torch
from av.error import FFmpegError
from PIL import Image

from tests.support.assertions import require_value
from unitunner.models.minimax_h3.data.media.output import (
    MiniMaxH3DecodedImage,
    save_minimax_h3_av,
    save_minimax_h3_media,
    synchronize_decoded_av,
)


def test_png_publication_preserves_pixels_and_existing_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "sample.png"
    pixels = torch.arange(4 * 6 * 3, dtype=torch.uint8).reshape(4, 6, 3)
    save_minimax_h3_media(MiniMaxH3DecodedImage(pixels), path)
    with Image.open(path) as image:
        assert torch.equal(torch.from_numpy(np.array(image)), pixels)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="uint8"):
        save_minimax_h3_media(MiniMaxH3DecodedImage(pixels.float()), path)

    def fail(*args, **kwargs):
        raise OSError("injected PNG failure")

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(OSError, match="injected"):
        save_minimax_h3_media(MiniMaxH3DecodedImage(pixels), path)
    assert path.read_bytes() == before and set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("suffix", [".mp4", ".mkv"])
def test_real_stereo_av_mux_timing_and_failed_encoder_preserves_destination(tmp_path, suffix):
    video = torch.linspace(-1.2, 1.2, 3 * 22 * 32 * 32).reshape(1, 3, 22, 32, 32)
    audio = torch.sin(torch.arange(32000, dtype=torch.float32) / 17).expand(1, 2, -1).clone()
    original_video, original_audio = video.clone(), audio.clone()
    rng = torch.get_rng_state().clone()
    decoded = synchronize_decoded_av(video, audio, frame_count=22)
    assert decoded.video.shape == (22, 32, 32, 3) and decoded.audio.shape == (2, 29333)
    path = tmp_path / ("output" + suffix)
    save_minimax_h3_av(decoded, path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == 24
        frames = list(container.decode(video=0))
        assert len(frames) == 22
        times = [require_value(frame.pts) * frame.time_base for frame in frames]
        assert all(
            abs((time - times[0]) - Fraction(index, 24)) <= Fraction(1, 1000) for index, time in enumerate(times)
        )
    with av.open(str(path)) as container:
        frames = list(container.decode(audio=0))
        assert frames and all(frame.sample_rate == 32000 and frame.layout.name == "stereo" for frame in frames)
        assert sum(frame.samples for frame in frames) >= 29333
    assert (
        torch.equal(video, original_video)
        and torch.equal(audio, original_audio)
        and torch.equal(rng, torch.get_rng_state())
    )
    before = path.read_bytes()
    with pytest.raises((ValueError, FFmpegError)):
        save_minimax_h3_av(replace(decoded, sample_rate=12345), path)
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path}
    decoded.audio.zero_()
    assert audio.count_nonzero()


def test_synchronization_short_audio_and_nonfinite_boundary():
    video = torch.zeros(1, 3, 23, 4, 6)
    audio = torch.full((1, 2, 20000), 2.0)
    result = synchronize_decoded_av(video, audio, frame_count=22)
    assert result.video.shape == (15, 4, 6, 3) and result.audio.shape == (2, 20000)
    assert torch.equal(result.audio, torch.ones_like(result.audio))
    with pytest.raises(ValueError, match="finite"):
        synchronize_decoded_av(video, audio * float("nan"), frame_count=22)
