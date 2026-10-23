"""Validate H3 sampling timelines and reject unsupported masks before compute imports."""

import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from unitunner.models.minimax_h3.config.sampling import MiniMaxH3SamplingConfig


def test_native_and_stretched_timeline_duration_validation():
    assert MiniMaxH3SamplingConfig().frames == 124
    assert MiniMaxH3SamplingConfig(frames=1).fps == 24
    assert MiniMaxH3SamplingConfig(frames=73, fps=12, temporal_fine_bands=3).fps == 12
    for values in [
        {"width": 31},
        {"frames": 7},
        {"frames": 5},
        {"frames": 1, "fps": 12},
        {"temporal_fine_bands": 1},
        {"video_shift": float("nan")},
        {"steps": True},
        {"audio_condition_clean": 1.01},
        {"fps": 25},
        {"rcm_mask": True},
    ]:
        with pytest.raises(ValidationError):
            MiniMaxH3SamplingConfig(**values)


def test_sampling_schema_does_not_import_torch():
    script = (
        "import sys; from unitunner.models.minimax_h3.config.sampling import MiniMaxH3SamplingConfig; "
        "MiniMaxH3SamplingConfig(); assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], env=os.environ.copy(), check=True)
