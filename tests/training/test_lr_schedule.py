"""Verify native update-index learning rates against independently tabulated values."""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from unituner.config.training import TrainConfig
from unituner.training.optimizers.schedules import learning_rate


def test_schedule_math_stays_compute_free():
    code = "import sys; import unituner.training.optimizers.schedules; assert 'torch' not in sys.modules"
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")}
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "schedule,expected",
    [
        ("constant", [0, 0.5, 1, 1, 1, 1, 1]),
        ("linear", [0, 0.5, 1, 0.75, 0.5, 0.25, 0]),
        ("cosine", [0, 0.5, 1, (2 + math.sqrt(2)) / 4, 0.5, (2 - math.sqrt(2)) / 4, 0]),
    ],
)
def test_warmup_and_decay_at_actual_update_boundaries(schedule, expected):
    config = TrainConfig(learning_rate=1.0, max_updates=6, warmup_updates=2, lr_schedule=schedule)
    assert [learning_rate(config, index) for index in range(7)] == pytest.approx(expected, abs=1e-15)
