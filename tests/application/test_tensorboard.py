"""Exercise public restoration training with real TensorBoard component events."""

import json

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tests.application.test_image_flow import image_assets as image_assets
from tests.support.text_assets import main_with_local_tokenizers


@pytest.mark.parametrize("accumulation", [1, 4])
def test_restoration_component_events_and_resume(image_assets, capsys, accumulation):
    training, _ = image_assets
    root = training.parent
    low = root / "low"
    low.mkdir()
    for name in ("a.png", "b.png"):
        (low / name).write_bytes((root / "images" / name).read_bytes())
    content = training.read_text().replace(
        "source = ", f'kind = "restoration"\nlow_quality_source = "{low.as_posix()}"\nsource = ', 1
    )
    content = content.replace("[output]", "[output]\nsave_every_updates = 1")
    content = content.replace("accumulation_steps = 2", f"accumulation_steps = {accumulation}")
    content += """
[method]
kind = "restoration"
[method.restoration]
encoder_rank = 2
[losses]
lrr = 1.0
l1 = 0.5
histogram = 0.0
dv3d = 0.0
gan = 0.0
fdl = 0.0
"""
    training.write_text(content)
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    logs = root / "events"
    args = ["train", str(training), "--output.tensorboard", logs.as_posix()]
    capsys.readouterr()
    assert main_with_local_tokenizers(args) == 0
    for update in (1, 2):
        assert (root / "run/weights" / f"model-{update:08d}.safetensors").is_file()
        assert (root / "run/weights" / f"model-{update:08d}.encoder.safetensors").is_file()
        assert (root / "run/resume" / f"model.state-{update:08d}.safetensors").is_file()
    continuous_logs = [
        entry for line in capsys.readouterr().out.splitlines() if "update" in (entry := json.loads(line))
    ]
    assert (
        main_with_local_tokenizers([*args, "--resume", (root / "run/resume/model.state-00000001.safetensors").as_posix()]) == 0
    )
    resumed_logs = [entry for line in capsys.readouterr().out.splitlines() if "update" in (entry := json.loads(line))]
    runs = sorted(logs.iterdir())
    assert [run.name for run in runs] == ["model"]
    events = EventAccumulator(str(runs[0])).Reload()
    assert [event.step for event in events.Scalars("loss/generator")] == [1, 2]
    assert [entry["update"] for entry in continuous_logs] == [1, 2]
    assert [entry["update"] for entry in resumed_logs] == [2]
    for entry in continuous_logs + resumed_logs:
        assert entry["loss"] == entry["metrics"]["loss/generator"]
        assert entry["metrics"]["train/update_microbatches"] == accumulation
        assert entry["metrics"]["train/update_samples"] == 2 * accumulation
        index = entry["update"] - 1
        for tag, value in entry["metrics"].items():
            if tag.startswith("performance/"):
                continue
            assert events.Scalars(tag)[index].value == pytest.approx(value)
    expected_steps = [1, 2]
    for name in ("l1", "lrr", "dv3d", "fdl", "gan", "histogram"):
        assert [event.step for event in events.Scalars(f"loss/generator/{name}")] == expected_steps
    for total, l1, lrr in zip(
        events.Scalars("loss/generator"),
        events.Scalars("loss/generator/l1"),
        events.Scalars("loss/generator/lrr"),
        strict=True,
    ):
        assert abs(total.value - (0.5 * l1.value + lrr.value)) < 1e-5
