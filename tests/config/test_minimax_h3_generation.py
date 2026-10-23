"""Reject invalid H3 generation inputs and dispatch video configs without Torch imports."""

import json
import os
import subprocess
import sys
import tomllib

import pytest
from pydantic import ValidationError

from tests.support.assertions import require_type
from unitunner.cli.main import main
from unitunner.models.minimax_h3.config.inputs import MiniMaxH3InputsConfig, validate_minimax_h3_inputs
from unitunner.models.minimax_h3.config.media import MiniMaxH3VideoReferenceInput
from unitunner.models.minimax_h3.config.sampling import MiniMaxH3SamplingConfig


def test_generation_inputs_task_contract_and_rcm_rejection():
    video = MiniMaxH3SamplingConfig()
    single = MiniMaxH3SamplingConfig(frames=1)
    validate_minimax_h3_inputs(MiniMaxH3InputsConfig(prompt="hello"), video, task="t2va")
    validate_minimax_h3_inputs(MiniMaxH3InputsConfig(prompt="hello", last_frame="image.png"), video, task="fl2va")
    controls = MiniMaxH3InputsConfig.model_validate_json(
        '{"prompt":"hello","controls":[{"path":"a.png","index":9},{"path":"b.png","index":2}],"target_index":7}'
    )
    validate_minimax_h3_inputs(controls, single, task="fl2va")
    for inputs, sampling, task in [
        (controls, video, "fl2va"),
        (controls, single, "t2va"),
        (MiniMaxH3InputsConfig(prompt=" "), video, "t2va"),
        (MiniMaxH3InputsConfig(prompt="a"), video, "fl2va"),
        (MiniMaxH3InputsConfig(prompt="a", first_frame="a"), single, "fl2va"),
        (MiniMaxH3InputsConfig(prompt="a"), video, "ref2va"),
    ]:
        with pytest.raises(ValueError):
            validate_minimax_h3_inputs(inputs, sampling, task=task)
    with pytest.raises(ValidationError):
        MiniMaxH3InputsConfig.model_validate({"prompt": "hello", "rcm_mask": True})


def test_video_configuration_dispatch_stays_compute_free():
    code = (
        "import sys; from unitunner.config.loading import load_video_generation_config; "
        "from unitunner.models.minimax_h3.config.media import MiniMaxH3RecordInput; "
        "assert not {'torch','av','transformers'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True)


def test_toml_reference_audio_disable_and_cached_training_dependency_boundary():
    document = tomllib.loads("""prompt="hello"
references=[{kind="video",path="video.mkv",audio_enabled=false}]
""")
    inputs = MiniMaxH3InputsConfig.model_validate_json(json.dumps(document))
    assert require_type(inputs.references[0], MiniMaxH3VideoReferenceInput).audio_enabled is False
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import unitunner.training.workflows.minimax_h3.run; assert 'transformers' not in sys.modules",
        ],
        env=os.environ.copy(),
        check=True,
    )


def test_h3_command_preflight_rejects_missing_conditions_before_model_loading(tmp_path, capsys):
    config = tmp_path / "bad.toml"
    config.write_text("""output="output.mp4"
[model]
family="minimax-h3"
variant="fl2va"
weights="absent-model.safetensors"
vae="absent-video.safetensors"
text_encoder="absent-text.safetensors"
[model.minimax_h3]
audio_vae="absent-audio.safetensors"
text_config="absent-text-config"
text_processor="absent-processor"
[inputs]
prompt="Hello"
first_frame="absent-condition.png"
""")
    assert main(["generate-video", str(config)]) == 2
    message = capsys.readouterr().err
    assert "condition first" in message and "absent-condition.png" in message
    assert not (tmp_path / "output.mp4").exists()
    image = tmp_path / "source.png"
    image.write_bytes(b"existing source must remain unchanged")
    collision = config.read_text().replace('output="output.mp4"', f'output="{image.as_posix()}"')
    collision = collision.replace(
        'first_frame="absent-condition.png"', f'controls=[{{path="{image.as_posix()}",index=9}}]'
    )
    config.write_text(collision + "\n[sampling]\nframes=1\n")
    assert main(["generate-video", str(config)]) == 2
    assert "output must differ" in capsys.readouterr().err
    assert image.read_bytes() == b"existing source must remain unchanged"
