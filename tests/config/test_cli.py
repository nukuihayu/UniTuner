"""Exercise file-free command dispatch, typed values and TOML override compatibility."""

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from unitunner.cli.main import main
from unitunner.config.loading import load_training_config

EXAMPLES = Path(__file__).resolve().parents[2] / "docs/examples"
TRAIN = [
    "--model.family",
    "z-image",
    "--model.variant",
    "base",
    "--model.weights",
    "weights/model.safetensors",
    "--set",
    "dataset.groups.main.kind=image",
    "--set",
    "dataset.groups.main.source=images",
    "--set",
    "dataset.groups.main.cache_directory=cache",
    "--train.adapter.algorithm",
    "lora",
]


@pytest.mark.parametrize("flag", ["--precision", "--execution.precision"])
@pytest.mark.parametrize("precision", ["auto", "bf16", "fp16", "fp32", "nvfp4", "int8_rowwise"])
def test_precision_cli_overrides_toml(tmp_path, monkeypatch, flag, precision):
    path = tmp_path / "train.toml"
    path.write_text((EXAMPLES / "flow.toml").read_text(encoding="utf-8"), encoding="utf-8")
    received = []

    def run(config, **kwargs):
        received.append(config)
        return 0

    monkeypatch.setattr("unitunner.cli.main.import_module", lambda name: SimpleNamespace(run=run))
    assert main(["train", str(path), flag, precision]) == 0
    expected = "auto" if precision in ("nvfp4", "int8_rowwise") else precision
    assert received[0].execution.precision == expected
    assert received[0].execution.quantization == (precision if precision in ("nvfp4", "int8_rowwise") else "none")


def _flags(document, prefix=""):
    result = []
    for key, value in document.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.extend(_flags(value, path))
        elif path.startswith("dataset.groups."):
            result.extend(["--set", f"{path}={json.dumps(value)}"])
        else:
            result.append(f"--{path.replace('_', '-')}={json.dumps(value)}")
    return result


@pytest.mark.parametrize(
    ("command", "example", "module"),
    [
        ("train", "flow", "unitunner.training.run"),
        ("cache", "image-cache", "unitunner.data.build"),
        ("mid-timestep", "mid-timestep", "unitunner.training.calibration"),
        ("generate", "generate", "unitunner.inference.generation"),
        ("restore", "restore", "unitunner.inference.restoration.run"),
        ("generate-video", "wan-generate-video", "unitunner.models.wan.inference.run"),
        ("generate-video", "minimax-h3-generate", "unitunner.models.minimax_h3.inference.run"),
    ],
)
def test_commands_accept_only_cli_configuration(command, example, module, monkeypatch, tmp_path):
    with (EXAMPLES / f"{example}.toml").open("rb") as stream:
        document = tomllib.load(stream)
    received = []

    def run(config, **kwargs):
        received.append((config, kwargs))
        return 7

    def imported(name):
        assert name == module
        return SimpleNamespace(run=run)

    original_open = Path.open

    def unexpected_read(path, mode="r", *args, **kwargs):
        if mode in ("a", "w"):
            return original_open(path, mode, *args, **kwargs)
        pytest.fail("CLI-only configuration must not read a TOML file")

    monkeypatch.setattr("unitunner.cli.main.import_module", imported)
    monkeypatch.setattr("unitunner.cli.main.find_spec", lambda name: object())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "open", unexpected_read)
    assert main([command, *_flags(document), "--device", "cuda"]) == 7
    config, kwargs = received[0]
    assert config.model.family == document["model"]["family"]
    assert kwargs["device"] == "cuda"


def test_validate_cli_only_and_schema_defaults(capsys):
    assert (
        main(["validate", *TRAIN, "--train.max-updates", "2", "--set", "dataset.groups.main.resolution=[32, 64]"]) == 0
    )
    assert json.loads(capsys.readouterr().out) == {"valid": True, "family": "z-image", "method": "flow-matching"}


@pytest.mark.parametrize("file_args", [[str(EXAMPLES / "flow.toml")], ["--config", str(EXAMPLES / "flow.toml")]])
def test_file_and_cli_values_merge_before_validation(file_args, monkeypatch):
    received = []

    def run(config, **kwargs):
        received.append(config)
        return 0

    monkeypatch.setattr("unitunner.cli.main.import_module", lambda name: SimpleNamespace(run=run))
    assert (
        main(
            [
                "train",
                *file_args,
                "--train.learning-rate",
                "0.003",
                "--set",
                "train.max_updates=3",
                "--train.max-updates",
                "4",
                "--set",
                "dataset.groups.main.enable_bucket=false",
                "--parameter-mode",
                "full",
                "--include-layers",
                "layers.*",
            ]
        )
        == 0
    )
    config = received[0]
    assert config.train.learning_rate == 0.003
    assert config.train.max_updates == 4
    assert config.train.seed == 42
    assert config.train.adapter is None
    assert config.train.include_layers == ("layers.*",)
    assert not config.dataset.groups["main"].enable_bucket


def test_set_objects_arrays_and_null(monkeypatch):
    received = []

    def run(config, **kwargs):
        received.append(config)
        return 0

    monkeypatch.setattr("unitunner.cli.main.import_module", lambda name: SimpleNamespace(run=run))
    assert (
        main(
            [
                "train",
                *TRAIN,
                "--set",
                'train={"adapter":{"algorithm":"lora","rank":8,"alpha":8.0}}',
                "--train.include_layers",
                '["layers.0", "layers.1"]',
                "--model.vae",
                "null",
            ]
        )
        == 0
    )
    assert received[0].train.adapter.rank == 8
    assert received[0].train.include_layers == ("layers.0", "layers.1")
    assert received[0].model.vae is None


@pytest.mark.parametrize(
    "options",
    [
        [],
        [*TRAIN, "--train.max-updates", "0"],
        [*TRAIN, "--set", "train.typo=1"],
        [*TRAIN, "--set", 'train.learning_rate="secret-value"'],
        [*TRAIN, "--set", "model=null", "--model.family", "z-image"],
        [str(EXAMPLES / "flow.toml"), "--config", str(EXAMPLES / "full.toml")],
    ],
)
def test_invalid_cli_configuration_is_reported_without_values(options, capsys):
    assert main(["validate", *options]) == 2
    error = capsys.readouterr().err
    assert error
    assert "secret-value" not in error


@pytest.mark.parametrize(
    "options",
    [
        ["--set", "missing-value"],
        ["--set", "model..family=x"],
        ["--set", "dataset.groups.main.resolution=[32,"],
        ["--train.typo", "1"],
    ],
)
def test_invalid_argument_syntax_exits_with_usage(options):
    with pytest.raises(SystemExit) as error:
        main(["validate", *options])
    assert error.value.code == 2


def test_help_lists_nested_cli_fields(capsys):
    with pytest.raises(SystemExit) as error:
        main(["generate-video", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--model.family" in output
    assert "--model.minimax-h3" in output
    assert "--high-noise.boundary" in output


def test_loader_can_validate_overrides_without_path():
    config = load_training_config(
        overrides={
            "model": {"family": "z-image", "variant": "base", "weights": "model.safetensors"},
            "dataset": {"groups": {"main": {"kind": "image", "source": "images", "cache_directory": "cache"}}},
            "train": {"adapter": {}},
        }
    )
    assert config.train.max_updates == 1000
