"""Ensure default installs retain every supported runtime capability."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = {
    "av",
    "comfy-kitchen",
    "ftfy",
    "huggingface-hub",
    "opencv-python-headless",
    "pillow",
    "sentencepiece",
    "torchvision",
    "transformers",
}


def test_default_install_and_lock_include_all_runtime_backends():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = [Requirement(value) for value in project["dependencies"]]
    assert not project.get("optional-dependencies")
    assert RUNTIME <= {value.name for value in requirements if value.marker is None}
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    package = next(value for value in lock["package"] if value["name"] == "unituner")
    assert not package.get("optional-dependencies")
    assert RUNTIME <= {value["name"] for value in package["dependencies"] if "marker" not in value}
    metadata = package["metadata"]
    assert not metadata.get("provides-extras")
    assert RUNTIME <= {value["name"] for value in metadata["requires-dist"] if "marker" not in value}
