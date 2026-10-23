"""Keep optional text providers outside restoration previews and load only the selected family."""

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

from tests.training.test_image_integrations import IMAGE_CASES


def test_importing_shared_preview_does_not_import_text_providers():
    code = (
        "import sys; import unitunner.training.previews.images; "
        "assert 'transformers' not in sys.modules; "
        "assert not any(n.startswith('unitunner.training.integrations.') and n.endswith('.previews') "
        "for n in sys.modules)"
    )
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")}
    subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=True)


@pytest.mark.parametrize("module, stem, options", IMAGE_CASES)
def test_factory_loads_only_selected_native_preview(monkeypatch, module, stem, options):
    from unitunner.training.previews import images

    topology = getattr(import_module(f"unitunner.models.{module}.config.model"), stem + "Config")(**options)
    model = getattr(import_module(f"unitunner.models.{module}.transformer"), stem + "Transformer")(topology)
    seen = []
    config = object()

    def publish(update):
        seen.append(update)

    def load(name):
        seen.append(name)

        def factory(actual_config, actual_model):
            assert actual_config is config and actual_model is model
            return publish

        return SimpleNamespace(ImagePreviews=factory)

    monkeypatch.setattr(images, "import_module", load)
    publisher = images.build_flow_preview(config, model)
    publisher(7)
    assert seen == [f"unitunner.training.integrations.{module}.previews", 7]
    with pytest.raises(TypeError, match="No image preview"):
        images.build_flow_preview(config, nn.Linear(1, 1))
    assert len(seen) == 2
