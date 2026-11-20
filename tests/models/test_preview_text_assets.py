"""Preview text assets are owned by one training run, across all text families."""

from pathlib import Path

import pytest

from unituner.config.model import ModelConfig
from unituner.config.text import TextEncodingConfig
from unituner.models import text_frontends as assets
from unituner.models.catalog import ModelFamily


@pytest.mark.parametrize("family", list(ModelFamily))
def test_preflight_reuses_every_role_and_releases_on_failure(tmp_path, monkeypatch, family):
    config = TextEncodingConfig()
    model = ModelConfig.model_construct(family=family, text=config, text_encoder=str(tmp_path), assets=str(tmp_path))
    reads = []
    monkeypatch.setattr(assets, "tokenizer_directory", lambda *args, **kwargs: tmp_path)

    def read(path, family, version, role):
        reads.append(role)
        return object()

    monkeypatch.setattr(assets, "_read_frontend", read)
    directory = None if family in ("wan", "minimax-h3", "minimax-h3-image") else Path(model.text_encoder)
    roles = ["tokenizer"] + ({"flux": ["clip_tokenizer"], "qwen-image": ["processor"]}.get(family, []))
    with pytest.raises(RuntimeError, match="render failed"), assets.preview_text_assets(model):
        assert reads == roles
        for role in roles:
            path = model.asset_path("clip_text") if role == "clip_tokenizer" else directory
            first = assets.load_frontend(config, family, path, role=role)
            assert assets.load_frontend(config, family, path, role=role) is first
        assert reads == roles
        raise RuntimeError("render failed")
    assets.load_frontend(config, family, directory)
    assert reads == [*roles, "tokenizer"]
