"""Own self-contained preview fixtures independently of external comparison artifacts."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def _minimax_h3_render_assets(tmp_path_factory):
    from tests.support.minimax_h3 import write_render_assets

    return write_render_assets(tmp_path_factory.mktemp("h3-render") / "assets")


@pytest.fixture
def minimax_h3_preview_assets(_minimax_h3_render_assets, monkeypatch):
    from tests.support.minimax_h3 import FixedTextEncoder
    from unitunner.models.minimax_h3.inference import loading

    def load_fixed_text(weights, **kwargs):
        assert weights == Path(_minimax_h3_render_assets["text_weights"])
        return FixedTextEncoder()

    monkeypatch.setattr(loading, "load_text_encoder", load_fixed_text)
    return _minimax_h3_render_assets
