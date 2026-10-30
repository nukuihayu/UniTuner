"""Detect source namespaces across shard headers without weakening validation."""

import pytest
import torch

from unituner.artifacts.tensors import save_tensors
from unituner.models.components.text_loading import language_weight_prefix, qwen_vl_source_format


@pytest.mark.parametrize(
    "prefix,expected",
    [("language_model.", "native"), ("model.language_model.", "transformers"), ("model.", "qwen")],
)
def test_detection_reads_all_shards_and_rejects_mixed_namespaces(tmp_path, prefix, expected):
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    save_tensors({"model.visual.weight": torch.ones(1)}, first)
    save_tensors({prefix + "embed_tokens.weight": torch.ones(1)}, second)
    assert qwen_vl_source_format(first) == expected
    assert qwen_vl_source_format(second) == expected
    save_tensors({"model.visual.weight": torch.ones(1), "norm.weight": torch.ones(1)}, first)
    with pytest.raises(ValueError, match="Ambiguous"):
        language_weight_prefix(second, ("", prefix))


@pytest.mark.parametrize("layout", ["native", "transformers", "qwen"])
@pytest.mark.parametrize("full,head", [(False, False), (True, False), (True, True)])
def test_h3_auto_layout_validates_full_or_truncated_inventory(tmp_path, monkeypatch, layout, full, head):
    from transformers import AutoProcessor, Qwen3VLConfig

    import unituner.models.minimax_h3.text.encoder as encoder_module
    import unituner.models.minimax_h3.text.loading as loading
    from tests.support.qwen_image_21 import write_text_assets
    from tests.support.text_assets import local_text_config
    from unituner.artifacts.tensors import iter_tensors

    write_text_assets(tmp_path, language_layers=3)
    config = Qwen3VLConfig.from_pretrained(tmp_path)
    config.image_token_id, config.video_token_id = 151655, 151656
    config.vision_start_token_id, config.vision_end_token_id = 151652, 151653
    config.save_pretrained(tmp_path)
    for module in (loading, encoder_module):
        monkeypatch.setattr(module, "TEXT_WIDTH", 16)
        monkeypatch.setattr(module, "TEXT_LAYERS", 2)
    processor = AutoProcessor.from_pretrained(tmp_path, local_files_only=True)
    monkeypatch.setattr(loading, "load_text_processor", lambda _: processor)
    values = {}
    for key, value in iter_tensors(tmp_path / "model.safetensors"):
        if key == "lm_head.weight" and not head:
            continue
        if not full and (
            key == "lm_head.weight"
            or key == "model.language_model.norm.weight"
            or key.startswith("model.language_model.layers.2.")
        ):
            continue
        if layout == "native":
            key = key.removeprefix("model.")
        elif layout == "qwen":
            key = key.replace("model.language_model.", "model.")
        values[key] = value
    path = tmp_path / "auto.safetensors"
    save_tensors(values, path)

    def load():
        return loading.load_text_encoder(
            path, config_directory=tmp_path, device=torch.device("cpu"), text_config=local_text_config(tmp_path)
        )

    rng = torch.get_rng_state().clone()
    owner = load()
    assert len(owner.model.language_model.layers) == 2
    assert isinstance(owner.model.language_model.norm, torch.nn.Identity)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in owner.model.state_dict().items():
        source = key if layout == "native" else "model." + key
        if layout == "qwen":
            source = source.replace("model.language_model.", "model.")
        torch.testing.assert_close(value, values[source], atol=0, rtol=0)
    if full:
        # Discarded language layers still undergo full source validation.
        discarded = next(key for key in values if ".layers.2." in key)
        save_tensors({**values, discarded: torch.full_like(values[discarded], float("nan"))}, path)
        with pytest.raises(ValueError, match="MiniMaxH3 text"):
            load()
