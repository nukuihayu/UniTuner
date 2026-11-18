"""Verify real H3 preview ownership, frozen snapshots and exception restoration."""

import json
import random
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from unituner.config.loading import load_training_config
from unituner.models.minimax_h3.inference import loading as minimax_h3_loading
from unituner.models.minimax_h3.loading import load_minimax_h3
from unituner.models.minimax_h3.vae import MiniMaxH3VideoVAE
from unituner.training.integrations.minimax_h3 import previews as minimax_h3_previews
from unituner.training.integrations.minimax_h3.previews import AudioVideoPreviews

PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("task", ["t2va", "ref2va"])
def test_real_preview_snapshots_modes_rng_gradients_and_failed_decode(
    tmp_path, monkeypatch, task, minimax_h3_preview_assets
):
    torch.set_num_threads(1)
    specs = minimax_h3_preview_assets
    inputs = {"prompt": "Hello"}
    if task == "ref2va":
        inputs = json.loads(json.dumps(specs["cases"]["single-ref2va"]["inputs"]))
        for reference in inputs["references"]:
            if reference["kind"] == "video":
                reference["audio_enabled"] = False
    config = load_training_config(
        PROJECT / "docs/examples/minimax-h3-train.toml",
        overrides={
            "model": {
                "variant": task,
                "weights": specs["model_weights"],
                "vae": specs["video_weights"],
                "text_encoder": specs["text_weights"],
                "minimax_h3": {
                    "audio_vae": specs["audio_weights"],
                    "text_config": specs["text_assets"],
                    "text_processor": specs["processor"],
                    "text_device": "cpu",
                },
            },
            "output": {"directory": str(tmp_path)},
            "preview": {
                "audio_video": [
                    {
                        "inputs": inputs,
                        "sampling": {"frames": 1, "width": 32, "height": 32, "steps": 2},
                    }
                ],
            },
        },
    )
    model = load_minimax_h3(Path(specs["model_weights"]), device=torch.device("cpu"))
    model.train()
    model.blocks[0].eval()
    branches = torch.nn.Linear(2, 2)
    branches.bias.grad = torch.full_like(branches.bias, 0.3)
    model_parameter = next(model.parameters())
    model_parameter.grad = torch.full_like(model_parameter, 0.2)
    before_weights = {key: value.clone() for key, value in model.state_dict().items()}
    before_grad = model_parameter.grad.clone()
    modes = [module.training for root in (model, branches) for module in root.modules()]
    owner = AudioVideoPreviews(config)
    loads, texts = [], []
    original_load = minimax_h3_previews.load_minimax_h3_render_resources
    original_text = minimax_h3_loading.load_text_encoder
    original_decode = MiniMaxH3VideoVAE.decode

    def load_text(*args, **kwargs):
        text = original_text(*args, **kwargs)
        texts.append(weakref.ref(text))
        return text

    def load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        assert texts and texts[-1]() is None
        assert result.audio is None
        loads.append(result)
        return result

    def fail_decode(*args, **kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError("injected preview decode failure")

    monkeypatch.setattr(minimax_h3_loading, "load_text_encoder", load_text)
    monkeypatch.setattr(minimax_h3_previews, "load_minimax_h3_render_resources", load)
    monkeypatch.setattr(MiniMaxH3VideoVAE, "decode", fail_decode)
    destination = tmp_path / "previews/step-00000000-00.png"
    destination.parent.mkdir()
    destination.write_bytes(b"existing preview")
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with sdpa_kernel(SDPBackend.MATH):
        with pytest.raises(RuntimeError, match="injected"):
            owner(0, model, branches)
        assert destination.read_bytes() == b"existing preview"
        monkeypatch.setattr(MiniMaxH3VideoVAE, "decode", original_decode)
        owner(0, model, branches)
        owner(1, model, branches)
    assert len(loads) == len(texts) == 1
    assert destination.read_bytes() == (tmp_path / "previews/step-00000001-00.png").read_bytes()
    assert random.getstate() == python_rng
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_rng[0] and np.array_equal(actual_numpy[1], numpy_rng[1])
    assert actual_numpy[2:] == numpy_rng[2:]
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert modes == [module.training for root in (model, branches) for module in root.modules()]
    assert torch.equal(model_parameter.grad, before_grad)
    assert torch.equal(branches.bias.grad, torch.full_like(branches.bias, 0.3))
    assert all(torch.equal(before_weights[key], value) for key, value in model.state_dict().items())
    assert all(value.device.type == "cpu" for value in loads[0].video.parameters())
    resources = weakref.ref(loads[0].video)
    loads.clear()
    owner.close()
    owner.close()
    assert resources() is None
    with pytest.raises(RuntimeError, match="closed"):
        owner(2, model, branches)
