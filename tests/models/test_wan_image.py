"""Exercise strict Wan CLIP artifact ownership, frame order and penultimate tokens."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from unitunner.models.wan.config.clip import WanClipConfig
from unitunner.models.wan.vision.encoder import WanImageEncoder, preprocess_clip_videos
from unitunner.models.wan.vision.loading import load_wan_clip, save_wan_clip
from unitunner.models.wan.vision.model import WanClipVision


@pytest.fixture
def assets(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    model = WanClipVision(
        WanClipConfig(image_size=28, patch_size=7, hidden_size=24, output_dim=16, num_heads=4, num_layers=3)
    )
    save_wan_clip(model, tmp_path / "visual.safetensors")
    yield tmp_path, model
    torch.set_num_threads(previous)


def test_visual_artifact_frozen_owner_and_penultimate_inactive_parameters(assets):
    root, model = assets
    rng = torch.get_rng_state().clone()
    loaded = load_wan_clip(root / "visual.safetensors")
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in loaded.state_dict().items())
    owner = WanImageEncoder(loaded)
    rows = (torch.linspace(-1, 1, 3 * 2 * 13 * 19).reshape(3, 2, 13, 19), torch.zeros(3, 1, 17, 11))
    originals = tuple(row.clone() for row in rows)
    identities = tuple(map(id, loaded.parameters()))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = owner.encode(rows)
    separate = torch.cat([owner.encode((row,)) for row in rows])
    assert actual.shape == (3, 17, 24) and actual.dtype == torch.float32
    torch.testing.assert_close(actual, separate, atol=1e-6, rtol=1e-5)
    assert all(torch.equal(row, original) for row, original in zip(rows, originals, strict=True))
    assert identities == tuple(map(id, loaded.parameters())) and torch.equal(rng, torch.get_rng_state())
    pixels = preprocess_clip_videos(rows, image_size=28).requires_grad_(True)
    model(pixels).square().mean().backward()
    for name, value in model.named_parameters():
        inactive = name.startswith(("transformer.2.", "post_norm.")) or name == "head"
        assert (value.grad is None) == inactive, name
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all()


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "nonfinite"])
def test_visual_artifact_rejects_invalid_inventory(assets, damage):
    root, model = assets
    tensors = load_file(root / "visual.safetensors")
    name = next(iter(tensors))
    if damage == "missing":
        tensors.pop(name)
    elif damage == "extra":
        tensors["textual.unexpected"] = torch.zeros(1)
    elif damage == "shape":
        tensors[name] = tensors[name].flatten()
    else:
        tensors[name] = torch.full_like(tensors[name], float("nan"))
    save_file(tensors, root / "broken.safetensors")
    with pytest.raises(ValueError, match=r"missing tensors|unexpected tensor|invalid tensor"):
        load_wan_clip(root / "broken.safetensors", config=model.config, source_format="visual")


def test_image_boundary_rejects_invalid_range_precision_and_training_state(assets):
    _, model = assets
    row = torch.zeros(3, 1, 17, 11)
    with pytest.raises(ValueError, match="frozen eval"):
        WanImageEncoder(model).encode((row,))
    with pytest.raises(ValueError, match="nonempty"):
        preprocess_clip_videos(())
    for invalid in (row + 2, row.double(), row[:, :0], torch.full_like(row, float("nan"))):
        with pytest.raises(ValueError, match="finite FP32 CTHW"):
            preprocess_clip_videos((invalid,))
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast disabled"):
        model(torch.zeros(1, 3, 28, 28))
