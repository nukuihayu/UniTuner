"""Check shared tensor reading across file formats and component boundaries."""

import pytest
import torch

from unitunner.artifacts.checkpoint_reading import checkpoint_tensors, is_pytorch_checkpoint
from unitunner.artifacts.tensors import save_tensors
from unitunner.models.components.vae_loading import vae_tensors


@pytest.mark.parametrize("extension", [".pt", ".pth", ".bin", ".safetensors", ".renamed"])
@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_formats_and_prefixes(tmp_path, extension, legacy):
    path = tmp_path / f"weights{extension}"
    state = {"vae.weight": torch.randn(2, 3)}
    if extension == ".safetensors":
        save_tensors(state, path)
    else:
        torch.save(state, path, _use_new_zipfile_serialization=not legacy)
    actual = dict(checkpoint_tensors(path))
    torch.testing.assert_close(actual["vae.weight"], state["vae.weight"], atol=0, rtol=0)
    torch.testing.assert_close(dict(vae_tensors(path, prefixes=("vae.",)))["weight"], state["vae.weight"])


def test_safetensors_header_length_is_not_a_pickle_signature(tmp_path):
    path = tmp_path / "weights.unknown"
    # Two names happen to produce a 128-byte JSON header in safetensors.
    save_tensors({"vae.weight": torch.ones(1), "weight": torch.zeros(1)}, path)
    assert not is_pytorch_checkpoint(path)
    assert set(dict(checkpoint_tensors(path))) == {"vae.weight", "weight"}


@pytest.mark.parametrize(
    "state", [[torch.ones(1)], {"epoch": 1}, {0: torch.ones(1)}, {"state_dict": {"a": torch.ones(1)}}]
)
def test_rejects_non_tensor_state_before_yielding(tmp_path, state):
    path = tmp_path / "invalid.pth"
    torch.save(state, path)
    with pytest.raises(ValueError, match="tensor state dict"):
        next(checkpoint_tensors(path))


@pytest.mark.parametrize("legacy", (False, True))
def test_checkpoint_mmap_selected_by_file_content(tmp_path, monkeypatch, legacy):
    path = tmp_path / "renamed.safetensors"
    value = torch.randn(3, 4)
    torch.save({"weight": value}, path, _use_new_zipfile_serialization=not legacy)
    original = torch.load
    calls = []

    def load(*args, **kwargs):
        assert kwargs["weights_only"] is True
        calls.append(kwargs["mmap"])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    actual = dict(checkpoint_tensors(path, memory_map=True))
    assert calls == [not legacy]
    torch.testing.assert_close(actual["weight"], value, rtol=0, atol=0)
