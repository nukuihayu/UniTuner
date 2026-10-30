"""Check external safetensors interoperability, storage and failure boundaries."""

import struct

import pytest
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from unitunner.artifacts.atomic import atomic_output
from unitunner.artifacts.tensors import TensorReader, iter_tensors, resolve_shards, save_tensors


def test_streamed_external_roundtrip(tmp_path):
    path = tmp_path / "weights.safetensors"
    tensors = {
        "view": torch.arange(24, dtype=torch.float32).reshape(4, 6).T,
        "scalar": torch.tensor(3.0, dtype=torch.bfloat16),
        "empty": torch.empty(0, 3),
        "boolean": torch.tensor([True, False]),
        "negative": torch._neg_view(torch.arange(4, dtype=torch.float32)),  # pyright: ignore[reportPrivateImportUsage] -- Test Torch negative-view serialization.
        "浮点": torch.tensor([1.0, -2.0], dtype=torch.float16),
    }
    save_tensors(tensors, path, metadata={"unitunner.provenance": "{}"})
    with safe_open(path, framework="pt") as reader:
        for key, tensor in tensors.items():
            torch.testing.assert_close(reader.get_tensor(key), tensor, rtol=0, atol=0)
        assert reader.metadata() == {"unitunner.provenance": "{}"}
    with TensorReader(path) as reader:
        kept = reader.read("view")
    reader.close()
    torch.testing.assert_close(kept, tensors["view"], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="closed"):
        reader.read("view")


def test_failed_write_preserves_destination(tmp_path):
    path = tmp_path / "weights.safetensors"
    save_tensors({"original": torch.ones(3)}, path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_tensors({"meta": torch.empty(3, device="meta")}, path)
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError):
        with atomic_output(path) as temporary:
            temporary.write_bytes(b"incomplete")
            raise RuntimeError("failure during payload write")
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_single_pass_writer_matches_external_reader_and_rolls_back_generator_failures(tmp_path):
    from unitunner.artifacts.tensors import save_tensor_stream

    path = tmp_path / "stream.safetensors"
    values = {"scalar": torch.tensor(2.0), "empty": torch.empty(0, 2), "view": torch.arange(12).view(3, 4).T}
    save_tensor_stream(iter(values.items()), path, metadata={"source": "stream"})
    with TensorReader(path) as reader:
        assert reader.metadata() == {"source": "stream"}
        for key, tensor in values.items():
            torch.testing.assert_close(reader.read(key), tensor)
    before = path.read_bytes()

    def failing():
        yield "new", torch.ones(2)
        raise ValueError("conversion failed")

    with pytest.raises(ValueError, match="conversion failed"):
        save_tensor_stream(failing(), path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="duplicate"):
        save_tensor_stream((("a", torch.ones(2)), ("a", torch.ones(2))), path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_shards_preserve_non_five_digit_numbering(tmp_path):
    first = tmp_path / "model-01-of-02.safetensors"
    second = tmp_path / "model-02-of-02.safetensors"
    save_file({"a": torch.ones(2)}, first)
    save_file({"b": torch.zeros(2)}, second)
    assert resolve_shards(second) == (first, second)
    values = dict(iter_tensors(first))
    torch.testing.assert_close(values["a"], torch.ones(2))
    save_file({"a": torch.zeros(2)}, second)
    with pytest.raises(ValueError, match="Duplicate"):
        next(iter_tensors(first))
    second.unlink()
    with pytest.raises(FileNotFoundError):
        next(iter_tensors(first))


def test_corrupt_header_rejected(tmp_path):
    path = tmp_path / "corrupt.safetensors"
    path.write_bytes(struct.pack("<Q", 1000) + b"{}")
    with pytest.raises(SafetensorError):
        TensorReader(path)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_publication_cast_matches_torch_and_keeps_live_tensors(tmp_path, dtype):
    tensors = {
        "weight": torch.linspace(-1, 1, 35, dtype=torch.float64).reshape(5, 7).T.requires_grad_(),
        "alpha": torch.tensor(7.1),
        "empty": torch.empty(0, 2),
        "ids": torch.tensor([2**40, -3]),
        "mask": torch.tensor([True, False]),
    }
    tensors["weight"].sum().backward()
    before = {key: value.detach().clone() for key, value in tensors.items()}
    assert tensors["weight"].grad is not None
    gradient = tensors["weight"].grad.clone()
    rng = torch.get_rng_state().clone()
    path = tmp_path / "cast.safetensors"
    save_tensors(tensors, path, dtype=dtype, metadata={"purpose": "publication"})
    with safe_open(path, framework="pt") as reader:
        assert set(reader.keys()) == set(tensors)
        assert reader.metadata() == {"purpose": "publication"}
        for key, source in tensors.items():
            expected = source.detach().to(dtype) if source.is_floating_point() else source
            torch.testing.assert_close(reader.get_tensor(key), expected, atol=0, rtol=0)
            torch.testing.assert_close(source, before[key], atol=0, rtol=0)
    assert tensors["weight"].grad is not None
    assert torch.equal(tensors["weight"].grad, gradient)
    assert torch.equal(torch.get_rng_state(), rng)
