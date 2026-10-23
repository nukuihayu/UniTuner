"""All restoration families must read text conditions without touching image tensors."""

import pytest
import torch

from unitunner.artifacts.tensors import TensorReader
from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.cache.storage import read_caption_record, save_cache_record
from unitunner.data.datasets.restoration import AlignedImagePair
from unitunner.models.components.conditioning import batch_captions
from unitunner.models.flux.conditioning import FluxTextFeatures
from unitunner.models.krea2.conditioning import batch_text
from unitunner.training.integrations.z_image.binding import ZImageBinding
from unitunner.training.methods import restoration

CASES = [
    (ZImageBinding, "z-image/CHW+LD"),
    (restoration.FluxBinding, "flux/dev/CHW+LD+V"),
    (restoration.Flux2Binding, "flux2/klein-4b/base/CHW+LD+R0"),
    (restoration.Flux3Binding, "flux3/base/CHW+LD+R0"),
    (restoration.MossImageBinding, "moss-image/CHW+LD"),
    (restoration.QwenImageBinding, "qwen-image/original/CLHW+LD+R0"),
    (restoration.QwenImage21Binding, "qwen-image-2-1/C1HW+LD+R0"),
    (restoration.Krea2Binding, "krea2/C1HW+SLD"),
]


@pytest.mark.parametrize("binding_type,layout", CASES)
@pytest.mark.parametrize("batch_size", (1, 2))
def test_native_restoration_caption_batches_skip_all_image_payloads(
    tmp_path, monkeypatch, binding_type, layout, batch_size
):
    pairs, rows, pooled = [], [], []
    for index in range(batch_size):
        length = 3 if binding_type is restoration.FluxBinding else index + 2
        caption = torch.randn((length, 3, 8) if binding_type is restoration.Krea2Binding else (length, 8))
        values = {"caption": caption, "latents": torch.randn(64, 1, 2, 2)}
        if binding_type is restoration.FluxBinding:
            values["pooled"] = torch.randn(6)
            pooled.append(values["pooled"])
        if binding_type is restoration.QwenImage21Binding:
            values.update(
                image_slots=torch.empty(0, dtype=torch.int64),
                reference_grids=torch.empty(0, 2, dtype=torch.int64),
                reference_hashes=torch.empty(0, 32, dtype=torch.uint8),
            )
        identity = CacheIdentity(
            source_sha256=str(index) * 64, encoder_sha256="a" * 64, transform_sha256="b" * 64, layout=layout
        )
        entry = save_cache_record(tmp_path, identity, values)
        pairs.append(AlignedImagePair(None, tmp_path / f"{index}.png", entry, tmp_path))
        rows.append(caption)
    original = TensorReader.read
    read_keys = []

    def read(self, key, **kwargs):
        assert key in {"caption", "pooled"}, "Restoration must leave cached image payloads on disk"
        read_keys.append(key)
        return original(self, key, **kwargs)

    monkeypatch.setattr(TensorReader, "read", read)
    method = object.__new__(restoration.RestorationMethod)
    method.binding = object.__new__(binding_type)
    method.vae = torch.nn.Linear(1, 1)
    actual = method.captions(pairs)
    if binding_type is restoration.FluxBinding:
        assert isinstance(actual, FluxTextFeatures)
        torch.testing.assert_close(actual.captions, torch.stack(rows), rtol=0, atol=0)
        torch.testing.assert_close(actual.pooled, torch.stack(pooled), rtol=0, atol=0)
    else:
        expected = batch_text(tuple(rows)) if binding_type is restoration.Krea2Binding else batch_captions(rows)
        torch.testing.assert_close(actual.features, expected.features, rtol=0, atol=0)
        assert actual.lengths == expected.lengths
    assert len(read_keys) == batch_size * (2 if binding_type is restoration.FluxBinding else 1)


def test_caption_reader_rejects_wrong_identity_and_invalid_text(tmp_path):
    identity = CacheIdentity(
        source_sha256="a" * 64, encoder_sha256="b" * 64, transform_sha256="c" * 64, layout="z-image/CHW+LD"
    )
    entry = save_cache_record(tmp_path, identity, {"caption": torch.ones(2), "latents": torch.ones(2, 2, 2)})
    with pytest.raises(ValueError, match="caption dimensions"):
        read_caption_record(tmp_path, entry, device=torch.device("cpu"))
