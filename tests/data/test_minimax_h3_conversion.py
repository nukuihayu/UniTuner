"""Check offline conversion precision, provenance and publication failure boundaries."""

import json
from fractions import Fraction

import av
import numpy as np
import pytest
import torch
from PIL import Image

from tests.support.text_assets import main_with_local_tokenizers
from unituner.artifacts.hashing import file_sha256
from unituner.artifacts.tensors import TensorReader, save_tensors
from unituner.models.minimax_h3.data.cache import conversion
from unituner.models.minimax_h3.data.cache.storage import MiniMaxH3CacheFormat, read_minimax_h3_record


@pytest.fixture
def inputs(tmp_path):
    with Image.new("RGBA", (48, 32), (97, 31, 17, 53)) as image:
        image.save(tmp_path / "target.png")
    latent = {
        "latents_1x2x2_float16": torch.arange(96, dtype=torch.float16).reshape(24, 1, 2, 2) / 7,
        "latents_audio_32x2x2_bfloat16": torch.arange(128, dtype=torch.bfloat16).reshape(32, 2, 2) / 13,
        "audio_present_float32": torch.tensor(0.0),
        "one_frame_target_index_int64": torch.tensor(7),
    }
    text = {
        "varlen_mmh3_hidden_states_bfloat16": torch.arange(2 * 5120, dtype=torch.bfloat16).reshape(2, 5120) / 37,
        "varlen_mmh3_token_tags_int64": torch.tensor([1, 0]),
    }
    document = {
        "source_format": "minimax-h3-split/1",
        "float_conversion": "exact-fp32",
        "encoder_sha256": "1" * 64,
        "format": {"task": "t2va", "frame_layout": "single-frame"},
        "entries": [
            {
                "latents": "latent.safetensors",
                "text": "text.safetensors",
                "source": {"target_path": "target.png", "caption": "雪", "target_index": 7},
                "resolution": [32, 32],
                "item": {"frame_count": 1, "audio_present": False, "target_index": 7},
            }
        ],
    }

    def write():
        save_tensors(latent, tmp_path / "latent.safetensors", metadata={"ss_old": "ignored at output"})
        save_tensors(text, tmp_path / "text.safetensors")
        document["entries"][0].update(
            latents_sha256=file_sha256(tmp_path / "latent.safetensors"),
            text_sha256=file_sha256(tmp_path / "text.safetensors"),
        )
        (tmp_path / "import.json").write_text(json.dumps(document))

    write()
    return latent, text, document, write, tmp_path / "import.json"


def test_exact_promotion_metadata_identity_and_public_command(inputs, tmp_path):
    latent, text, document, write, path = inputs
    original = {
        name: (tmp_path / name).read_bytes() for name in ["latent.safetensors", "text.safetensors", "target.png"]
    }
    rng = torch.get_rng_state().clone()
    directory = tmp_path / "cache"
    assert main_with_local_tokenizers(["caches", "import", str(path), str(directory)]) == 0
    manifest = conversion.import_minimax_h3_cache(path, directory)
    format = MiniMaxH3CacheFormat(task="t2va", frame_layout="single-frame")
    record = read_minimax_h3_record(directory, manifest.entries[0], format=format, device=torch.device("cpu"))
    assert torch.equal(record.video, latent["latents_1x2x2_float16"].float())
    assert torch.equal(record.audio, latent["latents_audio_32x2x2_bfloat16"].float())
    assert torch.equal(record.text, text["varlen_mmh3_hidden_states_bfloat16"].float())
    assert record.audio.count_nonzero() and not record.item.audio_present
    with TensorReader(directory / manifest.entries[0].file) as reader:
        assert set(reader.metadata()) == {"unituner.cache.identity", "unituner.cache.item", "unituner.cache.format"}
        assert not any("ss_" in key or "rcm" in key for key in reader.keys())
    assert original == {name: (tmp_path / name).read_bytes() for name in original}
    assert torch.equal(rng, torch.get_rng_state())
    document["entries"][0]["source"]["caption"] = "changed caption"
    write()
    revised = conversion.import_minimax_h3_cache(path, directory)
    assert revised.entries[0].file != manifest.entries[0].file
    record.video.zero_()
    assert latent["latents_1x2x2_float16"].count_nonzero()


@pytest.mark.parametrize("fault", ["rcm", "teacher", "suffix", "coordinates", "presence", "hash"])
def test_reject_invalid_inputs_preserves_existing_manifest(inputs, tmp_path, fault):
    latent, text, _document, write, path = inputs
    directory = tmp_path / "cache"
    conversion.import_minimax_h3_cache(path, directory)
    before = (directory / "manifest.json").read_bytes(), set(directory.glob("*.safetensors"))
    if fault == "rcm":
        latent["rcm_mask"] = torch.ones(1)
    if fault == "teacher":
        text["varlen_mmh3_teacher_hidden_states_float32"] = torch.ones(2, 5120)
    if fault == "suffix":
        latent["latents_2x2x2_float16"] = latent.pop("latents_1x2x2_float16")
    if fault == "coordinates":
        latent["one_frame_target_index_int64"] = torch.tensor(8)
    if fault == "presence":
        latent["audio_present_float32"] = torch.tensor(1.0)
    write()
    if fault == "hash":
        (tmp_path / "latent.safetensors").write_bytes((tmp_path / "latent.safetensors").read_bytes() + b"changed")
    with pytest.raises(ValueError):
        conversion.import_minimax_h3_cache(path, directory)
    assert before == ((directory / "manifest.json").read_bytes(), set(directory.glob("*.safetensors")))


def test_source_changes_during_decode_never_publish_stale_record(inputs, tmp_path, monkeypatch):
    _, _, document, write, path = inputs
    directory = tmp_path / "cache"
    conversion.import_minimax_h3_cache(path, directory)
    before = (directory / "manifest.json").read_bytes(), set(directory.glob("*.safetensors"))
    document["entries"][0]["source"]["caption"] = "new"
    write()
    traverse = conversion.traverse_minimax_h3

    def changed(*args, **kwargs):
        yield from traverse(*args, **kwargs)
        image = tmp_path / "target.png"
        image.write_bytes(image.read_bytes() + b"changed during decode")

    monkeypatch.setattr(conversion, "traverse_minimax_h3", changed)
    with pytest.raises(ValueError, match="changed during"):
        conversion.import_minimax_h3_cache(path, directory)
    assert before == ((directory / "manifest.json").read_bytes(), set(directory.glob("*.safetensors")))


def test_materially_short_target_audio_rejects_before_record_publication(inputs, tmp_path):
    latent, _, document, write, path = inputs
    video = tmp_path / "short.mkv"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "bgr0"
        audio = container.add_stream("pcm_f32le", rate=32000)
        audio.layout = "stereo"
        for index in range(24):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        frame = av.AudioFrame.from_ndarray(np.zeros((2, 1000), np.float32), format="fltp", layout="stereo")
        frame.sample_rate = 32000
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    latent.clear()
    latent.update(
        {
            "latents_7x2x2_float32": torch.zeros(24, 7, 2, 2),
            "latents_audio_32x2x37_float32": torch.ones(32, 2, 37),
            "audio_present_float32": torch.tensor(1.0),
        }
    )
    document["format"]["frame_layout"] = "video"
    document["policy"] = {"allow_experimental_duration": True}
    document["entries"][0].update(
        source={"target_path": "short.mkv", "caption": ""}, item={"frame_count": 22, "audio_present": True}
    )
    write()
    with pytest.raises(ValueError, match="short by"):
        conversion.import_minimax_h3_cache(path, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()
