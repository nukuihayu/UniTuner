"""Exercise H3 production with real compact VAEs and explicit synthetic text features."""

from dataclasses import replace

import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.support.assertions import require_value
from unituner.models.minimax_h3.conditioning import encode_audio, encode_condition_video
from unituner.models.minimax_h3.config.audio_vae import MiniMaxH3AudioVAEConfig
from unituner.models.minimax_h3.config.video_vae import MiniMaxH3VideoVAEConfig
from unituner.models.minimax_h3.data.cache.encoding import (
    MiniMaxH3CacheEncoders,
    cache_minimax_h3,
    encode_cache_record,
)
from unituner.models.minimax_h3.data.cache.inputs import MiniMaxH3CachePolicy, MiniMaxH3DecodedItem
from unituner.models.minimax_h3.data.cache.storage import (
    MiniMaxH3CacheFormat,
    build_record_layout,
    read_minimax_h3_record,
)
from unituner.models.minimax_h3.data.cache.traversal import MiniMaxH3TraversalPolicy, traverse_minimax_h3
from unituner.models.minimax_h3.data.cache.video import encode_target_video
from unituner.models.minimax_h3.data.media.inputs import prepare_pixels
from unituner.models.minimax_h3.data.sources import MiniMaxH3Control, MiniMaxH3Record
from unituner.models.minimax_h3.text.encoder import MiniMaxH3TextFeatures
from unituner.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE


@pytest.fixture
def vaes():
    with torch.random.fork_rng():
        torch.manual_seed(29)
        video = (
            MiniMaxH3VideoVAE(
                MiniMaxH3VideoVAEConfig(
                    channels=32,
                    channel_multipliers=(1, 1, 1, 1, 1, 1),
                    residual_blocks=(1, 1, 1, 1, 1, 1),
                    tiling=False,
                ),
                include_decoder=False,
            )
            .eval()
            .requires_grad_(False)
        )
        audio = (
            MiniMaxH3AudioVAE(
                MiniMaxH3AudioVAEConfig(
                    encoder_channels=2,
                    hidden_channels=256,
                    decoder_channels=128,
                ),
                include_decoder=False,
            )
            .eval()
            .requires_grad_(False)
        )
        with torch.no_grad():
            for name, value in audio.named_parameters():
                if name.endswith(".alpha"):
                    value.fill_(1.0)
                elif name.endswith((".beta", "q_bias", "v_bias")):
                    value.zero_()
            audio.pre_block.attn.zero_k_bias.zero_()
            audio.latents_mean.copy_(torch.linspace(-0.1, 0.1, 32))
            audio.latents_std.copy_(torch.linspace(0.7, 1.3, 32))

    def target(pixels, seed, key):
        with torch.autocast("cpu", enabled=False):
            return encode_target_video(video, pixels, cache_seed=seed, canonical_item_key=key)

    def condition(pixels):
        with torch.autocast("cpu", enabled=False):
            return encode_condition_video(video, pixels)

    def waveform(samples):
        with torch.autocast("cpu", enabled=False):
            return encode_audio(audio, samples)

    def text(presentation):
        # Deliberately synthetic: integration tests qualify real Qwen features separately.
        return MiniMaxH3TextFeatures(
            torch.full((1, 5120), len(presentation.text) / 37), torch.ones(1, dtype=torch.int64)
        )

    return video, audio, MiniMaxH3CacheEncoders(target, condition, waveform, text, "f" * 64)


def _decoded(tmp_path, *, single=False, references=()):
    target = tmp_path / "target.bin"
    target.write_bytes(b"source pixels")
    count = 1 if single else 22
    return MiniMaxH3DecodedItem(
        MiniMaxH3Record(target, "hello", references),
        torch.arange(count * 32 * 32 * 3, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(count, 32, 32, 3),
        torch.zeros(2, 1600) if single else torch.linspace(-0.5, 0.5, 29600).repeat(2, 1),
        not single,
        target_index=7 if single else None,
    )


def _fail(*args):
    raise RuntimeError("encoding deliberately unavailable")


def test_actual_vae_flow_cache_reuse_and_caption_content_identity(tmp_path, vaes):
    video, audio, encoders = vaes
    decoded = _decoded(tmp_path)
    format = MiniMaxH3CacheFormat(task="fl2va")
    policy = MiniMaxH3CachePolicy(cache_seed=53, allow_experimental_duration=True)
    before = decoded.frames.clone(), decoded.waveform.clone(), torch.get_rng_state().clone()
    directory = tmp_path / "cache"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        manifest = cache_minimax_h3(
            (decoded,), directory, encoders, format=format, policy=policy, encoder_sha256="1" * 64
        )
    record = read_minimax_h3_record(directory, manifest.entries[0], format=format, device=torch.device("cpu"))
    expected = encode_target_video(
        video, prepare_pixels(decoded.frames), cache_seed=53, canonical_item_key=f"{decoded.record.target_path}#0:22"
    )[0]
    assert torch.equal(record.video, expected)
    assert torch.equal(record.audio, encode_audio(audio, decoded.waveform)[0])
    assert torch.equal(
        record.visual_conditions[0], encode_condition_video(video, prepare_pixels(decoded.frames[:1]))[0]
    )
    assert torch.equal(
        record.visual_conditions[1], encode_condition_video(video, prepare_pixels(decoded.frames[-1:]))[0]
    )
    failing = replace(encoders, target_video=_fail, condition_video=_fail, audio=_fail, text=_fail)
    assert (
        cache_minimax_h3((decoded,), directory, failing, format=format, policy=policy, encoder_sha256="1" * 64)
        == manifest
    )
    changed = replace(decoded, record=replace(decoded.record, caption="different"))
    previous = (directory / "manifest.json").read_bytes()
    with pytest.raises(RuntimeError, match="unavailable"):
        cache_minimax_h3((changed,), directory, failing, format=format, policy=policy, encoder_sha256="1" * 64)
    assert (directory / "manifest.json").read_bytes() == previous
    assert all(
        torch.equal(actual, original)
        for actual, original in zip((decoded.frames, decoded.waveform, torch.get_rng_state()), before, strict=True)
    )


def test_single_frame_controls_keep_order_indices_seed_and_nonzero_silence_latent(tmp_path, vaes):
    video, audio, encoders = vaes
    decoded = _decoded(tmp_path, single=True)
    decoded = replace(
        decoded, control_frames=(decoded.frames[0], decoded.frames[0].flip(1)), control_indices=(9, 2), item_key="row-7"
    )
    decoded = replace(
        decoded, frames=torch.cat((decoded.frames, torch.full((1, 32, 32, 1), 128, dtype=torch.uint8)), dim=-1)
    )
    format = MiniMaxH3CacheFormat(task="fl2va", frame_layout="single-frame")
    policy = MiniMaxH3CachePolicy(cache_seed=17)
    record = encode_cache_record(decoded, encoders, format=format, policy=policy)
    assert record.item.control_indices == (9, 2)
    assert require_value(build_record_layout(record, format=format).time_overrides).condition_times == (15.0, 10 / 3)
    assert torch.equal(record.audio, encode_audio(audio, torch.zeros(2, 1600))[0])
    assert torch.count_nonzero(record.audio) > 0 and not record.item.audio_present
    expected = encode_target_video(video, prepare_pixels(decoded.frames), cache_seed=17, canonical_item_key="row-7#1f")[
        0
    ]
    assert torch.equal(record.video, expected)
    directory = tmp_path / "cache"
    first = cache_minimax_h3((decoded,), directory, encoders, format=format, policy=policy, encoder_sha256="1" * 64)
    second = cache_minimax_h3(
        (replace(decoded, item_key="row-8"),),
        directory,
        encoders,
        format=format,
        policy=policy,
        encoder_sha256="1" * 64,
    )
    assert first.entries[0].file != second.entries[0].file


def test_invalid_requests_fail_before_encoding_and_source_changes_preserve_manifest(tmp_path, vaes):
    _, _, encoders = vaes
    decoded = _decoded(tmp_path)
    failing = replace(encoders, target_video=_fail)
    format = MiniMaxH3CacheFormat(task="t2va")
    with pytest.raises(ValueError, match="5-15 seconds"):
        encode_cache_record(decoded, failing, format=format, policy=MiniMaxH3CachePolicy())
    with pytest.raises(ValidationError):
        MiniMaxH3CachePolicy.model_validate({"audio_decode": {"sample_rate": 16000, "channels": 2}})
    policy = MiniMaxH3CachePolicy(allow_experimental_duration=True)
    with pytest.raises(ValueError, match="zero waveform"):
        encode_cache_record(replace(decoded, audio_present=False), failing, format=format, policy=policy)
    directory = tmp_path / "cache"
    cache_minimax_h3((decoded,), directory, encoders, format=format, policy=policy, encoder_sha256="1" * 64)
    before = (directory / "manifest.json").read_bytes()

    def changing(presentation):
        decoded.record.target_path.write_bytes(b"changed source")
        return encoders.text(presentation)

    with pytest.raises(ValueError, match="changed during"):
        cache_minimax_h3(
            (replace(decoded, record=replace(decoded.record, caption="new")),),
            directory,
            replace(encoders, text=changing),
            format=format,
            policy=policy,
            encoder_sha256="1" * 64,
        )
    assert (directory / "manifest.json").read_bytes() == before


def test_completed_orphans_are_reusable_and_invocation_backend_changes_identity(tmp_path, vaes):
    _, _, encoders = vaes
    decoded = _decoded(tmp_path)
    format = MiniMaxH3CacheFormat(task="t2va")
    policy = MiniMaxH3CachePolicy(allow_experimental_duration=True)
    directory = tmp_path / "cache"

    def interrupted():
        yield decoded
        raise RuntimeError("interrupted traversal")

    with pytest.raises(RuntimeError, match="interrupted traversal"):
        cache_minimax_h3(interrupted(), directory, encoders, format=format, policy=policy, encoder_sha256="1" * 64)
    assert not (directory / "manifest.json").exists() and len(list(directory.glob("*.safetensors"))) == 1
    failing = replace(encoders, target_video=_fail, condition_video=_fail, audio=_fail, text=_fail)
    cache_minimax_h3((decoded,), directory, failing, format=format, policy=policy, encoder_sha256="1" * 64)
    with sdpa_kernel(SDPBackend.MATH), pytest.raises(RuntimeError, match="unavailable"):
        cache_minimax_h3((decoded,), directory, failing, format=format, policy=policy, encoder_sha256="1" * 64)


def test_traversal_to_real_vae_cache_and_control_file_identity(tmp_path, vaes):
    video, audio, encoders = vaes
    target, control = tmp_path / "target.png", tmp_path / "control.png"
    for path, color in ((target, (127, 83, 41)), (control, (31, 11, 97))):
        with Image.new("RGB", (48, 32), color) as image:
            image.save(path)
    source = MiniMaxH3Record(target, "hello", (), controls=(MiniMaxH3Control(control, 9),), target_index=7)
    format = MiniMaxH3CacheFormat(task="fl2va", frame_layout="single-frame")
    policy = MiniMaxH3CachePolicy(cache_seed=53)
    traversal = MiniMaxH3TraversalPolicy(resolution=(32, 32))
    directory = tmp_path / "cache"
    decoded = tuple(traverse_minimax_h3([source], format=format, traversal=traversal, policy=policy))
    manifest = cache_minimax_h3(decoded, directory, encoders, format=format, policy=policy, encoder_sha256="1" * 64)
    record = read_minimax_h3_record(directory, manifest.entries[0], format=format, device=torch.device("cpu"))
    assert torch.equal(
        record.video,
        encode_target_video(video, prepare_pixels(decoded[0].frames), cache_seed=53, canonical_item_key=f"{target}#1f")[
            0
        ],
    )
    assert torch.equal(record.audio, encode_audio(audio, torch.zeros(2, 1600))[0])
    assert torch.equal(
        record.visual_conditions[0],
        encode_condition_video(video, prepare_pixels(decoded[0].control_frames[0].unsqueeze(0)))[0],
    )
    # PNG ancillary bytes do not alter decoded pixels but must alter source identity.
    control.write_bytes(control.read_bytes() + b"source revision")
    revised = tuple(traverse_minimax_h3([source], format=format, traversal=traversal, policy=policy))
    assert torch.equal(revised[0].control_frames[0], decoded[0].control_frames[0])
    updated = cache_minimax_h3(revised, directory, encoders, format=format, policy=policy, encoder_sha256="1" * 64)
    assert updated.entries[0].file != manifest.entries[0].file
    before = (directory / "manifest.json").read_bytes(), set(directory.glob("*.safetensors"))

    def changed_artifacts():
        raise ValueError("encoder assets changed")

    with pytest.raises(ValueError, match="assets changed"):
        cache_minimax_h3(
            [replace(revised[0], record=replace(source, caption="different"))],
            directory,
            encoders,
            format=format,
            policy=policy,
            encoder_sha256="1" * 64,
            validate_encoder_artifacts=changed_artifacts,
        )
    assert before == ((directory / "manifest.json").read_bytes(), set(directory.glob("*.safetensors")))
    with pytest.raises(ValueError, match="assets changed"):
        cache_minimax_h3(
            revised,
            directory,
            replace(encoders, target_video=_fail, audio=_fail, condition_video=_fail, text=_fail),
            format=format,
            policy=policy,
            encoder_sha256="1" * 64,
            validate_encoder_artifacts=changed_artifacts,
        )
    assert before == ((directory / "manifest.json").read_bytes(), set(directory.glob("*.safetensors")))
