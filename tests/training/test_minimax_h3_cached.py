"""Exercise actual complete H3 caches through native flow, shared updates and resume."""

import json
from contextlib import ExitStack, closing
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.support.text_assets import main_with_local_tokenizers
from unituner.adapters.config import AdapterConfig
from unituner.artifacts.tensors import TensorReader
from unituner.checkpoints.restore import restore_checkpoint
from unituner.checkpoints.saving import save_checkpoint
from unituner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unituner.data.cache.schema import CacheIdentity
from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.config.training import MiniMaxH3FlowConfig
from unituner.models.minimax_h3.data.batching import build_minimax_h3_batches
from unituner.models.minimax_h3.data.cache.storage import (
    MiniMaxH3CacheFormat,
    MiniMaxH3CacheItem,
    MiniMaxH3CacheManifest,
    MiniMaxH3CacheRecord,
    MiniMaxH3CacheReference,
    load_minimax_h3_manifest,
    read_minimax_h3_record,
    save_minimax_h3_manifest,
    save_minimax_h3_record,
)
from unituner.models.minimax_h3.loading import load_minimax_h3, save_minimax_h3
from unituner.models.minimax_h3.transformer import MiniMaxH3Transformer
from unituner.training.contracts import RunState
from unituner.training.engine import update_phase
from unituner.training.integrations.minimax_h3.binding import MiniMaxH3Binding, build_cached_request
from unituner.training.parameters import ParameterInstallation


def test_named_image_video_groups_train_and_resume(tmp_path):
    from unituner.config.loading import load_training_config
    from unituner.training.workflows.minimax_h3.run import train_minimax_h3

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name, single in (("images", True), ("videos", False)):
            _cache(tmp_path / name, "t2va", single)
        weights = tmp_path / "model.safetensors"
        save_minimax_h3(_model(), weights)
        config = load_training_config(
            overrides={
                "model": {"family": "minimax-h3", "variant": "t2va", "weights": str(weights)},
                "dataset": {
                    "defaults": {"resolution": [32, 32]},
                    "groups": {
                        "images": {"source": "unused", "cache_directory": str(tmp_path / "images")},
                        "videos": {
                            "kind": "video",
                            "source": "unused",
                            "cache_directory": str(tmp_path / "videos"),
                            "repeats": 2,
                            "video": {"target_frames": [22], "target_fps": 24.0, "fps_resampling": "timestamps"},
                        },
                    },
                },
                "train": {"adapter": {"rank": 2}, "max_updates": 3, "accumulation_steps": 2},
                "output": {
                    "max_resume_checkpoints": 0,
                    "directory": (tmp_path / "continuous").as_posix(),
                    "name": "h3",
                    "save_every_updates": 1,
                },
            }
        )
        with sdpa_kernel(SDPBackend.MATH):
            expected = train_minimax_h3(config, device=torch.device("cpu"))
            resumed = config.model_copy(
                update={"output": config.output.model_copy(update={"directory": (tmp_path / "resumed").as_posix()})}
            )
            actual = train_minimax_h3(
                resumed, device=torch.device("cpu"), resume=tmp_path / "continuous/resume/h3.state-00000001.safetensors"
            )
        assert expected == actual == RunState(updates=3, microbatches=6, samples=6)
        for name in ("weights/h3.safetensors", "resume/h3.state-00000003.safetensors"):
            assert (tmp_path / "continuous" / name).read_bytes() == (tmp_path / "resumed" / name).read_bytes()
    finally:
        torch.set_num_threads(previous)


def _cache(directory, task, single):
    format = MiniMaxH3CacheFormat(task=task, frame_layout="single-frame" if single else "video")
    visual, audio, fields = (), (), {}
    if single:
        fields.update(target_index=7)
    if task == "fl2va":
        if single:
            fields.update(control_indices=(9, 2, 9))
            visual = tuple(torch.full((24, 1, 2, 2), (i + 1) / 10) for i in range(3))
        else:
            fields.update(condition_roles=("last",))
            visual = (torch.full((24, 1, 2, 2), 0.1),)
    if task == "ref2va":
        fields.update(references=(MiniMaxH3CacheReference(kind="video", has_audio=True),))
        visual = (torch.full((24, 2, 2, 2), 0.1),)
        audio = (torch.full((32, 2, 8), 0.2),)
    item = MiniMaxH3CacheItem(frame_count=1 if single else 22, audio_present=not single, **fields)
    record = MiniMaxH3CacheRecord(
        item,
        torch.full((24, 1 if single else 7, 2, 2), 0.05),
        torch.full((32, 2, 2 if single else 37), 0.17),
        torch.linspace(-0.1, 0.1, 3 * 5120).reshape(3, 5120),
        torch.tensor([1, 0, 1]),
        visual,
        audio,
    )
    identity = CacheIdentity(
        source_sha256="1" * 64, encoder_sha256="2" * 64, transform_sha256="3" * 64, layout=format.layout
    )
    entry = save_minimax_h3_record(directory, identity, record, format=format)
    other = save_minimax_h3_record(
        directory,
        identity.model_copy(update={"source_sha256": "4" * 64}),
        replace(record, video=record.video + 0.01),
        format=format,
    )
    manifest = MiniMaxH3CacheManifest(**format.model_dump(), entries=(entry, other, entry))
    save_minimax_h3_manifest(manifest, directory)
    return format, load_minimax_h3_manifest(directory)


def _model():
    model = MiniMaxH3Transformer(
        MiniMaxH3Config(
            hidden_size=24,
            num_layers=2,
            token_refiner_num_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            ffn_hidden_size=40,
            timestep_input_dim=8,
            time_embed_hidden_size=16,
            time_embed_dim=8,
            rope_inv_freq_len=2,
        )
    )
    with torch.no_grad():
        model.rope.inv_freq.copy_(torch.tensor([1.0, 0.01]))
    return model


@pytest.mark.parametrize("task,algorithm", [("t2va", "lora"), ("fl2va", "loha"), ("ref2va", "lokr")])
@pytest.mark.parametrize("single", [False, True])
def test_cached_updates_and_exact_resume(tmp_path, task, algorithm, single):
    torch.set_num_threads(1)
    format, manifest = _cache(tmp_path / "cache", task, single)
    plan = build_minimax_h3_batches(manifest, batch_size=1, repeats=2)
    assert sorted(plan.batch(i)[0].file for i in range(plan.num_batches)) == sorted(
        entry.file for entry in manifest.entries * 2
    )
    assert plan.samples_before(13) == 13
    with pytest.raises(ValueError):
        build_minimax_h3_batches(manifest, batch_size=2, repeats=1)
    with pytest.raises(ValueError):
        plan.samples_before(-1)
    torch.manual_seed(751)
    model = _model()
    binding = MiniMaxH3Binding(model, MiniMaxH3FlowConfig(audio_condition_clean=0.9))
    frozen = {key: value.clone() for key, value in model.state_dict().items()}
    identity = ResumeIdentity(config_sha256="1" * 64, dataset_manifest_sha256="2" * 64, model_artifact_sha256="3" * 64)
    with ExitStack() as resources:
        resources.enter_context(sdpa_kernel(SDPBackend.MATH))
        owner = resources.enter_context(
            closing(
                ParameterInstallation(
                    model,
                    adapter=AdapterConfig(algorithm=algorithm, rank=2, alpha=7.0),
                    targets=binding.adapter_targets(),
                )
            )
        )
        optimizer = torch.optim.AdamW(owner.plan.parameters, lr=0.003)
        modules = {"model": model, "adapter": owner.branches}
        initial = {key: value.clone() for key, value in owner.branches.state_dict().items()}
        state = RunState()

        def steps():
            for offset in range(2):
                entry = plan.batch(state.microbatches + offset)[0]
                record = read_minimax_h3_record(tmp_path / "cache", entry, format=format, device=torch.device("cpu"))
                video_noise, audio_noise = torch.randn_like(record.video), torch.randn_like(record.audio)
                uniform = torch.rand(1)
                visuals = tuple(torch.randn_like(value) for value in record.visual_conditions)
                audios = tuple(torch.randn_like(value) for value in record.audio_conditions)
                before = torch.get_rng_state().clone()
                request = build_cached_request(
                    record,
                    format=format,
                    video_noise=video_noise,
                    audio_noise=audio_noise,
                    uniform=uniform,
                    visual_condition_noise=visuals,
                    audio_condition_noise=audios,
                )
                assert torch.equal(before, torch.get_rng_state())
                assert torch.equal(request.audio[0], record.audio) and request.audio.count_nonzero()
                assert request.audio_present == (not single)
                if single and task == "fl2va":
                    assert request.layout.time_overrides is not None
                    assert request.layout.time_overrides.condition_times == (15.0, 10 / 3, 15.0)
                yield binding.step(request)

        update_phase(steps(), optimizer, state, max_grad_norm=1.0)
        checkpoint = tmp_path / "state.safetensors"
        save_checkpoint(checkpoint, modules, optimizer, CheckpointCursor(**asdict(state)), identity)
        for _ in range(2):
            update_phase(steps(), optimizer, state, max_grad_norm=1.0)
        continuous = {key: value.clone() for key, value in owner.branches.state_dict().items()}
        expected_optimizer = optimizer.state_dict()
        expected_rng = torch.get_rng_state().clone()
        expected_state = replace(state)
        cursor = restore_checkpoint(checkpoint, modules, optimizer, identity)
        state = RunState(**cursor.model_dump())
        for _ in range(2):
            update_phase(steps(), optimizer, state, max_grad_norm=1.0)
        assert state == expected_state == RunState(updates=3, microbatches=6, samples=6)
        assert all(torch.equal(value, owner.branches.state_dict()[key]) for key, value in continuous.items())
        assert any(not torch.equal(value, continuous[key]) for key, value in initial.items())
        assert torch.equal(expected_rng, torch.get_rng_state())
        actual_optimizer = optimizer.state_dict()
        assert expected_optimizer["param_groups"] == actual_optimizer["param_groups"]
        for index, values in expected_optimizer["state"].items():
            for key, value in values.items():
                assert torch.equal(value, actual_optimizer["state"][index][key])
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in frozen.items())


@pytest.mark.parametrize(
    "task,algorithm,single",
    [
        ("t2va", "lora", False),
        ("fl2va", "loha", True),
        ("ref2va", "lokr", False),
        ("t2va", "full", False),
        ("fl2va", "full", True),
        ("ref2va", "full", False),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("with_previews", [False, True])
@pytest.mark.parametrize("residency", ["resident", "frozen-blocks"])
def test_public_training_export_and_resume(
    tmp_path, task, algorithm, single, device, with_previews, residency, request
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if residency == "frozen-blocks" and device != "cuda":
        pytest.skip("H2D requires CUDA")
    torch.set_num_threads(1)
    _cache(tmp_path / "cache", task, single)
    torch.manual_seed(431)
    weights = tmp_path / "model.safetensors"
    save_minimax_h3(_model(), weights)
    original = weights.read_bytes()
    document = f'''[model]
family="minimax-h3"
variant="{task}"
weights="{weights.as_posix()}"
[dataset.groups.main]
kind="{"image" if single else "video"}"
source="unused"
cache_directory="{(tmp_path / "cache").as_posix()}"
resolution=[32,32]
repeats=2
'''
    if not single:
        document += """[dataset.groups.main.video]
target_frames=[22]
target_fps=24.0
fps_resampling="timestamps"
"""
    document += f'''[method.minimax_h3]
sigma_min=0.1
sigma_max=0.9
focus_min=0.2
focus_max=0.7
focus_probability=0.4
visual_condition_clean=0.85
audio_condition_clean=0.9
audio_loss_weight=0.7
[train]
max_updates=3
accumulation_steps=2
learning_rate=0.001
lr_schedule="cosine"
warmup_updates=1
[train.adapter]
algorithm="{algorithm}"
rank=2
alpha=7.0
[execution]
recompute="block"
[output]
max_resume_checkpoints = 0
directory="{(tmp_path / "continuous").as_posix()}"
name="h3"
save_every_updates=1
save_precision="bf16"
'''
    if algorithm == "full":
        if residency != "resident":
            pytest.skip("Full parameters stay resident")
        document = document.replace("[train]\n", '[train]\nparameter_mode="full"\n').replace(
            '[train.adapter]\nalgorithm="full"\nrank=2\nalpha=7.0\n', ""
        )
    if residency == "frozen-blocks":
        document = document.replace(
            'recompute="block"', 'recompute="block"\nresidency="frozen-blocks"\nuse_pinned_memory_for_block_swap=true'
        )
        document = document.replace("[output]", '[output]\ncheckpoint_storage="frozen-reference"')
    specs = None
    if with_previews:
        specs = request.getfixturevalue("minimax_h3_preview_assets")
        document = document.replace(
            f'weights="{weights.as_posix()}"',
            f'''weights="{weights.as_posix()}"
vae="{Path(specs["video_weights"]).as_posix()}"
text_encoder="{Path(specs["text_weights"]).as_posix()}"
[model.minimax_h3]
audio_vae="{Path(specs["audio_weights"]).as_posix()}"
text_config="{Path(specs["text_assets"]).as_posix()}"
text_processor="{Path(specs["processor"]).as_posix()}"
text_device="cpu"''',
        )
    path = tmp_path / "train.toml"
    path.write_text(document)
    resume_path = tmp_path / "resume.toml"
    resume_path.write_text(document.replace((tmp_path / "continuous").as_posix(), (tmp_path / "resumed").as_posix()))
    with sdpa_kernel(SDPBackend.MATH):
        assert main_with_local_tokenizers(["train", str(path), "--device", device]) == 0
        if with_previews:
            preview = """[preview]
every_updates=1
at_start=true
"""
            names = {"t2va": "video-t2va", "fl2va": "single-fl2va", "ref2va": "video-ref2va"}
            assert specs is not None
            spec = specs["cases"][names[task]]
            samples = [(spec["inputs"], spec["frames"], spec.get("fps", 24))]
            if not single:
                image_inputs = specs["cases"]["single-" + task]["inputs"]
                if task == "ref2va":
                    image_inputs = json.loads(json.dumps(image_inputs))
                    for reference in image_inputs["references"]:
                        if reference["kind"] == "video":
                            reference["audio_enabled"] = False
                samples.append((image_inputs, 1, 24))

            def inline(value):
                if isinstance(value, dict):
                    return "{" + ",".join(key + "=" + inline(item) for key, item in value.items()) + "}"
                if isinstance(value, list):
                    return "[" + ",".join(inline(item) for item in value) + "]"
                return json.dumps(value)

            for inputs, frames, fps in samples:
                preview += f"""[[preview.audio_video]]
inputs={inline(inputs)}
[preview.audio_video.sampling]
width=32
height=32
frames={frames}
fps={fps}
steps=2
seed=71
allow_experimental_duration=true
temporal_fine_bands={1 if fps != 24 else 0}
"""
            preview_path = tmp_path / "preview.toml"
            preview_document = (
                document.replace((tmp_path / "continuous").as_posix(), str(tmp_path / "previewed")) + preview
            )
            preview_path.write_text(preview_document)
            assert main_with_local_tokenizers(["train", str(preview_path), "--device", device]) == 0
            for name in [
                "weights/h3.safetensors",
                *[f"resume/h3.state-{update:08d}.safetensors" for update in range(1, 4)],
            ]:
                assert (tmp_path / "continuous" / name).read_bytes() == (tmp_path / "previewed" / name).read_bytes()
            assert len(list((tmp_path / "previewed/previews").iterdir())) == 4 * len(samples)
            resume_path.write_text(
                preview_document.replace(str(tmp_path / "previewed"), (tmp_path / "resumed").as_posix())
            )
        assert (
            main_with_local_tokenizers(
                [
                    "train",
                    str(resume_path),
                    "--device",
                    device,
                    "--resume",
                    str(tmp_path / "continuous/resume/h3.state-00000001.safetensors"),
                ]
            )
            == 0
        )
    for name in ["weights/h3.safetensors", "resume/h3.state-00000003.safetensors"]:
        assert (tmp_path / "continuous" / name).read_bytes() == (tmp_path / "resumed" / name).read_bytes()
    assert weights.read_bytes() == original
    if algorithm == "full":
        restored = load_minimax_h3(tmp_path / "resumed/weights/h3.safetensors")
        assert restored.config == _model().config
        with TensorReader(tmp_path / "resumed/weights/h3.safetensors") as reader:
            assert all(reader.read(key).dtype == torch.bfloat16 for key in reader.keys())
        assert (tmp_path / "resumed/weights/h3.safetensors").read_bytes() != original
        return
    with TensorReader(tmp_path / "resumed/weights/h3.safetensors") as reader:
        prefix = "diffusion_model."
        assert all(key.startswith(prefix) for key in reader.keys())
        assert all(reader.read(key).dtype == torch.bfloat16 for key in reader.keys())
        assert any(key.endswith(".alpha") for key in reader.keys())
        assert reader.metadata()["network_alpha"] == "7.0"
        assert reader.metadata()["unituner.adapter.layout"] == "minimax-h3-comfyui"
        assert not any(key.startswith("ss_") for key in reader.metadata())
    previous = (tmp_path / "resumed/weights/h3.safetensors").read_bytes()
    changed = tmp_path / "changed.toml"
    changed.write_text(resume_path.read_text().replace("focus_probability=0.4", "focus_probability=0.5"))
    assert (
        main_with_local_tokenizers(
            [
                "train",
                str(changed),
                "--device",
                device,
                "--resume",
                str(tmp_path / "continuous/resume/h3.state-00000001.safetensors"),
            ]
        )
        == 2
    )
    assert (tmp_path / "resumed/weights/h3.safetensors").read_bytes() == previous
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        document.replace(f'variant="{task}"', f'variant="{"ref2va" if task == "t2va" else "t2va"}"').replace(
            (tmp_path / "continuous").as_posix(), str(tmp_path / "invalid")
        )
    )
    assert main_with_local_tokenizers(["train", str(invalid), "--device", device]) == 2
    assert not list((tmp_path / "invalid").rglob("*.safetensors"))
