"""Verify native MOSS masking, gradients, rotary geometry and strict artifact ownership."""

import math

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from unitunner.artifacts.importing import import_weights
from unitunner.artifacts.tensors import save_tensors
from unitunner.cli.main import main
from unitunner.models.moss_image.config.model import MossImageConfig
from unitunner.models.moss_image.layers import RotaryEmbedding, SiTUGLU
from unitunner.models.moss_image.loading import load_moss_image, save_moss_image
from unitunner.models.moss_image.transformer import MossImageTransformer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config():
    return MossImageConfig(dim=32, ffn_dim=48, num_heads=2, num_layers=2, text_dim=12, freq_dim=8, latent_channels=8)


def test_moss_padded_batch_matches_individual_values_and_gradients():
    torch.manual_seed(118)
    model = MossImageTransformer(small_config())
    latents = torch.randn(2, 8, 2, 3, requires_grad=True)
    context = torch.randn(2, 5, 12, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    times = torch.tensor([244.0, 751.0])
    with sdpa_kernel(SDPBackend.MATH):
        batched = model(latents, times, context, mask)
        individual = torch.cat(
            [model(latents[i : i + 1], times[i : i + 1], context[i : i + 1, :size]) for i, size in enumerate((3, 5))]
        )
        torch.testing.assert_close(batched, individual, atol=1e-6, rtol=1e-5)
        variables = (latents, context, *model.parameters())
        left = torch.autograd.grad(batched.square().mean(), variables, retain_graph=True)
        right = torch.autograd.grad(individual.square().mean(), variables)
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    assert not left[1][0, 3:].count_nonzero()
    for invalid in (torch.zeros_like(mask), torch.tensor([[1, 0, 1, 0, 0], [1, 1, 1, 1, 1]]), torch.full((2, 5), 0.5)):
        with pytest.raises(ValueError, match="MOSS context mask"):
            model(latents, times, context, invalid)
    with pytest.raises(ValueError, match="inference-only"):
        model(latents, times, context, mask, skip_layers=(0,))
    model.eval()
    assert torch.isfinite(model(latents, times, context, mask, skip_layers=(0,))).all()


def test_moss_rotary_coordinates_and_bounded_gate_match_independent_formulas():
    rope = RotaryEmbedding(16, 10000.0)
    actual = rope(2, 3, 4, torch.tensor([2]))
    expected = []
    coordinates = [(0, 2, h - 0.5, w - 1) for h in range(2) for w in range(3)] + [(0, i, 0, 0) for i in range(4)]
    for point in coordinates:
        phases = [value * band for value in point for band in (1.0, 0.01)]
        expected.append([math.sin(x) for x in phases] + [math.cos(x) for x in phases])
    torch.testing.assert_close(actual, torch.tensor([expected]), atol=1e-6, rtol=1e-5)
    before = actual.clone()
    rope.to(torch.bfloat16)
    torch.testing.assert_close(rope(2, 3, 4, torch.tensor([2])), before, atol=0, rtol=0)
    hidden = torch.tensor([-40.0, 0.0, 30.0], requires_grad=True)
    gate = torch.tensor([-10.0, 0.0, 10.0], requires_grad=True)
    output = SiTUGLU()(hidden, gate)
    reference = torch.tensor(
        [
            (4 * math.tanh(g / 4) / (1 + math.exp(-g))) * (25 * math.tanh(h / 25))
            for h, g in zip(hidden.tolist(), gate.tolist(), strict=True)
        ]
    )
    torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-6)
    assert all(torch.isfinite(value).all() for value in torch.autograd.grad(output.sum(), (hidden, gate)))


def test_moss_native_reload_and_complete_artifact_preflight(tmp_path):
    model = MossImageTransformer(small_config())
    path = tmp_path / "moss.safetensors"
    save_moss_image(model, path)
    actual = load_moss_image(path)
    assert actual.config == model.config
    for key, value in model.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, atol=0, rtol=0)
    state = dict(model.state_dict())
    for invalid in (
        {k: v for k, v in state.items() if k != "rope.bands"},
        {**state, "unexpected": torch.ones(1)},
        {**state, "video_embedder.weight": torch.full_like(state["video_embedder.weight"], float("nan"))},
    ):
        save_tensors(invalid, path)
        with pytest.raises(ValueError, match="MOSS"):
            load_moss_image(path, config=model.config, source_format="native")


def test_moss_external_fused_qkv_import_preserves_values_and_rejects_mixed_layout(tmp_path):
    model = MossImageTransformer(small_config())
    original = model.state_dict()
    external = {k: v for k, v in original.items() if k != "rope.bands"}
    for index in range(model.config.num_layers):
        prefix = f"layers.{index}.proj."
        external[prefix + "qkv.weight"] = torch.cat(
            [external.pop(prefix + part + ".weight") for part in ("q", "k", "v")]
        )
    path = tmp_path / "external.safetensors"
    prefix = "model.diffusion_model."
    save_tensors({prefix + k: v for k, v in external.items()}, path)
    topology, destination = tmp_path / "topology.json", tmp_path / "native.safetensors"
    topology.write_text(model.config.model_dump_json())
    source_bytes = path.read_bytes()
    assert (
        main(
            [
                "weights",
                "import",
                str(path),
                str(destination),
                "--family",
                "moss-image",
                "--kind",
                "transformer",
                "--topology",
                str(topology),
                "--source-format",
                "external",
                "--prefix",
                prefix,
                "--fused-qkv",
            ]
        )
        == 0
    )
    assert path.read_bytes() == source_bytes
    loaded = load_moss_image(destination)
    for key, value in original.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="requires external"):
        load_moss_image(path, config=model.config, source_format="native", prefix=prefix)
    with pytest.raises(ValueError, match="unselected MOSS fused"):
        load_moss_image(path, config=model.config, source_format="external", prefix=prefix)
    mixed = {**external, "layers.0.proj.q.weight": original["layers.0.proj.q.weight"]}
    save_tensors(mixed, path)
    with pytest.raises(ValueError, match="contains split projection"):
        load_moss_image(path, config=model.config, source_format="external", fused_qkv=True)
    save_tensors(external, path)
    with pytest.raises(ValueError, match="lacks selected prefix"):
        load_moss_image(path, config=model.config, source_format="external", prefix=prefix, fused_qkv=True)


def test_moss_sampling_preserves_explicit_noise_rng_modes_and_fp32_under_autocast():
    from unitunner.models.moss_image.inference.sampling import build_schedule, sample_latents

    model = MossImageTransformer(small_config()).eval()
    noise = torch.randn(2, 8, 2, 3)
    original = noise.clone()
    text = torch.randn(2, 5, 12)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    rng = torch.get_rng_state().clone()
    options = {
        "positive_mask": mask,
        "steps": 4,
        "guidance": 1.0,
        "uncond_skip_layers": (),
        "slg_layers": (1,),
        "slg_scale": 1.0,
    }
    actual = sample_latents(model, noise, text, **options)
    baseline = sample_latents(model, noise, text, positive_mask=mask, steps=4, guidance=1.0, uncond_skip_layers=())
    torch.testing.assert_close(actual, baseline, atol=1e-6, rtol=1e-5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = sample_latents(model, noise, text, **options)
    torch.testing.assert_close(actual, mixed, atol=0, rtol=0)
    torch.testing.assert_close(noise, original, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert not actual.requires_grad and actual.dtype == torch.float32 and not model.training
    times = build_schedule(1500)
    assert times[0] == 1 and times[-1] == 0 and torch.all(times[:-1] > times[1:]) and len(times) <= 1001
    with pytest.raises(ValueError, match="encoded negative"):
        sample_latents(model, noise, text, uncond_skip_layers=())
    with pytest.raises(ValueError, match="skip-layer index"):
        sample_latents(model, noise, text, guidance=1.0)
    with pytest.raises(ValueError, match="right-padded"):
        sample_latents(model, noise, text, guidance=1.0, uncond_skip_layers=(), positive_mask=~mask)
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        sample_latents(model, noise, text)


def test_weight_import_rejects_other_source_shard_and_topology_destination(tmp_path):
    first, second = tmp_path / "model-01-of-02.safetensors", tmp_path / "model-02-of-02.safetensors"
    topology = tmp_path / "config.json"
    save_tensors({"a": torch.ones(1)}, first)
    save_tensors({"b": torch.ones(1)}, second)
    topology.write_text("{}")
    before = {path: path.read_bytes() for path in (first, second, topology)}
    for destination in (second, topology):
        with pytest.raises(ValueError, match="differ from all source files"):
            import_weights(
                first, destination, family="moss-image", kind="transformer", topology=topology, source_format="native"
            )
    assert all(path.read_bytes() == data for path, data in before.items())
