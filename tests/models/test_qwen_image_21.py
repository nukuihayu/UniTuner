"""Check Qwen Image 2.1 causal visibility, strict artifact IO and adapter ownership."""

from contextlib import ExitStack

import pytest
import torch
from torch.nn import functional as F

from tests.support.assertions import require_type
from unituner.adapters.config import AdapterConfig
from unituner.adapters.loha import LoHa
from unituner.adapters.lokr import LoKr
from unituner.adapters.lora import LoRA
from unituner.artifacts.tensors import save_tensors
from unituner.cli.main import main
from unituner.execution.recompute import RecomputeInstallation
from unituner.execution.regions import ModelRegion
from unituner.models.qwen_image_2_1.config.model import QwenImage21Config
from unituner.models.qwen_image_2_1.layers import segmented_attention
from unituner.models.qwen_image_2_1.loading import load_qwen_image_21, save_qwen_image_21
from unituner.models.qwen_image_2_1.projections import adapter_projections
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.training.parameters import ParameterInstallation


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config():
    return QwenImage21Config(
        in_channels=8,
        out_channels=8,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=12,
        axes_dims_rope=(4, 6, 6),
    )


@pytest.mark.parametrize("repeat_rows", (1, 2))
def test_segmented_attention_matches_independent_dense_visibility_and_gradients(repeat_rows):
    q, k, v = (torch.randn(2 * repeat_rows, 11, 2, 6, dtype=torch.float64, requires_grad=True) for _ in range(3))
    segments = (
        ((0, 2, True), (2, 5, False), (5, 7, True), (7, 11, False)),
        ((0, 3, False), (3, 5, True), (5, 9, False)),
    )
    segments = tuple(row for row in segments for _ in range(repeat_rows))
    # Explicit oracle: text queries are causal, each image is internally bidirectional.
    allowed = torch.zeros(2 * repeat_rows, 11, 11, dtype=torch.bool)
    for index, image_spans, valid_length in ((0, ((2, 5), (7, 11)), 11), (1, ((0, 3), (5, 9)), 9)):
        for repeat in range(repeat_rows):
            row = index * repeat_rows + repeat
            for query in range(valid_length):
                allowed[row, query, : query + 1] = True
                for start, end in image_spans:
                    if start <= query < end:
                        allowed[row, query, :end] = True
    actual = segmented_attention(q, k, v, segments)
    expected = (
        F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed[:, None]
        )
        .transpose(1, 2)
        .flatten(2)
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    upstream = torch.randn_like(actual)
    observed_grads = torch.autograd.grad(actual, (q, k, v), upstream)
    reference_grads = torch.autograd.grad(expected, (q, k, v), upstream)
    for observed, reference in zip(observed_grads, reference_grads, strict=True):
        torch.testing.assert_close(observed, reference, atol=1e-10, rtol=1e-8)
    assert actual[repeat_rows:, 9:].count_nonzero() == 0


def test_native_artifact_import_sequence_padding_and_geometry(tmp_path):
    model = QwenImage21Transformer(small_config())
    source, destination, topology = (
        tmp_path / name for name in ("raw.safetensors", "native.safetensors", "topology.json")
    )
    save_tensors(model.state_dict(), source)
    topology.write_text(model.config.model_dump_json())
    rng = torch.get_rng_state().clone()
    assert (
        main(
            [
                "weights",
                "import",
                str(source),
                str(destination),
                "--family",
                "qwen-image-2-1",
                "--kind",
                "transformer",
                "--topology",
                str(topology),
            ]
        )
        == 0
    )
    restored = load_qwen_image_21(destination)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
    target, text = torch.randn(2, 6, 8), torch.randn(2, 5, 12)
    refs = (torch.randn(2, 2, 8), torch.randn(2, 3, 8))
    grids = (((1, 1, 2), (1, 3, 1), (1, 2, 3)), ((1, 2, 1), (1, 1, 3), (1, 3, 2)))
    options = {"text_lengths": (3, 5), "references": refs, "image_slots": ((0, 2), (2, 2))}
    times = torch.tensor([0.244, 0.751])
    # Reloaded tensor storage can change CPU kernel rounding despite identical weights.
    torch.testing.assert_close(
        model(target, text, times, grids, **options),
        restored(target, text, times, grids, **options),
        atol=1e-6,
        rtol=1e-5,
    )
    batched = model(target, text, times, grids, **options)
    for index in range(2):
        independent = model(
            target[index : index + 1],
            text[index : index + 1, : options["text_lengths"][index]],
            times[index : index + 1],
            (grids[index],),
            references=tuple(ref[index : index + 1] for ref in refs),
            image_slots=(options["image_slots"][index],),
        )
        torch.testing.assert_close(batched[index : index + 1], independent, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="ordered integer"):
        model(target, text, times, grids, **{**options, "image_slots": ((2, 1), (2, 2))})
    with pytest.raises(ValueError, match="ordered integer"):
        model(target, text, times, grids, **{**options, "image_slots": ((0.5, 2), (2, 2))})
    with pytest.raises(ValueError, match="normalized flow"):
        model(target, text, times * 1000, grids, **options)
    accepted = destination.read_bytes()
    invalid = dict(model.state_dict())
    invalid["img_in.weight"] = torch.full_like(invalid["img_in.weight"], float("nan"))
    save_tensors(invalid, source)
    assert (
        main(
            [
                "weights",
                "import",
                str(source),
                str(destination),
                "--family",
                "qwen-image-2-1",
                "--kind",
                "transformer",
                "--topology",
                str(topology),
            ]
        )
        == 2
    )
    assert destination.read_bytes() == accepted
    save_qwen_image_21(model, source)
    with pytest.raises(ValueError, match="topology differs"):
        load_qwen_image_21(source, config=model.config.model_copy(update={"num_layers": 3}))
    invalid = dict(model.state_dict())
    invalid.pop("img_in.weight")
    save_tensors(invalid, source)
    with pytest.raises(ValueError, match="keys differ"):
        load_qwen_image_21(source, config=model.config)


@pytest.mark.parametrize("algorithm", ("lora", "loha", "lokr"))
def test_active_adapters_recompute_and_merged_prediction(algorithm):
    model = QwenImage21Transformer(small_config())
    merged = QwenImage21Transformer(model.config)
    merged.load_state_dict(model.state_dict(), strict=True)
    targets = adapter_projections(model)
    assert len(targets) == 14
    identities, keys = tuple(map(id, model.parameters())), tuple(model.state_dict())
    inputs = (torch.randn(2, 6, 8), torch.randn(2, 5, 12), torch.tensor([0.244, 0.751]), (((1, 2, 3),),))
    options = {"text_lengths": (3, 5)}
    baseline = model(*inputs, **options)
    with ExitStack() as resources:
        owner = ParameterInstallation(
            model, adapter=AdapterConfig(algorithm=algorithm, rank=2, alpha=7.0), targets=targets
        )
        resources.callback(owner.close)
        with torch.no_grad():
            for parameter in owner.branches.parameters():
                parameter.normal_(0, 0.1)
            for index, name in enumerate(targets):
                branch = owner.branches[index]
                assert isinstance(branch, (LoRA, LoHa, LoKr))
                require_type(merged.get_submodule(name), torch.nn.Linear).weight.add_(branch.delta())
        for index, block in enumerate(model.transformer_blocks):
            installation = RecomputeInstallation(ModelRegion(block, index, "visual"))
            resources.callback(installation.close)
        prediction = model(*inputs, **options)
        torch.testing.assert_close(prediction, merged(*inputs, **options), atol=1e-5, rtol=1e-4)
        assert not torch.equal(baseline, prediction)
        prediction.square().mean().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.count_nonzero()
            for parameter in owner.plan.parameters
        )
    assert identities == tuple(map(id, model.parameters())) and keys == tuple(model.state_dict())
    assert torch.equal(baseline, model(*inputs, **options))


def test_published_topology_inventory():
    config = QwenImage21Config()
    with torch.device("meta"):
        model = QwenImage21Transformer(config)
    assert len(model.transformer_blocks) == 32 and config.hidden_size == 4096
    assert model.img_in.weight.shape == (4096, 64)
    assert model.txt_in.in_layer.weight.shape == (4096, 4096)
    assert model.get_parameter("transformer_blocks.0.img_mlp.proj.weight").shape == (12288, 4096)
    assert model.proj_out.weight.shape == (64, 4096)
    assert len(adapter_projections(model)) == 224
    assert all(not name.endswith(".bias") for name in model.state_dict())
