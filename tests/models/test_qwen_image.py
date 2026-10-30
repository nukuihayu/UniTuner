"""Verify Qwen Image artifact ownership, active adapter paths and native positions."""

from contextlib import closing

import pytest
import torch

from tests.support.assertions import require_value
from unituner.adapters.config import AdapterConfig
from unituner.artifacts.tensors import save_tensors
from unituner.cli.main import main
from unituner.models.qwen_image.config.model import QwenImageConfig
from unituner.models.qwen_image.loading import load_qwen_image, save_qwen_image
from unituner.models.qwen_image.positions import build_rotary
from unituner.models.qwen_image.projections import adapter_projections
from unituner.models.qwen_image.transformer import QwenImageTransformer
from unituner.training.parameters import ParameterInstallation


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("variant,algorithm", [("original", "lora"), ("edit-2511", "loha"), ("layered", "lokr")])
def test_qwen_image_strict_artifacts_and_active_adapters(tmp_path, variant, algorithm):
    config = QwenImageConfig(
        variant=variant,
        in_channels=8,
        out_channels=2,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=12,
        axes_dims_rope=(4, 6, 6),
    )
    model = QwenImageTransformer(config)
    path = tmp_path / "model.safetensors"
    rng = torch.get_rng_state().clone()
    save_qwen_image(model, path)
    source = tmp_path / "external.safetensors"
    topology = tmp_path / "topology.json"
    save_tensors(model.state_dict(), source)
    topology.write_text(config.model_dump_json())
    imported = tmp_path / "imported.safetensors"
    assert (
        main(
            [
                "weights",
                "import",
                str(source),
                str(imported),
                "--family",
                "qwen-image",
                "--kind",
                "transformer",
                "--topology",
                str(topology),
            ]
        )
        == 0
    )
    assert load_qwen_image(imported).config == config
    restored = load_qwen_image(path)
    assert torch.equal(rng, torch.get_rng_state())
    assert restored.config == config
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
    values = (torch.randn(2, 8, 8), torch.randn(2, 5, 12), torch.tensor([0.244, 0.751]), ((1, 2, 3), (1, 1, 2)))
    options = {
        "text_mask": torch.arange(5)[None] < torch.tensor([3, 5])[:, None],
        "layer_condition": torch.tensor([0, 1]) if variant == "layered" else None,
    }
    expected = model(*values, **options)
    assert torch.equal(expected, restored(*values, **options))
    targets = adapter_projections(restored)
    assert len(targets) == 20
    assert "transformer_blocks.1.attn.add_q_proj" not in targets
    assert "transformer_blocks.1.attn.add_k_proj" in targets
    assert "transformer_blocks.1.attn.to_add_out" not in targets
    identities, keys = tuple(map(id, restored.parameters())), tuple(restored.state_dict())
    with closing(
        ParameterInstallation(restored, adapter=AdapterConfig(algorithm=algorithm, rank=2, alpha=2.0), targets=targets)
    ) as owner:
        with torch.no_grad():
            for parameter in owner.branches.parameters():
                parameter.normal_(0, 0.03)
        actual = restored(*values, **options)
        assert not torch.equal(expected, actual)
        actual.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in owner.plan.parameters)
        assert all(torch.count_nonzero(require_value(p.grad)) for p in owner.plan.parameters)
    assert identities == tuple(map(id, restored.parameters())) and keys == tuple(restored.state_dict())
    assert torch.equal(expected, restored(*values, **options))
    weights = model.state_dict()
    weights["img_in.weight"] = torch.full_like(weights["img_in.weight"], float("nan"))
    save_tensors(weights, path)
    with pytest.raises(ValueError, match="invalid tensor"):
        load_qwen_image(path, config=config)
    weights = model.state_dict()
    weights.pop("img_in.weight")
    save_tensors(weights, path)
    with pytest.raises(ValueError, match="keys differ"):
        load_qwen_image(path, config=config)
    save_qwen_image(model, path)
    with pytest.raises(ValueError, match="differs from artifact"):
        load_qwen_image(path, config=config.model_copy(update={"joint_attention_dim": 14}))


def test_qwen_image_published_inventory_and_literal_rotary_bounds():
    for variant in ("original", "edit", "edit-2509", "edit-2511", "layered"):
        config = QwenImageConfig(variant=variant)
        with torch.device("meta"):
            model = QwenImageTransformer(config)
        assert len(model.transformer_blocks) == 60
        assert model.img_in.weight.shape == (3072, 64)
        assert model.txt_in.weight.shape == (3072, 3584)
        assert model.proj_out.weight.shape == (64, 3072)
        assert ("time_text_embed.addition_t_embedding.weight" in model.state_dict()) == (variant == "layered")
        assert config.zero_reference_time == (variant == "edit-2511")
    cpu = torch.device("cpu")
    grids = ((1, 2, 3), (1, 1, 2))
    image, text = build_rotary(grids, 3, (2, 2, 2), layered=False, device=cpu)
    positions = torch.tensor(
        [[0, -1, -2], [0, -1, -1], [0, -1, 0], [0, 0, -2], [0, 0, -1], [0, 0, 0], [1, -1, -1], [1, -1, 0]]
    ).float()
    assert torch.equal(image, torch.polar(torch.ones_like(positions), positions))
    assert torch.equal(text, torch.polar(torch.ones(3, 3), torch.arange(1, 4).float()[:, None].expand(-1, 3)))
    layered, _ = build_rotary(grids, 3, (2, 2, 2), layered=True, device=cpu)
    assert torch.equal(layered[-2:, 0], torch.polar(torch.ones(2), -torch.ones(2)))
    rng = torch.get_rng_state().clone()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        alternate = build_rotary(grids, 3, (2, 2, 2), layered=False, device=cpu)
    finally:
        torch.set_default_dtype(previous)
    assert torch.equal(image, alternate[0]) and torch.equal(text, alternate[1])
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="bound4096"):
        build_rotary(((1, 2, 2),), 4096, (2, 2, 2), layered=False, device=cpu)
    with pytest.raises(ValueError, match="one frame"):
        build_rotary(((2, 1, 1),), 3, (2, 2, 2), layered=True, device=cpu)
