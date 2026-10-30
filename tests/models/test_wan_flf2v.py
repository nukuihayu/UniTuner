"""Exercise explicit FLF2V topology, artifact position weights and native recomputation."""

from contextlib import ExitStack

import pytest
import torch
from pydantic import ValidationError
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.support.assertions import require_type
from unituner.artifacts.tensors import save_tensors
from unituner.cli.main import main
from unituner.execution.recompute import RecomputeInstallation
from unituner.execution.regions import ModelRegion
from unituner.models.wan.config.model import WanConfig
from unituner.models.wan.layers import Attention
from unituner.models.wan.loading import load_wan
from unituner.models.wan.transformer import WanTransformer


def test_flf2v_artifact_and_native_recompute(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = WanConfig(
            variant="flf2v",
            hidden_size=48,
            num_heads=4,
            num_layers=2,
            ffn_dim=80,
            timestep_dim=16,
            text_dim=32,
            in_channels=36,
        )
        model = WanTransformer(config)
        with torch.no_grad():
            model.head.head.weight.normal_(std=0.03)
            model.img_emb.emb_pos.normal_(std=0.1)
        external, topology, native = (
            tmp_path / "external.safetensors",
            tmp_path / "topology.json",
            tmp_path / "native.safetensors",
        )
        save_tensors(model.state_dict(), external)
        topology.write_text(config.model_dump_json())
        rng = torch.get_rng_state().clone()
        assert (
            main(
                [
                    "weights",
                    "import",
                    str(external),
                    str(native),
                    "--family",
                    "wan",
                    "--kind",
                    "transformer",
                    "--topology",
                    str(topology),
                ]
            )
            == 0
        )
        loaded = load_wan(native)
        assert config == loaded.config and torch.equal(rng, torch.get_rng_state())
        assert model.state_dict().keys() == loaded.state_dict().keys()
        for name, value in model.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[name])
        assert model.img_emb.emb_pos.shape == (1, 514, 1280)
        assert require_type(model.blocks[0].cross_attn, Attention).image_token_count == 514
        assert WanConfig(variant="i2v").image_token_count == 257
        inputs = {
            "clip": torch.randn(16, 2, 4, 4),
            "condition": torch.randn(20, 2, 4, 4),
            "caption": torch.randn(3, 32),
            "images": torch.randn(1, 2, 257, 1280),
            "time": torch.tensor([244.0]),
        }
        observations = []
        for current, recompute in [(loaded, False), (load_wan(native), True)]:
            current.train()
            values = {name: value.clone().requires_grad_() for name, value in inputs.items()}
            with ExitStack() as owners, sdpa_kernel(SDPBackend.MATH):
                if recompute:
                    for index, block in enumerate(current.blocks):
                        owners.callback(RecomputeInstallation(ModelRegion(block, index, "visual")).close)
                result = current(
                    (values["clip"],),
                    values["time"],
                    (values["caption"],),
                    sequence_length=8,
                    conditioning=(values["condition"],),
                    image_features=values["images"],
                )[0]
                result.square().mean().backward()
            gradients = {}
            for name, value in [*current.named_parameters(), *values.items()]:
                assert value.grad is not None and torch.isfinite(value.grad).all(), name
                gradients[name] = value.grad.detach().clone()
            assert values["images"].grad is not None
            assert torch.count_nonzero(values["images"].grad[:, 0]) and torch.count_nonzero(values["images"].grad[:, 1])
            observations.append((result.detach(), gradients))
        assert torch.equal(observations[0][0], observations[1][0])
        assert observations[0][1].keys() == observations[1][1].keys()
        for name, value in observations[0][1].items():
            assert torch.equal(value, observations[1][1][name]), name
        with pytest.raises(ValueError, match="image_features shape"):
            model(
                (inputs["clip"],),
                inputs["time"],
                (inputs["caption"],),
                sequence_length=8,
                conditioning=(inputs["condition"],),
                image_features=inputs["images"].flatten(0, 1),
            )
        broken = dict(model.state_dict())
        broken.pop("img_emb.emb_pos")
        save_tensors(broken, external)
        with pytest.raises(ValueError, match="keys differ"):
            load_wan(external, config=config)
        with pytest.raises(ValidationError, match=r"version2\.1"):
            WanConfig(version="2.2", variant="flf2v")
    finally:
        torch.set_num_threads(previous)
