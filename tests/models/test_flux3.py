"""Check native packed FLUX3 sample isolation, flow-map time rows and strict artifacts."""

from dataclasses import replace

import pytest
import torch

from unitunner.artifacts.tensors import save_tensors
from unitunner.cli.main import main
from unitunner.models.flux3.conditioning import Flux3Sample, image_tokens, pack_samples, reference_tokens, text_tokens
from unitunner.models.flux3.config.model import Flux3Config
from unitunner.models.flux3.loading import load_flux3, save_flux3
from unitunner.models.flux3.transformer import Flux3Transformer


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(variant):
    return Flux3Config(
        variant=variant,
        hidden_size=32,
        num_heads=2,
        depth=2,
        depth_single=2,
        axes_dim=(4, 4, 4, 4),
        image_channels=8,
        text_channels=12,
        mlp_ratio=2.0,
    )


@pytest.mark.parametrize("variant", ["base", "distilled"])
def test_flux3_samples_and_empty_reference_do_not_leak_attention_or_round_time(variant):
    torch.manual_seed(21)
    model = Flux3Transformer(config(variant))
    image = torch.randn(8, 2, 3, requires_grad=True)
    reference = torch.randn(8, 1, 2, requires_grad=True)
    samples = [
        Flux3Sample(image_tokens(image), text_tokens(torch.randn(3, 12)), reference_tokens([reference])),
        Flux3Sample(image_tokens(torch.randn(8, 1, 2)), text_tokens(torch.randn(5, 12))),
    ]
    sigma = torch.tensor([0.244, 0.751], requires_grad=True)
    destinations = torch.tensor([0.1, 0.4], requires_grad=True) if variant == "distilled" else None
    packed = pack_samples(samples, sigma)
    assert packed.data["image"].shape == (8, 8)
    assert packed.bounds["image_cond"] == ((0, 2), (2, 2))
    assert packed.timesteps["image"].dtype == torch.float32
    torch.testing.assert_close(packed.timesteps["image"][:6], sigma[0].expand(6), atol=0, rtol=0)
    assert not torch.equal(sigma, sigma.to(torch.bfloat16).float())
    torch.testing.assert_close(
        samples[0].image.ids[:, 1:3], torch.tensor([[r / 2, c / 2] for r in range(2) for c in range(3)]), atol=0, rtol=0
    )
    assert samples[0].references is not None
    assert torch.all(samples[0].references.ids[:, 3] == -1)
    rng = torch.get_rng_state().clone()
    actual = model(packed, destinations)
    separate = torch.cat(
        [
            model(
                pack_samples([sample], sigma[index : index + 1]),
                None if destinations is None else destinations[index : index + 1],
            )
            for index, sample in enumerate(samples)
        ]
    )
    torch.testing.assert_close(actual, separate, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    grad_image, grad_reference = torch.autograd.grad(actual[:6].square().mean(), (image, reference), retain_graph=True)
    assert grad_image.norm() > 0 and grad_reference.norm() > 0
    actual[6:].square().mean().backward()
    torch.testing.assert_close(image.grad, torch.zeros_like(image), atol=0, rtol=0)
    torch.testing.assert_close(reference.grad, torch.zeros_like(reference), atol=0, rtol=0)
    empty = pack_samples([samples[1]], sigma[1:])
    assert model(empty, None if destinations is None else destinations[1:]).shape == (2, 8)
    with pytest.raises(ValueError, match="bounds"):
        model(replace(packed, bounds={**packed.bounds, "image": ((0, 6), (5, 8))}), destinations)
    with pytest.raises(ValueError, match="destination"):
        model(packed, torch.zeros(3) if variant == "distilled" else torch.zeros(2))


def test_flux3_native_reload_import_and_shape_value_contracts(tmp_path):
    topology = config("distilled")
    model = Flux3Transformer(topology)
    path = tmp_path / "model.safetensors"
    save_flux3(model, path)
    rng = torch.get_rng_state().clone()
    loaded = load_flux3(path)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert loaded.config == topology
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
    external = tmp_path / "external.safetensors"
    save_tensors(model.state_dict(), external)
    with pytest.raises(ValueError, match="keys differ"):
        load_flux3(external)
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(topology.model_dump_json())
    imported = tmp_path / "imported.safetensors"
    assert (
        main(
            [
                "weights",
                "import",
                str(external),
                str(imported),
                "--family",
                "flux3",
                "--kind",
                "transformer",
                "--topology",
                str(topology_path),
                "--source-format",
                "native",
            ]
        )
        == 0
    )
    assert load_flux3(imported).config == topology
    for values in (
        {key: value for key, value in model.state_dict().items() if key != "txt_in.weight"},
        {**model.state_dict(), "txt_in.weight": torch.zeros(1)},
        {**model.state_dict(), "txt_in.weight": torch.full_like(model.txt_in.weight, float("nan"))},
    ):
        save_tensors(values, external)
        with pytest.raises(ValueError, match="FLUX3"):
            load_flux3(external, config=topology)
    with pytest.raises(ValueError, match="differs"):
        load_flux3(path, config=topology.model_copy(update={"variant": "base"}))
