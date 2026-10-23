"""Compare encoder adapters with merged-weight equations and verify ownership."""

import copy
from contextlib import closing

import pytest
import torch
from torch import nn

from unitunner.adapters.encoder import EncoderAdapters, EncoderWeightLoRA, load_encoder_adapters
from unitunner.artifacts.tensors import save_tensors
from unitunner.models.qwen_image.vae import CausalConv3d


@pytest.mark.parametrize(
    "projection,shape",
    [
        (nn.Linear(4, 6), (2, 3, 4)),
        (nn.Conv2d(4, 6, 3, padding=1, groups=2), (2, 4, 5, 6)),
        (nn.Conv3d(2, 4, 3, padding=1), (1, 2, 3, 4, 5)),
    ],
)
def test_encoder_adapter_matches_merged_weight_value_and_all_gradients(projection, shape):
    torch.manual_seed(38)
    reference = copy.deepcopy(projection).requires_grad_(False)
    base_state = {key: value.clone() for key, value in projection.state_dict().items()}
    parameter_ids = [id(parameter) for parameter in projection.parameters()]
    inputs = torch.randn(shape, requires_grad=True)
    installation = EncoderAdapters(projection, rank=2)
    branch = installation.layers[0]
    assert isinstance(branch, EncoderWeightLoRA)
    with torch.no_grad():
        branch.up.normal_(std=0.1)
    actual = projection(inputs)
    weight = reference.weight + branch.delta()
    if isinstance(reference, nn.Linear):
        expected = torch.nn.functional.linear(inputs, weight, reference.bias)
    elif isinstance(reference, nn.Conv2d):
        expected = torch.nn.functional.conv2d(
            inputs, weight, reference.bias, reference.stride, reference.padding, reference.dilation, reference.groups
        )
    else:
        expected = torch.nn.functional.conv3d(
            inputs, weight, reference.bias, reference.stride, reference.padding, reference.dilation, reference.groups
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    variables = (inputs, *installation.parameters())
    actual_grad = torch.autograd.grad(actual.square().mean(), variables, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().mean(), variables)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    assert [id(parameter) for parameter in projection.parameters()] == parameter_ids
    for key, value in projection.state_dict().items():
        torch.testing.assert_close(value, base_state[key], rtol=0, atol=0)
    with installation.unadapted():
        with installation.unadapted():
            torch.testing.assert_close(projection(inputs), reference(inputs), rtol=0, atol=0)
        assert not branch.enabled
    assert branch.enabled
    installation.close()
    installation.close()
    assert all(parameter.requires_grad for parameter in projection.parameters())
    torch.testing.assert_close(projection(inputs), reference(inputs), rtol=0, atol=0)


def test_partial_encoder_install_failure_restores_prior_forwards_and_flags():
    model = nn.Sequential(nn.Linear(3, 3), nn.Conv2d(3, 3, 3, padding=1, padding_mode="reflect"))
    model[0].bias.requires_grad_(False)
    flags = [parameter.requires_grad for parameter in model.parameters()]
    with pytest.raises(ValueError, match="padding"):
        EncoderAdapters(model, rank=2)
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert all("forward" not in layer.__dict__ for layer in model)


def test_encoder_artifact_reload_matches_nonzero_merged_weights_and_rejects_invalid_inputs(tmp_path):
    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 3))
    reference = copy.deepcopy(model).requires_grad_(False)
    inputs = torch.randn(2, 4)
    state = {}
    for index, layer in enumerate(reference):
        assert isinstance(layer, nn.Linear)
        down = torch.randn(2, layer.in_features) * 0.1
        up = torch.randn(layer.out_features, 2) * 0.1
        state[f"layers.{index}.down"] = down
        state[f"layers.{index}.up"] = up
        with torch.no_grad():
            layer.weight.add_(up @ down)
    path = tmp_path / "encoder.safetensors"
    metadata = {"unitunner.encoder.targets": '["0", "1"]', "unitunner.encoder.rank": "2"}
    save_tensors(state, path, metadata=metadata)
    ids = [id(p) for p in model.parameters()]
    owner = load_encoder_adapters(model, path)
    try:
        torch.testing.assert_close(model(inputs), reference(inputs), atol=1e-6, rtol=1e-5)
        assert not any(p.requires_grad for p in owner.parameters())
        assert [id(p) for p in model.parameters()] == ids
    finally:
        owner.close()
    assert all(p.requires_grad for p in model.parameters())
    baseline = model(inputs).detach()
    variants = [
        ({**state, "unexpected": torch.ones(1)}, metadata),
        ({**state, "layers.0.up": torch.ones(1, 2)}, metadata),
        ({**state, "layers.0.up": torch.full((6, 2), float("nan"))}, metadata),
        (state, {**metadata, "unitunner.encoder.targets": '["1", "0"]'}),
        (state, {**metadata, "unitunner.encoder.rank": "0"}),
    ]
    for tensors, info in variants:
        save_tensors(tensors, path, metadata=info)
        with pytest.raises(ValueError, match="Encoder artifact"):
            load_encoder_adapters(model, path)
        torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)
        assert all("forward" not in layer.__dict__ for layer in model)


def test_encoder_causal_convolution_preserves_history_padding_and_all_gradients():
    """Compare the adapter with an independent merged kernel after native history padding."""
    torch.manual_seed(827)
    convolution = CausalConv3d(2, 4, 3, padding=1)
    state = {name: value.detach().clone() for name, value in convolution.state_dict().items()}
    identities = [id(value) for value in convolution.parameters()]
    image = torch.randn(1, 2, 1, 4, 6, requires_grad=True)
    history = torch.randn(1, 2, 2, 4, 6, requires_grad=True)
    with closing(EncoderAdapters(convolution, rank=2)) as owner:
        branch = owner.layers[0]
        assert isinstance(branch, EncoderWeightLoRA)
        with torch.no_grad():
            branch.up.normal_(std=0.03)
        actual = convolution(image, history)
        padded = torch.nn.functional.pad(torch.cat((history, image), dim=2), (1, 1, 1, 1, 0, 0))
        expected = torch.nn.functional.conv3d(padded, convolution.weight + branch.delta(), convolution.bias)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        variables = (image, history, *owner.parameters())
        observed = torch.autograd.grad(actual.square().mean(), variables, retain_graph=True)
        reference = torch.autograd.grad(expected.square().mean(), variables)
        for value, target in zip(observed, reference, strict=True):
            torch.testing.assert_close(value, target, atol=1e-6, rtol=1e-5)
        assert [id(value) for value in convolution.parameters()] == identities
        assert all(torch.equal(value, state[name]) for name, value in convolution.state_dict().items())
    assert "_conv_forward" not in convolution.__dict__ and "forward" not in convolution.__dict__
