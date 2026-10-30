"""Check fused optimizer numerics and the VAE activation's saved graph storage."""

import pytest
import torch

from unituner.models.components.autoencoder import swish
from unituner.models.z_image.vae import ResidualBlock
from unituner.training.contracts import RunState, StepResult
from unituner.training.engine import update_phase
from unituner.training.optimizers.adamw import create_adamw
from unituner.training.parameters import ParameterGroup


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adamw_matches_native_updates_and_restores_state(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    actual = torch.nn.Parameter(torch.linspace(-1, 1, 64, device=device))
    expected = torch.nn.Parameter(actual.detach().clone())
    optimizer = create_adamw((ParameterGroup("model", (actual,)),), lr=0.002)
    reference = torch.optim.AdamW([expected], lr=0.002, weight_decay=0.01, fused=False, foreach=False)
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["fused"] == (device == "cuda")
    for step in range(3):
        actual.grad = expected.grad = torch.sin(expected.detach() + step)
        optimizer.step()
        reference.step()
    torch.testing.assert_close(actual, expected)
    restored = create_adamw((ParameterGroup("model", (actual,)),), lr=0.002)
    restored.load_state_dict(optimizer.state_dict())
    assert restored.state[actual]["step"] == 3
    torch.testing.assert_close(restored.state[actual]["exp_avg"], reference.state[expected]["exp_avg"])


def test_fused_swish_preserves_values_and_gradients_with_less_saved_storage():
    value = torch.linspace(-8, 8, 1024, requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        actual = swish(value)
    old_saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: old_saved.append(tensor) or tensor, lambda tensor: tensor
    ):
        expected = reference * reference.sigmoid()
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(value.grad, reference.grad)
    assert len({tensor.data_ptr() for tensor in saved}) < len({tensor.data_ptr() for tensor in old_saved})


def test_z_image_residual_matches_unfused_activation_backward():
    torch.manual_seed(11)
    block = ResidualBlock(4, 8, 2)
    value = torch.randn(1, 4, 8, 8, requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    actual = block(value)
    hidden = block.norm1(reference)
    hidden = block.conv1(hidden * hidden.sigmoid())
    hidden = block.norm2(hidden)
    assert block.conv_shortcut is not None
    expected = block.conv2(block.dropout(hidden * hidden.sigmoid())) + block.conv_shortcut(reference)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(value.grad, reference.grad)


def test_fp16_loss_scale_is_removed_before_clipping_and_update():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    state = RunState()
    update_phase(
        [StepResult(parameter.square(), 1)],
        optimizer,
        state,
        precision="fp16",
        fp16_loss_scale=512.0,
        max_grad_norm=1.0,
    )
    torch.testing.assert_close(parameter, torch.tensor(1.9))
    assert state.updates == 1
