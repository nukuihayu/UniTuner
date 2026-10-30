"""Protect published precision boundaries through casts, loading and backward."""

import copy
import math

import pytest
import torch
from torch.nn import functional as F

from unituner.models.qwen_image_2_1.layers import FinalNorm, RMSNorm, TimeEmbedding

DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


@pytest.mark.parametrize("device", DEVICES)
def test_time_features_survive_bf16_and_meta_loading(device):
    model = TimeEmbedding(16).to(device=device, dtype=torch.bfloat16)
    with torch.device("meta"):
        loaded = TimeEmbedding(16)
    loaded.load_state_dict(model.state_dict(), assign=True)
    times = torch.tensor([0.244, 0.751, 0.0], device=device)
    frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32, device="cpu") / 128)
    phase = (1000 * times.float())[:, None] * frequencies.to(device)[None]
    expected_features = torch.cat((phase.cos(), phase.sin()), -1).bfloat16()
    features = []
    handle = loaded.timestep_embedder.register_forward_pre_hook(lambda _module, args: features.append(args[0]))
    try:
        loaded(times, torch.bfloat16)
    finally:
        handle.remove()
    torch.testing.assert_close(features[0], expected_features, atol=0, rtol=0)
    assert loaded.state_dict().keys() == model.state_dict().keys()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weight_dtype", (torch.float32, torch.bfloat16))
def test_qk_norm_matches_published_cast_graph_backward(device, weight_dtype):
    torch.manual_seed(951)
    norm = RMSNorm(16, 1e-6).to(device=device, dtype=weight_dtype)
    values = torch.randn(2, 11, 2, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference_values = values.detach().clone().requires_grad_()
    reference_weight = norm.weight.detach().clone().requires_grad_()
    # Diffusers RMSNorm: variance in FP32, original input on the multiply
    # branch, conversion to half weight dtype BEFORE multiplying that weight.
    variance = reference_values.float().pow(2).mean(-1, keepdim=True)
    expected = reference_values * torch.rsqrt(variance + 1e-6)
    if weight_dtype == torch.bfloat16:
        expected = expected.bfloat16()
    expected = expected * reference_weight
    actual = norm(values)
    upstream = torch.randn_like(expected)
    observed = torch.autograd.grad(actual, (values, norm.weight), upstream)
    reference = torch.autograd.grad(expected, (reference_values, reference_weight), upstream)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(observed, reference, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_target_head_preserves_per_token_bf16_scale_gradients(device):
    torch.manual_seed(123)
    head = FinalNorm(16, 1e-6).to(device)
    reference = copy.deepcopy(head)
    # FP32 residual stream and BF16 autocast modulation are the training case
    # where broadcasting would round after rather than before token reduction.
    values = torch.randn(2, 23, 16, device=device, requires_grad=True)
    time = torch.randn(3, 16, device=device, requires_grad=True)
    joint = values.detach().clone().requires_grad_()
    reference_time = time.detach().clone().requires_grad_()
    mask = torch.arange(23, device=device) >= 11
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = head(values[:, 11:], time[:-1])
        scale = reference.linear(F.silu(reference_time).to(joint.dtype))
        scale = torch.where(mask[None, :, None], scale[:-1, None], scale[-1:, None])
        expected = (reference.norm(joint) * (1 + scale))[:, 11:]
    upstream = torch.randn_like(actual)
    observed = torch.autograd.grad(actual, (values, time, head.linear.weight), upstream)
    reference_grads = torch.autograd.grad(expected, (joint, reference_time, reference.linear.weight), upstream)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # The linear GEMM sees B vs B+1 rows; allow its FP32 accumulation error.
    for a, b in zip(observed, reference_grads, strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
