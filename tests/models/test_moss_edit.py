"""Check isolated reference attention, gradients, cache lifetime and edit artifact contracts."""

import copy

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from unituner.models.moss_image.config.model import MossImageConfig
from unituner.models.moss_image.editing import reference_rope
from unituner.models.moss_image.inference.sampling import sample_latents
from unituner.models.moss_image.layers import MossImageBlock, apply_rope
from unituner.models.moss_image.loading import load_moss_image, save_moss_image
from unituner.models.moss_image.transformer import MossImageTransformer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model():
    torch.manual_seed(119)
    return MossImageTransformer(
        MossImageConfig(
            variant="edit",
            dim=32,
            ffn_dim=48,
            num_heads=2,
            num_layers=3,
            text_dim=12,
            freq_dim=8,
            latent_channels=8,
        )
    )


def test_edit_block_matches_dense_visibility_forward_and_all_gradients():
    torch.manual_seed(18)
    block = MossImageBlock(32, 48, 2)
    dense = copy.deepcopy(block)
    value = torch.randn(2, 9, 32, requires_grad=True)
    rope = torch.randn(2, 9, 16)
    time = torch.randn(2, 6, 32, requires_grad=True)
    zero = torch.randn(2, 6, 32, requires_grad=True)
    with sdpa_kernel(SDPBackend.MATH):
        actual = block(value, rope, time, (6, 6), 3, zero)
        mod = torch.cat((time[:, :, None].expand(-1, -1, 6, -1), zero[:, :, None].expand(-1, -1, 3, -1)), 2)
        x = dense.norm1(value) * (1 + mod[:, 1]) + mod[:, 0]
        q = apply_rope(dense.q_norm(dense.proj.q(x).unflatten(-1, (2, -1))), rope)
        k = apply_rope(dense.k_norm(dense.proj.k(x).unflatten(-1, (2, -1))), rope)
        v = dense.proj.v(x).unflatten(-1, (2, -1))
        visibility = torch.ones(9, 9, dtype=torch.bool)
        visibility[6:, :6] = False
        attention = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), visibility)
        x = value + dense.proj.o(attention.transpose(1, 2).flatten(2)) * mod[:, 2]
        y = dense.norm2(x) * (1 + mod[:, 4]) + mod[:, 3]
        h, g = dense.proj.ffn_up_gate(y).chunk(2, -1)
        expected = x + dense.proj.ffn_down(dense.activation(h, g)) * mod[:, 5]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        upstream = torch.randn_like(actual)
        left = torch.autograd.grad(actual, (value, time, zero, *block.parameters()), upstream, retain_graph=True)
        right = torch.autograd.grad(expected, (value, time, zero, *dense.parameters()), upstream)
        for a, b in zip(left, right, strict=True):
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)


def test_edit_batch_padding_references_and_recompute_gradients():
    from torch.utils.checkpoint import checkpoint

    native = model()
    x = torch.randn(2, 8, 2, 3, requires_grad=True)
    text = torch.randn(2, 5, 12, requires_grad=True)
    refs = (torch.randn(2, 8, 2, 2, requires_grad=True), torch.randn(1, 8, 1, 3, requires_grad=True))
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    times = torch.tensor([200.0, 700.0])
    batched = native(x, times, text, mask, references=refs)
    individual = torch.cat(
        [
            native(x[i : i + 1], times[i : i + 1], text[i : i + 1, :n], references=(refs[0][i : i + 1], refs[1]))
            for i, n in enumerate((2, 5))
        ]
    )
    torch.testing.assert_close(batched, individual, atol=2e-6, rtol=2e-5)
    variables = (x, text, *refs, *native.parameters())
    gradients = torch.autograd.grad(batched.square().sum(), variables, retain_graph=True)
    assert all(g.abs().sum() > 0 for g in gradients[2:4])
    assert not gradients[1][0, 2:].count_nonzero()
    recomputed = checkpoint(
        lambda a, b, c, d: native(a, times, b, mask, references=(c, d)), x, text, *refs, use_reentrant=False
    )
    other = torch.autograd.grad(recomputed.square().sum(), variables)
    for a, b in zip(gradients, other, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("skip", [(), (0,), (1,)])
def test_cached_cfg_slg_matches_live_references_and_rebuilds_after_weight_update(skip):
    native = model().eval()
    noise = torch.randn(1, 8, 2, 3)
    positive, negative = torch.randn(1, 5, 12), torch.randn(1, 2, 12)
    refs = (torch.randn(1, 8, 2, 2), torch.randn(1, 8, 1, 3))

    def sample(cached: bool):
        return sample_latents(
            native,
            noise,
            positive,
            negative=negative,
            references=refs,
            steps=3,
            guidance=3.0,
            uncond_skip_layers=skip,
            slg_layers=(1,),
            slg_scale=0.2,
            slg_start=0.2,
            slg_stop=0.8,
            use_reference_cache=cached,
        )

    outputs = []
    for update in range(2):
        actual = sample(True)
        expected = sample(False)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        outputs.append(actual)
        if update == 0:
            with torch.no_grad():
                block = native.layers[0]
                assert isinstance(block, MossImageBlock)
                block.proj.v.weight.add_(0.01)
    assert not torch.equal(outputs[0], outputs[1])
    with pytest.raises(ValueError, match="inference-only"):
        native._prepare_reference_kv(refs, 1)


def test_reference_positions_and_edit_weight_contract(tmp_path):
    native = model()
    bands = torch.tensor([1.0, 0.01])
    actual = reference_rope(((2, 2), (1, 1)), bands)
    ids = torch.tensor([[1, 0, -0.5, -0.5], [1, 0, -0.5, 0.5], [1, 0, 0.5, -0.5], [1, 0, 0.5, 0.5], [2, 0, 0, 0]])
    phase = ids[..., None] * bands
    torch.testing.assert_close(actual, torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)[None])
    path = tmp_path / "edit.safetensors"
    save_moss_image(native, path)
    restored = load_moss_image(path)
    assert restored.config.variant == "edit"
    assert set(restored.state_dict()) == set(native.state_dict())
    with pytest.raises(ValueError, match="reference images"):
        restored(torch.randn(1, 8, 2, 2), torch.tensor([200.0]), torch.randn(1, 3, 12))
