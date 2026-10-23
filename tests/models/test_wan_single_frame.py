"""Exercise explicit temporal binding, independent frame histories and VAE-local placeholders."""

import pytest
import torch

from unitunner.models.qwen_image.config.vae import QwenImageVAEConfig
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.models.wan.single_frame import build_single_frame_layout, encode_single_frame_conditions


def test_single_frame_binding_gradients_and_vae_local_neutral_condition():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vae = QwenImageVAE(QwenImageVAEConfig(base_channels=4, layers_per_block=2)).eval()
        target = torch.zeros(2, 3, 1, 16, 16).requires_grad_(True)
        clean = torch.linspace(-1, 1, 2 * 3 * 2 * 16 * 16).reshape(2, 3, 2, 16, 16).requires_grad_(True)
        rng = torch.get_rng_state().clone()
        first = encode_single_frame_conditions(vae, target, clean, clean_indices=(24, 0), target_index=12)
        second = encode_single_frame_conditions(vae, target, clean.flip(2), clean_indices=(0, 24), target_index=12)
        assert first.layout.frame_indices == (0, 12, 24) and first.layout.clean_order == (1, 0)
        assert first.layout.target_position == 1
        assert torch.equal(first.latents, second.latents) and torch.equal(first.image_condition, second.image_condition)
        assert first.latents.shape == (2, 16, 3, 2, 2) and first.image_condition.shape == (2, 20, 3, 2, 2)
        assert torch.equal(first.image_condition[:, 4:, :1], first.latents[:, :, :1])
        assert torch.count_nonzero(first.image_condition[:, :4, 1]) == 0
        loss = first.latents.square().mean() + first.image_condition.square().mean()
        loss.backward()
        assert target.grad is not None and clean.grad is not None
        assert torch.isfinite(target.grad).all() and torch.isfinite(clean.grad).all()
        assert torch.count_nonzero(target.grad) and torch.count_nonzero(clean.grad)
        with torch.no_grad():
            assert vae.quant_conv.bias is not None
            vae.quant_conv.bias.add_(0.25)
            changed = encode_single_frame_conditions(vae, target, clean, clean_indices=(24, 0), target_index=12)
            neutral = vae.encode(torch.zeros_like(target[:1]))
        assert not torch.equal(first.image_condition[:, 4:, 1:2], changed.image_condition[:, 4:, 1:2])
        assert torch.equal(changed.image_condition[:, 4:, 1:2], neutral.expand(2, -1, -1, -1, -1))
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("indices,target", [((), 1), ((0, 0), 1), ((0,), 0), ((0, 1024), 1), ((False,), 1), ((0,), -1)])
def test_reject_ambiguous_temporal_coordinates(indices, target):
    with pytest.raises(ValueError, match=r"one/two|distinct"):
        build_single_frame_layout(indices, target)
