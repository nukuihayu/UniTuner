"""Check the released 2D image decoder geometry without heavyweight artifacts."""

import torch
from safetensors.torch import save_file

from unituner.models.minimax_h3.config.video_vae import MiniMaxH3VideoDecoderConfig, MiniMaxH3VideoVAEConfig
from unituner.models.minimax_h3.image_vae import MiniMaxH3ImageVAE, image_vae_config, load_image_vae
from unituner.models.minimax_h3.vae import VideoDecoder


def test_image_decoder_has_one_frame_and_2d_rotary() -> None:
    decoder = VideoDecoder(
        patch_size=2, patch_size_t=1, num_layers=1, heads=2, dim_head=16, rope_dim_ratio=0.5, rope_dims=2
    )
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
        decoder.mask_token.zero_()
    pixels = decoder(torch.randn(1, 24, 1, 2, 3))
    assert pixels.shape == (1, 3, 1, 4, 6)
    assert torch.isfinite(pixels).all()


def test_image_vae_inventory_omits_video_decoder() -> None:
    with torch.device("meta"):
        model = MiniMaxH3ImageVAE()
    state = model.state_dict()
    assert model.config == image_vae_config()
    assert state["post_quant_conv.weight"].ndim == 4
    assert state["decoder.proj_out.weight"].shape[0] == 768
    assert "decoder.mask_token" in state


def test_image_vae_checkpoint_and_decode(tmp_path) -> None:
    config = MiniMaxH3VideoVAEConfig(
        channels=32,
        channel_multipliers=(1,) * 6,
        residual_blocks=(1,) * 6,
        tiling=False,
        token_drop=0,
        decoder=MiniMaxH3VideoDecoderConfig(
            num_layers=1,
            num_heads=2,
            head_dim=16,
            rope_dim_ratio=0.5,
            rope_dims=2,
        ),
    )
    model = MiniMaxH3ImageVAE(config)
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if name not in ("latents_mean", "latents_std"):
                value.zero_()
    path = tmp_path / "vae.safetensors"
    save_file({name: value.contiguous() for name, value in model.state_dict().items()}, path)
    loaded = load_image_vae(path, config=config)
    assert loaded.decoder.pos_embed.n_dim == 2
    pixels = loaded.decode(torch.zeros(1, 24, 1, 2, 2))
    assert pixels.shape == (1, 3, 1, 32, 32)
    assert torch.isfinite(pixels).all()
