"""Build portable H3 render fixtures: real VAEs/DiT and an explicit fixed text provider.

Preview lifecycle tests must not depend on previously generated benchmark files.
"""

from pathlib import Path

import torch
from PIL import Image
from pydantic import JsonValue

from tests.models.test_minimax_h3_audio_vae import _vae as audio_vae
from tests.support.media import write_media
from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.config.video_vae import MiniMaxH3VideoDecoderConfig, MiniMaxH3VideoVAEConfig
from unituner.models.minimax_h3.loading import save_minimax_h3
from unituner.models.minimax_h3.text.encoder import MiniMaxH3TextFeatures
from unituner.models.minimax_h3.transformer import MiniMaxH3Transformer
from unituner.models.minimax_h3.vae import MiniMaxH3VideoVAE, save_audio_vae, save_video_vae


class FixedTextEncoder:
    """Return deterministic Lx5120 snapshots without retaining model/visual tensors."""

    def encode(self, presentation):
        features = torch.linspace(-0.1, 0.1, 4 * 5120).reshape(4, 5120)
        features = features + sum(presentation.processor_text.encode("utf-8")) % 31 / 1000
        tags = (
            torch.tensor([0, 1, 0, 1])
            if presentation.images or presentation.videos
            else torch.zeros(4, dtype=torch.int64)
        )
        return MiniMaxH3TextFeatures(features, tags)


def write_render_assets(root: Path):
    """Write small real VAE/transformer artifacts without changing caller RNG state."""
    root.mkdir(parents=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9244)
        video = MiniMaxH3VideoVAE(
            MiniMaxH3VideoVAEConfig(
                channels=32,
                channel_multipliers=(1, 1, 1, 1, 1, 1),
                residual_blocks=(1, 1, 1, 1, 1, 1),
                tile_size=64,
                tile_overlap=32,
                tiling=False,
                decoder=MiniMaxH3VideoDecoderConfig(num_layers=2, num_heads=2, head_dim=8),
            )
        )
        with torch.no_grad():
            for name, value in video.named_parameters():
                if name.endswith(("scale1", "scale2", "register_tokens")):
                    value.fill_(0.1)
            assert video.decoder is not None
            video.decoder.mask_token.fill_(0.02)
        save_video_vae(video, root / "video.safetensors")
        save_audio_vae(audio_vae(), root / "audio.safetensors")
        model = MiniMaxH3Transformer(
            MiniMaxH3Config(
                hidden_size=24,
                num_layers=2,
                token_refiner_num_layers=2,
                num_attention_heads=2,
                attention_head_dim=16,
                ffn_hidden_size=40,
                text_dim=5120,
                timestep_input_dim=8,
                time_embed_hidden_size=16,
                time_embed_dim=8,
                rope_inv_freq_len=2,
            )
        )
        with torch.no_grad():
            model.rope.inv_freq.copy_(torch.tensor([1.0, 0.01]))
        save_minimax_h3(model, root / "model.safetensors")
    for name, color in (("picture", (20, 60, 80)), ("control0", (80, 30, 20)), ("control1", (40, 80, 10))):
        Image.new("RGB", (32, 32), color).save(root / f"{name}.png")
    write_media(
        root / "reference.mkv",
        video_ticks=tuple(range(24, 63)),
        audio_rate=32000,
        audio_start=32960,
        audio_samples=56000,
    )
    write_media(root / "audio.wav", audio_rate=44100, audio_channels=2, audio_samples=16001)
    # Only the injected text provider consumes this placeholder identity. It is
    # deliberately not presented as a valid Qwen3-VL weight file.
    (root / "fixed-text-provider.fixture").write_text("FixedTextEncoder\n")
    references: list[JsonValue] = [
        {"kind": "image", "path": str(root / "picture.png")},
        {"kind": "video", "path": str(root / "reference.mkv")},
    ]
    cases = {}
    for task in ("t2va", "fl2va", "ref2va"):
        for kind in ("single", "video"):
            inputs: dict[str, JsonValue] = {"prompt": "Hello"}
            if task == "fl2va":
                if kind == "single":
                    inputs.update(
                        controls=[
                            {"path": str(root / "control0.png"), "index": 9},
                            {"path": str(root / "control1.png"), "index": 2},
                        ],
                        target_index=7,
                    )
                else:
                    inputs["first_frame"] = str(root / "control0.png")
            if task == "ref2va":
                selected_references: list[JsonValue] = list(references)
                if kind == "video":
                    selected_references.append({"kind": "audio", "path": str(root / "audio.wav")})
                inputs["references"] = selected_references
            cases[f"{kind}-{task}"] = {
                "inputs": inputs,
                "frames": 1 if kind == "single" else (22 if task == "ref2va" else 5),
                "fps": 12 if kind == "video" and task == "ref2va" else 24,
            }
    return {
        "video_weights": str(root / "video.safetensors"),
        "audio_weights": str(root / "audio.safetensors"),
        "model_weights": str(root / "model.safetensors"),
        "text_weights": str(root / "fixed-text-provider.fixture"),
        "text_assets": str(root),
        "processor": str(root),
        "cases": cases,
    }
