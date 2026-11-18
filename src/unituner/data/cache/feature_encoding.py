"""Encode exact native target crops with an owned frozen DINOv3 ViT."""

from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from unituner.artifacts.hashing import file_sha256
from unituner.config.features import RepaConfig
from unituner.data.cache.representation import RepaEntry, RepaManifest, save_repa_manifest, save_repa_target
from unituner.data.cache.storage import CacheEntry
from unituner.models.features.loading import load_dino_vit
from unituner.models.features.vit import DinoViT


class RepaCacheWriter:
    """Own a frozen encoder and publish targets from the VAE's identical spatial/temporal crop."""

    def __init__(self, config: RepaConfig) -> None:
        if config.encoder_weights is None:
            raise ValueError("REPA cache production requires method.repa.encoder_weights")
        self.path: Path = Path(config.encoder_weights)
        if file_sha256(self.path) != config.encoder_sha256:
            raise ValueError("REPA encoder weight digest differs from method.repa.encoder_sha256")
        self.config: RepaConfig = config
        self.device: torch.device = torch.device(config.encoder_device)
        self.model: DinoViT = load_dino_vit(self.path, device=self.device)
        if self.model.config.width != config.encoder_dim:
            raise ValueError("REPA encoder output width differs from method.repa.encoder_dim")
        self.entries: dict[str, RepaEntry] = {}

    @torch.inference_mode()
    def publish(self, source: CacheEntry, pixels: Tensor) -> None:
        """Encode uint8 THWC RGB(A) crops, retaining all frames and excluding prefix tokens.

        Resize the full crop using the original bicubic/antialias path and ImageNet
        normalization. One frame at a time bounds activations. Alpha is excluded
        from encoder input; it remains part of native cache identity and VAE
        supervision. No latent decoding or independent crop is introduced.
        """
        if pixels.ndim != 4 or pixels.dtype != torch.uint8 or pixels.shape[-1] not in (3, 4) or min(pixels.shape) < 1:
            raise ValueError("REPA encoder requires nonempty uint8 THWC RGB(A) crops")
        if source.sha256 in self.entries:
            return
        size, patch = self.model.config.image_size, self.model.config.patch
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        features = []
        with torch.autocast(self.device.type, enabled=False):
            for frame in pixels:
                value = frame[..., :3].permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=torch.float32) / 255
                value = F.interpolate(value, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
                tokens = self.model((value - mean) / std)
                features.append(tokens.reshape(size // patch, size // patch, self.config.encoder_dim).cpu())
        if file_sha256(self.path) != self.config.encoder_sha256:
            raise ValueError("REPA encoder artifact changed during cache production")
        self.entries[source.sha256] = save_repa_target(
            Path(self.config.cache_directory), source, torch.stack(features), encoder_sha256=self.config.encoder_sha256
        )

    def publish_normalized(self, source: CacheEntry, pixels: Tensor) -> None:
        """Invert native uint8/127.5-1 CTHW preprocessing with nearest-byte rounding."""
        if pixels.ndim != 4 or pixels.shape[0] not in (3, 4) or not pixels.is_floating_point():
            raise ValueError("REPA normalized crops require native floating CTHW RGB(A)")
        self.publish(source, ((pixels.detach().cpu() + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0))

    def finish(self) -> None:
        """Publish the complete collected feature inventory after successful native caching."""
        save_repa_manifest(
            RepaManifest(
                encoder_sha256=self.config.encoder_sha256,
                encoder_dim=self.config.encoder_dim,
                entries=tuple(self.entries.values()),
            ),
            Path(self.config.cache_directory),
        )
