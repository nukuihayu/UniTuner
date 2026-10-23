"""Anti-aliased multilevel visual discriminator with a shared frozen backbone.

Head topology follows equations 8-15 of arXiv:2508.08227v2.
Inputs are RGB BCHW images in [-1, 1]; only spectral-normalized heads train.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from unitunner.models.features.convnext import DinoConvNeXt
from unitunner.models.features.normalization import normalize_image


class BlurPool(nn.Module):
    """Depthwise binomial low-pass filter with stride two and fixed buffer weights."""

    kernel: torch.Tensor

    def __init__(self, channels: int) -> None:
        super().__init__()
        window = torch.tensor([1.0, 3.0, 3.0, 1.0])
        self.register_buffer("kernel", (window[:, None] * window[None, :] / 64).expand(channels, 1, 4, 4).clone())

    def compute(self, image: torch.Tensor) -> torch.Tensor:
        """Return depthwise BCHW low-pass features at half spatial resolution."""
        return F.conv2d(image, self.kernel.to(image.dtype), padding=1, stride=2, groups=image.shape[1])

    forward = compute


def augment(image: torch.Tensor) -> torch.Tensor:
    """Differentiable color, zero-padded integer translation and cutout (DiffAugment)."""
    batch, _, h, w = image.shape
    image = image + (torch.rand(batch, 1, 1, 1, device=image.device, dtype=image.dtype) - 0.5)
    mean = image.mean(1, keepdim=True)
    image = (image - mean) * (2 * torch.rand(batch, 1, 1, 1, device=image.device, dtype=image.dtype)) + mean
    mean = image.mean((1, 2, 3), keepdim=True)
    image = (image - mean) * (torch.rand(batch, 1, 1, 1, device=image.device, dtype=image.dtype) + 0.5) + mean
    dy = torch.randint(-int(h * 0.125 + 0.5), int(h * 0.125 + 0.5) + 1, (batch, 1, 1), device=image.device)
    dx = torch.randint(-int(w * 0.125 + 0.5), int(w * 0.125 + 0.5) + 1, (batch, 1, 1), device=image.device)
    y = torch.arange(h, device=image.device)[None, :, None] + dy
    x = torch.arange(w, device=image.device)[None, None, :] + dx
    valid = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    image = image.permute(0, 2, 3, 1)[
        torch.arange(batch, device=image.device)[:, None, None], y.clamp(0, h - 1), x.clamp(0, w - 1)
    ]
    image = image.permute(0, 3, 1, 2) * valid[:, None]
    cutout_h, cutout_w = int(h * 0.5 + 0.5), int(w * 0.5 + 0.5)
    cy = torch.randint(0, h + (1 - cutout_h % 2), (batch, 1, 1), device=image.device)
    cx = torch.randint(0, w + (1 - cutout_w % 2), (batch, 1, 1), device=image.device)
    # Trainer clamps each cutout index at the boundary, including even-size
    # centers just outside the image. Preserve those exact asymmetric extents.
    y = (torch.arange(cutout_h, device=image.device)[None, :, None] + cy - cutout_h // 2).clamp(0, h - 1)
    x = (torch.arange(cutout_w, device=image.device)[None, None, :] + cx - cutout_w // 2).clamp(0, w - 1)
    keep = torch.ones(batch, h, w, device=image.device, dtype=image.dtype)
    keep[torch.arange(batch, device=image.device)[:, None, None], y, x] = 0
    return image * keep[:, None]


class MultilevelDiscriminator(nn.Module):
    """Frozen first-three-stage DINO backbone and trainable spectral-normalized heads."""

    def __init__(
        self, backbone: DinoConvNeXt, resolution: int, diffaug: bool = True, *, bf16_compute: bool = False
    ) -> None:
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.diffaug = diffaug
        self.bf16_compute = bf16_compute
        if resolution >= 1024:
            ratios = ((0.5, 1.0, 1.0, 1.0), (0.5, 0.5, 1.0, 1.0), (0.5, 0.5, 0.5, 1.0))
            strides = ((1, 2, 2, 2), (1, 1, 2, 2), (1, 1, 1, 2))
        else:
            ratios = ((0.5, 1.0, 1.0), (0.5, 0.5, 1.0), (0.5, 0.5, 0.5))
            strides = ((2, 2, 2), (1, 2, 2), (1, 1, 2))
        self._head_strides = strides
        self.validate_resolution(resolution, resolution)
        heads = []
        for width, levels, steps in zip(backbone.embed_dims[:3], ratios, strides, strict=True):
            layers = []
            for ratio, stride in zip(levels, steps, strict=True):
                output = max(1, int(width * ratio))
                layers.extend(
                    (BlurPool(width), spectral_norm(nn.Conv2d(width, output, 3, stride, 1)), nn.LeakyReLU(0.2))
                )
                width = output
            layers.extend((BlurPool(width), spectral_norm(nn.Conv2d(width, 1, 1))))
            heads.append(nn.Sequential(*layers))
        self.heads = nn.ModuleList(heads)

    def train(self, mode: bool = True) -> "MultilevelDiscriminator":
        """Set head modes while keeping the shared frozen backbone in evaluation mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def validate_resolution(self, height: int, width: int) -> None:
        """Reject grids that collapse inside the original anti-aliased head geometry."""
        for side in (height, width):
            for index, strides in enumerate(self._head_strides):
                size = side // (4 * 2**index)
                for stride in strides:
                    size //= 2
                    if size < 1:
                        raise ValueError("Image resolution is too small for the discriminator head geometry")
                    size = (size - 1) // stride + 1
                if size // 2 < 1:
                    raise ValueError("Image resolution is too small for the discriminator head geometry")

    def compute(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return three spatial logits; preserve pixels and head parameter gradients."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Discriminator requires RGB BCHW images")
        self.validate_resolution(*image.shape[-2:])
        with torch.autocast(image.device.type, dtype=torch.bfloat16, enabled=self.bf16_compute, cache_enabled=False):
            image = augment(image.float()) if self.diffaug else image.float()
            features = self.backbone.features(normalize_image(image))
            return tuple(head(value) for head, value in zip(self.heads, features, strict=True))

    forward = compute
