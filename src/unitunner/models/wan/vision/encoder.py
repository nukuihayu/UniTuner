"""Own frozen Wan image conditions and exact native resize/range/normalization order."""

import torch
import torch.nn.functional as F
from torch import Tensor

from unitunner.models.wan.vision.model import WanClipVision

_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_clip_videos(videos: tuple[Tensor, ...], *, image_size: int = 224) -> Tensor:
    """Flatten CTHW rows to frame-major N3SS FP32 native-normalized pixels.

    Inputs contain finite RGB values in [-1,1], share a CPU/CUDA device and may
    have unequal frame counts or image sizes. Bicubic resize precedes conversion
    to [0,1] and normalization, without clamping overshoot or modifying inputs.
    This differentiable boundary uses no antialiasing or visibility masks.
    """
    if type(image_size) is not int or image_size < 1 or not videos:
        raise ValueError("Wan CLIP requires nonempty videos and positive integer image_size")
    device = videos[0].device
    for video in videos:
        if (
            video.ndim != 4
            or video.shape[0] != 3
            or min(video.shape[1:]) < 1
            or video.dtype != torch.float32
            or video.device != device
            or device.type not in ("cpu", "cuda")
            or not torch.isfinite(video).all()
            or video.min() < -1
            or video.max() > 1
        ):
            raise ValueError("Wan CLIP requires same-device finite FP32 CTHW RGB videos in [-1,1]")
    pixels = torch.cat(
        [
            F.interpolate(video.transpose(0, 1), (image_size, image_size), mode="bicubic", align_corners=False)
            for video in videos
        ]
    )
    pixels.mul_(0.5).add_(0.5)
    mean = pixels.new_tensor(_MEAN).view(3, 1, 1)
    std = pixels.new_tensor(_STD).view(3, 1, 1)
    return (pixels - mean) / std


class WanImageEncoder:
    """Own frozen visual extraction; preserve frame order, modes, RNG and residency."""

    def __init__(self, model: WanClipVision) -> None:
        self.model = model

    @torch.no_grad()
    def encode(self, videos: tuple[Tensor, ...]) -> Tensor:
        """Return detached CPU NLD penultimate tokens, concatenating frames in row order.

        Require a frozen eval model. Inputs stay on their original device;
        resize/normalize there before transferring pixels to the model device.
        Disable ambient autocast for the supported FP32 path only.
        """
        if any(module.training for module in self.model.modules()) or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise ValueError("Wan image encoding requires a frozen eval encoder")
        if not videos:
            raise ValueError("Wan CLIP requires nonempty videos")
        with torch.autocast(videos[0].device.type, enabled=False):
            pixels = preprocess_clip_videos(videos, image_size=self.model.config.image_size)
        device = self.model.patch_embedding.weight.device
        with torch.autocast(device.type, enabled=False):
            features = self.model(pixels.to(device))
        if not torch.isfinite(features).all():
            raise ValueError("Wan image encoder must return finite visual features")
        return features.detach().cpu()
