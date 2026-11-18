"""RGB range conversion for native ImageNet-pretrained visual backbones."""

import torch


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """Map RGB BCHW [-1, 1] to ImageNet-normalized FP32 without clipping/detaching.

    Mean/std tensors are created on the input device; gradients flow to image.
    """
    image = image.float()
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return ((image.float() + 1) / 2 - mean) / std
