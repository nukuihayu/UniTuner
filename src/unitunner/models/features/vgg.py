"""Construct VGG19 spatial stages with canonical feature-only artifact keys."""

from itertools import pairwise

from torch import nn


class Vgg19Features(nn.Module):
    """Own the non-BN VGG19 convolution stack, excluding the unused classifier.

    Preserve torchvision feature indices and stage cuts (0,4,9,18,27,36).
    Widths are explicit; defaults match published ImageNet VGG19. The final pool
    has no parameters and is unused by FDL. No pretrained weights are implicit.
    """

    def __init__(self, widths: tuple[int, ...] = (64, 128, 256, 512, 512)) -> None:
        super().__init__()
        if len(widths) != 5 or min(widths) <= 0:
            raise ValueError("VGG19 requires five positive stage widths")
        layers: list[nn.Module] = []
        channels = 3
        for width, depth in zip(widths, (2, 2, 4, 4, 4), strict=True):
            for _ in range(depth):
                layers.extend((nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True)))
                channels = width
            layers.append(nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(*layers)
        self.widths = widths

    def feature_stages(self) -> nn.ModuleList:
        """Return five borrowed parameter-sharing stage views for FDL computation.

        Each stage after the first begins with pooling. The caller must not
        register both this model and these views as separate checkpoint owners.
        """
        cuts = (0, 4, 9, 18, 27, 36)
        return nn.ModuleList([self.features[left:right] for left, right in pairwise(cuts)])
