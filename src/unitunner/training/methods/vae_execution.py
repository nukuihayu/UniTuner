"""Install restoration-only recomputation on stateless VAE image blocks."""

from torch import nn

from unitunner.execution.recompute import RecomputeInstallation
from unitunner.execution.regions import ModelRegion
from unitunner.models.components.autoencoder import Downsample2d, ResidualConvBlock, Upsample2d
from unitunner.models.components.autoencoder import SpatialAttention as FluxAttention
from unitunner.models.qwen_image.vae import VAEAttention
from unitunner.models.qwen_image_2_1.vae import ImageResidual
from unitunner.models.z_image.vae import Downsample, ResidualBlock, SpatialAttention, Upsample
from unitunner.training.session import TrainingSession


def checkpoint_vae_blocks(vae: nn.Module, session: TrainingSession) -> None:
    """Preserve eval mode and parameter keys while replaying differentiable blocks.

    Qwen's temporal residual/resampling blocks mutate history cursors and must
    not be replayed here. Its spatial attention is stateless and safe. The image
    Qwen 2.1 residuals, Z-Image and FLUX-family blocks have no temporal history.
    No-grad HQ encoding, calibration and previews bypass recomputation.
    """
    block_types = (
        ResidualBlock,
        SpatialAttention,
        Downsample,
        Upsample,
        ResidualConvBlock,
        FluxAttention,
        Downsample2d,
        Upsample2d,
        ImageResidual,
        VAEAttention,
    )
    for index, module in enumerate(vae.modules()):
        if isinstance(module, block_types):
            session.own(RecomputeInstallation(ModelRegion(module, index, "visual"), require_training=False))
