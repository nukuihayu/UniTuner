"""Install restoration-only recomputation on stateless VAE image blocks."""

from torch import nn

from unituner.artifacts.formats.precision import FloatingPrecision
from unituner.execution.compiler import CompilerInstallation
from unituner.execution.recompute import RecomputeInstallation
from unituner.execution.regions import ModelRegion
from unituner.models.components.autoencoder import Downsample2d, ResidualConvBlock, Upsample2d
from unituner.models.components.autoencoder import SpatialAttention as FluxAttention
from unituner.models.qwen_image.vae import VAEAttention
from unituner.models.qwen_image_2_1.vae import ImageResidual
from unituner.models.z_image.vae import Downsample, ResidualBlock, SpatialAttention, Upsample
from unituner.training.session import TrainingSession


def checkpoint_vae_blocks(
    vae: nn.Module,
    session: TrainingSession,
    *,
    checkpoint_encoder: bool = True,
    compile_decoder: bool = False,
    precision: FloatingPrecision = "fp32",
) -> None:
    """Preserve eval mode and parameter keys while replaying differentiable blocks.

    Qwen's temporal residual/resampling blocks mutate history cursors and must
    not be replayed here. Its spatial attention is stateless and safe. The image
    Qwen 2.1 residuals, Z-Image and FLUX-family blocks have no temporal history.
    No-grad HQ encoding, calibration and previews bypass recomputation.
    Optional decoder compilation includes checkpoint replay and preserves eval
    mode; no-grad inference still uses the native callable.
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
    modules = vae.modules() if checkpoint_encoder else vae.get_submodule("decoder").modules()
    for index, module in enumerate(modules):
        if isinstance(module, block_types):
            session.own(RecomputeInstallation(ModelRegion(module, index, "visual"), require_training=False))
    if compile_decoder:
        for module in vae.get_submodule("decoder").modules():
            # Keep SDPA native: Inductor layout propagation can violate Flash
            # Attention's contiguous head-dimension contract during replay.
            if isinstance(module, (ResidualBlock, Upsample, ResidualConvBlock, Upsample2d, ImageResidual)):
                session.own(CompilerInstallation(module, precision=precision, require_training=False))
