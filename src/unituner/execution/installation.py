"""Install shared region execution owners consistently for image and AV workflows."""

from contextlib import ExitStack

import torch

from unituner.artifacts.formats.quantization import canonical_quantization
from unituner.config.execution import ExecutionConfig
from unituner.execution.activations import AttentionActivations, AttentionOffloadInstallation
from unituner.execution.compiler import CompilerInstallation
from unituner.execution.frozen_blocks import FrozenBlockInstallation
from unituner.execution.prefetch import WeightPrefetch
from unituner.execution.recompute import RecomputeInstallation
from unituner.execution.regions import ModelRegions
from unituner.execution.weight_cache import H2dCache, H2dStatistics
from unituner.precision import FLOAT_DTYPES


def training_base_dtype(config: ExecutionConfig, *, adapter: bool) -> torch.dtype:
    """Use BF16 frozen storage for AMP adapters; retain FP32 masters/quantization input."""
    if adapter and config.quantization == "none" and config.precision in ("bf16", "fp16"):
        return FLOAT_DTYPES[config.precision]
    return torch.float32


def prepare_base_weights(
    model: torch.nn.Module, config: ExecutionConfig, *, family: str | None = None, device: torch.device | None = None
) -> None:
    """Prepare frozen base storage before attaching adapters and execution owners."""
    if config.quantization != "none":
        from unituner.models.components.weight_quantization import quantize_model

        precision = canonical_quantization(config.quantization)
        if family is None:
            raise ValueError("Training quantization requires an explicit model family")
        quantize_model(
            model,
            precision,
            family=family,
            rotation_size=config.quantization_rotation_size,
            group_size=config.quantization_group_size,
            device=device,
        )
    if config.convrot_kernel != "eager" or config.convrot_backward != "float":
        from unituner.models.components.quantized import QuantizedLinear

        selected = [
            module
            for module in model.modules()
            if isinstance(module, QuantizedLinear) and module.quantization_format == "int8_convrot"
        ]
        if not selected:
            raise ValueError("Fused ConvRot execution requires ConvRot INT8 base weights")
        for module in selected:
            module.configure_convrot(config.convrot_kernel, config.convrot_backward)


def validate_training_execution(config: ExecutionConfig, *, device: torch.device, adapter: bool) -> None:
    """Reject unavailable execution policies before artifact IO without silently disabling them."""
    if config.quantization != "none" and not adapter:
        raise ValueError("Quantized bases require adapter training")
    if config.convrot_kernel == "triton" or config.convrot_backward == "int8":
        if device.type != "cuda":
            raise ValueError("Fused ConvRot INT8 execution requires CUDA")
        from importlib.util import find_spec

        if find_spec("triton") is None:
            raise ValueError("Fused ConvRot INT8 execution requires Triton")
    if config.precision == "auto":
        raise ValueError("Automatic training precision must be resolved from the model artifact before execution")
    if config.precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training compute requires CUDA")
    if config.precision == "bf16":
        if device.type != "cuda":
            raise ValueError("BF16 training compute requires a BF16-capable CUDA device")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 training compute requires a BF16-capable CUDA device")
    if config.residency == "trainable-blocks":
        raise ValueError("Trainable block swapping is not yet supported; H2D-only requires frozen-blocks")
    if config.compile is True and device.type != "cuda":
        raise ValueError("Training Inductor execution requires CUDA")
    if config.attention_activation_offload and device.type != "cuda":
        raise ValueError("Attention activation offload requires CUDA")
    if config.offloads_frozen_weights:
        if device.type != "cuda" or not adapter:
            raise ValueError("Frozen block H2D residency requires CUDA adapter training")
        if config.recompute != "block" or config.recompute_interval != 1:
            raise ValueError("Frozen block H2D residency requires block recompute with interval=1")


class ExecutionInstallation:
    """Own recompute/H2D/compile in construction order and close them in reverse order."""

    def __init__(self, regions: ModelRegions, config: ExecutionConfig, *, device: torch.device) -> None:
        self._owners: ExitStack = ExitStack()
        self._activations: AttentionActivations | None = None
        prefetch: WeightPrefetch | None = None
        self._cache: H2dCache | None = None
        compiled = config.compile is True or (config.compile == "auto" and device.type == "cuda")
        try:
            allocation_workspace_bytes = (
                max(
                    (p.nbytes for region in regions.regions for p in region.module.parameters() if not p.requires_grad),
                    default=0,
                )
                if config.offloads_frozen_weights
                else 0
            )
            if config.attention_activation_offload:
                if device.type != "cuda":
                    raise ValueError("Attention activation offload requires CUDA")
                self._activations = AttentionActivations(device)
                self._owners.callback(self._activations.close)
            if config.residency == "frozen-blocks":
                self._cache = H2dCache(
                    device=device,
                    workspace_bytes=max(
                        (
                            sum(
                                p.nbytes
                                for p in region.module.parameters()
                                if not p.requires_grad and p.is_contiguous()
                            )
                            for region in regions.regions
                        ),
                        default=0,
                    ),
                    cast_workspace_bytes=max(
                        (
                            sum(
                                p.numel() * 2
                                for p in region.module.parameters()
                                if not p.requires_grad
                                and p.is_contiguous()
                                and p.is_floating_point()
                                and p.dtype != (FLOAT_DTYPES[config.precision])
                            )
                            for region in regions.regions
                        ),
                        default=0,
                    )
                    if config.precision in ("bf16", "fp16")
                    else 0,
                    allocation_workspace_bytes=allocation_workspace_bytes,
                    retained_bytes=None
                    if config.frozen_weight_cache_mb is None
                    else config.frozen_weight_cache_mb * 1024**2,
                )
                self._owners.callback(self._cache.close)
                prefetch = WeightPrefetch(device)
            for region in regions.regions:
                if compiled:
                    compiler = CompilerInstallation(
                        region.module,
                        precision=config.precision,
                    )
                    self._owners.callback(compiler.close)
                if self._activations is not None:
                    offload = AttentionOffloadInstallation(region, self._activations)
                    self._owners.callback(offload.close)
                if config.offloads_frozen_weights:
                    owner = FrozenBlockInstallation(
                        region,
                        device=device,
                        use_pinned_memory=config.use_pinned_memory_for_block_swap is not False,
                        cache=self._cache,
                        prefetch=prefetch,
                    )
                    self._owners.callback(owner.close)
                elif config.recompute == "block":
                    recompute = RecomputeInstallation(
                        region,
                        interval=config.recompute_interval,
                        attention_offload=config.attention_activation_offload,
                    )
                    self._owners.callback(recompute.close)
            if prefetch is not None:
                self._owners.callback(prefetch.close)
        except BaseException:
            if prefetch is not None:
                prefetch.close()
            self._owners.close()
            raise

    def close(self) -> None:
        """Release installed callables and H2D masters after all backward work completes."""
        self._owners.close()

    def h2d_statistics(self) -> H2dStatistics | None:
        """Report the shared cache including current/lookahead and active-graph storage."""
        return None if self._cache is None else self._cache.statistics()
