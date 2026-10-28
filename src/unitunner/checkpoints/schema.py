"""Separate numerical run identity from publication metadata and payloads."""

from pydantic import BaseModel, ConfigDict, Field

from unitunner.artifacts.hashing import canonical_sha256
from unitunner.config.optimizer import DEFAULT_ADAMW_CONFIG
from unitunner.config.training import TrainingConfig
from unitunner.models.catalog import ModelFamily


class ResumeIdentity(BaseModel):
    """Pin numerical choices and content manifests, excluding output scheduling."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: int = Field(default=1, ge=1, le=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_resume_identity(
    config: TrainingConfig, *, dataset_manifest_sha256: str, model_artifact_sha256: str
) -> ResumeIdentity:
    """Include seed, method, horizon, accumulation, data and execution in identity.

    Exclude output and preview settings. The horizon is pinned because schedules can depend on
    it. This schema does not convert old optimizer/RNG/teacher state or claim
    old checkpoints can resume in the new project.
    """
    # These defaults made previously implicit behavior explicit. Omit only the
    # listed values to retain established hashes; pin all numerical overrides.
    default_policies = {
        "train": {"include_layers": ((),), "exclude_layers": ((),), "optimizer": (DEFAULT_ADAMW_CONFIG,)},
        "method": {"layersync": (None,), "tread": (None,), "repa": (None,), "wavelet": (None,)},
        "execution": {"use_pinned_memory_for_block_swap": (None, False), "attention_activation_offload": (False,)},
    }
    fields = {
        section: {name for name, omitted in defaults.items() if getattr(getattr(config, section), name) in omitted}
        for section, defaults in default_policies.items()
    }
    # Startup validation does not change the numerical training policy.
    fields["train"].update(("skip_dataset_validation", "dataset_validation_workers"))
    if config.model.family != ModelFamily.MINIMAX_H3:
        fields["method"].add("minimax_h3")
    if config.execution.quantization == "none":
        # Inactive quantization geometry cannot change an unquantized trajectory.
        fields["execution"].update(("quantization_rotation_size", "quantization_group_size"))
    # Cache execution is already pinned by dataset records, not training policy.
    exclusions: dict[str, bool | set[str]] = {"output": True, "preview": True, "cache": True, **fields}
    values = config.model_dump(mode="json", exclude=exclusions)
    if any(group.degradation is not None and group.degradation.enabled for group in config.dataset.groups.values()):
        values["degradation_policy"] = "exposure-grain/continuous-processing/low-zero/handheld/balanced-severity/independent-scale/11"
    values["dataset_sampling_policy"] = "named-groups/independent-buckets/repeat-before-chunk/short-tail/2"
    values["dataset_group_order"] = list(config.dataset.groups)
    if config.method.kind == "restoration":
        if config.losses is not None and not config.losses.ms_ssim:
            values["losses"].pop("ms_ssim")
        else:
            values["ms_ssim_policy"] = "rgb/fp32/valid-gaussian11-sigma1.5/5-scales/parity-pool/floor1e-6/1"
        if values["method"]["restoration"]["calibration_sampling"] == "random":
            values["method"]["restoration"].pop("calibration_sampling")
        if config.losses is not None and not config.losses.phase:
            # A disabled supplementary loss must not invalidate earlier runs.
            values["losses"].pop("phase")
            for name in ("phase_period", "phase_cycles", "phase_epsilon"):
                values["method"]["restoration"].pop(name)
        else:
            values["phase_loss_policy"] = "local-affine/target-ridge-1e-6/charbonnier/overlap-edges/2"
        # Posterior defaults and draw order are part of exact continuation,
        # even when the user left posterior unspecified in the TOML.
        values["restoration_sampling_policy"] = "sampled-z-image/hq-posterior-lrr-noise-lq-posterior/top-left-crop/3"
    if config.execution.compile:
        values["compiler_shape_policy"] = "native-evaluation/native-sdpa/block-only/dynamic-from-first-call/5"
        values["compiler_guard_policy"] = "scoped-accumulated-budget/reject-eager-fallback/1"
        values["compiler_backward_policy"] = "outside-forward-autocast/1"
        if config.execution.precision in ("bf16", "fp16"):
            values["compiler_precision_policy"] = "emulate-native-precision-casts/1"
    if config.execution.precision in ("bf16", "fp16") and config.model.family in {
        ModelFamily.WAN,
        ModelFamily.MINIMAX_H3,
    }:
        values["precision_islands_policy"] = "native-time-and-projection-fp32-under-amp/1"
    values["optimizer_implementation_policy"] = "adamw/fused-cuda/native-cpu/named-groups/2"
    if config.train.optimizer.backend != "auto":
        values["optimizer_implementation_policy"] = f"adamw/{config.train.optimizer.backend}/named-groups/2"
    values["vae_activation_policy"] = "fused-silu/1"
    values["training_weight_storage_policy"] = "floating-frozen-base-follows-compute/fp32-trainable-masters/1"
    if config.execution.precision == "fp16":
        values["fp16_gradient_policy"] = "fixed-scale/unscale-before-normalize-and-clip/fail-on-nonfinite/1"
    if config.model.family == ModelFamily.QWEN_IMAGE and config.method.kind == "restoration":
        # Direct std arithmetic and logical-order posterior RNG change the
        # numerical continuation of earlier reciprocal-scaled checkpoints.
        values["restoration_vae_policy"] = "qwen/direct-std-logical-noise/1"
    if config.model.family == ModelFamily.MINIMAX_H3:
        values["minimax_h3_batch_policy"] = "ordered-records/repeat-epoch/batch-one/1"
        values["minimax_h3_rng_policy"] = "video-uniform-audio-visual-conditions-audio-conditions/1"
        values["minimax_h3_time_policy"] = "video-initial-base/model-and-audio-recovered-multiply1000-add1/1"
    if config.model.family == ModelFamily.WAN:
        values["wan_batch_policy"] = "stable-grid-buckets/repeat-before-chunk/short-tails/1"
        values["wan_time_policy"] = "separate-sigma/direct-multiply1000-add1/1"
    return ResumeIdentity(
        config_sha256=canonical_sha256(values),
        dataset_manifest_sha256=dataset_manifest_sha256,
        model_artifact_sha256=model_artifact_sha256,
    )


class CheckpointCursor(BaseModel):
    """Persist completed local phase counts without importing the training engine."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    updates: int = Field(ge=0)
    microbatches: int = Field(ge=0)
    samples: int = Field(ge=0)
