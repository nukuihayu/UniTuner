"""Validate cross-component training policy separately from dataset and preview contracts."""

from typing import TYPE_CHECKING

from unituner.config.datasets import DatasetConfig
from unituner.config.diagnostics import ConfigRuleError
from unituner.config.sampling import validate_image_sampling
from unituner.models.catalog import ModelFamily, supports_restoration
from unituner.models.minimax_h3.config.inputs import validate_minimax_h3_inputs
from unituner.models.wan.config.sampling import validate_wan_sampling

if TYPE_CHECKING:
    from unituner.config.training import TrainingConfig


def validate_training_policy(config: "TrainingConfig") -> None:
    """Validate model/method/execution choices once for the whole run."""
    if config.method.kind == "restoration" and config.train.timestep_sampling != "shifted":
        raise ConfigRuleError(
            "train.timestep_sampling",
            "train.timestep_sampling requires flow-matching",
            "Use method.restoration sigma/calibration for fixed-timestep restoration.",
        )
    if config.output.adapter_format == "diffusers" and config.train.adapter is not None:
        if config.train.adapter.algorithm != "lora":
            raise ConfigRuleError(
                "output.adapter_format",
                "Official Diffusers export supports vanilla LoRA only",
                "Use adapter_format='comfyui' for LoHa/LoKr, or train.adapter.algorithm='lora'.",
            )
    if (
        config.model.family == ModelFamily.MOSS_IMAGE
        and config.model.variant == "edit"
        and config.method.tread is not None
    ):
        raise ConfigRuleError("method.tread", "MOSS edit does not support TREAD", "Remove method.tread for MOSS edit.")
    if config.method.wavelet is not None and config.model.family in (ModelFamily.WAN, ModelFamily.MINIMAX_H3):
        raise ConfigRuleError(
            "method.wavelet",
            "method.wavelet currently supports image families only",
            "Remove method.wavelet for Wan/MiniMaxH3, or select an image model.",
        )
    if config.method.minimax_h3 is not None and config.model.family != ModelFamily.MINIMAX_H3:
        raise ConfigRuleError(
            "method.minimax_h3",
            "method.minimax_h3 requires MiniMaxH3",
            "Remove this section or set model.family to minimax-h3.",
        )
    if config.method.qwen_image is not None and (
        config.model.family != ModelFamily.QWEN_IMAGE or config.model.variant != "layered"
    ):
        raise ConfigRuleError(
            "method.qwen_image",
            "method.qwen_image requires Qwen Image layered flow training",
            "Use model.family='qwen-image', model.variant='layered' and method.kind='flow-matching', or remove this section.",
        )
    if config.model.family == ModelFamily.QWEN_IMAGE:
        if config.model.variant not in ("original", "edit", "edit-2509", "edit-2511", "layered"):
            raise ConfigRuleError(
                "model.variant",
                "Qwen Image requires original/edit/edit-2509/edit-2511/layered variant",
                "Select original, edit, edit-2509, edit-2511 or layered.",
            )
        if config.method.kind == "restoration" and config.model.variant != "original":
            raise ConfigRuleError(
                "model.variant",
                "Qwen restoration requires the original variant",
                "Set model.variant='original' for Qwen restoration.",
            )
    if config.model.family == ModelFamily.FLUX3 and config.model.variant not in ("base", "distilled"):
        raise ConfigRuleError(
            "model.variant", "FLUX3 requires base or distilled variant", "Set model.variant to 'base' or 'distilled'."
        )
    if config.method.guidance is not None and not (
        config.model.family == ModelFamily.FLUX
        or (config.model.family == ModelFamily.FLUX2 and config.model.version == "dev")
    ):
        raise ConfigRuleError(
            "method.guidance",
            "method.guidance requires a learned guidance embedding (Flux or FLUX.2 Dev)",
            "Remove guidance unless using Flux or FLUX.2 Dev.",
        )
    if (config.train.parameter_mode == "adapter") != (config.train.adapter is not None):
        raise ConfigRuleError(
            "train.adapter",
            "train.adapter must be supplied exactly for adapter parameter mode",
            "Supply [train.adapter] for adapter mode; remove it for full mode.",
        )
    if config.train.parameter_mode == "full" and config.execution.quantization != "none":
        raise ConfigRuleError(
            "execution.quantization",
            "Full training requires execution.quantization='none'",
            "Set execution.quantization='none' for full training, or use train.parameter_mode='adapter'.",
        )
    if config.method.kind == "restoration":
        losses, restoration = config.losses, config.method.restoration
        if losses is None or restoration is None:
            raise ConfigRuleError(
                "method.restoration",
                "Restoration requires losses and method.restoration settings",
                "Supply both [method.restoration] and [losses] for restoration.",
            )
        wavelet = config.method.wavelet
        threshold = None if wavelet is None else wavelet.ll_level_threshold
        if wavelet is not None and threshold is not None and threshold <= 0:
            threshold += wavelet.levels
        wavelet_objective = wavelet is not None and (
            any((wavelet.lh, wavelet.hl, wavelet.hh))
            or (wavelet.ll > 0 and (threshold is None or threshold < wavelet.levels))
        )
        if not any(
            (losses.dv3d, losses.l1, losses.gan, losses.fdl, losses.histogram, losses.ms_ssim, wavelet_objective)
        ):
            raise ConfigRuleError(
                "losses",
                "At least one image objective is required",
                "Enable dv3d, l1, gan, fdl, histogram, ms_ssim or an active method.wavelet band; lrr/phase alone are insufficient.",
            )
        if losses.ms_ssim and any(min(group.resolution) <= 160 for group in config.dataset.groups.values()):
            raise ConfigRuleError(
                "losses.ms_ssim",
                "Five-scale MS-SSIM requires both image dimensions >160",
                "Increase every dataset resolution (e.g. [256,256]) or disable losses.ms_ssim.",
            )
        if (losses.dv3d or losses.gan) and not restoration.dino_weights:
            raise ConfigRuleError(
                "method.restoration.dino_weights",
                "dino_weights is required for dv3d or gan",
                "Set a local DINO .pt/.pth/.safetensors checkpoint, or set losses.dv3d=0 and losses.gan=0.",
            )
        if losses.fdl and not restoration.vgg_weights:
            raise ConfigRuleError(
                "method.restoration.vgg_weights",
                "vgg_weights is required for fdl",
                "Set a local VGG19 .pt/.pth/.safetensors checkpoint, or set losses.fdl=0.",
            )
        if not supports_restoration(config.model.family, config.model.variant):
            raise ConfigRuleError(
                "method.kind",
                "Restoration is not declared for this model/variant",
                "Use flow-matching or a model/variant with restoration support.",
            )
    elif config.losses is not None:
        raise ConfigRuleError(
            "losses",
            "losses settings require the restoration method",
            "Remove [losses] for flow-matching, or select method.kind='restoration'.",
        )


def validate_training_dataset(config: "TrainingConfig", dataset: DatasetConfig) -> None:
    """Validate one data group against its selected native model and method."""
    if dataset.minimax_h3 is not None and config.model.family != ModelFamily.MINIMAX_H3:
        raise ConfigRuleError(
            "minimax_h3",
            "dataset.minimax_h3 requires MiniMaxH3",
            "Remove this dataset section or select model.family=minimax-h3.",
        )
    if config.model.family == ModelFamily.MINIMAX_H3:
        if dataset.kind not in ("image", "video") or dataset.batch_size != 1:
            raise ConfigRuleError(
                "batch_size",
                "MiniMaxH3 requires image/video data and batch_size=1",
                "Use dataset kind image/video and batch_size=1 for MiniMaxH3.",
            )
        if any(size % 32 for size in dataset.resolution):
            raise ConfigRuleError(
                "resolution",
                "MiniMaxH3 resolution requires32-aligned dimensions",
                "Set both width and height to positive multiples of 32, e.g. [512,512].",
            )
        video = dataset.video
        if dataset.kind == "video":
            if video is None or video.target_fps != 24 or video.fps_resampling != "timestamps":
                raise ConfigRuleError(
                    "video",
                    "MiniMaxH3 video requires explicit24Hz timestamp policy",
                    "Set video.target_fps=24 and video.fps_resampling='timestamps'.",
                )
            if any(count < 5 or (count - 5) % 17 for count in video.target_frames) or video.max_frames < 5:
                raise ConfigRuleError(
                    "video.target_frames",
                    "MiniMaxH3 target_frames must follow17*n+5 and max_frames must be at least5",
                    "Use frame counts 5, 22, 39, ... and video.max_frames >= 5.",
                )
    if dataset.wan is not None and (config.model.family != ModelFamily.WAN or dataset.kind not in ("image", "video")):
        raise ConfigRuleError(
            "wan",
            "dataset.wan requires Wan image/video data",
            "Remove this dataset section or use a Wan image/video data group.",
        )
    if config.model.family == ModelFamily.WAN:
        if dataset.kind not in ("image", "video"):
            raise ConfigRuleError("kind", "Wan requires image/video data", "Use image or video data for Wan.")
        video = dataset.video
        if video is not None and (any((value - 1) % 4 for value in video.target_frames) or (video.max_frames - 1) % 4):
            raise ConfigRuleError(
                "video.target_frames",
                "Wan video target_frames/max_frames must follow1+4k",
                "Use target_frames and max_frames of 1, 5, 9, ..., e.g. target_frames=[81], max_frames=129.",
            )
        if (
            dataset.wan is not None
            and dataset.wan.single_frame is not None
            and (dataset.kind != "image" or config.model.variant != "i2v")
        ):
            raise ConfigRuleError(
                "wan.single_frame",
                "Wan single_frame requires I2V image data",
                "Use kind='image' with model.variant='i2v', or remove wan.single_frame.",
            )
    if dataset.wan is not None and dataset.wan.single_frame is not None and dataset.control_source is not None:
        raise ConfigRuleError(
            "control_source",
            "dataset.wan.single_frame and dataset.control_source are mutually exclusive",
            "Choose either wan.single_frame or control_source, not both.",
        )
    if config.model.family not in (ModelFamily.WAN, ModelFamily.MINIMAX_H3) and dataset.kind == "video":
        raise ConfigRuleError(
            "kind",
            "Selected image model requires image or restoration data",
            "Use image or restoration data for the selected image model.",
        )
    dataset.validate_for(config.model.family, config.model.variant, kind=dataset.kind)
    if config.model.family == ModelFamily.QWEN_IMAGE:
        data = dataset
        references = data.reference_sources
        layers = data.layer_sources
        if config.model.variant == "layered" and config.method.kind == "flow-matching" and not layers:
            raise ConfigRuleError(
                "layer_sources",
                "Qwen layered flow requires dataset.layer_sources",
                "Supply the per-layer image sources for layered training.",
            )
        if config.model.variant == "edit" and config.method.kind == "flow-matching" and not references:
            raise ConfigRuleError(
                "reference_sources",
                "Original Qwen edit flow requires dataset.reference_sources",
                "Supply reference image sources for Qwen edit training.",
            )
    if config.model.family == ModelFamily.FLUX and config.method.kind == "flow-matching":
        data = dataset
        if config.model.variant in ("kontext", "fill") and not data.reference_sources:
            raise ConfigRuleError(
                "reference_sources",
                "Flux Kontext/Fill training requires dataset.reference_sources",
                "Supply reference image sources for Kontext/Fill training.",
            )
    if config.method.kind == "restoration":
        if dataset.kind != "restoration":
            raise ConfigRuleError(
                "kind",
                "Restoration requires dataset.kind='restoration'",
                "Set this dataset group's kind='restoration'.",
            )
    elif dataset.kind == "restoration":
        raise ConfigRuleError(
            "kind",
            "Image restoration data requires the restoration method",
            "Select method.kind='restoration' with restoration settings, or use an image/video data group.",
        )


def validate_training_preview(config: "TrainingConfig") -> None:
    """Validate image, video or audio/video previews once per run."""
    if config.preview is not None:
        if config.preview.audio_video and config.model.family != ModelFamily.MINIMAX_H3:
            raise ConfigRuleError(
                "preview.audio_video",
                "Audio/video previews require MiniMaxH3",
                "Remove audio_video or select a MiniMaxH3 model.",
            )
        if config.model.family == ModelFamily.MINIMAX_H3:
            if (
                config.preview.samples
                or config.preview.videos
                or not config.preview.audio_video
                or config.preview.num_images != 1
            ):
                raise ConfigRuleError(
                    "preview",
                    "MiniMaxH3 previews require preview.audio_video without image/Wan samples or num_images",
                    "Supply preview.audio_video, remove samples/videos and leave num_images at 1.",
                )
            if config.model.minimax_h3 is None:
                raise ConfigRuleError(
                    "model.minimax_h3",
                    "MiniMaxH3 previews require model.minimax_h3 encoder artifacts",
                    "Supply MiniMaxH3 encoder artifacts for preview generation.",
                )
            for sample_av in config.preview.audio_video:
                validate_minimax_h3_inputs(sample_av.inputs, sample_av.sampling, task=config.model.variant)
        if config.preview.videos and config.model.family != ModelFamily.WAN:
            raise ConfigRuleError(
                "preview.videos", "Video previews currently require Wan", "Remove videos or select a Wan model."
            )
        if config.model.family == ModelFamily.WAN:
            if config.preview.samples or not config.preview.videos or config.preview.num_images != 1:
                raise ConfigRuleError(
                    "preview",
                    "Wan previews require preview.videos and do not accept image samples/num_images",
                    "Supply preview.videos, remove preview.samples and leave preview.num_images at 1.",
                )
            for video_sample in config.preview.videos:
                validate_wan_sampling(video_sample, version=config.model.version, variant=config.model.variant)
        for sample in config.preview.samples:
            validate_image_sampling(
                sample, family=config.model.family, variant=config.model.variant, version=config.model.version
            )
        if config.model.vae is None:
            raise ConfigRuleError(
                "model.vae", "Training previews require model.vae", "Set the VAE checkpoint path, or remove [preview]."
            )
        if config.method.kind == "flow-matching":
            if (
                not config.preview.samples and config.model.family not in (ModelFamily.WAN, ModelFamily.MINIMAX_H3)
            ) or config.preview.num_images != 1:
                raise ConfigRuleError(
                    "preview",
                    "Flow previews require samples and do not accept num_images",
                    "Set preview.samples for image flow previews and leave preview.num_images at 1.",
                )
            if config.model.text_encoder is None:
                raise ConfigRuleError(
                    "model.text_encoder",
                    "Flow previews require model.text_encoder",
                    "Set the text encoder artifacts required by this family, or remove [preview].",
                )
        elif config.preview.samples:
            raise ConfigRuleError(
                "preview.samples",
                "Restoration previews use dataset items, not prompt samples",
                "Remove preview.samples; use preview.num_images to select restoration dataset items.",
            )
