"""Validate configuration or lazily dispatch implemented application commands."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from unitunner.artifacts.formats.precision_policy import FAMILIES
from unitunner.artifacts.formats.quantization import DEFAULT_GROUP_SIZE, DEFAULT_ROTATION_SIZE, PRECISIONS
from unitunner.cli.configuration import add_config_arguments, config_input, set_override
from unitunner.config.cache import CacheConfig
from unitunner.config.calibration import CalibrationConfig
from unitunner.config.generation import GenerationConfig
from unitunner.config.loading import (
    load_cache_config,
    load_calibration_config,
    load_generation_config,
    load_restoration_inference_config,
    load_training_config,
    load_video_generation_config,
)
from unitunner.config.restoration_inference import RestorationInferenceConfig
from unitunner.config.training import TrainingConfig
from unitunner.config.video_generation import VideoGenerationConfig
from unitunner.logging import (
    configure_logging,
    get_logger,
    logging_session,
    report_error,
    set_context,
    start_training_log,
)
from unitunner.progress import status


class ArgumentParser(argparse.ArgumentParser):
    """Keep argparse usage/exit semantics while routing argument errors through logging."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        get_logger("cli").error("Invalid command arguments: %s", message)
        self.exit(2)


def main(argv: Sequence[str] | None = None) -> int:
    """Own one diagnostic session and report every application exception once."""
    with logging_session():
        try:
            return _main(argv)
        except KeyboardInterrupt:
            get_logger("cli").warning("Interrupted by user; only completed checkpoints are available for resume.")
            return 130
        except Exception as error:
            return report_error(error)


def _main(argv: Sequence[str] | None = None) -> int:
    """Validate grouped configuration or run an explicitly selected training application."""
    parser = ArgumentParser(prog="unitunner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate TOML or CLI configuration; does not train")
    validate.add_argument(
        "--for",
        dest="validation_target",
        default="train",
        choices=("train", "cache", "mid-timestep", "generate", "restore", "generate-video"),
    )
    add_config_arguments(
        validate,
        (
            TrainingConfig,
            CacheConfig,
            CalibrationConfig,
            GenerationConfig,
            RestorationInferenceConfig,
            VideoGenerationConfig,
        ),
    )
    train = subparsers.add_parser(
        "train", help="Train native image flow matching or low/high-quality image restoration"
    )
    add_config_arguments(train, TrainingConfig)
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume", type=Path, help="Training state file from the output resume/ directory")
    train.add_argument(
        "--parameter-mode",
        choices=("adapter", "full"),
        help="Override train.parameter_mode; full clears adapter configuration",
    )
    train.add_argument(
        "--include-layers", nargs="+", metavar="PATTERN", help="Native module/parameter glob patterns to train"
    )
    train.add_argument(
        "--exclude-layers", nargs="+", metavar="PATTERN", help="Native glob patterns to freeze; exclusions win"
    )
    train.add_argument(
        "--midtimestep",
        type=int,
        metavar="T",
        help="Restoration middle timestep (1..998; default 244); overrides method.restoration.midtimestep in TOML",
    )
    middle = subparsers.add_parser("mid-timestep", help="Estimate restoration time with only the frozen native vae")
    add_config_arguments(middle, CalibrationConfig)
    middle.add_argument("--device", default="cpu")
    cache = subparsers.add_parser("cache", help="Encode local image/video/text sources into native caches")
    add_config_arguments(cache, CacheConfig)
    cache.add_argument("--device", default="cpu")
    caches = subparsers.add_parser("caches", help="Explicitly convert external cache artifacts")
    cache_commands = caches.add_subparsers(dest="cache_command", required=True)
    cache_import = cache_commands.add_parser("import", help="Import explicitly described split MiniMaxH3 caches")
    cache_import.add_argument("specification", type=Path)
    cache_import.add_argument("destination", type=Path)
    generate = subparsers.add_parser("generate", help="Generate an image from native local artifacts")
    add_config_arguments(generate, GenerationConfig)
    generate.add_argument("--device", default="cpu")
    video = subparsers.add_parser(
        "generate-video", help="Generate native Wan video or MiniMaxH3 image/AV from local artifacts"
    )
    add_config_arguments(video, VideoGenerationConfig)
    video.add_argument("--device", default="cpu")
    restore = subparsers.add_parser("restore", help="Restore a low-quality image with one native prediction")
    add_config_arguments(restore, RestorationInferenceConfig)
    restore.add_argument("--device", default="cpu")
    weights = subparsers.add_parser("weights", help="Import weights or convert model precision")
    weight_commands = weights.add_subparsers(dest="weight_command", required=True)
    conversion = weight_commands.add_parser("convert", help="Stream float/Kitchen precision conversion")
    conversion.add_argument("source", type=Path)
    conversion.add_argument("destination", type=Path)
    conversion.add_argument("--precision", choices=PRECISIONS, required=True)
    conversion.add_argument(
        "--output-format",
        choices=("comfyui", "source"),
        default="comfyui",
        help="Export ComfyUI denoiser keys (default), or retain the source layout",
    )
    conversion.add_argument("--device", default="cpu")
    conversion.add_argument("--family", choices=FAMILIES, default="auto", help="Model-specific layer protection")
    conversion.add_argument("--include", nargs="+", default=(), metavar="GLOB")
    conversion.add_argument("--exclude", nargs="+", default=(), metavar="GLOB")
    conversion.add_argument("--rotation-size", type=int, default=DEFAULT_ROTATION_SIZE)
    conversion.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    conversion.add_argument("--layout-spec", type=Path, help="JSON module-prefix to calibrated layout mapping")
    conversion.add_argument("--on-incompatible", choices=("error", "keep"), default="error")
    conversion.add_argument("--backend", choices=("eager", "cuda", "triton"))
    import_weights = weight_commands.add_parser("import", help="Validate and write reloadable native weights")
    import_weights.add_argument("source", type=Path)
    import_weights.add_argument("destination", type=Path)
    import_weights.add_argument(
        "--family",
        choices=(
            "z-image",
            "moss-image",
            "flux",
            "flux2",
            "flux3",
            "qwen-image",
            "qwen-image-2-1",
            "krea2",
            "wan",
            "minimax-h3",
        ),
        required=True,
    )
    import_weights.add_argument("--kind", choices=("transformer", "vae"), required=True)
    import_weights.add_argument("--topology", type=Path)
    import_weights.add_argument("--output-format", choices=("comfyui", "native"), default="comfyui")
    import_weights.add_argument(
        "--source-format", choices=("auto", "native", "comfyui", "external", "diffusers"), default="auto"
    )
    import_weights.add_argument(
        "--prefix", choices=("", "model.diffusion_model.", "diffusion_model.", "vae."), default=""
    )
    import_weights.add_argument("--fused-qkv", action="store_true")
    adapters = subparsers.add_parser("adapters", help="Convert adapter artifacts for explicit consumers")
    adapter_commands = adapters.add_subparsers(dest="adapter_command", required=True)
    comfyui = adapter_commands.add_parser("export-comfyui", help="Convert a native model adapter for ComfyUI")
    comfyui.add_argument("source", type=Path)
    comfyui.add_argument("destination", type=Path)
    comfyui.add_argument("--topology", type=Path, required=True)
    comfyui.add_argument("--algorithm", choices=("lora", "loha", "lokr"), required=True)
    comfyui.add_argument("--family", choices=("wan", "minimax-h3"), default="wan")
    analyze = subparsers.add_parser(
        "analyze", help="Analyze final exported weights and effective adapters (excludes periodic/resume files)"
    )
    analyze.add_argument("project", type=Path, nargs="?", help="Run directory, safetensors file or shard index")
    analyze.add_argument("--list", dest="list_runs", action="store_true", help="List available training runs")
    analyze.add_argument("--runs-root", type=Path, default=Path("runs"))
    analyze.add_argument("--base", type=Path, help="Reference/base weights for matched-layer comparisons")
    analyze.add_argument(
        "--compare", dest="compare_path", type=Path, help="Additional full-weight artifact for three-way subspace plots"
    )
    analyze.add_argument("--output", type=Path, help="Analysis output directory (default: project/analysis)")
    analyze.add_argument("--scope", choices=("trained", "all"), default="trained")
    analyze.add_argument(
        "--key-map", type=Path, help="JSON logical-layer to baseline-key mapping for different layouts"
    )
    analyze.add_argument(
        "--layer", dest="layers", action="append", default=[], help="Logical layer glob for subspace plots; repeatable"
    )
    analyze.add_argument("--top-k", type=int, default=32, help="Number of nonzero singular directions to compare")
    analyze.add_argument(
        "--top-layers", type=int, default=3, help="Default number of largest-RMS layers per latest snapshot"
    )
    analyze.add_argument("--vector-side", choices=("left", "right"), default="right")
    analyze.add_argument("--heads", type=int, help="Explicit contiguous output-row head count for selected layers")
    analyze.add_argument("--svd-max-elements", type=int, default=262144)
    analyze.add_argument("--factor-budget", type=int, default=8_000_000)
    analyze.add_argument("--matrix-budget", type=int, default=32_000_000)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument(
        "--show",
        choices=("auto", "always", "never"),
        default="auto",
        help="Display figures in notebook/local GUI when available; always save results",
    )
    for command_parser in (
        parser,
        validate,
        train,
        middle,
        cache,
        caches,
        cache_import,
        generate,
        video,
        restore,
        weights,
        conversion,
        import_weights,
        adapters,
        comfyui,
        analyze,
    ):
        command_parser.add_argument(
            "--log-level",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            default=argparse.SUPPRESS,
            help="Console verbosity; DEBUG includes full tracebacks",
        )
    args = parser.parse_args(argv)
    set_context(
        command=args.command,
        device=getattr(args, "device", None),
        config=getattr(args, "config_file", None) or getattr(args, "config", None),
        resume=getattr(args, "resume", None),
        stage="validating configuration",
    )
    set_context(
        source=getattr(args, "source", None),
        destination=getattr(args, "destination", None),
        specification=getattr(args, "specification", None),
    )
    configure_logging(level=getattr(args, "log_level", "INFO"))
    get_logger("cli").debug("Runtime Python=%s platform=%s cwd=%s", sys.version.split()[0], sys.platform, Path.cwd())
    if args.command == "analyze":
        return import_module("unitunner.analysis.run").run(args)
    if args.command == "caches":
        if any(find_spec(name) is None for name in ("PIL", "cv2", "av")):
            raise ValueError("Cache import requires optional dependencies: uv sync --extra media")
        entry = import_module("unitunner.models.minimax_h3.data.cache.conversion").run
        return cast(Callable[..., int], entry)(args.specification, args.destination)
    if args.command == "adapters":
        entry = import_module("unitunner.artifacts.exporting").export_comfyui
        return cast(Callable[..., int], entry)(
            args.source, args.destination, topology=args.topology, algorithm=args.algorithm, family=args.family
        )
    if args.command == "weights":
        if args.weight_command == "convert":
            entry = import_module("unitunner.artifacts.conversion").convert_weights
            report = entry(
                args.source,
                args.destination,
                precision=args.precision,
                device=args.device,
                include=tuple(args.include),
                exclude=tuple(args.exclude),
                rotation_size=args.rotation_size,
                group_size=args.group_size,
                layout_spec=args.layout_spec,
                on_incompatible=args.on_incompatible,
                backend=args.backend,
                family=args.family,
                output_format=args.output_format,
            )
            print(json.dumps(report, ensure_ascii=False))
            return 0
        entry = import_module("unitunner.artifacts.importing").import_weights
        if not callable(entry):
            raise TypeError("Weight command entry must be callable")
        return cast(Callable[..., int], entry)(
            args.source,
            args.destination,
            family=args.family,
            kind=args.kind,
            topology=args.topology,
            source_format=args.source_format,
            prefix=args.prefix,
            fused_qkv=args.fused_qkv,
            output_format=args.output_format,
        )
    config_path, overrides = config_input(args)
    if args.command == "train" and args.midtimestep is not None:
        set_override(overrides, "method.restoration.midtimestep", args.midtimestep)
    if args.command == "train":
        train_overrides: dict[str, JsonValue] = {}
        if args.parameter_mode is not None:
            train_overrides["parameter_mode"] = args.parameter_mode
            if args.parameter_mode == "full":
                train_overrides["adapter"] = None
        if args.include_layers is not None:
            train_overrides["include_layers"] = args.include_layers
        if args.exclude_layers is not None:
            train_overrides["exclude_layers"] = args.exclude_layers
        if train_overrides:
            for key, value in train_overrides.items():
                set_override(overrides, f"train.{key}", value)
    if args.command == "validate" and args.validation_target != "train":
        loaders = {
            "cache": load_cache_config,
            "mid-timestep": load_calibration_config,
            "generate": load_generation_config,
            "restore": load_restoration_inference_config,
            "generate-video": load_video_generation_config,
        }
        validated = loaders[args.validation_target](config_path, overrides=overrides)
        print(json.dumps({"valid": True, "family": validated.model.family, "command": args.validation_target}))
        return 0
    if args.command == "generate-video":
        config_video = load_video_generation_config(config_path, overrides=overrides)
        set_context(family=config_video.model.family, variant=config_video.model.variant)
        if any(find_spec(name) is None for name in ("PIL", "cv2", "av", "transformers")):
            raise ValueError("Video generation requires uv sync --extra media --extra text")
        family = "minimax_h3" if config_video.model.family == "minimax-h3" else "wan"
        entry = import_module(f"unitunner.models.{family}.inference.run").run
        return cast(Callable[..., int], entry)(config_video, device=args.device)
    if args.command in {"cache", "mid-timestep", "generate", "restore"}:
        command_loaders = {
            "cache": load_cache_config,
            "mid-timestep": load_calibration_config,
            "generate": load_generation_config,
            "restore": load_restoration_inference_config,
        }
        command_config = command_loaders[args.command](config_path, overrides=overrides)
        set_context(family=command_config.model.family, variant=getattr(command_config.model, "variant", None))
        required = ("PIL", "cv2") if args.command == "mid-timestep" else ("PIL", "cv2", "transformers")
        if isinstance(command_config, CacheConfig) and any(
            group.kind == "video" for group in command_config.dataset.groups.values()
        ):
            required = (*required, "av")
        if any(find_spec(name) is None for name in required):
            raise ValueError(f"{args.command} requires optional dependencies: " + ", ".join(required))
        module = {
            "cache": "unitunner.data.build",
            "mid-timestep": "unitunner.training.calibration",
            "generate": "unitunner.inference.generation",
            "restore": "unitunner.inference.restoration.run",
        }[args.command]
        if args.command == "cache":
            status("[cache] Configuration loaded; importing encoder runtime")
        entry = import_module(module).run
        if not callable(entry):
            raise TypeError("Application command entry must be callable")
        return cast(Callable[..., int], entry)(command_config, device=args.device)
    config = load_training_config(config_path, overrides=overrides)
    set_context(
        family=config.model.family,
        variant=config.model.variant,
        method=config.method.kind,
        precision=config.execution.precision,
        residency=config.execution.residency,
    )
    if args.command == "train":
        from unitunner.training.output import prepare_training_output

        config = prepare_training_output(config, resume=args.resume)
        path = start_training_log(
            config.output.directory,
            config.output.name,
            run_name=config.output._run_name,
            append=args.resume is not None,
        )
        status(f"[train] Output directory: {config.output.directory}")
        status(f"[train] Log file: {path}")
        status("[train] Configuration loaded; importing training runtime")
        if config.execution.residency == "vbar":
            bootstrap = import_module("unitunner.execution.allocator_bootstrap").initialize_vbar
            bootstrap()
        entry = import_module("unitunner.training.run").run
        if not callable(entry):
            raise TypeError("Training command entry must be callable")
        return cast(Callable[..., int], entry)(config, device=args.device, resume=args.resume)
    print(json.dumps({"valid": True, "family": config.model.family, "method": config.method.kind}))
    return 0
