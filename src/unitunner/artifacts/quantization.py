"""Quantize and reconstruct canonical Kitchen weights through optional public APIs."""

import dataclasses
import json
from collections.abc import Callable, Mapping
from importlib import import_module

import torch
from safetensors import SafetensorError
from torch import Tensor

from unitunner.artifacts.formats.quantization import DEFAULT_GROUP_SIZE, DEFAULT_ROTATION_SIZE, FORMATS
from unitunner.precision import FLOAT_DTYPES as DTYPES


def kitchen_types():
    """Load optional tensor layouts only when a quantized operation is requested."""
    try:
        module = import_module("comfy_kitchen.tensor")
        for layout in {item.layout for item in FORMATS.values()}:
            getattr(module, layout)
        return module
    except (ImportError, AttributeError) as error:
        raise ValueError("Kitchen conversion requires current comfy-kitchen: uv sync --extra kitchen") from error


def json_tensor(value: Mapping) -> Tensor:
    """Encode ComfyUI's per-layer UTF-8 JSON as a one-dimensional uint8 tensor."""
    return torch.tensor(list(json.dumps(dict(value), separators=(",", ":")).encode()), dtype=torch.uint8)


def tensor_json(value: Tensor) -> dict:
    """Decode bounded, non-executable per-layer metadata."""
    if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() > 1024 * 1024:
        raise ValueError("comfy_quant must be a uint8 JSON vector of at most 1 MiB")
    try:
        result = json.loads(value.cpu().numpy().tobytes())
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Invalid comfy_quant JSON") from error
    if not isinstance(result, dict):
        raise ValueError("comfy_quant must contain a JSON object")
    return result


def quantize_weight(
    values: Tensor, precision: str, *, rotation_size: int = DEFAULT_ROTATION_SIZE, group_size: int = DEFAULT_GROUP_SIZE
):
    """Delegate packing, rotation, codebooks and block-scale ordering to Kitchen."""
    format = FORMATS[precision]
    if format.calibrated:
        raise ValueError(
            f"{precision} requires offline-calibrated weights; import with --layout-spec and --precision keep"
        )
    module = kitchen_types()
    layout = getattr(module, format.layout)
    options = dict(format.options)
    if precision.startswith("float8_"):
        dtype = getattr(torch, precision)
        # Calculate in FP32; zero matrices need a nonzero divisor.
        scale = (values.float().abs().amax() / torch.finfo(dtype).max).clamp_min(1e-30)
        if values.dtype == torch.float16:
            scale = scale.clamp_min(1 / torch.finfo(torch.float16).max)
        options.update(dtype=dtype, scale=scale)
    elif precision == "nvfp4":
        options["scale"] = (values.float().abs().amax() / (448 * 6)).clamp_min(1e-30)
    if format.uses_rotation:
        options["convrot_groupsize"] = rotation_size
    if format.uses_group_size:
        options["group_size"] = group_size
    if precision == "int8_convrot":
        # Match Musubi/published ConvRot checkpoints: rotate and choose INT8
        # codes in FP32, retaining the requested logical compute dtype.
        data, params = layout.quantize(values.float(), **options)
        return module.QuantizedTensor(data, format.layout, dataclasses.replace(params, orig_dtype=values.dtype))
    # Call the public layout directly: consumers such as ComfyUI may register
    # their own quantizers under Kitchen's original global layout names.
    if values.dtype == torch.float32 and precision in ("mxfp8", "nvfp4"):
        # Native block quantizers currently accept only FP16/BF16. Preserve
        # FP32 input rounding with Kitchen's eager implementation on this device.
        with import_module("comfy_kitchen").use_backend("eager"):
            data, params = layout.quantize(values, **options)
    else:
        data, params = layout.quantize(values, **options)
    return module.QuantizedTensor(data, format.layout, params)


def serialize_weight(key: str, weight, precision: str) -> dict[str, Tensor]:
    """Use layout-owned suffixes; preserve all scale and correction tensors."""
    config: dict[str, object] = {"format": FORMATS[precision].wire_format}
    if precision == "float8_e5m2":
        # PyTorch scaled_mm rejects E5M2 x E5M2. ComfyUI must use its supported
        # dequantized-matmul route instead of attempting that kernel.
        config["full_precision_matrix_mult"] = True
    for field in dataclasses.fields(weight._params):
        value = getattr(weight._params, field.name)
        if isinstance(value, Tensor) or value is None or field.name == "transposed":
            continue
        if field.name == "orig_dtype":
            value = next(name for name, dtype in DTYPES.items() if value == dtype)
        config[field.name] = value
    tensors = weight.state_dict(key)
    # E8M0 safetensors support varies; ComfyUI explicitly accepts raw bytes.
    tensors = {
        name: tensor.view(torch.uint8) if tensor.dtype == torch.float8_e8m0fnu else tensor
        for name, tensor in tensors.items()
    }
    tensors[key.removesuffix("weight") + "comfy_quant"] = json_tensor(config)
    return tensors


def read_weight(key: str, read: Callable[[str], Tensor], config: Mapping, *, device: torch.device):
    """Rebuild registered layouts, retaining logical shapes and per-layout options."""
    name = config.get("format")
    if name not in FORMATS:
        raise ValueError(f"Unsupported Kitchen format for {key}: {name!r}")
    module = kitchen_types()
    format = FORMATS[name]
    layout = getattr(module, format.layout)
    data = read(key).to(device)
    if name.startswith("float8_") and data.dtype == torch.uint8:
        data = data.view(getattr(torch, name))
    shape = config.get("orig_shape")
    if shape is None and name in ("float8_e4m3fn", "float8_e5m2", "int8_tensorwise", "int8_rowwise", "int8_convrot"):
        shape = list(data.shape)
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError(f"{key} requires positive two-dimensional orig_shape; supply --layout-spec for legacy files")
    dtype = config.get("orig_dtype", "bf16")
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported orig_dtype for {key}: {dtype!r}")
    options = config.get("params", {})
    if not isinstance(options, dict):
        raise ValueError(f"{key} params must be an object")
    fields = {field.name for field in dataclasses.fields(layout.Params)} - {"scale", "orig_shape", "orig_dtype"}
    params = {
        field: config.get(field, options[field]) if field in options else config[field]
        for field in fields
        if field in config or field in options
    }
    if params.get("transposed") or params.get("mma_packed"):
        raise ValueError(f"{key}: resident/transposed storage must be exported in canonical layout first")
    for tensor in format.tensors:
        try:
            value = read(key + "_" + tensor.suffix).to(device)
        except (KeyError, SafetensorError):
            if not tensor.optional:
                raise
            continue
        if value.dtype == torch.uint8 and tensor.byte_dtype is not None:
            value = value.view(getattr(torch, tensor.byte_dtype))
        params[tensor.parameter] = value
    params = layout.Params(orig_shape=tuple(shape), orig_dtype=DTYPES[dtype], **params)
    _validate_storage(key, name, data, params)
    return module.QuantizedTensor(data, layout.__name__, params)


def _validate_storage(key: str, name: str, data: Tensor, params) -> None:
    """Reject malformed canonical shapes before preservation or dequantization."""
    n, k = params.orig_shape
    if name.startswith("float8_"):
        expected_dtype, expected_shape = getattr(torch, name), (n, k)
    elif name == "mxfp8":
        expected_dtype, expected_shape = torch.float8_e4m3fn, ((n + 31) // 32 * 32, (k + 31) // 32 * 32)
    elif name == "nvfp4":
        expected_dtype, expected_shape = torch.uint8, ((n + 15) // 16 * 16, (k + 15) // 16 * 8)
    elif name in ("asym_w4a8_int8", "w6a8_int8"):
        expected_dtype, expected_shape = torch.int8, (n, k * (6 if name == "w6a8_int8" else 4) // 8)
    elif name in ("awq_w4a16", "svdquant_w4a4", "convrot_w4a4", "convrot_w4a8"):
        expected_dtype, expected_shape = torch.int8, (n, k // 2)
        if k % 2 or (name != "awq_w4a16" and k % 64):
            raise ValueError(f"{key}: packed group layout requires K divisible by 64")
        if name == "svdquant_w4a4" and data.ndim == 4:
            # Kitchen also accepts tiled N-interleaved SVDQuant storage.
            expected_shape = (n // 128, k // 64, 32, 128)
    else:
        expected_dtype, expected_shape = torch.int8, (n, k)
    if data.dtype == torch.uint8 and name.startswith("float8_"):
        raise ValueError(f"{key}: legacy FP8 byte storage must be viewed as {name} before loading")
    if data.dtype != expected_dtype or tuple(data.shape) != expected_shape:
        raise ValueError(
            f"{key}: expected {expected_dtype} storage {expected_shape}, got {data.dtype} {tuple(data.shape)}"
        )
    scale = params.scale
    if not torch.isfinite(scale.float()).all() or (scale.float() < 0).any():
        raise ValueError(f"{key}: scales must be finite and nonnegative")
    if name in ("mxfp8", "nvfp4"):
        block = 32 if name == "mxfp8" else 16
        scales = scale if name == "mxfp8" else params.block_scale
        expected_scales = ((n + 127) // 128 * 128, ((k + block - 1) // block + 3) // 4 * 4)
        expected_scale_dtype = torch.float8_e8m0fnu if name == "mxfp8" else torch.float8_e4m3fn
        if tuple(scales.shape) != expected_scales or scales.dtype != expected_scale_dtype:
            raise ValueError(f"{key}: invalid blocked scale shape or dtype")
        if not torch.isfinite(scales.float()).all() or (scales.float() < 0).any():
            raise ValueError(f"{key}: block scales must be finite and nonnegative")
        if name == "nvfp4" and (scale.numel() != 1 or scale.dtype != torch.float32):
            raise ValueError(f"{key}: NVFP4 requires a scalar FP32 global scale")
    if name.startswith("float8_") and scale.numel() != 1:
        raise ValueError(f"{key}: FP8 requires a scalar scale")
    if name.startswith("int8_") and scale.numel() != 1 and tuple(scale.shape) != (n, 1):
        raise ValueError(f"{key}: INT8 requires scalar or [out_features, 1] scale")
    if name.startswith("int8_") and params.convrot:
        group = params.convrot_groupsize
        if type(group) is not int or group < 4 or group & (group - 1) or (group.bit_length() - 1) % 2 or k % group:
            raise ValueError(f"{key}: ConvRot requires a power-of-four group dividing input width")
        if tuple(scale.shape) != (n, 1) or scale.dtype != torch.float32:
            raise ValueError(f"{key}: ConvRot requires FP32 per-output-row scales")
    if name in ("convrot_w4a4", "convrot_w4a8"):
        if tuple(scale.shape) != (n,) or params.quant_group_size != 64:
            raise ValueError(f"{key}: ConvRot W4 uses per-output-row scales and quant_group_size=64")
    if name == "awq_w4a16":
        group = params.group_size
        if type(group) is not int or group <= 0 or k % group or tuple(scale.shape) != (k // group, n):
            raise ValueError(f"{key}: invalid grouped scale shape")
        if name == "awq_w4a16" and params.zeros.shape != scale.shape:
            raise ValueError(f"{key}: AWQ zeros and scale shapes must match")
    if name == "svdquant_w4a4":
        tiled = data.ndim == 4
        expected_scale = (n // 128, k // 64, 128) if tiled else (k // 64, n)
        rank = params.proj_down.shape[-1] if params.proj_down.ndim == 2 else 0
        expected_up = (n // 128, rank, 128) if tiled else (n, rank)
        if (
            tuple(scale.shape) != expected_scale
            or params.proj_down.shape != (k, rank)
            or tuple(params.proj_up.shape) != expected_up
            or params.smooth_factor.shape != (k,)
            or rank <= 0
        ):
            raise ValueError(f"{key}: invalid SVDQuant scales, smoothing or low-rank shapes")
        if not torch.isfinite(params.smooth_factor).all() or (params.smooth_factor <= 0).any():
            raise ValueError(f"{key}: SVDQuant smoothing must be finite and positive")
