"""Frozen quantized linears with one floating compute and input-gradient path."""

from dataclasses import fields

import torch
from torch import Tensor, nn


def linear_dtype(projection: nn.Linear) -> torch.dtype:
    """Return floating adapter precision independently of the base storage dtype."""
    if isinstance(projection, QuantizedLinear):
        return projection.compute_dtype
    return projection.weight.dtype


class _StoredLinear(torch.autograd.Function):
    """Reconstruct weights per call, saving only packed storage for backward."""

    @staticmethod
    def forward(ctx, inputs, layer, *storage):
        ctx.layer = layer
        ctx.save_for_backward(*storage)
        ctx.input_dtype = inputs.dtype
        if torch.is_autocast_enabled(inputs.device.type):
            inputs = inputs.to(torch.get_autocast_dtype(inputs.device.type))
        ctx.compute_dtype = inputs.dtype
        with torch.autocast(inputs.device.type, enabled=False):
            weight = layer.dequantized(inputs.dtype, storage=dict(zip(layer._storage_names, storage, strict=True)))
            bias = None if layer.bias is None else layer.bias.to(inputs.dtype)
            return torch.nn.functional.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, *gradients):
        (gradient,) = gradients
        # Touch saved tensors to enforce autograd's mutation/version checks.
        storage = ctx.saved_tensors
        result = None
        if ctx.needs_input_grad[0]:
            with torch.autocast(gradient.device.type, enabled=False):
                weight = ctx.layer.dequantized(
                    ctx.compute_dtype, storage=dict(zip(ctx.layer._storage_names, storage, strict=True))
                )
                result = (
                    (gradient.to(ctx.compute_dtype).reshape(-1, weight.shape[0]) @ weight)
                    .reshape(*gradient.shape[:-1], weight.shape[1])
                    .to(ctx.input_dtype)
                )
        return (result, None, *(None for _ in storage))


class QuantizedLinear(nn.Linear):
    """Canonical packed weights with exact floating-matmul input gradients.

    Unlike inference-only low-bit kernels, this path quantizes weights only.
    Dense weights are transient and are reconstructed again during backward.
    All sidecars are frozen parameters so residency owners stage them together.
    """

    _compute_anchor: Tensor
    comfy_quant: Tensor

    def __init__(self, tensors: dict[str, Tensor], *, dtype: torch.dtype) -> None:
        from unitunner.artifacts.formats.quantization import FORMATS
        from unitunner.artifacts.quantization import kitchen_types, read_weight, tensor_json

        nn.Module.__init__(self)
        self.config = tensor_json(tensors["comfy_quant"])
        packed = read_weight("weight", tensors.__getitem__, self.config, device=tensors["weight"].device)
        self.out_features, self.in_features = packed.shape
        format = FORMATS[self.config["format"]]
        self._layout = getattr(kitchen_types(), format.layout)
        self._param_options = {}
        self._param_storage = {}
        for field in fields(packed._params):
            value = getattr(packed._params, field.name)
            if not isinstance(value, Tensor):
                self._param_options[field.name] = value
                continue
            suffix = format.tensor_suffix(field.name)
            self._param_storage[field.name] = ("weight_" + suffix, value.dtype)
        if not torch.isfinite(packed._qdata.float()).all():
            raise ValueError("Quantized weight codes must be finite")
        self._storage_names = tuple(key for key in tensors if key not in ("comfy_quant", "bias"))
        for key in self._storage_names:
            self.register_parameter(key, nn.Parameter(tensors[key], requires_grad=False))
        bias = tensors.get("bias")
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.register_buffer("comfy_quant", tensors["comfy_quant"])
        self.register_buffer(
            "_compute_anchor", torch.empty(0, device=self.weight.device, dtype=dtype), persistent=False
        )

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._compute_anchor.dtype

    @property
    def quantization_format(self) -> str:
        """Identify storage variants from validated metadata without device reads."""
        name = self.config["format"]
        if name in ("int8_tensorwise", "int8_rowwise", "int8_convrot"):
            if self._param_options.get("convrot", False):
                return "int8_convrot"
            return (
                "int8_rowwise"
                if tuple(self.get_parameter("weight_scale").shape) == (self.out_features, 1)
                else "int8_tensorwise"
            )
        if name == "convrot_w4a4" and self._param_options.get("linear_dtype", "int4") == "int8":
            return "convrot_w4a8"
        return name

    def validate_quantization(self, precision: str, *, rotation_size: int, group_size: int) -> None:
        """Require existing storage to match the complete requested quantizer policy."""
        from unitunner.artifacts.formats.quantization import FORMATS

        if self.quantization_format != precision:
            raise ValueError(f"existing quantization differs from requested {precision}")
        format = FORMATS[precision]
        if format.uses_rotation and self._param_options.get("convrot_groupsize", 256) != rotation_size:
            raise ValueError(f"existing rotation size differs from requested {rotation_size}")
        if format.uses_group_size and self._param_options.get("group_size", 16) != group_size:
            raise ValueError(f"existing group size differs from requested {group_size}")

    def _apply(self, fn, recurse=True):
        original = {name: getattr(self, name).detach() for name in self._storage_names}
        result = super()._apply(fn, recurse=recurse)
        for name, value in original.items():
            setattr(self, name, nn.Parameter(value.to(device=self.weight.device), requires_grad=False))
        return result

    def dequantized(self, dtype: torch.dtype, *, storage: dict[str, Tensor] | None = None) -> Tensor:
        from comfy_kitchen import use_backend

        if storage is None:
            storage = {name: getattr(self, name) for name in self._storage_names}
        parameters = {
            field: storage[name].view(original_dtype) for field, (name, original_dtype) in self._param_storage.items()
        }
        params = self._layout.Params(**self._param_options, **parameters)
        # Backend overrides are thread-local, while autograd may run on another
        # thread. Pin canonical dequantization in both directions so backward
        # and replay reconstruct the same rounded weights, without revalidating
        # scales or synchronizing device tensors on every call.
        with use_backend("eager"):
            return self._layout.dequantize(storage["weight"], params).to(dtype)

    @torch.compiler.disable()
    def forward(self, input: Tensor) -> Tensor:
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise ValueError("Quantized base layers must remain frozen; train adapters instead")
        return _StoredLinear.apply(input, self, *(getattr(self, name) for name in self._storage_names))
