"""Own reversible linear attachment without changing base parameter identity or keys."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from unitunner.adapters.loha import LoHa
from unitunner.adapters.lokr import LoKr
from unitunner.adapters.lora import LoRA
from unitunner.adapters.ops.lora import fused_lora_linear
from unitunner.models.components.forward_override import ForwardOverride
from unitunner.models.components.quantized import linear_dtype

type LinearBranch = LoRA | LoHa | LoKr


class AdapterAttachment:
    """Own exactly one forward replacement; keep branch state outside the base model.

    Install before optimizer/compile. Close only after backward and transfers
    have completed. Nested owners must close in reverse order. No attribute
    scanning or old wrapper discovery occurs. The caller owns branch parameters.
    """

    def __init__(
        self, projection: nn.Linear, branch: LinearBranch, *, allow_cpu_frozen: bool = False, fused: bool = True
    ) -> None:
        factor = next(branch.parameters())
        staged = (
            allow_cpu_frozen
            and projection.weight.device.type == "cpu"
            and not projection.weight.requires_grad
            and factor.device.type == "cuda"
        )
        if (
            branch.out_features,
            branch.in_features,
        ) != (projection.out_features, projection.in_features) or (
            factor.device != projection.weight.device and not staged
        ):
            raise ValueError("Adapter and projection weight shape/device must agree")
        base_dtype = linear_dtype(projection)
        if factor.dtype != base_dtype and not (
            factor.dtype == torch.float32 and base_dtype in (torch.float16, torch.bfloat16)
        ):
            raise ValueError("Adapter dtype must match the projection or use FP32 masters for a low precision base")
        self.projection: nn.Linear = projection
        self.branch: LinearBranch = branch
        original: Callable[[Tensor], Tensor] = projection.forward

        def installed(inputs: Tensor) -> Tensor:
            if (
                fused
                and type(projection) is nn.Linear
                and projection.weight.is_floating_point()
                and isinstance(branch, LoRA)
                and not projection.weight.requires_grad
                and (projection.bias is None or not projection.bias.requires_grad)
                and inputs.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
                and torch.is_grad_enabled()
            ):
                return fused_lora_linear(
                    inputs, projection.weight, projection.bias, branch.down, branch.up, branch.scale
                )
            return original(inputs) + branch(inputs)

        self._forward = ForwardOverride(projection, installed, owner="AdapterAttachment")

    def close(self) -> None:
        """Restore the exact owned callable; repeated close is harmless."""
        self._forward.close()
