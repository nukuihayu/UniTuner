"""Own a typed compiled callable without scanning or rewriting unrelated module state."""

from collections.abc import Callable
from functools import partial
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch._dynamo import config as dynamo_config
from torch._functorch import config as autograd_config
from torch._inductor import compile as compile_inductor
from torch.fx import GraphModule

from unitunner.artifacts.formats.precision import FloatingPrecision
from unitunner.execution.attention import compiled_attention
from unitunner.models.components.forward_override import ForwardOverride

type CompilerInput = Tensor | int | torch.SymInt | None


def _compile_graph(
    graph: GraphModule,
    inputs: list[CompilerInput],
    *,
    preserve_precision: bool,
) -> Callable[..., tuple[CompilerInput, ...]]:
    """Scope precision and memory policy to this compilation, including backward."""
    return cast(
        Callable[..., tuple[CompilerInput, ...]],
        compile_inductor(
            graph,
            inputs,
            options={
                "emulate_precision_casts": preserve_precision,
                # Avoid private graph pools with variable buckets and swapped weights.
                "triton.cudagraphs": False,
                # Keep pointwise epilogues fused instead of storing intermediates.
                "epilogue_fusion": True,
            },
        ),
    )


class CompilerInstallation[R: Tensor | tuple[Tensor, ...], **P]:
    """Compile one typed tensor or tensor-tuple callable on stable parameter identities.

    Construction registers only the named callable mutation. Compilation is lazy:
    first-call failures propagate and the session context restores the callable.
    Close after outstanding backward/transfer work, in reverse ownership order.
    Tensor arguments and keyword policies retain their original callable contract.
    Each owner has an isolated Python compilation boundary, with dynamic shapes
    requested from the first training call. Evaluation uses the native callable
    so previews match standalone generation and cannot generalize the training
    graph. Compiler settings are scoped to the training call and restored on exit.
    fullgraph=True rejects graph breaks instead of falling back. Compiled graphs
    use native buffer donation: one backward per graph, without retain_graph.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        backend: Literal["eager", "inductor"] = "inductor",
        fullgraph: bool = False,
        precision: FloatingPrecision = "fp32",
    ) -> None:
        if not torch.__version__.startswith("2.13."):
            raise RuntimeError("Training compilation requires the qualified Torch2.13 provider")
        self.module: nn.Module = module
        original: Callable[P, R] = module.forward

        def training_boundary(*args: P.args, **kwargs: P.kwargs) -> R:
            return original(*args, **kwargs)

        # Dynamo caches guards per Python code object. Shared installation closures
        # otherwise exhaust one recompile budget across unrelated model regions.
        training_boundary.__code__ = training_boundary.__code__.replace(
            co_name=f"compiled_training_{id(self)}", co_qualname=f"compiled_training_{id(self)}"
        )
        provider = (
            partial(_compile_graph, preserve_precision=precision in ("bf16", "fp16"))
            if backend == "inductor"
            else backend
        )
        training = torch.compile(training_boundary, backend=provider, fullgraph=fullgraph, dynamic=True)

        def installed(*args: P.args, **kwargs: P.kwargs) -> R:
            if module.training and torch.is_grad_enabled():
                with compiled_attention(fullgraph=fullgraph):
                    return compiled_training(*args, **kwargs)
            return original(*args, **kwargs)

        def compiled_training(*args: P.args, **kwargs: P.kwargs) -> R:
            """Keep compiler guard policy outside the traced tensor boundary."""
            # The engine leaves forward AMP before invoking backward. AOT's
            # default assumes backward uses forward AMP, which changes gradients.
            with autograd_config.patch(backward_pass_autocast="off"):
                if (
                    dynamo_config.recompile_limit == dynamo_config.accumulated_recompile_limit
                    and dynamo_config.fail_on_recompile_limit_hit
                ):
                    return training(*args, **kwargs)
                with dynamo_config.patch(
                    recompile_limit=dynamo_config.accumulated_recompile_limit,
                    fail_on_recompile_limit_hit=True,
                ):
                    return training(*args, **kwargs)

        self._forward = ForwardOverride(module, torch.compiler.disable()(installed), owner="CompilerInstallation")

    def close(self) -> None:
        """Restore only this installation's callable; repeated close is harmless."""
        self._forward.close()
