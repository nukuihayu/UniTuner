"""Own encoder weight adapters without changing base artifact keys or parameters."""

import json
import math
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.artifacts.tensors import TensorReader


class EncoderWeightLoRA(nn.Module):
    """Factor flattened input/kernel axes as [O,R]@[R,I*kernel] with unit scaling.

    Preserve the original encoder-adapter rule: FP32 factors, zero-initialized
    up matrix, no alpha/rank multiplier. Delta is differentiable and unmerged.
    """

    def __init__(self, weight: Tensor, rank: int) -> None:
        super().__init__()
        if rank < 1 or weight.ndim not in (2, 4, 5) or weight.dtype != torch.float32:
            raise ValueError("Encoder adapters require positive rank and FP32 linear/2D/3D weights")
        self.shape = tuple(weight.shape)
        width = math.prod(weight.shape[1:])
        self.down = nn.Parameter(torch.randn(rank, width, device=weight.device, dtype=torch.float32) / math.sqrt(width))
        self.up = nn.Parameter(torch.zeros(weight.shape[0], rank, device=weight.device, dtype=torch.float32))
        self.enabled = True

    def delta(self) -> Tensor:
        """Return a newly allocated FP32 weight delta in the original projection layout."""
        return (self.up @ self.down).reshape(self.shape)


class _WeightAttachment:
    """Own one ephemeral merged-weight path with exact original callable restoration."""

    def __init__(self, projection: nn.Linear | nn.Conv2d | nn.Conv3d, branch: EncoderWeightLoRA) -> None:
        if isinstance(projection, (nn.Conv2d, nn.Conv3d)) and projection.padding_mode != "zeros":
            raise ValueError("Encoder adapters require zero convolution padding")
        self.projection = projection
        self._attribute = "forward" if isinstance(projection, nn.Linear) else "_conv_forward"
        self._had_override = self._attribute in projection.__dict__
        self._closed = False
        if isinstance(projection, nn.Linear):
            original_linear = projection.forward
            self._original = original_linear

            def linear(inputs: Tensor) -> Tensor:
                return (
                    F.linear(inputs, projection.weight + branch.delta(), projection.bias)
                    if branch.enabled
                    else original_linear(inputs)
                )

            self._installed = linear
        else:
            # Torch 2.13 convolutions call this boundary after custom causal padding.
            # Keep the subclass forward signature/history intact and never swap weights.
            original_convolution = projection._conv_forward
            self._original = original_convolution

            def convolution(inputs: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
                # Merge the ephemeral weight before convolution to preserve native
                # parametrization rounding; never mutate the stored base parameter.
                return original_convolution(inputs, weight + branch.delta() if branch.enabled else weight, bias)

            self._installed = convolution
        projection.__dict__[self._attribute] = self._installed

    def close(self) -> None:
        """Restore only the owned callable, rejecting incorrect nested-owner order."""
        if self._closed:
            return
        current = self.projection.forward if isinstance(self.projection, nn.Linear) else self.projection._conv_forward
        if current is not self._installed:
            raise RuntimeError("Encoder weight attachment must close after later forward owners")
        if self._had_override:
            # The original inherited/bound callable was captured before installation.
            self.projection.__dict__[self._attribute] = self._original
        else:
            del self.projection.__dict__[self._attribute]
        self._closed = True


class EncoderAdapters(nn.Module):
    """Own only trainable encoder branches and reversible base trainability changes.

    Base encoder parameters/keys remain unchanged, including while installed.
    Decoder is untouched. Install before optimizer/compiler and close after all
    dependent backward/recompute calls. The caller retains encoder ownership.
    """

    def __init__(self, encoder: nn.Module, *, rank: int) -> None:
        super().__init__()
        projections = {
            name: module
            for name, module in encoder.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d))
        }
        if not projections or rank < 1:
            raise ValueError("Encoder adapters require projections and positive rank")
        self.names = tuple(projections)
        self.layers = nn.ModuleList()
        self._branches: tuple[EncoderWeightLoRA, ...] = ()
        self._owners = ExitStack()
        for parameter in encoder.parameters():
            self._owners.callback(parameter.requires_grad_, parameter.requires_grad)
        try:
            encoder.requires_grad_(False)
            branches = []
            for projection in projections.values():
                branch = EncoderWeightLoRA(projection.weight, rank)
                attachment = _WeightAttachment(projection, branch)
                self._owners.callback(attachment.close)
                self.layers.append(branch)
                branches.append(branch)
            self._branches = tuple(branches)
        except BaseException:
            self._owners.close()
            raise

    @contextmanager
    def unadapted(self) -> Generator[None]:
        """Temporarily disable deltas for a frozen HQ target, restoring nested states."""
        flags = tuple(branch.enabled for branch in self._branches)
        try:
            for branch in self._branches:
                branch.enabled = False
            with torch.no_grad():
                yield
        finally:
            for branch, enabled in zip(self._branches, flags, strict=True):
                branch.enabled = enabled

    def close(self) -> None:
        """Detach all owned forwards in reverse order and restore base gradient flags."""
        self._owners.close()


def load_encoder_adapters(encoder: nn.Module, path: Path) -> EncoderAdapters:
    """Load floating encoder factors into FP32 with exact target and rank metadata.

    Return an owned reversible installation; caller must close it. Validate all
    keys, shapes, floating finite values and metadata before installing forwards.
    No base weights/parameter identities change. A failed load leaves the encoder
    unchanged. Only native safetensors exports are accepted, without aliases.
    """
    with TensorReader(path) as reader:
        metadata = reader.metadata()
        try:
            names = json.loads(metadata["unitunner.encoder.targets"])
            rank = int(metadata["unitunner.encoder.rank"])
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("Encoder artifact requires valid native targets and rank metadata") from error
        projections = {
            name: module
            for name, module in encoder.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d))
        }
        if not isinstance(names, list) or names != list(projections) or rank < 1:
            raise ValueError("Encoder artifact targets/rank differ from the supplied encoder")
        expected = {}
        for index, projection in enumerate(projections.values()):
            expected[f"layers.{index}.down"] = (rank, math.prod(projection.weight.shape[1:]))
            expected[f"layers.{index}.up"] = (projection.weight.shape[0], rank)
        if set(reader.keys()) != set(expected):
            raise ValueError("Encoder artifact tensor keys differ from the complete target inventory")
        tensors = {}
        for key, shape in expected.items():
            tensor = reader.read(key)
            if tuple(tensor.shape) != shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"Encoder artifact tensor shape/type/values invalid: {key}")
            tensor = tensor.float()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Encoder artifact tensor exceeds FP32 range: {key}")
            tensors[key] = tensor
        devices = sorted(
            {
                parameter.device.index if parameter.device.index is not None else torch.cuda.current_device()
                for parameter in encoder.parameters()
                if parameter.device.type == "cuda"
            }
        )
        with torch.random.fork_rng(devices=devices):
            owner = EncoderAdapters(encoder, rank=rank)
        try:
            owner.load_state_dict(tensors, strict=True)
            owner.eval().requires_grad_(False)
        except BaseException:
            owner.close()
            raise
    return owner
