"""TREAD token selection and differentiable residual bypass for native DiTs.

Reference: TREAD, Token Routing for Efficient Architecture-agnostic Diffusion
Training (https://arxiv.org/abs/2501.04765). Only target-image tokens are sampled; conditioning and padding retain
their order. A selection is immutable for one forward and its checkpoint replay.
"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from unitunner.config.routing import TreadConfig


@dataclass(frozen=True, slots=True)
class TargetSpan:
    """A half-open target-token interval in one native sequence, excluding conditioning."""

    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class TokenSelection:
    """Static target geometry and kept-token count for one forward's bypass."""

    span: TargetSpan
    kept: int

    @property
    def removed(self) -> int:
        return self.span.length - self.kept


@dataclass(frozen=True, slots=True)
class TokenRoute:
    """One forward's sorted indices and static offsets for each packed sample.

    ``indices`` is int64 [R,K] on the compute device. R=1 shares a selection
    across a dense batch; ragged layouts supply one row per sample. Each row's
    targets hold ordered intervals and explicit kept-token counts. Unselected tokens
    bypass blocks through their original residual, retaining their gradients.
    """

    config: TreadConfig
    indices: Tensor
    targets: tuple[tuple[TokenSelection, ...], ...]

    def applies(self, layer: int) -> bool:
        """Whether this visual block belongs to the routing window."""
        return self.config.start_layer <= layer < self.config.end_layer

    def boundary(self, position: int, row: int = 0) -> int:
        """Translate a segment boundary; boundaries inside target spans are invalid."""
        removed = 0
        for target in self.targets[row]:
            if target.span.start < position < target.span.stop:
                raise ValueError("Routing boundaries must not split a sampled target span")
            if target.span.stop <= position:
                removed += target.removed
        return position - removed

    def relocate(self, length: int, spans: tuple[TargetSpan, ...]) -> Self:
        """Carry a packed target selection across a text/image stream concatenation.

        Target spans must retain their order and original lengths. No RNG is
        consumed: the same target tokens bypass every block in the window.
        """
        if len(self.targets) != 1 or len(spans) != len(self.targets[0]):
            raise ValueError("Packed relocation must retain the target span count")
        parts = []
        layout = []
        cursor = 0
        for previous, span in zip(self.targets[0], spans, strict=True):
            if not cursor <= span.start < span.stop <= length or span.length != previous.span.length:
                raise ValueError("Packed relocation must preserve ordered target span lengths")
            offset = self.boundary(previous.span.start)
            parts.append(torch.arange(cursor, span.start, device=self.indices.device))
            parts.append(self.indices[0, offset : offset + previous.kept] - previous.span.start + span.start)
            layout.append(TokenSelection(span, previous.kept))
            cursor = span.stop
        parts.append(torch.arange(cursor, length, device=self.indices.device))
        return type(self)(self.config, torch.cat(parts)[None], (tuple(layout),))

    def select(self, value: Tensor, *, dim: int = 1, batched: bool = True) -> Tensor:
        """Gather token-aligned states/RoPE/masks while preserving other dimensions.

        Batched tensors have batch axis zero and token axis ``dim``; a singleton
        route broadcasts across the batch. Unbatched tensors require one route
        row and use ``dim`` as their token axis. Dtype and autograd are retained.
        """
        if not batched:
            if self.indices.shape[0] != 1:
                raise ValueError("Unbatched metadata requires a shared routing selection")
            return value.index_select(dim, self.indices[0].to(value.device))
        moved = value.movedim(dim, 1)
        indices = self.indices.to(value.device)
        # index_select backward saves sizes/indices rather than the full residual
        # tensor retained by gather backward. This matters across a deep route.
        if indices.shape[0] == 1:
            return moved.index_select(1, indices[0]).movedim(1, dim)
        if indices.shape[0] != moved.shape[0]:
            raise ValueError("Per-sample routing must match the tensor batch size")
        offsets = torch.arange(moved.shape[0], device=value.device)[:, None] * moved.shape[1]
        selected = moved.flatten(0, 1).index_select(0, (indices + offsets).flatten())
        return selected.reshape(moved.shape[0], indices.shape[1], *moved.shape[2:]).movedim(1, dim)

    def restore(self, original: Tensor, updated: Tensor, *, batched: bool = True) -> Tensor:
        """Scatter updated tokens into the residual; skipped tokens follow identity.

        Inputs are floating [B,L,D]/[B,K,D], or [L,D]/[K,D] for packed models.
        Output has the original shape in the updated tensor's dtype/device. No in-place
        mutation or detached copy may sever the skipped-token gradient path.
        """
        if not batched:
            return original.to(updated).index_copy(0, self.indices[0].to(updated.device), updated)
        indices = self.indices.to(updated.device).expand(updated.shape[0], -1)
        return original.to(updated).scatter(1, indices[..., None].expand_as(updated), updated)


def sample_token_route(
    config: TreadConfig | None,
    *,
    training: bool,
    length: int,
    spans: tuple[tuple[TargetSpan, ...], ...],
    device: torch.device,
) -> TokenRoute | None:
    """Sample a uniform subset per target span using the run's checkpointed RNG.

    ``spans`` contains sorted, disjoint target intervals per dense batch row, or
    one row containing all target intervals of a packed sequence. All rows must
    remove the same number of tokens. Keep at least one token per nonempty span.
    Evaluation and no-grad EMA teachers neither route tokens nor consume RNG.
    """
    if config is None or not training or not torch.is_grad_enabled():
        return None
    if not spans or any(not row for row in spans):
        raise ValueError("Token routing requires nonempty target spans")
    layouts: list[tuple[TokenSelection, ...]] = []
    for row in spans:
        cursor = 0
        layout: list[TokenSelection] = []
        for span in row:
            if not cursor <= span.start < span.stop <= length:
                raise ValueError("Routing targets must be sorted, nonempty and within the sequence")
            kept = max(1, int(span.length * (1 - config.drop_fraction)))
            layout.append(TokenSelection(span, kept))
            cursor = span.stop
        layouts.append(tuple(layout))
    removed = {sum(target.removed for target in row) for row in layouts}
    if len(removed) != 1:
        raise ValueError("Dense routing rows must remove equal token counts")
    selections = []
    for row in layouts:
        cursor = 0
        parts = []
        for target in row:
            parts.append(torch.arange(cursor, target.span.start, device=device))
            # Sorted indices preserve causal/segment order without a second inverse permutation.
            chosen = (
                torch.rand(target.span.length, device=device).topk(target.kept, largest=False).indices.sort().values
            )
            parts.append(chosen + target.span.start)
            cursor = target.span.stop
        parts.append(torch.arange(cursor, length, device=device))
        selections.append(torch.cat(parts))
    return TokenRoute(config, torch.stack(selections), tuple(layouts))
