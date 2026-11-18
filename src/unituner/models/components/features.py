"""Retain explicitly selected target-visual features for one student forward scope."""

from torch import Tensor


class FeatureRecorder:
    """Own selected BLD target features while active; never register model parameters.

    Native model loops remove text, padding, audio and references before calling
    record. Capture ends before backward, so checkpoint replay cannot append
    another graph. Multiple native calls append in sample order.
    """

    def __init__(self, layers: frozenset[int]) -> None:
        if not layers or min(layers) < 0:
            raise ValueError("Feature capture requires nonnegative selected layers")
        self.layers: frozenset[int] = layers
        self.active: bool = False
        self.features: dict[int, list[Tensor]] = {}

    def record(self, index: int, visual: Tensor) -> None:
        """Retain native BLD target tokens without copying or detaching their graph."""
        if self.active and index in self.layers:
            if visual.ndim != 3 or min(visual.shape) < 1:
                raise ValueError("Feature capture requires nonempty BLD target tokens")
            self.features.setdefault(index, []).append(visual)

    def clear(self) -> None:
        """Deactivate and release retained graphs idempotently."""
        self.active = False
        self.features.clear()
