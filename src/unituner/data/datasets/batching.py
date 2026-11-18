"""Replay repeat-expanded shape buckets as shuffled epochs with exact resume cursors."""

import random
from collections.abc import Mapping, Sequence


class EpochBatches[T]:
    """Own repeat-expanded groups and one cached deterministic shuffled epoch.

    Groups are independent datasets/buckets. Shuffle samples within each group,
    chunk without dropping or wrapping tails, then shuffle batches. Local Python
    RNG leaves augmentation and model RNG untouched. A microbatch cursor alone
    reconstructs the epoch on resume; no mutable sampler checkpoint is needed.
    """

    def __init__(
        self,
        groups: Mapping[tuple[int, ...], Sequence[T]],
        *,
        batch_size: int | Mapping[tuple[int, ...], int],
        seed: int,
    ) -> None:
        if not groups or any(not values for values in groups.values()):
            raise ValueError("Epoch batches require nonempty groups")
        keys = sorted(groups)
        self._groups = tuple(tuple(groups[key]) for key in keys)
        self._batch_sizes = tuple(batch_size if isinstance(batch_size, int) else batch_size[key] for key in keys)
        if min(self._batch_sizes) < 1:
            raise ValueError("Epoch batch sizes must be positive")
        self._seed = seed
        self._epoch = -1
        self._batches: tuple[tuple[T, ...], ...] = ()
        self.num_batches = sum(
            (len(group) + size - 1) // size for group, size in zip(self._groups, self._batch_sizes, strict=True)
        )
        self.num_samples = sum(map(len, self._groups))

    def batch(self, microbatch: int) -> tuple[T, ...]:
        """Return one uniform bucket, retaining its partial tail, at an absolute cursor."""
        if microbatch < 0:
            raise ValueError("Microbatch cursor must be nonnegative")
        epoch, offset = divmod(microbatch, self.num_batches)
        if epoch != self._epoch:
            rng = random.Random(self._seed + epoch)
            batches = []
            for group, size in zip(self._groups, self._batch_sizes, strict=True):
                items = list(group)
                rng.shuffle(items)
                batches.extend(tuple(items[start : start + size]) for start in range(0, len(items), size))
            rng.shuffle(batches)
            self._batches, self._epoch = tuple(batches), epoch
        return self._batches[offset]

    def samples_before(self, microbatches: int) -> int:
        """Count completed samples, including epoch-dependent short-tail positions."""
        if microbatches < 0:
            raise ValueError("Microbatch cursor must be nonnegative")
        epochs, tail = divmod(microbatches, self.num_batches)
        return epochs * self.num_samples + sum(len(self.batch(epochs * self.num_batches + i)) for i in range(tail))


class DatasetBatches[T]:
    """Shuffle independent dataset batches together without combining their partial tails."""

    def __init__(
        self, plans: Mapping[str, EpochBatches[T]], *, seed: int, batch_weights: Mapping[str, int] | None = None
    ) -> None:
        if not plans:
            raise ValueError("Dataset batches require at least one dataset")
        if batch_weights is not None and (
            set(batch_weights) != set(plans) or any(type(weight) is not int or weight < 1 for weight in batch_weights.values())
        ):
            raise ValueError("Dataset batch weights require positive integers for every group")
        self._plans = dict(plans)
        self._batch_weights = dict(batch_weights) if batch_weights is not None else None
        self._seed = seed
        self._epoch = -1
        self._order: list[tuple[str, int]] = []
        self.num_batches = (
            sum(batch_weights.values()) if batch_weights is not None else sum(plan.num_batches for plan in plans.values())
        )
        self.num_samples = (
            sum(plan.samples_before(batch_weights[name]) for name, plan in plans.items())
            if batch_weights is not None
            else sum(plan.num_samples for plan in plans.values())
        )

    def batch(self, microbatch: int) -> tuple[str, tuple[T, ...]]:
        """Return a named group and its batch at an exact resumable cursor."""
        if microbatch < 0:
            raise ValueError("Microbatch cursor must be nonnegative")
        epoch, offset = divmod(microbatch, self.num_batches)
        if epoch != self._epoch:
            self._order = [
                (name, index)
                for name, plan in self._plans.items()
                for index in range(self._batch_weights[name] if self._batch_weights is not None else plan.num_batches)
            ]
            random.Random(self._seed + epoch).shuffle(self._order)
            self._epoch = epoch
        name, index = self._order[offset]
        plan = self._plans[name]
        cycle_count = self._batch_weights[name] if self._batch_weights is not None else plan.num_batches
        return name, plan.batch(epoch * cycle_count + index)

    def samples_before(self, microbatches: int) -> int:
        """Count completed samples including independent short tails in every dataset."""
        if microbatches < 0:
            raise ValueError("Microbatch cursor must be nonnegative")
        epochs, tail = divmod(microbatches, self.num_batches)
        if self._batch_weights is not None:
            completed = sum(
                plan.samples_before(epochs * self._batch_weights[name]) for name, plan in self._plans.items()
            )
            return completed + sum(len(self.batch(epochs * self.num_batches + i)[1]) for i in range(tail))
        return epochs * self.num_samples + sum(len(self.batch(epochs * self.num_batches + i)[1]) for i in range(tail))
