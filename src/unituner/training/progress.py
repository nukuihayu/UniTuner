"""Display training metrics in place while retaining every update in file logs."""

from collections.abc import Mapping
from time import monotonic

from unituner.logging import get_logger
from unituner.progress import Progress


class TrainingProgress:
    """Use a live bar or bounded summaries, sharing the telemetry scalar snapshot."""

    def __init__(self, total: int, *, initial: int = 0, interval: float = 60.0) -> None:
        self.progress = Progress("[train] updates", total, initial=initial, unit="update")
        self.total = total
        self.interval = interval
        self.last_summary: float | None = None

    def report(self, update: int, samples: int, scalars: Mapping[str, float]) -> None:
        phase = "generator" if "loss/generator" in scalars else "main"
        loss = scalars[f"loss/{phase}"]
        detail = (
            f"loss={loss:.5g} time={scalars['performance/update_seconds']:.2f}s lr={scalars[f'train/{phase}/lr/0']:.3g}"
        )
        if "loss/discriminator" in scalars:
            detail += f" D={scalars['loss/discriminator']:.5g}"
        if "performance/samples_per_second" in scalars:
            detail += f" throughput={scalars['performance/samples_per_second']:.2f}sample/s"
        if f"train/{phase}/grad_norm" in scalars:
            detail += f" grad={scalars[f'train/{phase}/grad_norm']:.4g}"
        if "memory/allocated_gib" in scalars:
            detail += (
                f" VRAM={scalars['memory/allocated_gib']:.1f}GiB peak={scalars['memory/peak_allocated_gib']:.1f}GiB"
            )
        self.progress.report(update, detail=detail)
        now = monotonic()
        due = self.last_summary is None or now - self.last_summary >= self.interval or update == self.total
        if due:
            self.last_summary = now
        summary = self.progress.bar.disable and due
        get_logger("training").info(
            "[train] update=%s/%s samples=%s %s",
            update,
            self.total,
            samples,
            detail,
            extra={
                "metrics": dict(scalars),
                "console": summary,
                "event": "progress",
                "update": update,
                "total": self.total,
                "samples": samples,
            },
        )

    def close(self) -> None:
        """Release the live display at completion or on failure."""
        self.progress.close()
