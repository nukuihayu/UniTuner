"""Preserve training flags and explicit CPU/CUDA RNG streams across temporary previews."""

import random
from collections.abc import Generator, Sequence
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn


@contextmanager
def preview_context(
    model: nn.Module, *, additional_models: Sequence[nn.Module] = (), cuda_devices: Sequence[int] = ()
) -> Generator[None]:
    """Enter no-grad evaluation and restore exact child flags and RNG even on failure.

    Call only after the training graph and transfers finish. CUDA RNG is captured
    only on explicitly supplied devices; no device enumeration/initialization is
    triggered for CPU previews. EMA/residency ownership belongs to their separate
    installations and is not silently changed here.
    """
    roots = (model, *additional_models)
    flags = tuple(
        (module, module.training) for module in dict.fromkeys(child for root in roots for child in root.modules())
    )
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=list(cuda_devices)), torch.no_grad():
        try:
            for root in roots:
                root.eval()
            yield
        finally:
            for module, training in flags:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
