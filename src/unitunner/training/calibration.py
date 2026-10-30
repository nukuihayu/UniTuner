"""Estimate image-restoration time without loading text, denoiser or training state."""

import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from unitunner.config.calibration import CalibrationConfig
from unitunner.data.media.image import discover_images, index_conditions
from unitunner.data.media.restoration import iter_aligned_images
from unitunner.execution.runtime import runtime_scope
from unitunner.inference.restoration.vae import (
    encode_restoration_image,
    restoration_posterior,
)
from unitunner.models.components.image_encoders import load_image_vae
from unitunner.training.methods.snr_calibration import calibration_rng, estimate_mid_timestep


@torch.no_grad()
@runtime_scope
def run(config: CalibrationConfig, *, device: str) -> int:
    """Print an OMGSR mid-timestep estimate from real or synthetic aligned image pairs.

    Read the training TOML's image sources, crop size, seed, degradation and
    calibration_pairs limit. Load only the frozen FP32 VAE. Seeded posteriors
    and item-local crop/degradation generators make calibration deterministic;
    global CPU/CUDA RNG is restored after VAE construction. No captions,
    caches, denoiser, text encoder or optimizer artifacts are required.
    Output JSON includes index, exact noise fraction, mean SNR error and count;
    nothing is written back into the caller's configuration or model files.
    """
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("Mid-timestep estimation supports CPU/CUDA")
    if selected.type == "cuda" and selected.index is None:
        selected = torch.device("cuda", torch.cuda.current_device())
    devices = [selected.index or 0] if selected.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        vae = load_image_vae(Path(config.model.vae), family=config.model.family, device=selected)
        posterior = restoration_posterior(config.model.family, config.posterior)

        def latent_pairs() -> Iterator[tuple[Tensor, Tensor]]:
            count = 0
            offset = 0
            for source in config.dataset.groups.values():
                targets = discover_images(Path(source.source))
                if len({path.stem for path in targets}) != len(targets):
                    raise ValueError("Calibration requires unique HQ image stems")
                conditions = (
                    None
                    if source.low_quality_source is None
                    else index_conditions(targets, Path(source.low_quality_source))
                )
                for index, high in enumerate(targets):
                    low = None if conditions is None else conditions[high.stem]
                    for pixels_low, pixels_high in iter_aligned_images(
                        low,
                        high,
                        resolution=source.resolution,
                        seed=config.seed + offset + index,
                        flip_probability=source.horizontal_flip_probability,
                        degradation=source.degradation,
                        sampling=config.calibration_sampling,
                    ):
                        yield (
                            encode_restoration_image(vae, pixels_low.unsqueeze(0).to(selected), posterior=posterior),
                            encode_restoration_image(vae, pixels_high.unsqueeze(0).to(selected), posterior=posterior),
                        )
                        count += 1
                        if count == config.calibration_pairs:
                            return
                offset += len(targets) * source.repeats

        with calibration_rng(config.seed, selected), torch.autocast(selected.type, enabled=False):
            estimate = estimate_mid_timestep(
                latent_pairs(),
                family=config.model.family,
                resolution=next(iter(config.dataset.groups.values())).resolution,
            )
    print(
        json.dumps(
            {
                "family": config.model.family,
                "resolution": next(iter(config.dataset.groups.values())).resolution,
                "posterior": posterior,
                "sampling": config.calibration_sampling,
                **asdict(estimate),
            }
        )
    )
    return 0
