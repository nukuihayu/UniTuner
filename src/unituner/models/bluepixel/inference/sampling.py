"""BluePixel's reference-conditioned Euler flow matching inference."""

import math
from itertools import pairwise

import torch
from torch import Tensor

from unituner.models.bluepixel.transformer import BluePixelTransformer
from unituner.models.flux2.conditioning import build_image_inputs, pack_text, unpack_image


def flow_schedule(steps: int, token_count: int) -> tuple[float, ...]:
    """Shift the published 1..1/steps sigma grid using FLUX.2's empirical mu."""
    if type(steps) is not int or steps < 1 or type(token_count) is not int or token_count < 1:
        raise ValueError("BluePixel requires positive steps and target token count")
    m200 = 0.00016927 * token_count + 0.45666666
    m10 = 8.73809524e-05 * token_count + 1.89833333
    mu = m200 if token_count > 4300 else m200 + (steps - 200) * (m200 - m10) / 190
    if not math.isfinite(mu) or mu > math.log(torch.finfo(torch.float32).max):
        raise ValueError("BluePixel empirical time shift exceeds FP32 range")
    shift = math.exp(mu)
    raw = torch.linspace(1, 1 / steps, steps, dtype=torch.float32)
    grid = (shift * raw) / (1 + (shift - 1) * raw)
    result = (*grid.tolist(), 0.0)
    if not all(a > b for a, b in pairwise(result)):
        raise ValueError("BluePixel schedule must strictly descend to zero")
    return result


@torch.no_grad()
def sample_latents(
    model: BluePixelTransformer,
    noise: Tensor,
    reference: Tensor,
    positive: Tensor,
    negative: Tensor | None = None,
    *,
    steps: int = 4,
    guidance: float = 3.0,
) -> Tensor:
    """Predict the target velocity while preserving clean reference tokens at time ID 10."""
    if (
        noise.ndim != 4 or noise.shape[0] != 1 or reference.shape != noise.shape
        or noise.shape[1] != model.config.in_channels or noise.device != reference.device
        or noise.dtype != reference.dtype or positive.ndim != 2
        or positive.shape[1] != model.config.context_in_dim or positive.device != noise.device
    ):
        raise ValueError("BluePixel requires matching BCHW target/reference and LD text features")
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("BluePixel CFG guidance must be finite and nonnegative")
    if guidance > 1 and (negative is None or negative.shape != positive.shape or negative.device != noise.device):
        raise ValueError("BluePixel CFG requires an empty-prompt caption matching positive features")
    inputs = build_image_inputs(noise[0], (reference[0],))
    text, text_ids = pack_text(positive)
    if negative is not None:
        uncond, uncond_ids = pack_text(negative)
    state = inputs.tokens[:, : inputs.target_count].clone()
    clean = inputs.tokens[:, inputs.target_count :]
    for current, following in pairwise(flow_schedule(steps, inputs.target_count)):
        tokens = torch.cat((state, clean), dim=1)
        time = state.new_full((1,), current)
        prediction = model(tokens, inputs.ids, time, text.unsqueeze(0), text_ids.unsqueeze(0), None)
        if guidance > 1 and negative is not None:
            unconditional = model(tokens, inputs.ids, time, uncond.unsqueeze(0), uncond_ids.unsqueeze(0), None)
            prediction = unconditional + guidance * (prediction - unconditional)
        state = state + (following - current) * prediction[:, : inputs.target_count].to(state.dtype)
    return unpack_image(state, height=noise.shape[-2], width=noise.shape[-1])
