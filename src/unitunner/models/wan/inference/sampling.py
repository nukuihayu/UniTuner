"""Sample native Wan latent videos with explicit CFG, solver and resident expert choices."""

import math
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from unitunner.execution.attention import inference_attention
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.models.wan.conditioning import validate_wan_vae
from unitunner.models.wan.config.sampling import WanSamplingConfig
from unitunner.models.wan.inference.solvers import WanSolver
from unitunner.models.wan.transformer import WanTransformer

if TYPE_CHECKING:
    from unitunner.models.wan.inference.inputs import WanVideoInputs


@torch.no_grad()
def sample_latents(
    model: WanTransformer,
    noise: Tensor,
    captions: tuple[Tensor, ...],
    *,
    steps: int,
    solver: Literal["euler", "dpm++", "unipc"] = "unipc",
    shift: float = 5.0,
    guidance: float | None = 5.0,
    negative_captions: tuple[Tensor, ...] | None = None,
    conditioning: Tensor | None = None,
    image_features: Tensor | None = None,
    frame_indices: tuple[tuple[int, ...], ...] | None = None,
    high_noise_model: WanTransformer | None = None,
    boundary: float | None = None,
    high_noise_guidance: float | None = None,
    allow_cpu_frozen: bool = False,
) -> Tensor:
    """Integrate caller-supplied FP32 B16FHW noise using one or two resident experts.

    Caption tuples contain unpadded LD text. Optional conditioning is an already
    assembled native BCFHW image/control input; CLIP features retain their native
    layout. Both CFG calls share these conditions. Guidance=None requests only
    the conditional prediction; otherwise compute negative + scale*(positive -
    negative) in separate native calls. All temporal slots are integrated.

    Two-expert sampling requires equal Wan2.2 topology, an explicit normalized
    boundary and optionally a distinct high-noise CFG scale. Choose high noise
    inclusively from native truncated model time/1000, without reconstructing
    sigma or adding the training offset1. Preserve solver history across the
    expert transition. Borrow models/conditions without changing mode, weights,
    RNG or input storage. Model offload, SLG and full FLF2V are unsupported here.
    """
    if noise.ndim != 5 or min(noise.shape) < 1 or noise.dtype != torch.float32:
        raise ValueError("Wan sampling requires nonempty FP32 BCFHW noise")
    if model.config.variant == "flf2v":
        raise ValueError("Full FLF2V sampling is not accepted by this application yet")
    if noise.shape[1] != model.config.out_channels:
        raise ValueError("Wan sampling noise channels differ from model output")
    if (guidance is None) != (negative_captions is None):
        raise ValueError("Wan CFG requires negative_captions exactly when guidance is supplied")
    if high_noise_model is None:
        if boundary is not None or high_noise_guidance is not None:
            raise ValueError("Wan expert boundary/high-noise guidance requires a high_noise_model")
    elif model.config.version != "2.2" or high_noise_model.config != model.config or boundary is None:
        raise ValueError("Wan dual experts require matching2.2 topology and an explicit boundary")
    if guidance is None and high_noise_guidance is not None:
        raise ValueError("Wan high-noise CFG requires enabled guidance")
    for value in (guidance, high_noise_guidance):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Wan CFG scales must be finite and nonnegative")
    integration = WanSolver(steps, solver=solver, shift=shift)
    high_steps = (False,) * steps if boundary is None else integration.schedule.high_noise_steps(boundary=boundary)
    if conditioning is not None and conditioning.shape[0] != noise.shape[0]:
        raise ValueError("Wan conditioning batch differs from noise")
    conditions = None if conditioning is None else tuple(conditioning.unbind())
    clips = tuple(noise.unbind())
    sequence_length = math.prod(noise.shape[2:]) // math.prod(model.config.patch_size)
    times = integration.schedule.timesteps.to(device=noise.device, dtype=torch.float32)
    with torch.autocast(noise.device.type, enabled=False):
        for expert in (model,) if high_noise_model is None else (model, high_noise_model):
            if any(
                parameter.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                or (
                    parameter.device != noise.device
                    and not (allow_cpu_frozen and parameter.device.type == "cpu" and not parameter.requires_grad)
                )
                for parameter in expert.parameters()
            ):
                raise ValueError("Wan sampling requires resident floating experts on the noise device")
            for text in (captions,) if negative_captions is None else (captions, negative_captions):
                expert.validate_inputs(
                    clips,
                    times[0].expand(noise.shape[0]),
                    text,
                    sequence_length,
                    conditions,
                    image_features,
                    frame_indices,
                )
    sample = noise
    compute_dtype = model.patch_embedding.weight.dtype
    with torch.autocast(
        noise.device.type,
        dtype=compute_dtype if compute_dtype != torch.float32 else torch.bfloat16,
        enabled=compute_dtype != torch.float32,
        cache_enabled=False,
    ):
        for timestep, high in zip(times.unbind(), high_steps, strict=True):
            expert = high_noise_model if high and high_noise_model is not None else model
            scale = high_noise_guidance if high and high_noise_guidance is not None else guidance
            timesteps = timestep.expand(noise.shape[0])
            clips = tuple(sample.unbind())
            positive = torch.stack(
                expert(
                    clips,
                    timesteps,
                    captions,
                    sequence_length=sequence_length,
                    conditioning=conditions,
                    image_features=image_features,
                    frame_indices=frame_indices,
                )
            )
            if negative_captions is None:
                prediction = positive
            else:
                negative = torch.stack(
                    expert(
                        clips,
                        timesteps,
                        negative_captions,
                        sequence_length=sequence_length,
                        conditioning=conditions,
                        image_features=image_features,
                        frame_indices=frame_indices,
                    )
                )
                if scale is None:
                    raise RuntimeError("Wan CFG scale was not resolved")
                prediction = negative + scale * (positive - negative)
            sample = integration.step(prediction, sample)
    return sample


@torch.no_grad()
def render_wan_video(
    model: WanTransformer,
    vae: QwenImageVAE,
    sampling: WanSamplingConfig,
    inputs: "WanVideoInputs",
    *,
    high_noise_model: WanTransformer | None = None,
    boundary: float | None = None,
    high_noise_guidance: float | None = None,
    allow_cpu_frozen: bool = False,
) -> Tensor:
    """Return CPU RGB CTHW pixels using a local noise generator and shared native sampler.

    Borrow a frozen eval CPU VAE and resident FP32 model(s). Move the VAE to
    the model device only for decode and return it to CPU even after failure.
    Preserve model mode/weights and global RNG; caller owns adapters/preview
    context. Input conditions stay on CPU and are copied explicitly per render.
    """
    validate_wan_vae(vae)
    if any(parameter.device.type != "cpu" for parameter in vae.parameters()):
        raise ValueError("Wan render requires a CPU vae before decode")
    device = next(model.parameters()).device
    width, height = inputs.resolution
    noise = torch.randn(
        (1, 16, (sampling.frames - 1) // 4 + 1, height // 8, width // 8),
        generator=torch.Generator(device=device).manual_seed(sampling.seed),
        dtype=torch.float32,
        device=device,
    )
    with inference_attention(sampling.attention_backend):
        latents = sample_latents(
            model,
            noise,
            (inputs.caption.to(device),),
            steps=sampling.steps,
            solver=sampling.solver,
            shift=sampling.shift,
            guidance=sampling.guidance,
            negative_captions=(inputs.negative_caption.to(device),),
            conditioning=None if inputs.conditioning is None else inputs.conditioning.to(device),
            image_features=None if inputs.image_features is None else inputs.image_features.to(device),
            high_noise_model=high_noise_model,
            boundary=boundary,
            high_noise_guidance=high_noise_guidance,
            allow_cpu_frozen=allow_cpu_frozen,
        )
    try:
        vae.to(device)
        pixels = vae.decode(latents).detach().cpu()[0]
    finally:
        vae.to("cpu")
    if pixels.shape != (3, sampling.frames, height, width) or not torch.isfinite(pixels).all():
        raise ValueError("Wan decoded video differs from requested RGB frame geometry or is nonfinite")
    return pixels
