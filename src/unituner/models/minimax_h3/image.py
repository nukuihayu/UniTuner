"""Native MiniMax-H3-Image denoiser and distilled PDD sampling."""

import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unituner.artifacts.formats.transformer import convert_transformer
from unituner.artifacts.tensors import TensorReader, iter_tensors, save_tensors
from unituner.execution.routing import TargetSpan, sample_token_route
from unituner.models.components.state_loading import load_model_state
from unituner.models.components.transformer_artifacts import transformer_tensors
from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.layers import AdalnProj, DiTBlock, RotationFrequencies, TimeEmbedder, TokenRefiner
from unituner.models.minimax_h3.layout import MiniMaxH3VideoGeometry
from unituner.models.minimax_h3.packing import pack_video_rows
from unituner.models.minimax_h3.positions import _frame_grid
from unituner.models.minimax_h3.timesteps import _build_modulation_segments
from unituner.models.minimax_h3.transformer import MiniMaxH3Transformer

_CPU = torch.device("cpu")
_PDD_FORMAT_KEY = "unituner.minimax_h3.image.pdd_format"
_PDD_SIGMAS_KEY = "unituner.minimax_h3.image.pdd_sigmas"


class ImageFinalLayer(nn.Module):
    """Project the target image with a full head and optional PDD delta heads."""

    def __init__(self, config: MiniMaxH3Config, steps: int, *, dtype: torch.dtype, device: str | torch.device | None):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(
            config.adaln_in_features,
            config.hidden_size,
            2,
            1,
            dtype=dtype,
            apply_silu=not config.is_pruned,
            device=device,
        )
        self.video_out = nn.Linear(
            config.hidden_size, config.video_patch_dim * steps, dtype=torch.float32, device=device
        )

    def forward(
        self,
        hidden: Tensor,
        times: Tensor,
        start: int,
        stop: int,
        sigmas: Sequence[float],
        *,
        fused: bool = False,
    ) -> Tensor:
        shift, scale = self.adaln_proj(times)
        image = self.norm(hidden) * (1 + scale[0]) + shift[0]
        image = image.float()
        steps = len(sigmas) - 1
        if steps == 1:
            return self.video_out(image)
        weights = image.new_tensor([sigmas[i] - sigmas[i + 1] for i in range(start, stop)])
        weights = weights / weights.sum()
        rows = self.video_out.weight.reshape(steps, -1, self.video_out.in_features)
        biases = self.video_out.bias.reshape(steps, -1)
        first = max(start, 1)
        if fused:
            weight = torch.einsum("n,noi->oi", weights, rows[start:stop])
            bias = torch.einsum("n,no->o", weights, biases[start:stop])
        else:
            weight = rows[0] + torch.einsum("n,noi->oi", weights[first - start :], rows[first:stop])
            bias = biases[0] + torch.einsum("n,no->o", weights[first - start :], biases[first:stop])
        return F.linear(image, weight, bias)


class MiniMaxH3ImageTransformer(MiniMaxH3Transformer):
    """The H3 single-stream transformer without audio weights or audio tokens."""

    def __init__(
        self,
        config: MiniMaxH3Config,
        *,
        pdd_sigmas: Sequence[float] = (1.0, 0.0),
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ) -> None:
        nn.Module.__init__(self)
        sigmas = tuple(float(value) for value in pdd_sigmas)
        if (
            len(sigmas) < 2
            or sigmas[0] != 1
            or sigmas[-1] != 0
            or any(not math.isfinite(a) or a <= b for a, b in pairwise(sigmas))
        ):
            raise ValueError("MiniMax-H3-Image PDD sigmas must decrease from 1 to 0")
        self.config = config
        self.pdd_sigmas = sigmas
        self.fused_pdd_heads = False
        self.feature_recorder = None
        self.token_routing = None
        self.feature_width = config.hidden_size
        self.feature_patch = config.patch_size
        self.video_patch_proj = nn.Linear(
            config.video_patch_dim, config.hidden_size, dtype=torch.float32, device=device
        )
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size, dtype=dtype, device=device)
        if config.adaln_curve_grid is None:
            self.time_embedder = TimeEmbedder(
                config.timestep_input_dim,
                config.time_embed_hidden_size,
                config.time_embed_dim,
                device=device,
            )
        else:
            self.time_embedder = None
            self.register_buffer(
                "adaln_t_table",
                torch.empty(config.adaln_curve_grid, config.time_embed_dim, dtype=torch.float32, device=device),
            )
        self.rope = RotationFrequencies(config.rope_inv_freq_len, device)
        self.token_refiner = TokenRefiner(config, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([DiTBlock(config, dtype=dtype, device=device) for _ in range(config.num_layers)])
        self.final_layer = ImageFinalLayer(config, len(sigmas) - 1, dtype=dtype, device=device)

    def predict(
        self,
        latents: Tensor,
        text_hidden_states: Tensor,
        text_token_tags: Tensor,
        *,
        sigma: float,
        sigma_next: float,
        references: Sequence[Tensor] = (),
        reference_seed: int = 0,
    ) -> Tensor:
        """Predict image velocity at one PDD interval; inputs are B24x1xHxW."""
        if latents.ndim != 5 or latents.shape[:3] != (1, 24, 1):
            raise ValueError("H3-Image needs a single B24x1xHxW target")
        if latents.shape[-2] % 2 or latents.shape[-1] % 2:
            raise ValueError("H3-Image latent spatial dimensions must be even")
        if (
            text_hidden_states.ndim != 3
            or text_hidden_states.shape[:1] != (1,)
            or text_hidden_states.shape[-1] != self.config.text_dim
        ):
            raise ValueError("H3-Image text width does not match the transformer")
        text_length = text_hidden_states.shape[1]
        if (
            text_token_tags.shape != (1, text_length)
            or text_token_tags.dtype != torch.int64
            or not torch.all((text_token_tags == 0) | (text_token_tags == 1))
        ):
            raise ValueError("H3-Image text tags must be int64 [1, text_length] with values 0 or 1")
        if not 0 <= sigma_next < sigma <= 1:
            raise ValueError("H3-Image sampling needs decreasing sigmas within [0,1]")
        grid = self.pdd_sigmas
        # Continuous training sigmas belong to the interval above the next lower grid point.
        # Exact grid points retain the published sampler's head selection.
        start = next(index for index in range(len(grid) - 1) if grid[index] >= sigma > grid[index + 1])
        stop = min(range(len(grid)), key=lambda index: abs(grid[index] - sigma_next))
        stop = max(stop, start + 1)

        device = self.device
        positions = [torch.zeros(text_length, 3, dtype=torch.float64)]
        positions[0][:, 0] = torch.arange(text_length)
        rows = []
        cursor = float(text_length)
        for reference in references:
            if (
                reference.ndim != 5
                or reference.shape[:3] != (1, 24, 1)
                or any(size % 2 for size in reference.shape[-2:])
            ):
                raise ValueError("H3-Image references must be B24x1xHxW with even spatial dimensions")
            geometry = MiniMaxH3VideoGeometry(frames=1, height=reference.shape[-2], width=reference.shape[-1])
            frame, _ = _frame_grid(geometry)
            reference_positions = torch.empty(frame.shape[0], 3, dtype=torch.float64)
            reference_positions[:, 0] = cursor
            reference_positions[:, 1:] = frame
            positions.append(reference_positions)
            reference_rows = pack_video_rows(reference.to(device))
            noise = torch.randn(
                reference_rows.shape, generator=torch.Generator("cpu").manual_seed(reference_seed), dtype=torch.float32
            )
            rows.append(0.999 * reference_rows + 0.001 * noise.to(device))
            cursor += 1
        target = MiniMaxH3VideoGeometry(frames=1, height=latents.shape[-2], width=latents.shape[-1])
        frame, _ = _frame_grid(target)
        target_positions = torch.empty(frame.shape[0], 3, dtype=torch.float64)
        target_positions[:, 0] = cursor
        target_positions[:, 1:] = frame
        positions.append(target_positions)
        rows.append(pack_video_rows(latents.to(device)))

        with torch.autocast(device_type=device.type, enabled=False):
            image = self.video_patch_proj(torch.cat(rows, dim=1).float()).to(self.dtype)
        text = self.token_refiner(self.condition_proj(text_hidden_states.to(device=device, dtype=self.dtype)))
        hidden = torch.cat((text, image), dim=1)
        time = 1.0 - sigma
        condition_time = max(time, 0.999)
        unique = sorted({time, condition_time} if references else {time})
        time_index = {value: index for index, value in enumerate(unique)}
        timesteps = self._timestep_embeddings(torch.tensor(unique, device=device), device)
        tags = torch.cat(
            (text_token_tags.reshape(-1).to(device), torch.zeros(image.shape[1], dtype=torch.long, device=device))
        )
        indices = torch.full((hidden.shape[1],), time_index[time] * 3, dtype=torch.long, device=device)
        indices[:text_length] += tags[:text_length]
        if references:
            indices[text_length : -frame.shape[0]] = time_index[condition_time] * 3
        segments = _build_modulation_segments(indices.cpu())
        rotation = self._rotation_table(torch.cat(positions).unsqueeze(0).to(device), hidden.dtype)
        target_start = hidden.shape[1] - frame.shape[0]
        route = sample_token_route(
            self.token_routing,
            training=self.training,
            length=hidden.shape[1],
            spans=((TargetSpan(target_start, hidden.shape[1]),),),
            device=device,
        )
        for index, block in enumerate(self.blocks):
            if route is not None and route.applies(index):
                routed_segments = tuple(
                    (route.boundary(start), route.boundary(stop), time) for start, stop, time in segments
                )
                updated = block(route.select(hidden), timesteps, routed_segments, route.select(rotation))
                hidden = route.restore(hidden, updated)
            else:
                hidden = block(hidden, timesteps, segments, rotation)
            if self.feature_recorder is not None:
                self.feature_recorder.record(index, hidden[:, target_start:])
        velocity = self.final_layer(
            hidden[:, -frame.shape[0] :],
            timesteps[time_index[time] : time_index[time] + 1],
            start,
            stop,
            grid,
            fused=self.fused_pdd_heads,
        )
        return (
            -velocity.reshape(1, target.height // 2, target.width // 2, 24, 1, 2, 2)
            .permute(0, 3, 4, 1, 5, 2, 6)
            .reshape_as(latents)
        )

    forward = predict


def load_image_transformer(
    path: Path,
    *,
    config: MiniMaxH3Config,
    pdd_config: Path,
    pdd_heads: Path | None = None,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.bfloat16,
) -> MiniMaxH3ImageTransformer:
    """Load strict native/ComfyUI denoiser weights and the published PDD grid."""
    with pdd_config.open(encoding="utf-8") as stream:
        schedule = json.load(stream)
    sigmas = schedule["sigmas"]
    if schedule["num_inference_steps"] != len(sigmas) - 1:
        raise ValueError("H3-Image PDD schedule does not match its head count")
    with torch.device("meta"):
        model = MiniMaxH3ImageTransformer(config, pdd_sigmas=sigmas, dtype=dtype)
    expected = model.state_dict()
    metadata = {}
    if path.is_file():
        with TensorReader(path) as reader:
            metadata = reader.metadata()
    if _PDD_SIGMAS_KEY in metadata and json.loads(metadata[_PDD_SIGMAS_KEY]) != sigmas:
        raise ValueError("H3-Image native PDD grid differs from the supplied schedule")
    tensors = convert_transformer("minimax_h3", transformer_tensors(path), expected)
    if pdd_heads is not None:
        heads = dict(iter_tensors(pdd_heads))
        if heads.keys() != {"video_out.weight", "video_out.bias"}:
            raise ValueError("H3-Image PDD head artifact has unexpected keys")
        for suffix in ("weight", "bias"):
            value = heads[f"video_out.{suffix}"]
            target = f"final_layer.video_out.{suffix}"
            head_shape = (len(sigmas) - 1, expected[target].shape[0] // (len(sigmas) - 1), *expected[target].shape[1:])
            if value.shape != head_shape:
                raise ValueError(f"H3-Image PDD {suffix} shape does not match the denoiser")
            tensors[target] = value.reshape(expected[target].shape)
    elif tensors["final_layer.video_out.weight"].shape != expected["final_layer.video_out.weight"].shape:
        raise ValueError("H3-Image base transformer needs pdd_fused_heads.safetensors for this schedule")
    load_model_state(
        model,
        tensors,
        device=device,
        dtype=dtype,
        label="MiniMax-H3-Image",
        floating_dtypes={name: value.dtype for name, value in expected.items()},
    )
    head_format = metadata.get(_PDD_FORMAT_KEY, "delta")
    if head_format not in {"delta", "fused"}:
        raise ValueError("H3-Image native PDD head format is invalid")
    model.fused_pdd_heads = pdd_heads is not None or head_format == "fused"
    return model


def save_image_transformer(
    model: MiniMaxH3ImageTransformer,
    path: Path,
    *,
    metadata: Mapping[str, str] | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    """Save native weights; keep the PDD grid in its separate published JSON file."""
    save_tensors(
        model.state_dict(),
        path,
        metadata={
            **({} if metadata is None else metadata),
            _PDD_FORMAT_KEY: "fused" if model.fused_pdd_heads else "delta",
            _PDD_SIGMAS_KEY: json.dumps(model.pdd_sigmas),
        },
        dtype=dtype,
    )


@torch.no_grad()
def sample_image(
    model: MiniMaxH3ImageTransformer,
    noise: Tensor,
    text: Tensor,
    tags: Tensor,
    *,
    references: Sequence[Tensor] = (),
    reference_seed: int = 0,
) -> Tensor:
    """Integrate the checkpoint's distilled six-step grid without CFG."""
    latents = noise
    for sigma, next_sigma in pairwise(model.pdd_sigmas):
        velocity = model.predict(
            latents,
            text,
            tags,
            sigma=sigma,
            sigma_next=next_sigma,
            references=references,
            reference_seed=reference_seed,
        )
        latents = latents + (sigma - next_sigma) * velocity
    return latents
