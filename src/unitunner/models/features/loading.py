"""Load frozen feature artifacts with exact keys and explicit topology."""

from pathlib import Path

import torch
from pydantic import TypeAdapter

from unitunner.artifacts.checkpoint_reading import checkpoint_tensors, is_pytorch_checkpoint
from unitunner.artifacts.tensors import TensorReader
from unitunner.models.features.convnext import DinoConvNeXt
from unitunner.models.features.vgg import Vgg19Features
from unitunner.models.features.vit import DinoViT, DinoViTConfig


def load_dino_convnext(
    path: Path,
    *,
    widths: tuple[int, ...],
    depths: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> DinoConvNeXt:
    """Load official PyTorch or safetensors ConvNeXt as owned frozen FP32 storage.

    Explicit four-stage geometry avoids inferring semantic topology from weights.
    Validate all keys/shapes and tied final-normalization values before loading.
    PyTorch checkpoints use weights_only=True. No stage is discarded or downloaded. Caller owns the
    returned eval model and all generated-image graphs using its frozen weights.
    """
    with torch.device("meta"):
        model = DinoConvNeXt(widths, depths)
    expected = model.state_dict()
    tensors = dict(checkpoint_tensors(path, label="Feature"))
    if tensors.keys() != expected.keys():
        raise ValueError("DINO ConvNeXt artifact keys differ from explicit topology")
    for name, value in tensors.items():
        if value.shape != expected[name].shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(
                f"DINO ConvNeXt artifact tensor contract differs for {name}: "
                f"expected shape={tuple(expected[name].shape)}, got shape={tuple(value.shape)}; "
                "check method.restoration.dino_widths and dino_depths against the checkpoint variant"
            )
    for suffix in ("weight", "bias"):
        if not torch.equal(tensors[f"norm.{suffix}"], tensors[f"norms.3.{suffix}"]):
            raise ValueError("DINO ConvNeXt tied final normalization values disagree")
    model.load_state_dict(
        {name: value.to(device=device, dtype=dtype) for name, value in tensors.items()},
        strict=True,
        assign=True,
    )
    return model.eval().requires_grad_(False)


def load_vgg19_features(
    path: Path,
    *,
    device: torch.device,
    widths: tuple[int, ...] = (64, 128, 256, 512, 512),
    dtype: torch.dtype = torch.float32,
) -> Vgg19Features:
    """Load official or feature-only VGG19 PyTorch/safetensors weights.

    Require every convolution weight and bias. Discard only the known official
    classifier inventory; reject unknown keys. PyTorch uses weights_only=True.
    Return owned frozen FP32 eval storage without downloads. Explicit
    widths permit controlled small-topology comparisons; defaults are official.
    """
    with torch.device("meta"):
        model = Vgg19Features(widths)
    expected = model.state_dict()
    tensors = dict(checkpoint_tensors(path, label="Feature"))
    classifier = {f"classifier.{index}.{suffix}" for index in (0, 3, 6) for suffix in ("weight", "bias")}
    if tensors.keys() - expected.keys() == classifier:
        tensors = {name: value for name, value in tensors.items() if name not in classifier}
    if tensors and all(name.split(".", 1)[0].isdigit() for name in tensors):
        tensors = {"features." + name: value for name, value in tensors.items()}
    if tensors.keys() != expected.keys():
        raise ValueError("VGG19 feature artifact keys differ from explicit topology")
    for name, value in tensors.items():
        if value.shape != expected[name].shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"VGG19 feature artifact tensor contract differs for {name}")
    model.load_state_dict(
        {name: value.to(device=device, dtype=dtype) for name, value in tensors.items()},
        strict=True,
        assign=True,
    )
    return model.eval().requires_grad_(False)


def load_dino_vit(path: Path, *, device: torch.device) -> DinoViT:
    """Load native or published MLP ViT-S/B/L PyTorch/safetensors, without downloads.

    Native bundles declare topology using unitunner.dinov3 metadata. Official
    weights use explicit supported width/head relations; unsupported giant and
    SwiGLU geometries fail. Strictly validate every key, shape and finite value
    before returning an owned frozen FP32 evaluation encoder.
    """
    tensors = dict(checkpoint_tensors(path, label="Feature"))
    metadata = {}
    if not is_pytorch_checkpoint(path):
        with TensorReader(path) as reader:
            metadata = reader.metadata()
    if "unitunner.dinov3" in metadata:
        config = TypeAdapter(DinoViTConfig).validate_json(metadata["unitunner.dinov3"], strict=True)
    else:
        width = tensors["cls_token"].shape[-1]
        if width not in (384, 768, 1024):
            raise ValueError("DINOv3 supports official MLP ViT-S/B/L or an explicit native topology")
        config = DinoViTConfig(
            width=width,
            depth=sum(key.endswith(".ls1.gamma") for key in tensors),
            heads=width // 64,
            patch=tensors["patch_embed.proj.weight"].shape[-1],
            registers=tensors["storage_tokens"].shape[1],
            hidden=tensors["blocks.0.mlp.fc1.weight"].shape[0],
        )
    with torch.device("meta"):
        model = DinoViT(config)
    expected = model.state_dict()
    if tensors.keys() != expected.keys():
        raise ValueError("DINOv3 ViT tensor inventory differs from topology")
    for name, value in tensors.items():
        if value.shape != expected[name].shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"DINOv3 ViT tensor contract differs for {name}")
    model.load_state_dict(
        {name: value.to(device=device, dtype=torch.float32) for name, value in tensors.items()},
        strict=True,
        assign=True,
    )
    return model.eval().requires_grad_(False)
