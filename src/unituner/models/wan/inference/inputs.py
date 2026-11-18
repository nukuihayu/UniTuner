"""Encode native Wan text and visual conditions at the optional asset boundary."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from unituner.config.model import ModelConfig
from unituner.models.qwen_image.vae import QwenImageVAE
from unituner.models.wan.conditioning import encode_first_frame_condition, validate_wan_vae
from unituner.models.wan.config.sampling import WanSamplingConfig, validate_wan_sampling
from unituner.models.wan.data.media.control import read_wan_control
from unituner.models.wan.data.media.image import read_wan_image
from unituner.models.wan.text.encoder import load_wan_text
from unituner.models.wan.vision.encoder import WanImageEncoder
from unituner.models.wan.vision.loading import load_wan_clip


@dataclass(frozen=True, slots=True)
class WanVideoInputs:
    """Own CPU unpadded text and optional native BCFHW/CLIP video conditions."""

    resolution: tuple[int, int]
    caption: Tensor
    negative_caption: Tensor
    conditioning: Tensor | None
    image_features: Tensor | None


@torch.no_grad()
def build_wan_video_inputs(
    model: ModelConfig, sampling: WanSamplingConfig, vae: QwenImageVAE, *, text_length: int
) -> WanVideoInputs:
    """Encode explicit local conditions on CPU with frozen eval mean-only encoders.

    Preserve native area-budget I2V geometry, original-image CLIP input and
    distinct inference normalization order; cache resize/crop is not reused.
    I2V encodes the first image followed by normalized zeros; Fun-Control uses
    control16 plus first-image16 with later latent slots zeroed. Align controls
    to the requested raw frame count before encoding, truncating excess frames
    and rejecting short sources without implicit FPS resampling. No target
    video or posterior noise is used.
    """
    validate_wan_sampling(sampling, version=model.version, variant=model.variant)
    validate_wan_vae(vae)
    if model.text_encoder is None:
        raise ValueError("Wan generation requires local text_encoder")
    if any(parameter.device.type != "cpu" for parameter in vae.parameters()):
        raise ValueError("Wan video condition vae must reside on CPU")
    text = load_wan_text(Path(model.text_encoder), max_length=text_length, text_config=model.text)
    caption, negative = text.encode([sampling.prompt, sampling.negative_prompt])
    caption, negative = caption.float(), negative.float()
    del text
    condition, features = None, None
    width, height = sampling.width, sampling.height
    if sampling.conditioning_image is not None:
        image_pixels = read_wan_image(Path(sampling.conditioning_image), (width, height))
        width, height = image_pixels.resolution
        image = image_pixels.resized
        pixels = torch.cat((image[:, None], image.new_zeros(3, sampling.frames - 1, height, width)), dim=1)
        condition = encode_first_frame_condition(vae, pixels.unsqueeze(0))
        if model.version == "2.1":
            image_model = load_wan_clip(model.asset_path("image_encoder"))
            if (
                image_model.config.hidden_size != 1280
                or image_model.config.image_size != 224
                or image_model.config.patch_size != 14
            ):
                raise ValueError("Wan2.1 generation requires1280-wide224/14 CLIP features")
            features = WanImageEncoder(image_model).encode((image_pixels.original[:, None],))
            del image_model
    if sampling.control_video is not None:
        control_pixels = read_wan_control(Path(sampling.control_video), (width, height), frames=sampling.frames)
        control = vae.encode(control_pixels.unsqueeze(0))
        first = (
            torch.zeros_like(control)
            if condition is None
            else torch.cat((condition[:, 4:, :1], torch.zeros_like(condition[:, 4:, 1:])), dim=2)
        )
        condition = torch.cat((control, first), dim=1)
    return WanVideoInputs((width, height), caption, negative, condition, features)
