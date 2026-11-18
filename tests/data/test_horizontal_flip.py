"""Check paired pixel augmentation before VAE, vision and immutable cache publication."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from unituner.config.dataset_inputs import DatasetInputConfig
from unituner.config.datasets import CacheDatasetConfig, DatasetConfig
from unituner.data.media.augmentation import sample_horizontal_flip
from unituner.data.media.image import resize_pixels
from unituner.models.components.conditioning import Caption
from unituner.models.qwen_image.config.vae import QwenImageVAEConfig
from unituner.models.qwen_image.data.cache import cache_qwen_image
from unituner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unituner.models.qwen_image_2_1.data.cache.encoding import cache_qwen_image_21


def test_flip_configuration_defaults_bounds_and_rng():
    for kind in ("image", "video", "restoration"):
        fields: dict[str, Any] = {"kind": kind, "source": "target", "cache_directory": "cache"}
        if kind == "restoration":
            fields["low_quality_source"] = "control"
        assert DatasetConfig(**fields).horizontal_flip_probability == (0.5 if kind == "restoration" else 0.0)
        for probability in (0.0, 0.5, 1.0):
            assert (
                DatasetConfig(**fields, horizontal_flip_probability=probability).horizontal_flip_probability
                == probability
            )
            assert (
                CacheDatasetConfig(
                    kind=kind, source="target", cache_directory="cache", horizontal_flip_probability=probability
                ).horizontal_flip_probability
                == probability
            )
    for probability in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            DatasetInputConfig(horizontal_flip_probability=probability)
    rng = torch.get_rng_state().clone()
    decisions = [sample_horizontal_flip(0.5, 42, str(i)) for i in range(100)]
    assert decisions == [sample_horizontal_flip(0.5, 42, str(i)) for i in range(100)]
    assert 25 < sum(decisions) < 75
    assert decisions != [sample_horizontal_flip(0.5, 43, str(i)) for i in range(100)]
    assert not sample_horizontal_flip(0, 42, "target") and sample_horizontal_flip(1, 42, "target")
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("variant", ("edit-2511", "layered", "2.1"))
def test_qwen_target_layers_references_and_vision_flip_together(tmp_path, variant):
    source, condition = tmp_path / "source", tmp_path / "condition"
    source.mkdir()
    condition.mkdir()
    rgba = np.random.default_rng(71).integers(0, 256, (32, 32, 4), dtype=np.uint8)
    Image.fromarray(rgba).save(source / "a.png")
    Image.fromarray(rgba).save(condition / "a.png")
    channels = 4 if variant in ("layered", "2.1") else 3
    observed, vision = [], []

    class PixelVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.config = QwenImage21VAEConfig() if variant == "2.1" else QwenImageVAEConfig(input_channels=channels)

        def encode(self, pixels, **kwargs):
            observed.extend(pixels.detach().clone().unbind(0))
            return pixels[:, :1, :, ::16, ::16].repeat(1, self.config.latent_channels, 1, 1, 1).contiguous()

        encode_layers = encode

    def encode_prompt(prompt, *, references):
        vision.extend(references or ())
        return Caption(torch.ones(1, 1, 8))

    def encode(prompts, *, references):
        vision.extend(references[0])
        return [SimpleNamespace(features=torch.ones(1, 8), image_slots=(0,), image_shapes=((2, 2),))]

    vae: Any = PixelVAE().eval()
    text: Any = SimpleNamespace(
        model=torch.nn.Linear(1, 1).eval().requires_grad_(False),
        variant=variant,
        encode_prompt=encode_prompt,
        encode=encode,
    )
    options = {"layer_sources": (str(condition),)} if variant == "layered" else {"reference_sources": (str(condition),)}
    producer = cache_qwen_image_21 if variant == "2.1" else cache_qwen_image
    manifests = []
    for probability in (0.0, 1.0):
        observed.clear()
        vision.clear()
        data = DatasetInputConfig(**options, horizontal_flip_probability=probability)
        manifest = producer(
            source, tmp_path / "cache", vae, text, resolution=(32, 32), encoder_sha256="a" * 64, data=data
        )
        expected = torch.from_numpy(rgba[..., :channels].copy()).permute(2, 0, 1).float() / 127.5 - 1
        if probability:
            expected = expected.flip(-1)
        assert torch.equal(observed[0][:, 0], expected)
        if variant == "layered":
            assert torch.equal(observed[0][:, 1], expected)
        else:
            assert torch.equal(observed[-1][:, 0], expected)
            if variant == "2.1":
                assert np.array_equal(vision[0], rgba[:, ::-1] if probability else rgba)
            else:
                # Vision shares the already flipped VAE reference before its independent resize.
                assert np.array_equal(
                    vision[0], resize_pixels((rgba[:, ::-1, :3] if probability else rgba[..., :3]).copy(), (384, 384))
                )
        assert (
            producer(source, tmp_path / "cache", vae, text, resolution=(32, 32), encoder_sha256="a" * 64, data=data)
            == manifest
        )
        manifests.append(manifest)
    assert manifests[0].entries[0].file != manifests[1].entries[0].file
