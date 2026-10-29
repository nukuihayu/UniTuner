"""Verify direct released VAE loading without an offline topology import."""

import pytest
import torch

from tests.models.test_flux2_vae import small_vae
from tests.models.test_minimax_h3_audio_vae import _vae as audio_vae
from tests.models.test_minimax_h3_video_vae import _vae as video_vae
from unitunner.artifacts.formats.qwen_image import convert_vae_tensors as convert_qwen
from unitunner.artifacts.formats.qwen_image import vae_key_map as qwen_key_map
from unitunner.artifacts.formats.z_image import convert_vae_tensors as convert_zimage
from unitunner.artifacts.formats.z_image import vae_key_map as zimage_key_map
from unitunner.artifacts.tensors import save_tensors
from unitunner.models.components.vae_loading import vae_tensors
from unitunner.models.flux.config.vae import FluxVAEConfig
from unitunner.models.flux.loading import load_flux_vae
from unitunner.models.flux.vae import FluxVAE
from unitunner.models.flux2.loading import load_flux2_vae
from unitunner.models.minimax_h3.vae import load_audio_vae, load_video_vae
from unitunner.models.qwen_image.config.vae import QwenImageVAEConfig
from unitunner.models.qwen_image.loading import load_qwen_image_vae
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21_vae
from unitunner.models.z_image.config.vae import ZImageVAEConfig
from unitunner.models.z_image.loading import load_z_image_vae
from unitunner.models.z_image.vae import ZImageVAE


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "loader",
    [
        load_z_image_vae,
        load_flux_vae,
        load_flux2_vae,
        load_qwen_image_vae,
        load_qwen_image_21_vae,
        load_audio_vae,
        load_video_vae,
    ],
)
def test_missing_metadata_uses_published_topology_and_still_rejects_missing_weights(tmp_path, loader):
    path = tmp_path / "external.safetensors"
    save_tensors({}, path)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=r"keys differ|inventory"):
        loader(path)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "family",
    ["z-image", "z-image-prefixed", "z-image-native-prefixed", "qwen-rgb", "qwen-rgba", "wan-pth", "flux", "flux2"],
)
def test_auto_layout_preserves_every_tensor_and_rejects_incomplete_weights(tmp_path, family):
    if family.startswith("z-image"):
        model = ZImageVAE(ZImageVAEConfig(block_channels=(8, 16), norm_groups=4, latent_channels=2))
        mapping = zimage_key_map(tuple(model.state_dict()), num_blocks=2)
        external = convert_zimage(model.state_dict(), mapping, to_native=False)
        if family == "z-image-native-prefixed":
            external = model.state_dict()
        if family.endswith("prefixed"):
            external = {"vae." + key: value for key, value in external.items()}
        loader = load_z_image_vae
    elif family.startswith("qwen") or family == "wan-pth":
        model = QwenImageVAE(
            QwenImageVAEConfig(
                input_channels=4 if family == "qwen-rgba" else 3,
                base_channels=4,
                latent_channels=2,
                layers_per_block=1,
                channel_multipliers=(1, 2, 2, 2),
                latents_mean=(0.0, 0.0),
                latents_std=(1.0, 1.0),
            )
        )
        mapping = qwen_key_map(tuple(model.state_dict()), num_blocks=4, layers_per_block=1)
        external = convert_qwen(model.state_dict(), mapping, to_native=False)
        if family != "wan-pth":
            external = {"vae." + key: value for key, value in external.items()}
        loader = load_qwen_image_vae
    else:
        model = (
            small_vae()
            if family == "flux2"
            else FluxVAE(
                FluxVAEConfig(
                    base_channels=8,
                    channel_multipliers=(1, 2),
                    norm_groups=4,
                    latent_channels=2,
                )
            )
        )
        external = {"vae." + key: value for key, value in model.state_dict().items()}
        loader = load_flux2_vae if family == "flux2" else load_flux_vae
    path = tmp_path / ("external.pth" if family == "wan-pth" else "external.safetensors")

    def save(state):
        if family == "wan-pth":
            torch.save(state, path)
        else:
            save_tensors(state, path)

    save(external)
    before = path.read_bytes()
    rng = torch.get_rng_state().clone()
    loaded = loader(path, config=model.config)
    assert path.read_bytes() == before and torch.equal(rng, torch.get_rng_state())
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
        assert loaded.state_dict()[key].dtype == value.dtype
    external.pop(next(iter(external)))
    save(external)
    with pytest.raises(ValueError, match="keys differ"):
        loader(path, config=model.config)


@pytest.mark.parametrize(
    "kind,prefix",
    [
        ("audio", "audio_vae."),
        ("audio", "vae."),
        ("video", "first_stage_model."),
        ("video", "video_vae."),
        ("video", "vae."),
    ],
)
def test_h3_documented_prefixes_and_encoder_only_loading(tmp_path, kind, prefix):
    model = audio_vae() if kind == "audio" else video_vae()
    loader = load_audio_vae if kind == "audio" else load_video_vae
    path = tmp_path / "external.safetensors"
    save_tensors({prefix + key: value for key, value in model.state_dict().items()}, path)
    for include_decoder in (True, False):
        loaded = loader(path, config=model.config, include_decoder=include_decoder)
        for key, value in loaded.state_dict().items():
            assert torch.equal(value, model.state_dict()[key]), key


def test_prefix_collisions_fail_instead_of_overwriting(tmp_path):
    path = tmp_path / "colliding.safetensors"
    save_tensors({"vae.weight": torch.ones(1), "weight": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="collide"):
        dict(vae_tensors(path, prefixes=("vae.",)))


@pytest.mark.parametrize("family", ["flux", "flux2"])
def test_flux_diffusers_auto_detection_and_bn_counter(tmp_path, family):
    model = (
        small_vae()
        if family == "flux2"
        else FluxVAE(
            FluxVAEConfig(
                base_channels=8,
                channel_multipliers=(1, 2),
                norm_groups=4,
                latent_channels=2,
            )
        )
    )
    config = model.config
    with torch.device("meta"):
        diffusers = ZImageVAE(
            ZImageVAEConfig(
                block_channels=tuple(config.base_channels * m for m in config.channel_multipliers),
                layers_per_block=config.layers_per_block,
                norm_groups=config.norm_groups,
                latent_channels=config.latent_channels,
                quant_conv=family == "flux2",
                post_quant_conv=family == "flux2",
            )
        )
    mapping = zimage_key_map(tuple(diffusers.state_dict()), num_blocks=len(config.channel_multipliers))
    native = model.state_dict()
    external = {}
    for key, source in mapping.items():
        if key.startswith("quant_conv."):
            source = "encoder." + key
        elif key.startswith("post_quant_conv."):
            source = "decoder." + key
        value = native[source]
        if ".attentions." in key and value.ndim == 4:
            value = value.squeeze(-1).squeeze(-1)
        external[key] = value
    external.update({key: value for key, value in native.items() if key.startswith("bn.")})
    external.pop("bn.num_batches_tracked", None)
    path = tmp_path / "diffusers.safetensors"
    save_tensors(external, path)
    loader = load_flux2_vae if family == "flux2" else load_flux_vae
    loaded = loader(path, config=config)
    for key, value in native.items():
        assert torch.equal(value, loaded.state_dict()[key]), key
