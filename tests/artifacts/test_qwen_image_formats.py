"""Verify strict Qwen external inventories through the public offline importer."""

import pytest
import torch

from unitunner.artifacts.formats.qwen_image import convert_vae_tensors, vae_key_map
from unitunner.artifacts.importing import import_weights
from unitunner.artifacts.tensors import save_tensors
from unitunner.models.qwen_image.config.model import QwenImageConfig
from unitunner.models.qwen_image.config.vae import QwenImageVAEConfig
from unitunner.models.qwen_image.loading import load_qwen_image, load_qwen_image_vae
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.models.qwen_image.vae import QwenImageVAE


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("channels,layers,prefix", [(3, 2, ""), (4, 1, "vae.")])
def test_vae_external_import_exact_and_atomic(tmp_path, channels, layers, prefix):
    config = QwenImageVAEConfig(
        input_channels=channels,
        base_channels=4,
        latent_channels=2,
        channel_multipliers=(1, 2, 2, 2),
        layers_per_block=layers,
        latents_mean=(-0.31, 0.73),
        latents_std=(1.73, 0.91),
    )
    model = QwenImageVAE(config)
    keys = vae_key_map(tuple(model.state_dict()), num_blocks=4, layers_per_block=layers)
    external = convert_vae_tensors(model.state_dict(), keys, to_native=False)
    assert all(external[keys[k]].data_ptr() == v.data_ptr() for k, v in model.state_dict().items())
    tensors = {prefix + key: value for key, value in external.items()}
    source, destination, topology = (
        tmp_path / p for p in ("external.safetensors", "native.safetensors", "topology.json")
    )
    topology.write_text(config.model_dump_json())
    save_tensors(tensors, source)
    before = source.read_bytes()
    rng = torch.get_rng_state().clone()
    import_weights(
        source,
        destination,
        family="qwen-image",
        kind="vae",
        topology=topology,
        source_format="comfyui",
        prefix=prefix,
    )
    restored = load_qwen_image_vae(destination)
    assert torch.equal(rng, torch.get_rng_state())
    assert source.read_bytes() == before
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    pixels, latents = torch.randn(1, channels, 1, 8, 16), torch.randn(1, 2, 1, 1, 2)
    assert torch.equal(model.encode(pixels), restored.encode(pixels))
    assert torch.equal(model.decode_raw(latents), restored.decode_raw(latents))
    accepted = destination.read_bytes()
    tensors["unexpected.weight"] = torch.ones(1)
    save_tensors(tensors, source)
    invalid = source.read_bytes()
    with pytest.raises(ValueError, match=r"keys differ|selected prefix"):
        import_weights(
            source,
            destination,
            family="qwen-image",
            kind="vae",
            topology=topology,
            source_format="comfyui",
            prefix=prefix,
        )
    assert destination.read_bytes() == accepted and source.read_bytes() == invalid


@pytest.mark.parametrize(
    "variant,prefix", [("original", "diffusion_model."), ("edit-2511", "model.diffusion_model."), ("layered", "")]
)
def test_transformer_external_import_and_marker(tmp_path, variant, prefix):
    config = QwenImageConfig(
        variant=variant,
        in_channels=8,
        out_channels=2,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=12,
        axes_dims_rope=(4, 6, 6),
    )
    model = QwenImageTransformer(config)
    tensors = {prefix + k: v for k, v in model.state_dict().items()}
    if variant == "edit-2511":
        tensors[prefix + "__index_timestep_zero__"] = torch.empty(0)
    source, destination, topology = (
        tmp_path / p for p in ("external.safetensors", "native.safetensors", "topology.json")
    )
    topology.write_text(config.model_dump_json())
    save_tensors(tensors, source)
    before = source.read_bytes()
    rng = torch.get_rng_state().clone()
    import_weights(
        source,
        destination,
        family="qwen-image",
        kind="transformer",
        topology=topology,
        source_format="comfyui",
        prefix=prefix,
    )
    restored = load_qwen_image(destination)
    assert torch.equal(rng, torch.get_rng_state()) and source.read_bytes() == before
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    values = (torch.randn(2, 8, 8), torch.randn(2, 5, 12), torch.tensor([0.244, 0.751]), ((1, 2, 3), (1, 1, 2)))
    options = {
        "text_mask": torch.arange(5)[None] < torch.tensor([3, 5])[:, None],
        "layer_condition": torch.tensor([0, 1]) if variant == "layered" else None,
    }
    assert torch.equal(model(*values, **options), restored(*values, **options))
    accepted = destination.read_bytes()
    tensors[prefix + "__index_timestep_zero__"] = torch.ones(1) if variant == "edit-2511" else torch.empty(0)
    save_tensors(tensors, source)
    with pytest.raises(ValueError, match="marker"):
        import_weights(
            source,
            destination,
            family="qwen-image",
            kind="transformer",
            topology=topology,
            source_format="comfyui",
            prefix=prefix,
        )
    assert destination.read_bytes() == accepted
    if prefix:
        tensors.pop(prefix + "__index_timestep_zero__")
        tensors["img_in.weight"] = tensors.pop(prefix + "img_in.weight")
        save_tensors(tensors, source)
        with pytest.raises(ValueError, match="selected prefix"):
            load_qwen_image(source, config=config, source_format="comfyui", prefix=prefix)
    save_tensors(model.state_dict(), source)
    diffusers = load_qwen_image(source, config=config, source_format="diffusers")
    assert all(torch.equal(v, diffusers.state_dict()[k]) for k, v in model.state_dict().items())
