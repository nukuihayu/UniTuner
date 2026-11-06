"""Check lazy provider contracts, exact derivative identity and native latent geometry."""

import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unituner.artifacts.conversion import convert_weights
from unituner.artifacts.formats.precision_policy import protection_reason
from unituner.artifacts.tensors import TensorReader
from unituner.config.model import ModelConfig
from unituner.models.bluepixel.loading import load_bluepixel, save_bluepixel
from unituner.models.bluepixel.transformer import BluePixelTransformer
from unituner.models.bluepixel.vae import BluePixelVAE
from unituner.models.catalog import IMAGE_SPECS, ModelFamily, image_family_spec
from unituner.models.components.image_geometry import ImageLatentGeometry, image_latent_geometry
from unituner.models.flux.config.vae import FluxVAEConfig
from unituner.models.flux.vae import FluxVAE
from unituner.models.flux2.config.model import Flux2Config
from unituner.models.flux2.config.vae import Flux2VAEConfig
from unituner.models.flux2.vae import Flux2VAE
from unituner.models.image import model_spec, validate_model_selection
from unituner.models.providers import load_provider


@pytest.mark.parametrize("spec", IMAGE_SPECS.values(), ids=lambda spec: spec.family)
def test_declared_providers_have_compatible_application_signatures(spec):
    arguments = {"device": torch.device("cpu"), "dtype": torch.float32}
    if spec.loader_variant:
        arguments["variant"] = "base"
    if spec.loader_version:
        arguments["version"] = "klein-9b"
    inspect.signature(load_provider(spec.loader)).bind(Path("unused"), **arguments)
    inspect.signature(load_provider(spec.saver)).bind(object(), Path("unused"), metadata={}, dtype=torch.float32)
    inspect.signature(load_provider(spec.binding)).bind(object())
    inspect.signature(load_provider(spec.projections)).bind(object())
    inspect.signature(load_provider(spec.generation)).bind(object(), device=torch.device("cpu"))
    inspect.signature(load_provider(spec.preview)).bind(object(), object())


def test_catalog_and_dispatch_imports_do_not_eagerly_load_model_implementations():
    source = Path(__file__).resolve().parents[2] / "src"
    script = """
import sys
import unituner.models.catalog
assert 'torch' not in sys.modules
import unituner.models.image
import unituner.training.integrations.image_models
assert not any(name.startswith('unituner.models.') and name.endswith('.transformer') for name in sys.modules)
assert 'diffusers' not in sys.modules and 'transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONPATH": str(source)}, check=True)


def test_bluepixel_keeps_public_identity_with_a_shared_klein_backbone():
    topology = Flux2Config(
        version="klein-9b",
        variant="base",
        hidden_size=16,
        num_heads=2,
        axes_dim=(2, 2, 2, 2),
        depth=1,
        depth_single_blocks=1,
    )
    with torch.device("meta"):
        model = BluePixelTransformer(topology)
    selected = ModelConfig(family=ModelFamily.BLUEPIXEL, variant="turbo", weights="unused")
    validate_model_selection(selected, model)
    spec = model_spec(model)
    assert spec.backbone == ModelFamily.FLUX2 and spec.vae == ModelFamily.BLUEPIXEL
    assert spec.binding == image_family_spec(ModelFamily.FLUX2).binding
    with pytest.raises(ValueError, match="family"):
        validate_model_selection(
            ModelConfig(family=ModelFamily.FLUX2, variant="base", version="klein-9b", weights="unused"), model
        )
    assert not any("family" in name for name in model.state_dict())


def test_latent_geometry_preserves_packing_and_restoration_alignment():
    with torch.device("meta"):
        flux = FluxVAE(FluxVAEConfig())
        flux2 = Flux2VAE(Flux2VAEConfig())
    backend = torch.nn.Module()
    backend.config = SimpleNamespace(latent_channels=128, encoder_block_out_channels=(1,) * 7, scaling_factor=0.4538)
    dcae = BluePixelVAE(backend)
    assert image_latent_geometry(flux) == ImageLatentGeometry(8, 16, 16)
    assert image_latent_geometry(flux2) == ImageLatentGeometry(16, 128, 32)
    assert image_latent_geometry(dcae) == ImageLatentGeometry(64, 128, 64)
    qwen21 = image_family_spec(ModelFamily.QWEN_IMAGE_2_1)
    assert qwen21.cache_alignment == 32 and qwen21.restoration_training_alignment == 16


@pytest.mark.parametrize("family", ["auto", "bluepixel"])
def test_derivative_weight_conversion_uses_backbone_policy_and_retains_identity(tmp_path, family):
    topology = Flux2Config(
        version="klein-9b",
        variant="base",
        in_channels=8,
        context_in_dim=12,
        hidden_size=16,
        num_heads=2,
        axes_dim=(2, 2, 2, 2),
        depth=1,
        depth_single_blocks=1,
    )
    model = BluePixelTransformer(topology)
    source, destination = tmp_path / "source.safetensors", tmp_path / "converted.safetensors"
    save_bluepixel(model, source)
    report = convert_weights(source, destination, precision="fp32", family=family, output_format="comfyui")
    assert report["family"] == "bluepixel"
    with TensorReader(destination) as reader:
        assert "unituner.bluepixel.config" in reader.metadata()
        assert "unituner.flux2.config" not in reader.metadata()
    restored = load_bluepixel(destination, dtype=torch.float32)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    for name in ("img_in.weight", "double_blocks.0.img_attn.qkv.weight", "time_in.in_layer.weight"):
        assert (protection_reason("bluepixel", name) is None) == (protection_reason("flux2", name) is None)
