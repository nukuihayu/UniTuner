"""Check lazy provider contracts and native latent geometry."""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from unituner.models.catalog import IMAGE_SPECS, ModelFamily, image_family_spec
from unituner.models.components.image_geometry import ImageLatentGeometry, image_latent_geometry
from unituner.models.flux.config.vae import FluxVAEConfig
from unituner.models.flux.vae import FluxVAE
from unituner.models.flux2.config.vae import Flux2VAEConfig
from unituner.models.flux2.vae import Flux2VAE
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


def test_latent_geometry_preserves_packing_and_restoration_alignment():
    with torch.device("meta"):
        flux = FluxVAE(FluxVAEConfig())
        flux2 = Flux2VAE(Flux2VAEConfig())
    assert image_latent_geometry(flux) == ImageLatentGeometry(8, 16, 16)
    assert image_latent_geometry(flux2) == ImageLatentGeometry(16, 128, 32)
    qwen21 = image_family_spec(ModelFamily.QWEN_IMAGE_2_1)
    assert qwen21.cache_alignment == 32 and qwen21.restoration_training_alignment == 16
