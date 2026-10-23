"""Exercise VAE-only calibration from training TOMLs without text-model fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from unitunner.config.loading import load_calibration_config, load_training_config
from unitunner.data.cache.storage import CacheEntry
from unitunner.data.datasets.restoration import AlignedImagePair, RestorationDataset
from unitunner.data.media.restoration import read_restoration_pair
from unitunner.inference.restoration.vae import load_restoration_vae
from unitunner.models.z_image.config.vae import ZImageVAEConfig
from unitunner.models.z_image.loading import save_z_image_vae
from unitunner.models.z_image.vae import ZImageVAE
from unitunner.training.calibration import run
from unitunner.training.methods.restoration import RestorationMethod


@pytest.mark.parametrize("sampling", ["random", "tiles"])
@pytest.mark.parametrize("synthetic", [False, True])
def test_toml_calibration_preprocessing_limit_and_rng(tmp_path, capsys, sampling, synthetic):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vae = ZImageVAE(
            ZImageVAEConfig(block_channels=(8, 8, 8, 8), latent_channels=2, norm_groups=4, layers_per_block=1)
        )
        save_z_image_vae(vae, tmp_path / "vae.safetensors")
        source = tmp_path / "hq"
        source.mkdir()
        Image.fromarray(np.random.default_rng(12).integers(0, 256, (67, 81, 3), dtype=np.uint8)).save(source / "a.png")
        inputs = "degradation = { enabled = true, scales = [2] }" if synthetic else f'low_quality_source = "{source}"'
        document = tmp_path / "train.toml"
        document.write_text(f'''
[model]
family = "z-image"
vae = "{tmp_path / "vae.safetensors"}"
weights = "absent-denoiser.safetensors"
[train]
seed = 17
[method.restoration]
calibrate = true
calibration_pairs = 3
calibration_sampling = "{sampling}"
posterior = "sample"
[dataset.defaults]
resolution = [32, 32]
[dataset.groups.main]
source = "{source}"
cache_directory = "absent-cache"
kind = "restoration"
{inputs}
''')
        config = load_calibration_config(document)
        assert config.calibration_sampling == sampling
        assert config.dataset.groups["main"].resolution == (32, 32)
        before = torch.get_rng_state().clone()
        outputs = []
        for _ in range(2):
            assert run(config, device="cpu") == 0
            outputs.append(json.loads(capsys.readouterr().out))
        assert outputs[0] == outputs[1]
        assert outputs[0]["pairs"] == (3 if sampling == "tiles" else 1)
        assert outputs[0]["sampling"] == sampling
        assert 0 < outputs[0]["sigma"] < 1
        assert np.isfinite(outputs[0]["mean_snr_error"])
        assert torch.equal(before, torch.get_rng_state())
        training = load_training_config(
            Path(__file__).resolve().parents[2] / "docs/examples/restoration.toml",
            overrides={
                "train": {"seed": config.seed},
                "method": {
                    "restoration": {
                        "calibrate": True,
                        "calibration_pairs": 3,
                        "calibration_sampling": sampling,
                        "posterior": "sample",
                    }
                },
                "dataset": {
                    "groups": {
                        "main": {
                            "source": str(source),
                            "cache_directory": "absent-cache",
                            "resolution": [32, 32],
                            "low_quality_source": None if synthetic else str(source),
                            "degradation": {"enabled": True, "scales": [2]} if synthetic else None,
                        }
                    }
                },
            },
        )
        source_config = config.dataset.groups["main"]
        pair = AlignedImagePair(
            None if synthetic else source / "a.png",
            source / "a.png",
            CacheEntry(file="0" * 64 + ".safetensors", sha256="0" * 64),
            Path("absent-cache"),
            source_config.degradation,
        )
        # Calibration needs only frozen VAE state, not the denoiser or loss networks.
        method = RestorationMethod.__new__(RestorationMethod)
        method.vae = load_restoration_vae(tmp_path / "vae.safetensors", family="z-image", device=torch.device("cpu"))
        method.encoder = None
        method.posterior = "sample"
        method.state = torch.nn.Module()
        method.state.register_buffer("sigma", torch.tensor(0.0, dtype=torch.float64))
        before = torch.get_rng_state().clone()
        method.calibrate(training, RestorationDataset((pair,), "unused", read_restoration_pair))
        assert method.sigma == outputs[0]["sigma"]
        assert torch.equal(before, torch.get_rng_state())
    finally:
        torch.set_num_threads(previous)
