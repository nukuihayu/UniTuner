"""Exercise public Qwen generation, adapter reload, references and RGBA layer publication."""

import json
import os
from contextlib import closing

import numpy as np
import pytest
import torch
from PIL import Image

from tests.support.qwen_image import write_text_assets
from tests.support.text_assets import main_with_local_tokenizers
from unituner.adapters.config import AdapterConfig
from unituner.adapters.loading import LoadedAdapter
from unituner.artifacts.formats.adapters import (
    AdapterTargetKeys,
    adapter_key_map,
    adapter_prefixes,
    convert_adapter_keys,
)
from unituner.artifacts.tensors import save_tensors
from unituner.config.datasets import DataConfig, DatasetConfig
from unituner.config.previews import PreviewConfig
from unituner.config.sampling import SamplingConfig
from unituner.config.training import ModelConfig, OutputConfig, TrainConfig, TrainingConfig
from unituner.models.catalog import ModelFamily
from unituner.models.qwen_image.config.model import QwenImageConfig
from unituner.models.qwen_image.config.vae import QwenImageVAEConfig
from unituner.models.qwen_image.loading import save_qwen_image, save_qwen_image_vae
from unituner.models.qwen_image.projections import adapter_projections
from unituner.models.qwen_image.transformer import QwenImageTransformer
from unituner.models.qwen_image.vae import QwenImageVAE
from unituner.training.parameters import ParameterInstallation
from unituner.training.previews.images import PreviewImages


@pytest.mark.parametrize("variant,algorithm", [("original", "lora"), ("edit-2511", "loha"), ("layered", "lokr")])
def test_qwen_public_generation_is_repeatable_and_publishes_each_layer(tmp_path, variant, algorithm):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(751)
        write_text_assets(tmp_path / "text")
        vae = QwenImageVAE(
            QwenImageVAEConfig(
                input_channels=4 if variant == "layered" else 3,
                base_channels=4,
                latent_channels=2,
                channel_multipliers=(1, 2, 2, 2),
                layers_per_block=1,
                latents_mean=(-0.31, 0.73),
                latents_std=(1.73, 0.91),
            )
        )
        save_qwen_image_vae(vae, tmp_path / "vae.safetensors")
        config = QwenImageConfig(
            variant=variant,
            in_channels=8,
            out_channels=2,
            num_layers=2,
            joint_attention_dim=16,
            attention_head_dim=16,
            num_attention_heads=2,
            axes_dims_rope=(4, 6, 6),
        )
        model = QwenImageTransformer(config)
        save_qwen_image(model, tmp_path / "model.safetensors")
        pixels = (np.arange(24 * 40 * 4, dtype=np.uint32).reshape(24, 40, 4) * 13).astype(np.uint8)
        with Image.fromarray(pixels) as image:
            image.save(tmp_path / "reference.png")
        with Image.fromarray(pixels[..., 0]) as image:
            image.save(tmp_path / "mask.png")
        adapter = AdapterConfig(algorithm=algorithm, rank=2, alpha=2.0)
        with closing(ParameterInstallation(model, adapter=adapter, targets=adapter_projections(model))) as installation:
            with torch.no_grad():
                for parameter in installation.branches.parameters():
                    parameter.uniform_(-0.04, 0.04)
            keys = adapter_key_map(
                [
                    AdapterTargetKeys(
                        str(i), adapter_prefixes(installation.plan.target_names, algorithm=algorithm)[name]
                    )
                    for i, name in enumerate(installation.plan.target_names)
                ],
                algorithm=algorithm,
            )
            save_tensors(
                convert_adapter_keys(installation.branches.state_dict(), keys), tmp_path / "adapter.safetensors"
            )
        sampling = {
            "prompt": "recolor shapes",
            "negative_prompt": "",
            "width": 32,
            "height": 16,
            "steps": 3,
            "seed": 753,
            "guidance": 2.0,
            "conditioning_resize": "output" if variant != "original" else "preferred",
        }
        if variant != "original":
            sampling["reference_images"] = (str(tmp_path / "reference.png"),)
        if variant == "edit-2511":
            sampling["mask_image"] = str(tmp_path / "mask.png")
        if variant == "layered":
            sampling["output_layers"] = 2
            sampling["prompt"] = " "
            sampling["caption_language"] = "zh"
        generation = {
            "model": {
                "family": "qwen-image",
                "variant": variant,
                "weights": str(tmp_path / "model.safetensors"),
                "vae": str(tmp_path / "vae.safetensors"),
                "text_encoder": str(tmp_path / "text"),
                "text_source_format": "qwen",
            },
            "sampling": sampling,
            "adapter": {"weights": str(tmp_path / "adapter.safetensors"), "config": adapter.model_dump()},
            "output": str(tmp_path / "generated.png"),
        }
        path = tmp_path / "generate.toml"
        lines = ["output = " + json.dumps(generation["output"])]
        for section in ("model", "sampling"):
            lines.append(f"[{section}]")
            lines.extend(f"{key} = {json.dumps(value)}" for key, value in generation[section].items())
        lines.extend(("[adapter]", "weights = " + json.dumps(generation["adapter"]["weights"]), "[adapter.config]"))
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in adapter.model_dump().items())
        path.write_text("\n".join(lines) + "\n")
        rng = torch.get_rng_state().clone()
        command = ["generate", str(path), "--device", os.environ.get("UNITUNER_TEST_DEVICE", "cpu")]
        assert main_with_local_tokenizers(command) == 0
        outputs = sorted(tmp_path.glob("generated*.png"))
        assert len(outputs) == (3 if variant == "layered" else 1)
        saved = {output.name: output.read_bytes() for output in outputs}
        for output in outputs:
            with Image.open(output) as image:
                assert image.size == (32, 16)
                assert image.mode == ("RGBA" if variant == "layered" else "RGB")
        assert main_with_local_tokenizers(command) == 0
        assert all(output.read_bytes() == saved[output.name] for output in outputs)
        assert torch.equal(rng, torch.get_rng_state())
        model.to(os.environ.get("UNITUNER_TEST_DEVICE", "cpu")).train()
        model.transformer_blocks[0].eval()
        configuration = TrainingConfig(
            model=ModelConfig(**{**generation["model"], "family": ModelFamily.QWEN_IMAGE}),
            dataset=DataConfig(
                groups={
                    "main": DatasetConfig(
                        source=str(tmp_path),
                        cache_directory=str(tmp_path / "cache"),
                        layer_sources=(str(tmp_path / "foreground"),) if variant == "layered" else (),
                    )
                }
            ),
            train=TrainConfig(adapter=adapter),
            output=OutputConfig(directory=str(tmp_path)),
            preview=PreviewConfig(samples=(SamplingConfig.model_validate(sampling),)),
        )
        targets = adapter_projections(model)
        with closing(
            LoadedAdapter(
                tmp_path / "adapter.safetensors",
                adapter,
                targets,
                external_prefixes=adapter_prefixes(targets, algorithm=algorithm),
            )
        ):
            identities = [id(parameter) for parameter in model.parameters()]
            flags = [module.training for module in model.modules()]
            renderer = PreviewImages(configuration, None)
            renderer(7, model, torch.nn.Identity(), None)
            previews = sorted((tmp_path / "previews").glob("*.png"))
            assert len(previews) == len(outputs)
            assert [path.read_bytes() for path in previews] == [path.read_bytes() for path in outputs]
            assert flags == [module.training for module in model.modules()]
            assert identities == [id(parameter) for parameter in model.parameters()]
            assert torch.equal(rng, torch.get_rng_state())
            if variant == "edit-2511":
                with Image.fromarray(255 - pixels) as image:
                    image.save(tmp_path / "reference.png")
                renderer(8, model, torch.nn.Identity(), None)
                assert (tmp_path / "previews/step-00000008-00.png").read_bytes() != previews[0].read_bytes()
    finally:
        torch.set_num_threads(previous)
