"""Exercise Qwen restoration cache, sampled encoder training, resume, previews and export inference."""

import os

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.support.qwen_image import write_text_assets
from tests.support.text_assets import main_with_local_tokenizers
from unitunner.artifacts.tensors import TensorReader, save_tensors
from unitunner.models.features.convnext import DinoConvNeXt
from unitunner.models.features.vgg import Vgg19Features
from unitunner.models.qwen_image.config.model import QwenImageConfig
from unitunner.models.qwen_image.config.vae import QwenImageVAEConfig
from unitunner.models.qwen_image.loading import save_qwen_image, save_qwen_image_vae
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.models.qwen_image.vae import QwenImageVAE


@pytest.fixture(autouse=True, scope="module")
def deterministic_convolutions():
    """Pin cuDNN algorithms for exact CUDA restoration update/resume comparisons."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.backends.cudnn.flags(deterministic=True, benchmark=False), sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("algorithm", ["full", "lora", "loha", "lokr"])
def test_qwen_restoration_shared_runner_resume_preview_and_export(tmp_path, algorithm):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(809)
        root = tmp_path
        width, height = (512, 512) if algorithm == "lora" else (32, 16)
        device = os.environ.get("UNITUNNER_TEST_DEVICE", "cpu")
        write_text_assets(root / "text")
        save_qwen_image_vae(
            QwenImageVAE(
                QwenImageVAEConfig(
                    base_channels=4,
                    latent_channels=2,
                    channel_multipliers=(1, 2, 2, 2),
                    layers_per_block=1,
                    latents_mean=(-0.31, 0.73),
                    latents_std=(1.73, 0.91),
                )
            ),
            root / "vae.safetensors",
        )
        save_qwen_image(
            QwenImageTransformer(
                QwenImageConfig(
                    in_channels=8,
                    out_channels=2,
                    num_layers=2,
                    joint_attention_dim=16,
                    attention_head_dim=16,
                    num_attention_heads=2,
                    axes_dims_rope=(4, 6, 6),
                )
            ),
            root / "model.safetensors",
        )
        for directory in ("high", "low"):
            (root / directory).mkdir()
            for index, name in enumerate(("a", "b")):
                pixels = (
                    np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3) * 13 + index * 31
                ).astype(np.uint8)
                if directory == "low":
                    pixels = pixels // 2 + 19
                with Image.fromarray(pixels) as image:
                    image.save(root / directory / f"{name}.png")
        (root / "high/a.txt").write_text("restore colors")
        (root / "high/b.txt").write_text("restore colors and detailed shapes " * 3)
        header = f'''[model]
family = "qwen-image"
variant = "original"
weights = "{root}/model.safetensors"
vae = "{root}/vae.safetensors"
text_encoder = "{root}/text"
text_source_format = "qwen"
'''
        text = (
            header
            + f'''[dataset.groups.main]
kind = "restoration"
source = "{root}/high"
low_quality_source = "{root}/low"
cache_directory = "{root}/cache"
resolution = [{width},{height}]
batch_size = {1 if algorithm == "lora" else 2}
horizontal_flip_probability = 0.0
[train]
parameter_mode = "{"full" if algorithm == "full" else "adapter"}"
max_updates = 3
accumulation_steps = {1 if algorithm == "lora" else 2}
seed = 811
learning_rate = 0.001
'''
        )
        if algorithm != "full":
            text += f'''[train.adapter]
algorithm = "{algorithm}"
rank = 2
alpha = 3.0
'''
        text += """[method]
kind = "restoration"
[method.restoration]
encoder_rank = 2
"""
        if algorithm == "full":
            text += 'posterior = "mean"\ncalibrate = true\ncalibration_pairs = 2\n'
        if algorithm == "lora":
            save_tensors(DinoConvNeXt((4, 8, 12, 16), (1, 1, 1, 1)).state_dict(), root / "dino.safetensors")
            save_tensors(Vgg19Features((4, 8, 12, 16, 16)).state_dict(), root / "vgg.safetensors")
            text += f'''dino_weights = "{root}/dino.safetensors"
dino_widths = [4,8,12,16]
dino_depths = [1,1,1,1]
vgg_weights = "{root}/vgg.safetensors"
vgg_widths = [4,8,12,16,16]
'''
        text += f'''[losses]
l1 = 0.5
lrr = 1.0
dv3d = {0.1 if algorithm == "lora" else 0.0}
gan = {0.1 if algorithm == "lora" else 0.0}
fdl = {0.1 if algorithm == "lora" else 0.0}
histogram = 0.1
[execution]
precision = "fp32"
recompute = "block"
[output]
max_resume_checkpoints = 0
directory = "{root}/run"
name = "model"
save_every_updates = 1
'''
        configuration = root / "train.toml"
        configuration.write_text(text)

        def command(args):
            return main_with_local_tokenizers([*args, "--device", device])

        rng = torch.get_rng_state().clone()
        assert command(["cache", str(configuration)]) == 0
        assert torch.equal(rng, torch.get_rng_state())
        assert command(["train", str(configuration)]) == 0
        files = {
            path.relative_to(root / "run").as_posix(): path.read_bytes()
            for path in (root / "run").rglob("*.safetensors")
        }
        configuration.write_text(text.replace(f"{root}/run", f"{root}/resume"))
        assert (
            command(
                ["train", str(configuration), "--resume", str(root / "run/resume/model.state-00000001.safetensors")]
            )
            == 0
        )
        for name in (
            "resume/model.state-00000003.safetensors",
            "weights/model.safetensors",
            "weights/model.encoder.safetensors",
        ):
            assert (root / "resume" / name).read_bytes() == files[name]
        preview = (
            text.replace(f"{root}/run", f"{root}/preview")
            + """[preview]
every_updates = 1
at_start = true
num_images = 1
"""
        )
        configuration.write_text(preview)
        assert command(["train", str(configuration)]) == 0
        for name in (
            "resume/model.state-00000003.safetensors",
            "weights/model.safetensors",
            "weights/model.encoder.safetensors",
        ):
            assert (root / "preview" / name).read_bytes() == files[name]
        # Reload the public full/adapter and encoder artifacts into staged inference.
        restore = (
            f'''input = "{root}/low/a.png"
output = "{root}/restored.png"
prompt = "restore colors"
encoder_weights = "{root}/run/weights/model.encoder.safetensors"
'''
            + header
        )
        if algorithm == "full":
            restore = restore.replace(f"{root}/model.safetensors", f"{root}/run/weights/model.safetensors")
        else:
            restore += f'''[adapter]
weights = "{root}/run/weights/model.safetensors"
[adapter.config]
algorithm = "{algorithm}"
rank = 2
alpha = 3.0
'''
        path = root / "restore.toml"
        path.write_text(restore)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(torch.device(device)) if device.startswith("cuda") else None
        assert command(["restore", str(path)]) == 0
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(torch.device(device)))
        saved = (root / "restored.png").read_bytes()
        assert command(["restore", str(path)]) == 0
        assert (root / "restored.png").read_bytes() == saved
        with (
            Image.open(root / "restored.png") as restored,
            Image.open(root / "preview/previews/step-00000003-00.png") as grid,
        ):
            assert restored.size == (width, height)
            assert np.array_equal(np.array(restored), np.array(grid)[2 : height + 2, width + 4 : 2 * width + 4])
        with Image.new("RGB", (19, 13), (37, 83, 149)) as image:
            image.save(root / "odd.png")
        path.write_text(
            restore.replace(f"{root}/low/a.png", f"{root}/odd.png").replace("restored.png", "odd-restored.png")
        )
        assert command(["restore", str(path)]) == 0
        with Image.open(root / "odd-restored.png") as image:
            assert image.size == (19, 13) and image.mode == "RGB"
        with TensorReader(root / "run/weights/model.safetensors") as reader:
            sigma = reader.metadata()["unitunner.restoration.sigma"]
        path.write_text(
            restore.replace('prompt = "restore colors"', f'prompt = "restore colors"\nsigma = {float(sigma) / 2}')
        )
        assert command(["restore", str(path)]) == 2
        assert (root / "restored.png").read_bytes() == saved
        # Input edits invalidate the indexed caption identity before resume.
        (root / "high/a.txt").write_text("changed caption")
        configuration.write_text(text)
        assert (
            command(
                ["train", str(configuration), "--resume", str(root / "run/resume/model.state-00000001.safetensors")]
            )
            == 2
        )
    finally:
        torch.set_num_threads(previous)
