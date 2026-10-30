"""Exercise Qwen2.1 restoration cache, sampled encoder training, resume, previews and export inference."""

import os

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.support.qwen_image_21 import write_text_assets
from tests.support.text_assets import main_with_local_tokenizers
from unituner.artifacts.tensors import TensorReader
from unituner.models.qwen_image_2_1.config.model import QwenImage21Config
from unituner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unituner.models.qwen_image_2_1.loading import save_qwen_image_21, save_qwen_image_21_vae
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.models.qwen_image_2_1.vae import QwenImage21VAE


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
def test_qwen21_restoration_shared_runner_resume_preview_and_export(tmp_path, algorithm, *, wavelet=False):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(809)
        root = tmp_path
        width, height = 64, 32
        device = os.environ.get("UNITUNER_TEST_DEVICE", "cpu")
        write_text_assets(root / "text")
        save_qwen_image_21_vae(
            QwenImage21VAE(
                QwenImage21VAEConfig(
                    base_channels=2, decoder_base_channels=4, channel_multipliers=(1, 2, 4, 4, 4), layers_per_block=1
                )
            ),
            root / "vae.safetensors",
        )
        save_qwen_image_21(
            QwenImage21Transformer(
                QwenImage21Config(
                    in_channels=64,
                    out_channels=64,
                    num_layers=2,
                    context_in_dim=16,
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
family = "qwen-image-2-1"
variant = "base"
weights = "{root}/model.safetensors"
vae = "{root}/vae.safetensors"
text_encoder = "{root}/text"
text_source_format = "transformers"
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
        text += f'''[losses]
l1 = 0.5
lrr = 1.0
dv3d = 0.0
gan = 0.0
fdl = 0.0
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
        if wavelet:
            text += '[method.wavelet]\nwavelet = "db4"\nlevels = 2\nweight = 0.5\n'
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
            sigma = reader.metadata()["unituner.restoration.sigma"]
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
