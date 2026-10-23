"""Check public conditioning caches from floating, prequantized, and packed text encoders."""

import pytest
import torch
from PIL import Image

from tests.support.qwen_image_21 import write_text_assets
from tests.support.text_assets import local_text_config, main_with_local_tokenizers
from unitunner.artifacts.conversion import convert_weights
from unitunner.config.text import TextEncodingConfig
from unitunner.data.cache.storage import load_latent_manifest
from unitunner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unitunner.models.qwen_image_2_1.data.cache.storage import read_qwen_image_21_record
from unitunner.models.qwen_image_2_1.loading import save_qwen_image_21_vae
from unitunner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text
from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE


@pytest.mark.parametrize("format", ("none", "float8_e4m3fn", "int8_convrot"))
@pytest.mark.parametrize("precision", ("fp32", "bf16"))
def test_conditioning_cache_quantization_reuse_and_precision_identity(tmp_path, format, precision):
    kitchen = pytest.importorskip("comfy_kitchen")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        write_text_assets(tmp_path / "text", language_layers=3)
        vae = QwenImage21VAE(
            QwenImage21VAEConfig(
                base_channels=2, decoder_base_channels=4, channel_multipliers=(1, 2, 4, 4, 4), layers_per_block=1
            )
        )
        save_qwen_image_21_vae(vae, tmp_path / "vae.safetensors")
        images = tmp_path / "images"
        images.mkdir()
        Image.new("RGBA", (32, 32), (80, 40, 180, 200)).save(images / "a.png")
        (images / "a.txt").write_text("a purple flower")
        config = tmp_path / "train.toml"
        document = f'''[model]
family = "qwen-image-2-1"
variant = "base"
weights = "not-loaded.safetensors"
vae = "{tmp_path}/vae.safetensors"
text_encoder = "{tmp_path}/text"
[model.text]
precision = "{precision}"
quantization = "{format}"
quantization_rotation_size = 4
[dataset.groups.main]
source = "{images}"
cache_directory = "{tmp_path}/cache"
resolution = [32, 32]
[train]
parameter_mode = "full"
'''
        config.write_text(document)
        options = TextEncodingConfig(precision=precision, quantization=format, quantization_rotation_size=4)
        with kitchen.use_backend("eager"):
            assert main_with_local_tokenizers(["cache", str(config), "--device", "cpu"]) == 0
            encoder = load_qwen_image_21_text(
                tmp_path / "text",
                device=torch.device("cpu"),
                text_config=local_text_config(tmp_path / "text", base=options),
            )
        entry = load_latent_manifest(tmp_path / "cache").entries[0]
        record = read_qwen_image_21_record(tmp_path / "cache", entry, device=torch.device("cpu"))
        torch.testing.assert_close(
            record.caption, encoder.encode(["a purple flower"])[0].features.float(), atol=0, rtol=0
        )
        assert record.caption.dtype == record.latents.dtype == torch.float32
        first = (tmp_path / "cache" / entry.file).read_bytes()
        with kitchen.use_backend("eager"):
            assert main_with_local_tokenizers(["cache", str(config), "--device", "cpu"]) == 0
        assert (tmp_path / "cache" / entry.file).read_bytes() == first
        if format != "none":
            # Convert the same effective floating precision to reproduce online codes.
            source = tmp_path / "text/model.safetensors"
            floating = tmp_path / "floating.safetensors"
            packed = tmp_path / "packed.safetensors"
            convert_weights(source, floating, precision=precision)
            with kitchen.use_backend("eager"):
                convert_weights(floating, packed, precision=format, family="qwen3-vl", rotation_size=4)
            from unitunner.artifacts.tensors import TensorReader

            with TensorReader(packed) as reader:
                stored = {
                    key.removeprefix("model."): reader.read(key) for key in reader.keys() if key != "lm_head.weight"
                }
            for key, tensor in encoder.model.state_dict().items():
                torch.testing.assert_close(
                    stored[key],
                    tensor.cpu(),
                    atol=0,
                    rtol=2e-7 if "scale" in key else 0,
                    msg=lambda msg, name=key: f"{name}: {msg}",
                )
            packed.replace(source)
            config.write_text(document.replace(f'quantization = "{format}"', 'quantization = "none"'))
            assert main_with_local_tokenizers(["cache", str(config), "--device", "cpu"]) == 0
            restored = load_latent_manifest(tmp_path / "cache").entries[0]
            actual = read_qwen_image_21_record(tmp_path / "cache", restored, device=torch.device("cpu"))
            # File materialization can change ConvRot scale reduction by one FP32 ULP.
            torch.testing.assert_close(actual.caption, record.caption, atol=1e-7, rtol=1e-4)
            assert restored.file != entry.file
        else:
            config.write_text(
                document.replace(
                    f'precision = "{precision}"', f'precision = "{"bf16" if precision == "fp32" else "fp32"}"'
                )
            )
            assert main_with_local_tokenizers(["cache", str(config), "--device", "cpu"]) == 0
            changed = load_latent_manifest(tmp_path / "cache").entries[0]
            assert changed.file != entry.file
    finally:
        torch.set_num_threads(previous)
