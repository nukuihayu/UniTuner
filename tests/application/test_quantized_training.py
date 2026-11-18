"""Run public conversion, cache, quantized training, resume and adapter inference."""

import pytest
import torch

from tests.application.test_image_flow import image_assets as image_assets
from tests.support.text_assets import main_with_local_tokenizers
from unituner.artifacts.tensors import TensorReader


@pytest.mark.parametrize("algorithm", ("lora", "loha", "lokr"))
def test_public_convrot_training_resume_and_generation(image_assets, algorithm):
    pytest.importorskip("comfy_kitchen")
    training, generation = image_assets
    root = training.parent
    source, packed = root / "model.safetensors", root / "model-int8.safetensors"
    assert (
        main_with_local_tokenizers(
            [
                "weights",
                "convert",
                str(source),
                str(packed),
                "--precision",
                "int8_convrot",
                "--rotation-size",
                "4",
                "--backend",
                "eager",
            ]
        )
        == 0
    )
    before = packed.read_bytes()
    body = (
        training.read_text()
        .replace(str(source), str(packed))
        .replace('parameter_mode = "full"', 'parameter_mode = "adapter"')
    )
    body = body.replace("[output]", "[output]\nsave_every_updates = 1")
    body += f'''\n[train.adapter]
algorithm = "{algorithm}"
rank = 2
alpha = 3.0
[execution]
precision = "fp32"
quantization = "convrot-int8"
quantization_rotation_size = 4
quantization_group_size = 4
recompute = "block"
compile = false
'''
    training.write_text(body)
    assert main_with_local_tokenizers(["cache", str(training)]) == 0
    assert main_with_local_tokenizers(["train", str(training)]) == 0
    checkpoint = root / "run/resume/model.state-00000002.safetensors"
    adapter = root / "run/weights/model.safetensors"
    expected = checkpoint.read_bytes(), adapter.read_bytes()
    assert (
        main_with_local_tokenizers(
            ["train", str(training), "--resume", str(root / "run/resume/model.state-00000001.safetensors")]
        )
        == 0
    )
    assert expected == (checkpoint.read_bytes(), adapter.read_bytes())
    assert packed.read_bytes() == before
    with TensorReader(adapter) as reader:
        assert not any(key.endswith(("comfy_quant", "weight_scale")) for key in reader.keys())
    generation.write_text(
        generation.read_text().replace(str(source), str(packed))
        + f'''
[adapter]
weights = "{adapter}"
[adapter.config]
algorithm = "{algorithm}"
rank = 2
alpha = 3.0
'''
    )
    assert main_with_local_tokenizers(["generate", str(generation)]) == 0
    image = (root / "sample.png").read_bytes()
    assert main_with_local_tokenizers(["generate", str(generation)]) == 0
    assert image == (root / "sample.png").read_bytes()


@pytest.mark.parametrize("device", ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
@pytest.mark.parametrize("precision", ("float8_e4m3fn", "int8_convrot"))
def test_public_training_quantizes_floating_base_before_adapters(image_assets, precision, device):
    pytest.importorskip("comfy_kitchen")
    training, _generation = image_assets
    root = training.parent
    source = root / "model.safetensors"
    before = source.read_bytes()
    body = training.read_text().replace('parameter_mode = "full"', 'parameter_mode = "adapter"')
    body = body.replace("[output]", "[output]\nsave_every_updates = 1")
    body += f'''
[train.adapter]
algorithm = "lora"
rank = 2
alpha = 3.0
[execution]
precision = "fp32"
quantization = "{precision}"
quantization_rotation_size = 4
quantization_group_size = 4
recompute = "block"
compile = false
'''
    training.write_text(body)
    assert main_with_local_tokenizers(["cache", str(training), "--device", device]) == 0
    assert main_with_local_tokenizers(["train", str(training), "--device", device]) == 0
    checkpoint, adapter = root / "run/resume/model.state-00000002.safetensors", root / "run/weights/model.safetensors"
    expected = checkpoint.read_bytes(), adapter.read_bytes()
    assert (
        main_with_local_tokenizers(
            ["train", str(training), "--resume", str(root / "run/resume/model.state-00000001.safetensors"), "--device", device]
        )
        == 0
    )
    assert expected == (checkpoint.read_bytes(), adapter.read_bytes())
    assert source.read_bytes() == before
    with TensorReader(adapter) as reader:
        assert not any("comfy_quant" in key or "weight_scale" in key for key in reader.keys())
