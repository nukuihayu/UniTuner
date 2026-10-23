"""Exercise newly connected full and mid-timestep image training lifecycles."""

import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.application import test_flux2_flow, test_flux3_flow, test_flux_flow, test_moss_flow
from tests.support.assertions import require_value
from tests.support.text_assets import local_text_config, main_with_local_tokenizers
from unitunner.artifacts.tensors import TensorReader
from unitunner.config.generation import GenerationAdapterConfig
from unitunner.config.loading import load_training_config
from unitunner.config.restoration_inference import RestorationInferenceConfig
from unitunner.config.training import TrainingConfig
from unitunner.inference.restoration.run import run as restore
from unitunner.training.methods.mid_timestep import restoration_sigma
from unitunner.training.run import run as train

flux_assets = test_flux_flow.assets
flux2_assets = test_flux2_flow.assets
flux3_assets = test_flux3_flow.assets
moss_assets = test_moss_flow.moss_assets


@pytest.fixture(autouse=True)
def deterministic_execution():
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.backends.cudnn.flags(deterministic=True, benchmark=False), sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.parametrize("parameter_mode", ["adapter", "full"])
@pytest.mark.parametrize(
    "family,variant,version",
    [
        ("flux", "dev", None),
        ("flux2", "base", "klein-4b"),
        ("flux2", "distilled", "dev"),
        ("flux3", "base", None),
        ("flux3", "distilled", None),
        ("moss-image", "base", None),
    ],
)
def test_mid_timestep_update_preview_resume_and_restore(request, family, variant, version, parameter_mode):
    root = request.getfixturevalue(
        {"flux": "flux_assets", "flux2": "flux2_assets", "flux3": "flux3_assets", "moss-image": "moss_assets"}[family]
    )
    device = os.environ.get("UNITUNNER_TEST_DEVICE", "cpu")
    if family == "flux":
        path = test_flux_flow.write_training(root, variant, "lora")
    elif family == "flux2":
        path = test_flux2_flow.configs(root, "lora", version, variant, 0)
    elif family == "flux3":
        path = test_flux3_flow.configs(root, "lora", variant, 0)
    else:
        path, _ = test_moss_flow.write_configs(root, "lora")
    assert main_with_local_tokenizers(["cache", str(path), "--device", device]) == 0
    loaded = load_training_config(path)
    document = loaded.model_dump(mode="json")
    document["model"]["text"] = local_text_config(
        Path(loaded.model.text_encoder),
        base=loaded.model.text,
        clip=root / "text_encoder" if family == "flux" else None,
    ).model_dump(mode="json")
    document["dataset"]["groups"]["main"].update(
        kind="restoration",
        low_quality_source=str(root / "images"),
        horizontal_flip_probability=0,
    )
    document["method"] = {
        "kind": "restoration",
        "restoration": {"midtimestep": 321, "encoder_rank": 2, "posterior": "sample" if family == "flux" else "mean"},
    }
    document["losses"] = {"lrr": 0.2, "dv3d": 0, "l1": 0.5, "gan": 0, "fdl": 0, "histogram": 0}
    document["train"].update(parameter_mode=parameter_mode, max_updates=3, learning_rate=0.001)
    if parameter_mode == "full":
        document["train"]["adapter"] = None
    document["output"].update(max_resume_checkpoints=0, directory=str(root / "continuous"), name="restored")
    document["preview"] = {"at_start": True, "every_updates": 1, "num_images": 1}
    config = TrainingConfig.model_validate_json(json.dumps(document))
    original = Path(config.model.weights).read_bytes()
    assert train(config, device=device, resume=None) == 0
    artifact = root / "continuous/weights/restored.safetensors"
    assert len(list((root / "continuous/previews").glob("*.png"))) == 4
    document["output"]["directory"] = str(root / "resumed")
    resumed = TrainingConfig.model_validate_json(json.dumps(document))
    assert train(resumed, device=device, resume=root / "continuous/resume/restored.state-00000001.safetensors") == 0
    for name in [
        "weights/restored.safetensors",
        "weights/restored.encoder.safetensors",
        "resume/restored.state-00000003.safetensors",
    ]:
        assert (root / "continuous" / name).read_bytes() == (root / "resumed" / name).read_bytes()
    assert Path(config.model.weights).read_bytes() == original
    with TensorReader(artifact) as reader:
        assert float(reader.metadata()["unitunner.restoration.sigma"]) == restoration_sigma(
            321, family=family, resolution=config.dataset.groups["main"].resolution
        )
        assert not any(key.startswith("ss_") for key in reader.metadata())
    inference = RestorationInferenceConfig(
        model=config.model.model_copy(update={"weights": str(artifact)}) if parameter_mode == "full" else config.model,
        input=str(root / "images/a.png"),
        output=str(root / "restored.png"),
        prompt="restore detail",
        encoder_weights=str(root / "continuous/weights/restored.encoder.safetensors"),
        adapter=None
        if parameter_mode == "full"
        else GenerationAdapterConfig(weights=str(artifact), config=require_value(config.train.adapter)),
    )
    assert restore(inference, device=device) == 0
    with Image.open(root / "restored.png") as result, Image.open(root / "images/a.png") as source:
        assert result.mode == "RGB" and result.size == source.size
