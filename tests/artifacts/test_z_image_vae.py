"""Verify exact external VAE conversion and actual application import semantics."""

from pathlib import Path

import pytest
import torch

from unitunner.artifacts.formats.z_image import convert_vae_tensors, vae_key_map
from unitunner.artifacts.tensors import save_tensors
from unitunner.cli.main import main
from unitunner.models.z_image.config.vae import ZImageVAEConfig
from unitunner.models.z_image.loading import load_z_image_vae
from unitunner.models.z_image.vae import ZImageVAE


@pytest.mark.parametrize("widths", [(4, 8), (4, 8, 8)])
def test_vae_external_roundtrip_and_import(tmp_path: Path, widths: tuple[int, ...]) -> None:
    """Preserve every weight and decoded pixel across non-four-block conversions."""
    torch.manual_seed(12)
    config = ZImageVAEConfig(block_channels=widths, norm_groups=2, layers_per_block=1, latent_channels=2)
    vae = ZImageVAE(config).eval()
    native = vae.state_dict()
    mapping = vae_key_map(tuple(native), num_blocks=len(widths))
    assert mapping["decoder.up_blocks.0.resnets.0.conv1.weight"] == (
        f"decoder.up.{len(widths) - 1}.block.0.conv1.weight"
    )
    external = convert_vae_tensors(native, mapping, to_native=False)
    assert external["encoder.mid.attn_1.q.weight"].ndim == 4
    restored = convert_vae_tensors(external, mapping, to_native=True)
    for key, value in native.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
    source, destination = tmp_path / "external.safetensors", tmp_path / "native.safetensors"
    topology = tmp_path / "topology.json"
    save_tensors(external, source)
    topology.write_text(config.model_dump_json())
    assert (
        main(
            [
                "weights",
                "import",
                str(source),
                str(destination),
                "--family",
                "z-image",
                "--kind",
                "vae",
                "--topology",
                str(topology),
                "--source-format",
                "comfyui",
            ]
        )
        == 0
    )
    loaded = load_z_image_vae(destination).eval()
    latents = torch.randn(1, 2, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(loaded.decode(latents), vae.decode(latents), rtol=0, atol=0)
    before = destination.read_bytes()
    external.pop(next(iter(external)))
    save_tensors(external, source)
    assert (
        main(
            [
                "weights",
                "import",
                str(source),
                str(destination),
                "--family",
                "z-image",
                "--kind",
                "vae",
                "--topology",
                str(topology),
                "--source-format",
                "comfyui",
            ]
        )
        == 2
    )
    assert destination.read_bytes() == before


def test_vae_conversion_rejects_ambiguous_or_spatial_projection() -> None:
    """Reject colliding maps and convolution weights that cannot be reshaped exactly."""
    with pytest.raises(ValueError, match="bijective"):
        convert_vae_tensors({}, {"a": "same", "b": "same"}, to_native=True)
    key = "encoder.mid_block.attentions.0.to_q.weight"
    with pytest.raises(ValueError, match="1x1"):
        convert_vae_tensors({"q": torch.zeros(4, 4, 3, 3)}, {key: "q"}, to_native=True)
