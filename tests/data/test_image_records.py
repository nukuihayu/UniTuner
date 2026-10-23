"""Verify complete Flux record contracts and immutable finite cache publication."""

import pytest
import torch

from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.cache.storage import (
    LatentManifest,
    load_latent_manifest,
    read_latent_record,
    save_latent_manifest,
    save_latent_record,
)


def test_flux_record_requires_native_conditions_and_rejects_nonfinite_collisions(tmp_path):
    identity = CacheIdentity(
        source_sha256="a" * 64, encoder_sha256="b" * 64, transform_sha256="c" * 64, layout="flux/fill/CHW+LD+V"
    )
    latent, caption, pooled, control, mask = (
        torch.randn(16, 2, 4),
        torch.randn(512, 8),
        torch.randn(6),
        torch.randn(16, 2, 4),
        torch.rand(64, 2, 4),
    )
    with pytest.raises(ValueError, match="native layout"):
        save_latent_record(tmp_path, identity, latent, caption, pooled=pooled)
    entry = save_latent_record(
        tmp_path, identity, latent, caption, pooled=pooled, control_latents=control, fill_mask=mask
    )
    manifest = LatentManifest(family="flux", variant="fill", entries=(entry,))
    save_latent_manifest(manifest, tmp_path)
    assert load_latent_manifest(tmp_path) == manifest
    record = read_latent_record(tmp_path, entry, device=torch.device("cpu"))
    torch.testing.assert_close(record.pooled, pooled, atol=0, rtol=0)
    torch.testing.assert_close(record.fill_mask, mask, atol=0, rtol=0)
    before = (tmp_path / entry.file).read_bytes()
    with pytest.raises(ValueError, match="finite"):
        save_latent_record(
            tmp_path,
            identity,
            latent,
            caption,
            pooled=torch.full_like(pooled, float("nan")),
            control_latents=control,
            fill_mask=mask,
        )
    with pytest.raises(ValueError, match="collision"):
        save_latent_record(
            tmp_path, identity, latent, caption, pooled=pooled + 1, control_latents=control, fill_mask=mask
        )
    assert (tmp_path / entry.file).read_bytes() == before
    with pytest.raises(ValueError, match="variant"):
        LatentManifest(family="flux", entries=(entry,))


def test_moss_edit_records_preserve_ordered_references_and_reject_base_cache(tmp_path):
    from unitunner.config.dataset_inputs import DatasetInputConfig
    from unitunner.config.sampling import SamplingConfig, validate_image_sampling

    identity = CacheIdentity(
        source_sha256="a" * 64, encoder_sha256="b" * 64, transform_sha256="c" * 64, layout="moss-image/edit/CHW+LD+R2"
    )
    refs = (torch.randn(8, 2, 3), torch.randn(8, 1, 2))
    entry = save_latent_record(tmp_path, identity, torch.randn(8, 2, 2), torch.randn(3, 12), references=refs)
    manifest = LatentManifest(family="moss-image", variant="edit", entries=(entry,))
    save_latent_manifest(manifest, tmp_path)
    assert load_latent_manifest(tmp_path) == manifest
    actual = read_latent_record(tmp_path, entry, device=torch.device("cpu"))
    for a, b in zip(actual.references, refs, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(ValueError):
        save_latent_record(tmp_path, identity, torch.randn(8, 2, 2), torch.randn(3, 12))
    DatasetInputConfig(reference_sources=("refs",)).validate_for("moss-image", "edit")
    with pytest.raises(ValueError, match="reference_sources"):
        DatasetInputConfig().validate_for("moss-image", "edit")
    sampling = SamplingConfig(prompt="edit", reference_images=("ref.png",))
    validate_image_sampling(sampling, family="moss-image", variant="edit")
    with pytest.raises(ValueError, match="reference_images"):
        validate_image_sampling(sampling, family="moss-image", variant="sft")
