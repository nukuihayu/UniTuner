"""Validate real frozen feature publication and strict cache/gradient boundaries."""

import json
from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from tests.support.assertions import require_value
from unitunner.artifacts.hashing import file_sha256
from unitunner.artifacts.tensors import save_tensors
from unitunner.config.features import RepaConfig
from unitunner.data.cache.feature_encoding import RepaCacheWriter
from unitunner.data.cache.representation import RepaTargets
from unitunner.data.cache.storage import CacheEntry
from unitunner.models.features.vit import DinoViT, DinoViTConfig
from unitunner.training.losses.representation import RepaProjection, VisualGrid


def test_real_encoder_exact_crop_frames_and_tamper_rejection(tmp_path):
    torch.manual_seed(52)
    model = (
        DinoViT(DinoViTConfig(width=16, depth=2, heads=2, patch=4, registers=2, image_size=16, hidden=24))
        .eval()
        .requires_grad_(False)
    )
    path = tmp_path / "encoder.safetensors"
    save_tensors(model.state_dict(), path, metadata={"unitunner.dinov3": json.dumps(asdict(model.config))})
    config = RepaConfig(
        encoder_dim=16,
        encoder_sha256=file_sha256(path),
        encoder_weights=str(path),
        cache_directory=str(tmp_path / "features"),
    )
    writer = RepaCacheWriter(config)
    source = CacheEntry(file="b" * 64 + ".safetensors", sha256="c" * 64)
    pixels = torch.randint(0, 256, (3, 9, 13, 3), dtype=torch.uint8)
    rng = torch.get_rng_state().clone()
    writer.publish(source, pixels)
    writer.finish()
    assert torch.equal(torch.get_rng_state(), rng)
    targets = RepaTargets(config)
    actual = targets.read(source)
    expected = []
    with torch.no_grad():
        for frame in pixels:
            value = frame.permute(2, 0, 1).unsqueeze(0).float() / 255
            value = F.interpolate(value, (16, 16), mode="bicubic", align_corners=False, antialias=True)
            value = (value - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
            expected.append(model(value).reshape(4, 4, 16))
    torch.testing.assert_close(actual, torch.stack(expected), atol=0, rtol=0)
    normalized = pixels.permute(3, 0, 1, 2).float() / 127.5 - 1
    other = CacheEntry(file="d" * 64 + ".safetensors", sha256="e" * 64)
    writer.publish_normalized(other, normalized)
    writer.finish()
    torch.testing.assert_close(RepaTargets(config).read(other), actual, atol=0, rtol=0)
    with pytest.raises(ValueError, match="cover"):
        targets.validate_sources((other,))
    entry = targets.entries[source.sha256]
    target_path = tmp_path / "features" / entry.file
    target_path.write_bytes(target_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        targets.read(source)
    with pytest.raises(ValueError, match="content"):
        RepaTargets(config)


def test_repa_original_projection_interpolation_and_stopped_target_gradients():
    torch.manual_seed(65)
    projection = RepaProjection(6, 7)
    hidden = torch.randn(2, 24, 6, requires_grad=True)
    target = torch.randn(2, 3, 2, 3, 7, requires_grad=True)
    grid = VisualGrid(2, 3, 4)
    actual = projection.alignment_loss(hidden, target, grid)
    actual_gradients = torch.autograd.grad(actual, (hidden, *projection.parameters(), target), allow_unused=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference = RepaProjection(6, 7)
    reference.load_state_dict(projection.state_dict())
    projected = reference.layers(reference_hidden.float())
    resized = F.interpolate(
        target.detach().permute(0, 4, 1, 2, 3), size=(2, 3, 4), mode="trilinear", align_corners=False
    )
    expected = -F.cosine_similarity(projected, resized.permute(0, 2, 3, 4, 1).reshape_as(projected), dim=-1).mean()
    expected_gradients = torch.autograd.grad(expected, (reference_hidden, *reference.parameters()))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual_gradients[-1] is None
    for left, right in zip(actual_gradients[:-1], expected_gradients, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_qwen_layered_remove_base_keeps_repa_targets_in_native_order(tmp_path):
    from tests.training import test_qwen_image
    from unitunner.config.routing import TreadConfig
    from unitunner.data.cache.representation import RepaManifest, save_repa_manifest, save_repa_target
    from unitunner.training.objectives import FeatureAlignment, TokenRoutingInstallation
    from unitunner.training.session import TrainingSession

    torch.set_num_threads(1)
    torch.manual_seed(73)
    with TrainingSession() as session:
        binding, modules, _optimizer = test_qwen_image.build(session, "lora", "layered")
        request = test_qwen_image.request("layered", True)
        session.own(TokenRoutingInstallation(binding.model, TreadConfig(end_layer=2), num_layers=2))
        sources = tuple(
            CacheEntry(file=f"{index + 1:064x}.safetensors", sha256=f"{index + 1:064x}") for index in range(2)
        )
        grids = [torch.randn(3, 2, 3, 7) for _ in sources]
        for grid in grids:
            grid[0].fill_(1000)
        entries = tuple(
            save_repa_target(tmp_path, source, grid, encoder_sha256="a" * 64)
            for source, grid in zip(sources, grids, strict=True)
        )
        save_repa_manifest(RepaManifest(encoder_sha256="a" * 64, encoder_dim=7, entries=entries), tmp_path)
        config = RepaConfig(weight=0.3, encoder_dim=7, encoder_sha256="a" * 64, cache_directory=str(tmp_path))
        alignment = session.own(
            FeatureAlignment(
                binding.model, None, num_layers=2, repa=config, targets=RepaTargets(config), device=torch.device("cpu")
            )
        )
        with alignment.capture():
            alignment.set_targets(sources, request.latents, skip_first_frame=True)
            result = binding.step(request)
            captures = torch.cat(alignment.recorder.features[0], dim=0)
            expected = result.loss + 0.3 * (
                sum(
                    require_value(alignment.projection).alignment_loss(
                        captures[index : index + 1], grid[1:].unsqueeze(0), VisualGrid(2, 2, 3)
                    )
                    for index, grid in enumerate(grids)
                )
                / 2
            )
            actual = alignment.step(result)
            torch.testing.assert_close(actual.loss, expected, atol=0, rtol=0)
            assert all(
                torch.equal(row, grid[1:]) for (row, _geometry), grid in zip(alignment._target_rows, grids, strict=True)
            )
        actual.loss.backward()
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in modules["adapter"].parameters()
        )
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in require_value(alignment.projection).parameters()
        )


def test_changed_repa_features_preserve_previous_published_inventory(tmp_path):
    from unitunner.data.cache.representation import RepaManifest, save_repa_manifest, save_repa_target

    source = CacheEntry(file="b" * 64 + ".safetensors", sha256="c" * 64)
    first = save_repa_target(tmp_path, source, torch.ones(1, 2, 3, 7), encoder_sha256="a" * 64)
    save_repa_manifest(RepaManifest(encoder_sha256="a" * 64, encoder_dim=7, entries=(first,)), tmp_path)
    previous = (tmp_path / "manifest.json").read_bytes()
    second = save_repa_target(tmp_path, source, torch.zeros(1, 2, 3, 7), encoder_sha256="a" * 64)
    assert first.artifact.file != second.artifact.file
    assert (tmp_path / "manifest.json").read_bytes() == previous
    config = RepaConfig(encoder_dim=7, encoder_sha256="a" * 64, cache_directory=str(tmp_path))
    torch.testing.assert_close(RepaTargets(config).read(source), torch.ones(1, 2, 3, 7), atol=0, rtol=0)
