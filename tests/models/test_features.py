"""Validate native feature artifact contracts and generated-image gradients."""

import pytest
import torch
from torch.nn import functional as F

from unitunner.artifacts.tensors import save_tensors
from unitunner.models.features.convnext import ChannelLayerNorm, DinoConvNeXt, l2_pool
from unitunner.models.features.loading import load_dino_convnext, load_dino_vit, load_vgg19_features
from unitunner.models.features.normalization import normalize_image
from unitunner.models.features.vgg import Vgg19Features
from unitunner.models.features.vit import DinoViT, DinoViTConfig
from unitunner.training.losses.adversarial import frozen_discriminator, gan_loss
from unitunner.training.losses.perceptual import DinoPerceptualLoss
from unitunner.training.methods.discriminator import MultilevelDiscriminator


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_channel_norm_and_pool_match_independent_spatial_formulas():
    value = torch.randn(2, 4, 7, 8, requires_grad=True)
    norm = ChannelLayerNorm(4)
    expected = F.layer_norm(value.permute(0, 2, 3, 1), (4,), norm.weight, norm.bias, 1e-6).permute(0, 3, 1, 2)
    actual = norm(value)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual_grad = torch.autograd.grad(actual.square().mean(), value, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().mean(), value)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
    padded = F.pad(value.square(), (1, 1, 1, 1))
    expected_pool = torch.zeros_like(value)
    for y, wy in enumerate((0.25, 0.5, 0.25)):
        for x, wx in enumerate((0.25, 0.5, 0.25)):
            expected_pool += padded[:, :, y : y + 7, x : x + 8] * wy * wx
    torch.testing.assert_close(l2_pool(value), (expected_pool + 1e-12).sqrt(), atol=1e-6, rtol=1e-5)
    low_precision = value[:, :3].detach().bfloat16().requires_grad_()
    expected_normalized = (
        (low_precision.float() + 1) / 2 - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    ) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    torch.testing.assert_close(normalize_image(low_precision), expected_normalized, rtol=0, atol=0)


def test_convnext_strict_reload_and_dists_image_gradient(tmp_path):
    torch.manual_seed(42)
    widths, depths = (4, 8, 12, 16), (1, 1, 1, 1)
    model = DinoConvNeXt(widths, depths)
    path = tmp_path / "dino.safetensors"
    save_tensors(model.state_dict(), path)
    loaded = load_dino_convnext(path, widths=widths, depths=depths, device=torch.device("cpu"))
    image = torch.randn(1, 3, 32, 32, requires_grad=True)
    for actual, expected in zip(loaded.features(image, pooled=True), model.features(image, pooled=True), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    target = torch.randn_like(image).requires_grad_()
    loss = DinoPerceptualLoss(loaded)(image, target)
    loss.backward()
    assert image.grad is not None and image.grad.norm() > 0
    assert target.grad is None
    assert all(parameter.grad is None for parameter in loaded.parameters())
    tensors = dict(model.state_dict())
    tensors["norms.3.weight"] = tensors["norms.3.weight"] + 1
    save_tensors(tensors, path)
    with pytest.raises(ValueError, match="tied"):
        load_dino_convnext(path, widths=widths, depths=depths, device=torch.device("cpu"))


def test_channel_norm_bf16_matches_fp32_statistics():
    torch.manual_seed(93)
    norm = ChannelLayerNorm(16).bfloat16().requires_grad_(False)
    image = torch.randn(2, 16, 9, 7).bfloat16().requires_grad_()
    reference_input = image.detach().float().requires_grad_()
    expected = F.layer_norm(
        reference_input.permute(0, 2, 3, 1), (16,), norm.weight.float(), norm.bias.float(), norm.eps
    ).permute(0, 3, 1, 2)
    actual = norm(image)
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient.float())
    torch.testing.assert_close(image.grad, reference_input.grad.bfloat16(), rtol=0.02, atol=0.016)


@pytest.mark.parametrize("pooled", [False, True])
def test_frozen_convnext_recompute_preserves_image_gradients_and_saves_activations(pooled):
    torch.manual_seed(31)
    model = DinoConvNeXt((8, 16, 24, 32), (2, 2, 3, 1)).eval().requires_grad_(False)
    source = torch.randn(1, 3, 64, 64)
    results = []
    for recompute in (False, True):
        model.recompute = recompute
        image = source.clone().requires_grad_()
        saved = []

        def pack(tensor, saved=saved):
            saved.append(tensor.numel() * tensor.element_size())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            features = model.features(image, pooled=pooled)
        sum(feature.square().mean() for feature in features).backward()
        results.append((tuple(f.detach() for f in features), image.grad, sum(saved)))
    for actual, expected in zip(results[1][0], results[0][0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(results[1][1], results[0][1], rtol=0, atol=0)
    assert results[1][2] < results[0][2] / 2
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        for actual, expected in zip(model.features(source, pooled=pooled), results[0][0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("suffix", [".pth", ".pt", ".safetensors"])
def test_convnext_official_pytorch_state_dict_and_misnamed_checkpoint(tmp_path, suffix):
    widths, depths = (4, 8, 12, 16), (1, 1, 1, 1)
    model = DinoConvNeXt(widths, depths)
    path = tmp_path / ("dino" + suffix)
    torch.save(model.state_dict(), path)
    before = path.read_bytes()
    rng = torch.get_rng_state().clone()
    loaded = load_dino_convnext(path, widths=widths, depths=depths, device=torch.device("cpu"))
    assert torch.equal(rng, torch.get_rng_state()) and path.read_bytes() == before
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    with pytest.raises(ValueError, match="dino_widths"):
        load_dino_convnext(path, widths=(5, 8, 12, 16), depths=depths, device=torch.device("cpu"))
    broken = dict(model.state_dict())
    broken.pop(next(iter(broken)))
    torch.save(broken, path)
    with pytest.raises(ValueError, match="keys differ"):
        load_dino_convnext(path, widths=widths, depths=depths, device=torch.device("cpu"))
    torch.save({"model": model.state_dict()}, path)
    with pytest.raises(ValueError, match="tensor state dict"):
        load_dino_convnext(path, widths=widths, depths=depths, device=torch.device("cpu"))


def test_vgg_stage_views_preserve_official_feature_indices(tmp_path):
    model = Vgg19Features((2, 3, 4, 5, 6))
    path = tmp_path / "vgg.safetensors"
    save_tensors(model.state_dict(), path)
    loaded = load_vgg19_features(path, device=torch.device("cpu"), widths=model.widths)
    hidden = torch.randn(1, 3, 80, 80, requires_grad=True)
    direct = hidden
    cut_outputs = []
    for index, layer in enumerate(model.features):
        direct = layer(direct)
        if index in (3, 8, 17, 26, 35):
            cut_outputs.append(direct)
    for stage, expected in zip(loaded.feature_stages(), cut_outputs, strict=True):
        hidden = stage(hidden)
        torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
    assert hidden.shape == (1, 6, 5, 5)


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".safetensors"])
@pytest.mark.parametrize("layout", ["features", "full", "sequential"])
def test_vgg_formats_and_official_classifier_handling(tmp_path, suffix, layout):
    model = Vgg19Features((2, 3, 4, 5, 6))
    state = dict(model.state_dict())
    if layout == "full":
        state.update({f"classifier.{i}.{s}": torch.ones(1) for i in (0, 3, 6) for s in ("weight", "bias")})
    elif layout == "sequential":
        state = {key.removeprefix("features."): value for key, value in state.items()}
    path = tmp_path / ("vgg" + suffix)

    def save(tensors):
        if suffix == ".safetensors":
            save_tensors(tensors, path)
        else:
            torch.save(tensors, path)

    save(state)
    rng = torch.get_rng_state().clone()
    loaded = load_vgg19_features(path, device=torch.device("cpu"), widths=model.widths)
    assert torch.equal(rng, torch.get_rng_state())
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    pixels = torch.randn(1, 3, 32, 32, requires_grad=True)
    loaded.features(pixels).sum().backward()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    save({**state, "unexpected.weight": torch.ones(1)})
    with pytest.raises(ValueError, match="keys differ"):
        load_vgg19_features(path, device=torch.device("cpu"), widths=model.widths)


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".safetensors"])
def test_dino_vit_published_topology_formats(tmp_path, suffix):
    model = DinoViT(DinoViTConfig(width=384, depth=1, heads=6, hidden=768))
    path = tmp_path / ("vit" + suffix)
    if suffix == ".safetensors":
        save_tensors(model.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
    loaded = load_dino_vit(path, device=torch.device("cpu"))
    assert loaded.config == model.config
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_real_multilevel_discriminator_generator_and_discriminator_gradients():
    torch.manual_seed(43)
    backbone = DinoConvNeXt((4, 8, 12, 16), (1, 1, 1, 1))
    discriminator = MultilevelDiscriminator(backbone, 512, diffaug=False).train()
    image = torch.randn(1, 3, 512, 512, requires_grad=True)
    with frozen_discriminator(discriminator):
        loss = gan_loss(discriminator(image), True)
    loss.backward()
    assert image.grad is not None and image.grad.norm() > 0
    assert all(parameter.grad is None for parameter in discriminator.parameters())
    assert not backbone.training and all(not parameter.requires_grad for parameter in backbone.parameters())
    gan_loss(discriminator(image.detach()), False).backward()
    assert all(parameter.grad is not None for parameter in discriminator.heads.parameters())
