"""Check restoration formulas, pixel gradients and isolated optimizer phases."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from unituner.adapters.encoder import EncoderWeightLoRA
from unituner.config.restoration import RestorationLossWeights
from unituner.models.flux.vae import FluxVAE
from unituner.models.z_image.config.model import ZImageConfig
from unituner.models.z_image.config.vae import ZImageVAEConfig
from unituner.models.z_image.transformer import ZImageTransformer
from unituner.models.z_image.vae import ZImageVAE
from unituner.training.contracts import RunState, StepResult
from unituner.training.engine import update_phase
from unituner.training.integrations.z_image.binding import ZImageBinding
from unituner.training.losses.adversarial import frozen_discriminator, gan_loss
from unituner.training.losses.histogram import _normalized_lab, histogram_loss
from unituner.training.losses.restoration import generator_loss
from unituner.training.methods.mid_timestep import flow_target, one_step


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_histogram_sparse_matches_dense_values_and_gradients():
    torch.manual_seed(141)
    generated = (torch.rand(2, 3, 7, 9) * 1.8 - 0.9).requires_grad_()
    target = torch.rand_like(generated).requires_grad_()
    bins = 8
    centers = torch.linspace(0, 1, bins)
    histograms = []
    for image in (generated, target.detach()):
        lab = _normalized_lab(image).flatten(2)
        dense = (1 - (lab[..., None] - centers).abs() * (bins - 1)).clamp_min(0).sum(2)
        histograms.append(dense / dense.sum(-1, keepdim=True))
    expected = (histograms[0].cumsum(-1) - histograms[1].cumsum(-1)).abs().mean()
    actual = histogram_loss(generated, target, bins)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual_grad = torch.autograd.grad(actual, (generated, target), allow_unused=True, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, generated)[0]
    torch.testing.assert_close(actual_grad[0], expected_grad, atol=1e-6, rtol=1e-5)
    assert actual_grad[1] is None


def test_gan_freeze_restores_flags_and_preserves_image_gradient():
    discriminator = nn.Linear(3, 2)
    discriminator.bias.requires_grad_(False)
    image = torch.randn(4, 3, requires_grad=True)
    with frozen_discriminator(discriminator):
        logits = discriminator(image)
        loss = gan_loss((logits, logits[:, :1]), True)
    expected = (F.softplus(logits) - 0.8 * logits).mean() + (F.softplus(logits[:, :1]) - 0.8 * logits[:, :1]).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert image.grad is not None and image.grad.norm() > 0
    assert discriminator.weight.grad is None and discriminator.bias.grad is None
    assert discriminator.weight.requires_grad and not discriminator.bias.requires_grad
    with pytest.raises(RuntimeError), frozen_discriminator(discriminator):
        raise RuntimeError("failure")
    assert discriminator.weight.requires_grad and not discriminator.bias.requires_grad
    fake = discriminator(image.detach())
    gan_loss((fake,), False).backward()
    assert discriminator.weight.grad is not None


def test_z_image_restoration_pixel_update_matches_direct_native_step():
    torch.manual_seed(17)
    model = ZImageTransformer(
        ZImageConfig(
            in_channels=2,
            dim=24,
            num_layers=1,
            num_refiner_layers=1,
            num_heads=4,
            num_kv_heads=2,
            caption_dim=8,
            timestep_hidden_dim=32,
            axes_dims=(2, 2, 2),
            axes_lengths=(128, 32, 32),
        )
    )
    vae = (
        ZImageVAE(
            ZImageVAEConfig(
                block_channels=(4, 8),
                norm_groups=2,
                layers_per_block=1,
                latent_channels=2,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    low = torch.randn(1, 3, 8, 8).requires_grad_()
    high = torch.randn_like(low).requires_grad_()
    low_latents = vae.normalize(vae.encode(low).mean)
    clean = vae.normalize(vae.encode(high).mean)
    noise = torch.randn_like(clean)
    sigma = 0.244
    captions = torch.randn(1, 4, 8)
    binding = ZImageBinding(model)
    restored = binding.restore(low_latents, captions, sigma=sigma)
    direct = (
        low_latents.float()
        + sigma * model(low_latents.unsqueeze(2), torch.tensor([1 - sigma]), captions).squeeze(2).float()
    )
    torch.testing.assert_close(restored, direct, atol=0, rtol=0)
    generated = vae.decode(vae.denormalize(restored))
    weights = RestorationLossWeights(l1=0.5, lrr=5.0, dv3d=0.0, gan=0.0, fdl=0.0, histogram=0.1)
    noisy_high = flow_target(clean, noise, sigma)
    result = generator_loss(generated, high, low_latents, noisy_high, weights, histogram_bins=8)
    expected = (
        0.5 * F.l1_loss(generated, high.detach())
        + 5 * F.mse_loss(low_latents, ((1 - sigma) * clean + sigma * noise).detach())
        + 0.1 * histogram_loss(generated, high, 8)
    )
    torch.testing.assert_close(result.total, expected, atol=0, rtol=0)
    assert all(not value.requires_grad for value in result.metrics.values())
    image_grad, target_grad = torch.autograd.grad(result.total, (low, high), retain_graph=True, allow_unused=True)
    assert image_grad is not None and image_grad.norm() > 0
    assert target_grad is None
    before = model.get_parameter("layers.0.attention.to_q.weight").detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    state = RunState()
    update_phase([StepResult(result.total, 1)], optimizer, state)
    assert state.updates == 1
    assert not torch.equal(before, model.get_parameter("layers.0.attention.to_q.weight"))
    assert all(parameter.grad is None for parameter in vae.parameters())


@pytest.mark.parametrize("sigma", [0.0, 1.0, float("nan"), float("inf")])
def test_restoration_rejects_invalid_time(sigma):
    with pytest.raises(ValueError):
        one_step(torch.ones(1, 2), torch.ones(1, 2), sigma, flow_sign=-1)


def test_generator_temporaries_are_released_before_discriminator_forward():
    import weakref
    from types import SimpleNamespace

    from unituner.training.methods.restoration import RestorationMethod

    method = object.__new__(RestorationMethod)
    method._first_step = False
    method.compute_precision = "fp32"
    method.weights = RestorationLossWeights(lrr=3, dv3d=5, l1=0.5, gan=0.5, fdl=0, phase=4)
    method.encoder = SimpleNamespace(unadapted=torch.no_grad)
    method.state = nn.Module()
    method.state.register_buffer("sigma", torch.tensor(0.244, dtype=torch.float64))
    method.histogram_bins, method.phase_period, method.phase_cycles, method.phase_epsilon = 32, 16, 8, 0.001
    method.posterior = "mean"
    method.wavelet = method.frequency = None
    scale = nn.Parameter(torch.tensor(0.8))
    refs = {}

    def track(name, value):
        refs[name] = weakref.ref(value)
        return value

    method.encode = lambda pixels, **kwargs: track("low" if pixels.requires_grad else "high", pixels * scale)
    method.restore = lambda latent, captions: track("restored", latent * scale)
    method.decode = lambda latent: latent + 0
    method.perceptual = lambda generated, high: F.mse_loss(generated, high)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Conv2d(3, 1, 1)
            self.calls = 0

        def forward(self, pixels):
            self.calls += 1
            if self.calls > 1:
                assert all(reference() is None for reference in refs.values())
            return (self.head(pixels),)

    method.discriminator = Discriminator()
    low = torch.randn(1, 3, 128, 128, requires_grad=True)
    high = torch.randn_like(low)
    steps = method.steps(low, high, None)
    generator = next(steps)
    generator.loss.backward()
    assert scale.grad is not None and scale.grad.abs() > 0
    assert low.grad is not None and torch.isfinite(low.grad).all()
    assert all(p.grad is None for p in method.discriminator.parameters())
    discriminator = next(steps)
    discriminator.loss.backward()
    assert all(p.grad is not None for p in method.discriminator.parameters())
    assert next(steps, None) is None


@pytest.mark.parametrize("family", ["z-image", "flux"])
@pytest.mark.parametrize("channels_last", [False, True])
def test_sampled_restoration_matches_trainer_draw_order_and_gradients(tmp_path, family, channels_last):
    """Match sampled HQ -> LRR noise -> sampled LQ, including noncontiguous moments."""
    from tests.training.test_flux import build as build_flux
    from unituner.config.loading import load_training_config
    from unituner.models.components.conditioning import Caption
    from unituner.models.flux.conditioning import FluxTextFeatures
    from unituner.models.flux.config.vae import FluxVAEConfig
    from unituner.models.flux.loading import save_flux_vae
    from unituner.models.z_image.loading import save_z_image_vae
    from unituner.training.methods.restoration import RestorationMethod
    from unituner.training.session import TrainingSession

    torch.manual_seed(71)
    vae_path = tmp_path / "vae.safetensors"
    if family == "flux":
        vae = FluxVAE(
            FluxVAEConfig(
                base_channels=4, channel_multipliers=(1, 2), layers_per_block=1, latent_channels=2, norm_groups=2
            )
        )
        save_flux_vae(vae, vae_path)
        captions = FluxTextFeatures(torch.randn(2, 3, 12), torch.randn(2, 6))
    else:
        vae = ZImageVAE(ZImageVAEConfig(block_channels=(4, 8), norm_groups=2, layers_per_block=1, latent_channels=2))
        save_z_image_vae(vae, vae_path)
        captions = Caption(torch.randn(2, 3, 8), lengths=(3, 3))
    config_path = tmp_path / "train.toml"
    config_path.write_text(f'''
[model]
family = "{family}"
variant = "{"dev" if family == "flux" else "base"}"
weights = "unused.safetensors"
vae = "{vae_path.as_posix()}"
[dataset.groups.main]
kind = "restoration"
source = "hq"
low_quality_source = "lq"
cache_directory = "cache"
resolution = [16,16]
[train]
parameter_mode = "full"
[method]
kind = "restoration"
[method.restoration]
sigma = 0.37
encoder_rank = 2
[losses]
lrr = 5.0
l1 = 0.5
dv3d = 0.0
gan = 0.0
fdl = 0.0
''')
    config = load_training_config(config_path)
    low = torch.rand(2, 3, 16, 16) * 2 - 1
    high = torch.rand_like(low) * 2 - 1
    if channels_last:
        low = low.to(memory_format=torch.channels_last)
        high = high.to(memory_format=torch.channels_last)
    low.requires_grad_()
    with TrainingSession() as session:
        if family == "flux":
            binding, _, _ = build_flux(session, "full")
        else:
            binding = ZImageBinding(
                ZImageTransformer(
                    ZImageConfig(
                        in_channels=2,
                        dim=24,
                        num_layers=1,
                        num_refiner_layers=1,
                        num_heads=4,
                        num_kv_heads=2,
                        caption_dim=8,
                        timestep_hidden_dim=32,
                        axes_dims=(2, 2, 2),
                        axes_lengths=(128, 32, 32),
                    )
                )
            )
        method = RestorationMethod(config, binding, torch.device("cpu"), session)
        assert method.posterior == "sample"
        with torch.no_grad():
            assert method.encoder is not None
            for branch in method.encoder.layers:
                assert isinstance(branch, EncoderWeightLoRA)
                branch.up.normal_(std=0.01)
        assert method.encoder is not None
        parameters = (low, *binding.model.parameters(), *method.encoder.parameters())
        torch.manual_seed(19)
        actual = next(method.steps(low, high, captions)).loss
        actual_rng = torch.get_rng_state()
        actual_gradients = torch.autograd.grad(actual, parameters, allow_unused=True)

        def trainer_encode(image):
            vae = method.vae
            assert isinstance(vae, (FluxVAE, ZImageVAE))
            moments = vae.posterior(image) if isinstance(vae, FluxVAE) else vae.encode(image)
            raw = moments.mean + torch.exp(0.5 * moments.log_variance.clamp(-30, 20)) * torch.randn(moments.mean.shape)
            return (raw - vae.config.shift_factor) * vae.config.scaling_factor

        torch.manual_seed(19)
        assert method.encoder is not None
        with method.encoder.unadapted():
            clean = trainer_encode(high)
        noise = torch.randn_like(clean)
        latent = trainer_encode(low)
        predicted = method.restore(latent, captions)
        generated = method.decode(predicted)
        expected = 5 * F.mse_loss(latent, ((1 - 0.37) * clean + 0.37 * noise).detach())
        expected = expected + 0.5 * F.l1_loss(generated, high)
        expected_gradients = torch.autograd.grad(expected, parameters, allow_unused=True)
        assert torch.equal(actual_rng, torch.get_rng_state())
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
            assert (actual_gradient is None) == (expected_gradient is None)
            if actual_gradient is not None:
                torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-5)
