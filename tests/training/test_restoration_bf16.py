"""Exercise BF16 restoration graphs, feature losses and shared optimizer updates."""

import pytest
import torch
from torch import nn

from tests.execution.test_native_strategies import _case
from tests.training.test_qwen_image import build as build_qwen
from unitunner.config.loading import load_training_config
from unitunner.models.components.conditioning import Caption
from unitunner.models.features.convnext import DinoConvNeXt
from unitunner.models.features.vgg import Vgg19Features
from unitunner.models.flux.conditioning import FluxTextFeatures
from unitunner.models.flux.config.vae import FluxVAEConfig
from unitunner.models.flux.loading import save_flux_vae
from unitunner.models.flux.vae import FluxVAE
from unitunner.models.flux2.config.vae import Flux2VAEConfig
from unitunner.models.flux2.loading import save_flux2_vae
from unitunner.models.flux2.vae import Flux2VAE
from unitunner.models.krea2.conditioning import Krea2TextFeatures
from unitunner.models.qwen_image.config.vae import QwenImageVAEConfig
from unitunner.models.qwen_image.loading import save_qwen_image_vae
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unitunner.models.qwen_image_2_1.loading import save_qwen_image_21_vae
from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE
from unitunner.models.z_image.config.vae import ZImageVAEConfig
from unitunner.models.z_image.loading import save_z_image_vae
from unitunner.models.z_image.vae import ZImageVAE
from unitunner.training.engine import update_phase
from unitunner.training.losses.adversarial import frozen_discriminator, gan_loss
from unitunner.training.losses.frequency_distribution import FrequencyDistributionLoss
from unitunner.training.losses.perceptual import DinoPerceptualLoss
from unitunner.training.methods.discriminator import MultilevelDiscriminator
from unitunner.training.methods.restoration import RestorationMethod
from unitunner.training.session import TrainingSession
from unitunner.training.state import RunState

FAMILIES = ["z-image", "flux", "flux2", "flux3", "moss-image", "qwen-image", "qwen-image-2-1", "krea2"]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _vae(family):
    if family == "z-image":
        return ZImageVAE(ZImageVAEConfig(block_channels=(4, 8), norm_groups=2, latent_channels=2)), save_z_image_vae
    if family == "flux":
        return FluxVAE(
            FluxVAEConfig(base_channels=4, channel_multipliers=(1, 2), norm_groups=2, latent_channels=2)
        ), save_flux_vae
    if family in ("flux2", "flux3", "moss-image"):
        return Flux2VAE(
            Flux2VAEConfig(base_channels=4, channel_multipliers=(1, 2), norm_groups=2, latent_channels=2)
        ), save_flux2_vae
    if family == "qwen-image-2-1":
        return QwenImage21VAE(
            QwenImage21VAEConfig(
                base_channels=2,
                decoder_base_channels=4,
                latent_channels=4,
                layers_per_block=1,
                latents_mean=(0.0,) * 4,
                latents_std=(1.0,) * 4,
            )
        ), save_qwen_image_21_vae
    channels = 16 if family == "krea2" else 2
    return QwenImageVAE(
        QwenImageVAEConfig(
            base_channels=4,
            latent_channels=channels,
            layers_per_block=1,
            latents_mean=(0.0,) * channels,
            latents_std=(1.0,) * channels,
        )
    ), save_qwen_image_vae


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
@pytest.mark.parametrize("recompute", ["none", "block"])
def test_all_image_restoration_bf16_updates(tmp_path, family, mode, device_name, precision, recompute):
    if device_name == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        pytest.skip("BF16-capable CUDA unavailable")
    device = torch.device(device_name)
    torch.manual_seed(818)
    vae, save = _vae(family)
    vae_path = tmp_path / "vae.safetensors"
    save(vae, vae_path)
    variant = {"flux": "dev", "qwen-image": "original", "krea2": "raw"}.get(family, "base")
    version = 'version = "klein-4b"' if family == "flux2" else ""
    path = tmp_path / "train.toml"
    path.write_text(f'''
[model]
family = "{family}"
variant = "{variant}"
{version}
weights = "unused.safetensors"
vae = "{vae_path.as_posix()}"
[dataset.groups.main]
kind = "restoration"
source = "hq"
low_quality_source = "lq"
cache_directory = "cache"
resolution = [16,16]
[method]
kind = "restoration"
[method.restoration]
sigma = 0.37
encoder_rank = 1
[losses]
lrr = 1.0
l1 = 0.5
dv3d = 0.0
gan = 0.0
fdl = 0.0
[method.wavelet]
wavelet = "haar"
weight = 0.5
levels = 2
[execution]
precision = "{precision}"
recompute = "{recompute}"
[train]
parameter_mode = "full"
''')
    config = load_training_config(path)
    with TrainingSession() as session:
        if family == "qwen-image":
            binding, modules, _ = build_qwen(session, mode, "original")
        else:
            binding, modules, _ = _case(family, session, mode)
        for module in modules.values():
            module.to(device)
        method = RestorationMethod(config, binding, device, session)
        if family == "flux":
            captions = FluxTextFeatures(torch.randn(1, 3, 12, device=device), torch.randn(1, 6, device=device))
        elif family == "krea2":
            captions = Krea2TextFeatures(torch.randn(1, 3, 3, 32, device=device), lengths=(3,))
        else:
            width = 8 if family == "z-image" else 6 if family == "qwen-image-2-1" else 12
            captions = Caption(torch.randn(1, 3, width, device=device), lengths=(3,))
        observed = []
        handles = []

        def record(module, args, output):
            if isinstance(output, torch.Tensor):
                observed.append(output.dtype)

        # Check actual VAE and transformer operation dtypes, not just config acceptance.
        for component in (method.vae, binding.model):
            layer = next(m for m in component.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)))
            handles.append(layer.register_forward_hook(record))
        low = torch.randn(1, 3, 16, 16, device=device).requires_grad_()
        high = torch.randn_like(low).requires_grad_()
        parameters = [p for p in binding.model.parameters() if p.requires_grad]
        assert method.encoder is not None
        parameters += list(method.encoder.parameters())
        before = [p.detach().clone() for p in parameters]
        state = RunState()
        result = update_phase(
            method.steps(low, high, captions),
            torch.optim.AdamW(parameters, lr=1e-3),
            state,
            precision=precision,
            phase="generator",
        )
        for handle in handles:
            handle.remove()
        assert high.grad is None
        assert state.updates == 1 and torch.isfinite(result.mean_loss)
        assert (torch.bfloat16 if precision == "bf16" else torch.float16) in observed
        assert low.grad is not None and torch.isfinite(low.grad).all() and low.grad.abs().sum() > 0
        assert any(not torch.equal(old, p) for old, p in zip(before, parameters, strict=True))


@pytest.mark.parametrize("recompute", [False, True])
def test_bf16_feature_losses_and_discriminator_preserve_gradients(recompute):
    backbone = DinoConvNeXt((4, 8, 12, 16), (1, 1, 1, 1))
    backbone.recompute = recompute
    perception = DinoPerceptualLoss(backbone, bf16_compute=True)
    vgg = Vgg19Features((2, 2, 2, 2, 2))
    frequency = FrequencyDistributionLoss(vgg.feature_stages(), vgg.widths, 3, bf16_compute=True)
    discriminator = MultilevelDiscriminator(backbone, 512, diffaug=False, bf16_compute=True)
    observed = []
    handles = []
    for component in (backbone, frequency, discriminator.heads):
        layer = next(m for m in component.modules() if isinstance(m, nn.Conv2d))
        handles.append(layer.register_forward_hook(lambda m, args, out: observed.append(out.dtype)))
    generated = torch.randn(1, 3, 512, 512, requires_grad=True)
    target = torch.randn_like(generated).requires_grad_()
    with frozen_discriminator(discriminator):
        loss = perception(generated, target) + frequency(generated[:, :, :80, :80], target[:, :, :80, :80])
        loss = loss + gan_loss(discriminator(generated), True)
    assert torch.isfinite(loss) and loss.dtype == torch.float32
    loss.backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all() and generated.grad.abs().sum() > 0
    assert target.grad is None
    assert all(p.grad is None for p in backbone.parameters())
    assert all(p.grad is None for p in discriminator.heads.parameters())
    gan_loss(discriminator(generated.detach()), False).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in discriminator.heads.parameters())
    assert observed and set(observed) == {torch.bfloat16}
    for handle in handles:
        handle.remove()
