"""Check native mid-timestep prediction, input/parameter gradients and three updates."""

import os

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence

from tests.training import test_flux, test_flux2, test_flux3, test_moss_image
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux.conditioning import FluxTextFeatures, image_ids, pack_latents, unpack_latents
from unitunner.models.flux2.conditioning import build_image_inputs, pack_text
from unitunner.models.flux3.conditioning import Flux3Sample, image_tokens, pack_samples, text_tokens
from unitunner.training.session import TrainingSession


def native_prediction(family, model, low, caption, sigma):
    """Evaluate the accepted native forward directly, retaining original time conversion."""
    times = torch.full((low.shape[0],), sigma * 1000, device=low.device)
    if family == "flux":
        velocity = model(
            pack_latents(low),
            image_ids(low.shape[0], low.shape[2] // 2, low.shape[3] // 2, device=low.device),
            caption.captions,
            low.new_zeros((*caption.captions.shape[:2], 3)),
            times / 1000,
            caption.pooled,
            low.new_full((low.shape[0],), 1.0),
        )
        velocity = unpack_latents(velocity, low.shape[2], low.shape[3])
    elif family == "flux2":
        rows = []
        for index, image in enumerate(low):
            packed = build_image_inputs(image, ())
            text, ids = pack_text(caption.features[index, : caption.lengths[index]])
            guidance = low.new_ones((1,)) if model.config.use_guidance_embed else None
            prediction = model(
                packed.tokens,
                packed.ids,
                times[index : index + 1] / 1000,
                text[None],
                ids[None],
                guidance,
                target_count=packed.target_count,
            )
            rows.append(prediction[0].T.reshape_as(image))
        velocity = torch.stack(rows)
    elif family == "flux3":
        packed = pack_samples(
            [
                Flux3Sample(image_tokens(image), text_tokens(caption.features[index, : caption.lengths[index]]), None)
                for index, image in enumerate(low)
            ],
            times / 1000,
        )
        velocity = model(packed, times / 1000 if model.config.variant == "distilled" else None)
        velocity = velocity.reshape(low.shape[0], low.shape[2], low.shape[3], low.shape[1]).permute(0, 3, 1, 2)
    else:
        velocity = model(low, times, caption.features, caption.mask)
    return low - sigma * velocity


@pytest.mark.parametrize("mode", ["lora", "full"])
@pytest.mark.parametrize(
    "family,variant",
    [
        ("flux", "dev"),
        ("flux2", "klein-4b"),
        ("flux2", "dev"),
        ("flux3", "base"),
        ("flux3", "distilled"),
        ("moss-image", "base"),
    ],
)
def test_native_restoration_gradients_and_three_updates(family, variant, mode):
    helper = {"flux": test_flux, "flux2": test_flux2, "flux3": test_flux3, "moss-image": test_moss_image}[family]
    device = os.environ.get("UNITUNNER_TEST_DEVICE", "cpu")
    torch.manual_seed(237)
    with TrainingSession() as actual_session, TrainingSession() as expected_session, sdpa_kernel(SDPBackend.MATH):
        bindings, modules = [], []
        for session in (actual_session, expected_session):
            if family in ("flux", "flux2", "flux3"):
                binding, owned, _ = helper.build(
                    session, "lora", variant="base" if family == "flux2" and variant == "klein-4b" else variant
                )
            else:
                binding, owned, _ = helper.build(session, "lora")
            if mode == "full":
                session.close()
                binding.model.requires_grad_(True)
                owned = {"model": binding.model}
            for module in owned.values():
                module.to(device)
            bindings.append(binding)
            modules.append(owned)
        for role, module in modules[0].items():
            modules[1][role].load_state_dict(module.state_dict(), strict=True)
        parameters = [
            {
                role + "." + name: value
                for role, module in group.items()
                for name, value in module.named_parameters()
                if value.requires_grad
            }
            for group in modules
        ]
        optimizers = [torch.optim.AdamW(group.values(), lr=0.0003, eps=1e-8, weight_decay=0.01) for group in parameters]
        request = helper.request()
        low = request.latents.to(device)
        if family == "flux":
            caption = FluxTextFeatures(request.captions.to(device), request.pooled.to(device))
        elif family in ("flux2", "flux3"):
            lengths = tuple(len(row) for row in request.captions)
            caption = Caption(pad_sequence(request.captions, batch_first=True).to(device), lengths=lengths)
        else:
            caption = Caption(request.captions.to(device), request.caption_mask.to(device))
        for _ in range(3):
            inputs = [low.detach().clone().requires_grad_() for _ in range(2)]
            predictions = [
                bindings[0].restore(inputs[0], caption, sigma=0.244),
                native_prediction(family, bindings[1].model, inputs[1], caption, 0.244),
            ]
            torch.testing.assert_close(predictions[0], predictions[1], atol=1e-5, rtol=1e-4)
            for prediction, optimizer in zip(predictions, optimizers, strict=True):
                optimizer.zero_grad(set_to_none=True)
                prediction.square().mean().backward()
            torch.testing.assert_close(inputs[0].grad, inputs[1].grad, atol=1e-5, rtol=1e-4)
            assert parameters[0].keys() == parameters[1].keys()
            for name, parameter in parameters[0].items():
                reference = parameters[1][name]
                assert (parameter.grad is None) == (reference.grad is None), name
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all(), name
                    torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-5, rtol=1e-4, msg=name)
            for optimizer in optimizers:
                optimizer.step()
            for name, parameter in parameters[0].items():
                torch.testing.assert_close(parameter, parameters[1][name], atol=1e-5, rtol=1e-4, msg=name)
