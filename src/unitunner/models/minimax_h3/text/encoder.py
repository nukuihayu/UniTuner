"""Extract native H3 layer50 pre-normalization features from a frozen Qwen3-VL owner."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import Qwen3VLModel, Qwen3VLProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from unitunner.models.minimax_h3.text.presentation import (
    TEXT_WIDTH,
    MiniMaxH3Presentation,
    build_token_tags,
    validate_text_rows,
)
from unitunner.models.minimax_h3.text.processor import process_presentation

TEXT_LAYERS = 50


@dataclass(frozen=True, slots=True)
class MiniMaxH3TextFeatures:
    """Own Lx5120 pre-norm features on the model device and CPU int64 modality tags."""

    hidden_states: Tensor
    token_tags: Tensor

    def __post_init__(self) -> None:
        validate_text_rows(self.hidden_states, self.token_tags)
        if self.token_tags.device.type != "cpu":
            raise ValueError("MiniMaxH3 text feature tags must reside on CPU")


class MiniMaxH3TextEncoder:
    """Own a frozen resident Qwen3-VL model truncated to50 pre-norm language layers.

    Vision uses the model-native deepstack branches. No vocabulary output head,
    generation cache, trainer or transfer installation belongs to this owner.
    """

    def __init__(self, processor: Qwen3VLProcessor, model: Qwen3VLModel) -> None:
        if (
            not isinstance(model.config.text_config, Qwen3VLTextConfig)
            or model.config.text_config.hidden_size != TEXT_WIDTH
        ):
            raise ValueError("MiniMaxH3 text encoder requires hidden width5120")
        if len(model.language_model.layers) != TEXT_LAYERS:
            raise ValueError("MiniMaxH3 text encoder requires exactly50 language layers")
        if not isinstance(model.language_model.norm, nn.Identity):
            raise ValueError("MiniMaxH3 text encoder requires layer50 pre-norm output")
        if model.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("MiniMaxH3 text encoder requires FP32/BF16/FP16 weights")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("MiniMaxH3 text encoder weights must be frozen")
        if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
            raise ValueError("MiniMaxH3 text encoder requires materialized weights and buffers")
        self.processor = processor
        self.model = model

    @torch.no_grad()
    def encode(self, presentation: MiniMaxH3Presentation) -> MiniMaxH3TextFeatures:
        """Encode one non-chat prompt with all vision rows retained and no padding.

        Use the embedding's resident device and the weights' dtype, disabling
        ambient autocast. Scoped eval mode suppresses dropout and is restored even
        on failure. Inputs, RNG, parameter identities and weights are unchanged;
        no forward hooks are installed and no hidden-state history is retained.
        """
        processed = process_presentation(self.processor, presentation)
        tags = build_token_tags(processed["input_ids"])
        device = self.model.language_model.embed_tokens.weight.device
        inputs = {key: value.to(device) for key, value in processed.items()}
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            with torch.autocast(device.type, enabled=False):
                output = self.model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
        finally:
            for module, training in modes:
                module.training = training
        hidden = output.last_hidden_state
        if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
            raise RuntimeError("MiniMaxH3 text encoder must return one BL5120 tensor")
        return MiniMaxH3TextFeatures(hidden[0], tags)
