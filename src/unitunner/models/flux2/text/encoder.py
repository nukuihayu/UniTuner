"""Provide optional local FLUX.2 Mistral3/Qwen3 language features and native prompt rules."""

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    PixtralProcessor,
    PreTrainedTokenizerBase,
)
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralModel, MistralRotaryEmbedding
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding

from unitunner.artifacts.tensors import iter_tensors
from unitunner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig
from unitunner.models.components.conditioning import Caption
from unitunner.models.components.text_artifacts import text_config_path
from unitunner.models.components.text_loading import load_text_state, prepare_text_model, text_dtype, text_weight_path
from unitunner.models.text_assets import tokenizer_directory

_SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\n"
    "attribution and actions without speculation."
)


class Flux2TextEncoder:
    """Own frozen language weights and tokenizer with native512-token padding.

    Dev concatenates Mistral layers10/20/30 with left padding and a system prompt.
    Klein concatenates Qwen3 layers9/18/27 with thinking disabled and a generation
    prompt. All512 hidden positions, including padding, reach the diffusion model.
    Multimodal vision towers and LM heads are neither constructed nor executed.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: MistralModel | Qwen3Model,
        *,
        chat_template: str,
        version: Literal["dev", "klein-4b", "klein-9b"],
    ) -> None:
        layers = (10, 20, 30) if isinstance(model, MistralModel) else (9, 18, 27)
        if model.config.num_hidden_layers < layers[-1]:
            raise ValueError("FLUX.2 text lacks required native hidden layers")
        if isinstance(model, MistralModel) and tokenizer.padding_side != "left":
            raise ValueError("FLUX.2 Mistral text requires left padding")
        self.version = version
        self.tokenizer = tokenizer
        self.model = model
        self.chat_template = chat_template
        self.output_layers = layers

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> Caption:
        """Return detached Bx512x3D language states on the model device in its dtype.

        Dev removes literal[IMG] placeholders before formatting its system/user
        conversation and disables generation prompts. Klein formats each prompt
        independently with thinking disabled before fixed-length tokenization.
        The text attention mask is used inside the language model only; returned
        diffusion captions have no mask. No RNG/mode or ambient autocast changes.
        """
        if not prompts:
            raise ValueError("FLUX.2 text encoding requires a nonempty prompt batch")
        if isinstance(self.model, MistralModel):
            conversations = [
                [
                    {"role": "system", "content": [{"type": "text", "text": _SYSTEM_MESSAGE}]},
                    {"role": "user", "content": [{"type": "text", "text": prompt.replace("[IMG]", "")}]},
                ]
                for prompt in prompts
            ]
            inputs = self.tokenizer.apply_chat_template(
                conversations,  # ty: ignore[invalid-argument-type] - Transformers omits multimodal text blocks from its stub.
                chat_template=self.chat_template,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            if not isinstance(inputs, BatchEncoding):
                raise TypeError("FLUX.2 Mistral tokenizer must return a BatchEncoding")
            ids, mask = inputs["input_ids"], inputs["attention_mask"]
        else:
            all_ids, all_masks = [], []
            for prompt in prompts:
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    chat_template=self.chat_template,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if not isinstance(text, str):
                    raise TypeError("FLUX.2 Qwen3 template must produce text")
                inputs = self.tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
                )
                token_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
                if not isinstance(token_ids, Tensor) or not isinstance(attention_mask, Tensor):
                    raise TypeError("FLUX.2 tokenizer must return tensor IDs/masks")
                all_ids.append(token_ids)
                all_masks.append(attention_mask)
            ids, mask = torch.cat(all_ids), torch.cat(all_masks)
        if (
            not isinstance(ids, Tensor)
            or not isinstance(mask, Tensor)
            or ids.shape != (len(prompts), 512)
            or mask.shape != ids.shape
        ):
            raise ValueError("FLUX.2 tokenizer must produce matching Bx512 IDs/masks")
        parameter = next(self.model.parameters())
        with torch.autocast(parameter.device.type, enabled=False):
            output = self.model(
                input_ids=ids.to(parameter.device),
                attention_mask=mask.to(parameter.device),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        states = output.hidden_states
        if states is None or len(states) <= self.output_layers[-1]:
            raise RuntimeError("FLUX.2 text encoder did not return required hidden layers")
        features = torch.stack([states[layer] for layer in self.output_layers], dim=1).permute(0, 2, 1, 3).flatten(2)
        return Caption(features.detach())

    def encode_prompt(self, prompt: str) -> Caption:
        """Return one CPU1LD caption, retaining every fixed-length padding token."""
        return Caption(self.encode([prompt]).features.cpu())


def load_flux2_text(
    directory: Path,
    *,
    version: Literal["dev", "klein-4b", "klein-9b"],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    text_config: TextEncodingConfig = DEFAULT_TEXT_CONFIG,
) -> Flux2TextEncoder:
    """Load owned local language-only Mistral3 or Qwen3 weights and prompt assets.

    Dev requires a Mistral3 conditional-generation directory and Pixtral processor;
    Klein accepts Qwen3Model or Qwen3ForCausalLM directories. Exact declared
    language namespaces, keys/shapes and finite conversion ranges are validated.
    Only known vision/projector/LM-head keys may be excluded. Meta construction
    avoids random allocations and the language rotary buffer is rebuilt in FP32.
    Support single/standard-sharded safetensors; no network or remote code.
    """
    dtype = text_dtype(text_config, dtype)
    if version not in ("dev", "klein-4b", "klein-9b") or dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("FLUX.2 text requires Dev/Klein version and FP32/BF16 dtype")
    source = AutoConfig.from_pretrained(
        str(text_config_path(directory, "flux2", version=version)), local_files_only=True, trust_remote_code=False
    )
    tokenizer: PreTrainedTokenizerBase
    model: MistralModel | Qwen3Model
    if version == "dev":
        if not isinstance(source, Mistral3Config) or not isinstance(source.text_config, MistralConfig):
            raise ValueError("FLUX.2 Dev text requires a local Mistral3 configuration")
        processor = AutoProcessor.from_pretrained(
            str(tokenizer_directory(text_config, "flux2", directory, version=version)),
            local_files_only=True,
            trust_remote_code=False,
        )
        if not isinstance(processor, PixtralProcessor) or not isinstance(processor.tokenizer, PreTrainedTokenizerBase):
            raise ValueError("FLUX.2 Dev requires a local Pixtral processor/tokenizer")
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        template = processor.chat_template
        config = source.text_config
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = MistralModel(config)
        prefix = "language_model.model."
        excluded = ("vision_tower.", "multi_modal_projector.", "language_model.lm_head.")
    else:
        if not isinstance(source, Qwen3Config):
            raise ValueError("FLUX.2 Klein text requires a local Qwen3 configuration")
        architectures = tuple(source.architectures or ())
        if architectures not in (("Qwen3Model",), ("Qwen3ForCausalLM",)):
            raise ValueError("FLUX.2 Klein text must declare Qwen3Model or Qwen3ForCausalLM architecture")
        loaded = AutoTokenizer.from_pretrained(
            str(tokenizer_directory(text_config, "flux2", directory, version=version)),
            local_files_only=True,
            trust_remote_code=False,
        )
        if not isinstance(loaded, PreTrainedTokenizerBase):
            raise ValueError("FLUX.2 Klein requires a local tokenizer")
        tokenizer = loaded
        template = tokenizer.chat_template
        source._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Qwen3Model(source)
        prefix = "model." if architectures == ("Qwen3ForCausalLM",) else ""
        excluded = ("lm_head.",) if prefix else ()
    if not isinstance(template, str) or not template:
        raise ValueError("FLUX.2 text requires one local chat template")
    if model.config.num_hidden_layers < (30 if version == "dev" else 27):
        raise ValueError("FLUX.2 text lacks required native hidden layers")
    path = text_weight_path(directory)
    tensors: dict[str, Tensor] = {}
    for original, tensor in iter_tensors(path):
        if original.startswith(excluded):
            continue
        if not original.startswith(prefix):
            raise ValueError(f"Unexpected FLUX.2 text weight namespace: {original}")
        key = original[len(prefix) :]
        if key in tensors:
            raise ValueError(f"Duplicate FLUX.2 text tensor: {original}")
        tensors[key] = tensor
    load_text_state(model, tensors, dtype=dtype, label="FLUX.2 text")
    with torch.device("cpu"):
        if isinstance(model, MistralModel):
            model.rotary_emb = MistralRotaryEmbedding(model.config)
        else:
            model.rotary_emb = Qwen3RotaryEmbedding(model.config)
    prepare_text_model(model, text_config, family="mistral" if version == "dev" else "qwen3", device=device)
    return Flux2TextEncoder(tokenizer, model, chat_template=template, version=version)
