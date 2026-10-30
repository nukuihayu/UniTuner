"""Provide optional local FLUX T5 final states and normalized CLIP pooled text."""

import re
from pathlib import Path

import ftfy
import torch
from torch import Tensor
from transformers import AutoConfig, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5Tokenizer

from unituner.artifacts.tensors import iter_tensors
from unituner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig
from unituner.models.components.text_artifacts import text_config_path
from unituner.models.components.text_loading import load_text_state, prepare_text_model, text_dtype, text_weight_path
from unituner.models.flux.conditioning import FluxTextFeatures
from unituner.models.text_assets import tokenizer_directory

_WHITESPACE = re.compile(r"\s+")


def clean_clip_text(text: str) -> str:
    """Apply published CLIP ftfy repair, whitespace collapse and outer stripping."""
    return _WHITESPACE.sub(" ", ftfy.fix_text(text)).strip()


class FluxTextEncoder:
    """Own frozen local T5/CLIP text models and their tokenizers.

    T5 uses raw prompts, right padding to 512 and no attention mask. CLIP uses
    repaired/collapsed text and right padding to 77; return pooler_output, not
    projected CLIP embeddings. The caller owns residency; no network access or
    preview/cache policy occurs here. FP32 numerical acceptance is independent
    of lower-precision storage support.
    """

    def __init__(
        self,
        t5_tokenizer: T5Tokenizer,
        t5: T5EncoderModel,
        clip_tokenizer: CLIPTokenizer,
        clip: CLIPTextModel,
        *,
        max_length: int = 512,
    ) -> None:
        if type(max_length) is not int or not 1 <= max_length <= 512:
            raise ValueError("FLUX T5 max_length must be between 1 and 512")
        if t5_tokenizer.padding_side != "right" or clip_tokenizer.padding_side != "right":
            raise ValueError("FLUX text tokenizers require right padding")
        if clip.config.max_position_embeddings < 77:
            raise ValueError("FLUX CLIP requires at least 77 position embeddings")
        self.t5_tokenizer = t5_tokenizer
        self.t5 = t5
        self.clip_tokenizer = clip_tokenizer
        self.clip = clip
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> FluxTextFeatures:
        """Return finite CPU native final/pooled features without consuming RNG.

        Empty strings are legal; an empty batch is rejected. Frozen eval models
        are required so stochastic dropout cannot silently change cache contents.
        Ambient autocast is disabled separately on each model device. No module
        mode or weight placement is changed, including after exceptions.
        """
        if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("FLUX text encoding requires a nonempty batch of strings")
        if any(module.training for model in (self.t5, self.clip) for module in model.modules()):
            raise ValueError("FLUX text encoding requires eval models")
        t5_ids = self.t5_tokenizer(
            prompts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )["input_ids"]
        clip_ids = self.clip_tokenizer(
            [clean_clip_text(prompt) for prompt in prompts],
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        if not isinstance(t5_ids, Tensor) or not isinstance(clip_ids, Tensor):
            raise TypeError("FLUX tokenizers must return tensor IDs")
        t5_device, clip_device = next(self.t5.parameters()).device, next(self.clip.parameters()).device
        with torch.autocast(t5_device.type, enabled=False):
            features = self.t5(
                input_ids=t5_ids.to(t5_device), attention_mask=None, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        with torch.autocast(clip_device.type, enabled=False):
            pooled = self.clip(input_ids=clip_ids.to(clip_device), attention_mask=None, return_dict=True).pooler_output
        if not isinstance(features, Tensor) or not isinstance(pooled, Tensor):
            raise TypeError("FLUX text models must return tensor features")
        if (
            features.shape[:2] != t5_ids.shape
            or pooled.shape[0] != len(prompts)
            or not torch.isfinite(features).all()
            or not torch.isfinite(pooled).all()
        ):
            raise ValueError("FLUX text features must be finite with matching batch/token geometry")
        return FluxTextFeatures(features.detach().cpu(), pooled.detach().cpu())


def _weight_path(directory: Path) -> Path:
    """Resolve one local safetensors file or a standard first numbered shard."""
    return text_weight_path(directory)


def _load_model_weights(
    model: T5EncoderModel | CLIPTextModel, directory: Path, *, device: torch.device, dtype: torch.dtype
) -> None:
    """Preflight exact local language tensors, then assign an owned meta model.

    T5 shared/encoder embeddings are an explicit tied pair: accept one published
    tensor or equal duplicate values, and retie after assignment. CLIP supports
    uniform historical text_model.* keys or native Transformers 5 keys, rejecting
    mixed layouts. No missing/extra key is ignored or randomly initialized.
    """
    tensors = dict(iter_tensors(_weight_path(directory)))
    if isinstance(model, T5EncoderModel):
        shared, encoder = tensors.get("shared.weight"), tensors.get("encoder.embed_tokens.weight")
        if (
            shared is not None
            and encoder is not None
            and (shared.dtype != encoder.dtype or not torch.equal(shared, encoder))
        ):
            raise ValueError("FLUX T5 tied embedding values conflict")
        if shared is None and encoder is not None:
            tensors["shared.weight"] = encoder
        if encoder is None and shared is not None:
            tensors["encoder.embed_tokens.weight"] = shared
    elif any(key.startswith("text_model.") for key in tensors):
        if not all(key.startswith("text_model.") for key in tensors):
            raise ValueError("FLUX CLIP artifact mixes native and text_model prefixes")
        tensors = {key.removeprefix("text_model."): value for key, value in tensors.items()}
    load_text_state(model, tensors, dtype=dtype, label="FLUX text")
    if isinstance(model, T5EncoderModel):
        model.tie_weights()
    else:
        model.embeddings.register_buffer(
            "position_ids",
            torch.arange(model.config.max_position_embeddings, device="cpu").expand((1, -1)),
            persistent=False,
        )


def load_flux_text(
    t5_directory: Path,
    clip_directory: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    max_length: int = 512,
    text_config: TextEncodingConfig = DEFAULT_TEXT_CONFIG,
) -> FluxTextEncoder:
    """Load owned local T5 encoder and CLIP text-only weights.

    Accept model directories or single files with local/bundled topology.
    Resolve tokenizer assets independently. All keys, shapes and
    finite dtype conversion ranges are strict. Only T5's declared tied embedding
    aliases and CLIP's uniform historical prefix are normalized. Tokenizers and
    both model topologies are checked before large weight IO; no visual tower,
    remote code or runtime trainer fallback is supported.
    """
    dtype = text_dtype(text_config, dtype)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("FLUX text loading supports FP32, FP16 or BF16")
    if type(max_length) is not int or not 1 <= max_length <= 512:
        raise ValueError("FLUX T5 max_length must be between 1 and 512")
    t5_config = AutoConfig.from_pretrained(
        str(text_config_path(t5_directory, "flux")), local_files_only=True, trust_remote_code=False
    )
    clip_config = AutoConfig.from_pretrained(
        str(text_config_path(clip_directory, "flux-clip")), local_files_only=True, trust_remote_code=False
    )
    if not isinstance(t5_config, T5Config) or not isinstance(clip_config, CLIPTextConfig):
        raise ValueError("FLUX text requires local T5 and CLIP text-only configurations")
    t5_tokenizer = T5Tokenizer.from_pretrained(
        str(tokenizer_directory(text_config, "flux", t5_directory)), local_files_only=True
    )
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        str(tokenizer_directory(text_config, "flux", clip_directory, role="clip_tokenizer")), local_files_only=True
    )
    if (
        t5_tokenizer.padding_side != "right"
        or clip_tokenizer.padding_side != "right"
        or clip_config.max_position_embeddings < 77
    ):
        raise ValueError("FLUX text requires right padding and CLIP's 77 positions")
    t5_config._attn_implementation = "eager"
    clip_config._attn_implementation = "sdpa"
    with torch.device("meta"):
        t5 = T5EncoderModel(t5_config)
        clip = CLIPTextModel(clip_config)
    _load_model_weights(t5, t5_directory, device=device, dtype=dtype)
    _load_model_weights(clip, clip_directory, device=device, dtype=dtype)
    prepare_text_model(t5, text_config, family="t5", device=device)
    prepare_text_model(clip, text_config, family="clip", device=device)
    return FluxTextEncoder(t5_tokenizer, t5, clip_tokenizer, clip, max_length=max_length)
