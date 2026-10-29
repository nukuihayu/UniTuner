"""Verify native FLUX.2 language-only local assets, fixed padding and strict failures."""

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    PixtralImageProcessor,
    PixtralProcessor,
    Qwen2Tokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from tests.support.text_assets import local_text_config
from unitunner.models.flux2.text.encoder import load_flux2_text


@pytest.mark.parametrize("version", ["dev", "klein-4b"])
@pytest.mark.parametrize("single_file", [False, True])
def test_flux2_text_language_only_and_retained_padding(tmp_path, version, single_file):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vocabulary = {value: index for index, value in enumerate(sorted(ByteLevel.alphabet()))}
        vocabulary.update({"<|endoftext|>": 256, "<|im_start|>": 257, "<|im_end|>": 258})
        tokenizer = Qwen2Tokenizer(vocab=vocabulary, merges=[], pad_token="<|endoftext|>")
        if version == "dev":
            template = "{% for message in messages %}{{ message['role'] + ': ' }}{% for content in message['content'] %}{{ content['text'] }}{% endfor %}{{ '\\n' }}{% endfor %}"
            processor = PixtralProcessor(
                image_processor=PixtralImageProcessor(), tokenizer=tokenizer, chat_template=template
            )
            processor.save_pretrained(tmp_path)
            config = Mistral3Config(
                text_config={
                    "model_type": "mistral",
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 32,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "vocab_size": 259,
                },
                vision_config={
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "image_size": 8,
                    "patch_size": 2,
                },
                image_token_index=259,
            )
            full = Mistral3ForConditionalGeneration(config).eval().requires_grad_(False)
        else:
            tokenizer.chat_template = "{{ '<|im_start|>user\\n' + messages[0]['content'] + '<|im_end|>' }}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}{% if enable_thinking %}{{ '<think>' }}{% endif %}"
            tokenizer.save_pretrained(tmp_path)
            full = (
                Qwen3ForCausalLM(
                    Qwen3Config(
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=28,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=8,
                        vocab_size=259,
                    )
                )
                .eval()
                .requires_grad_(False)
            )
        full.save_pretrained(tmp_path)
        rng = torch.get_rng_state().clone()
        encoder = load_flux2_text(
            (tmp_path / "model.safetensors" if single_file else tmp_path),
            text_config=local_text_config(tmp_path),
            version=version,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert torch.equal(rng, torch.get_rng_state())
        assert not any("vision" in name or "lm_head" in name for name, _ in encoder.model.named_modules())
        assert not encoder.model.training and all(not p.requires_grad for p in encoder.model.parameters())
        assert all(not b.is_meta for b in encoder.model.buffers())
        prompts = ["", "caption [IMG] here", "word " * 200]
        caption = encoder.encode(prompts)
        assert caption.features.shape == (3, 512, 48) and caption.mask is None
        assert torch.count_nonzero(caption.features[0, -1]) > 0
        assert torch.equal(encoder.encode_prompt(prompts[1]).features, caption.features[1:2])
        if version == "dev":
            assert torch.equal(encoder.encode(["caption  here"]).features, caption.features[1:2])
            assert encoder.tokenizer.padding_side == "left"
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert torch.equal(encoder.encode(prompts).features, caption.features)
        assert torch.equal(rng, torch.get_rng_state())
        values = load_file(tmp_path / "model.safetensors")
        source_prefix = "language_model.model." if version == "dev" else "model."
        language = {k.removeprefix(source_prefix): v for k, v in values.items() if k.startswith(source_prefix)}
        for prefix in ("", "model.", "model.language_model.") if version == "dev" else ("",):
            save_file({prefix + k: v for k, v in language.items()}, tmp_path / "model.safetensors")
            loaded = load_flux2_text(
                tmp_path,
                text_config=local_text_config(tmp_path),
                version=version,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            torch.testing.assert_close(loaded.encode(prompts).features, caption.features, atol=0, rtol=0)
        save_file(values, tmp_path / "model.safetensors")
        key = "language_model.model.embed_tokens.weight" if version == "dev" else "model.embed_tokens.weight"
        for bad in (
            {k: v for k, v in values.items() if k != key},
            {**values, key: torch.full_like(values[key], float("nan"))},
            {**values, "unexpected.weight": torch.zeros(1)},
        ):
            save_file(bad, tmp_path / "model.safetensors")
            with pytest.raises(ValueError, match=r"FLUX\.2 text"):
                load_flux2_text(
                    tmp_path,
                    text_config=local_text_config(tmp_path),
                    version=version,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
        with pytest.raises(ValueError, match="nonempty"):
            encoder.encode([])
    finally:
        torch.set_num_threads(previous)
