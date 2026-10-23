"""Check native FLUX3 language-only artifacts, retained padding and strict loading failures."""

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2Tokenizer, Qwen2VLImageProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from tests.support.text_assets import local_text_config
from unitunner.models.flux3.text.encoder import load_flux3_text


@pytest.mark.parametrize("single_file", [False, True])
def test_flux3_text_only_directory_preserves_padding_rng_and_rejects_invalid_language_weights(tmp_path, single_file):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vocabulary = {value: index for index, value in enumerate(sorted(ByteLevel.alphabet()))}
        vocabulary.update({"<|endoftext|>": 256, "<|im_start|>": 257, "<|im_end|>": 258})
        tokenizer = Qwen2Tokenizer(vocab=vocabulary, merges=[], pad_token="<|endoftext|>")
        template = "{{ '<|im_start|>user\\n' + messages[0]['content'] + '<|im_end|>\\n<|im_start|>assistant\\n' }}"
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(),
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(),
            chat_template=template,
        )
        processor.save_pretrained(tmp_path)
        config = Qwen3VLTextConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=259,
            rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [1, 1, 2]},
        )
        Qwen3VLTextModel(config).save_pretrained(tmp_path)
        rng = torch.get_rng_state().clone()
        encoder = load_flux3_text(
            (tmp_path / "model.safetensors" if single_file else tmp_path),
            text_config=local_text_config(tmp_path),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        rows = encoder.encode(["", "caption " * 20])
        assert rows[0].values.shape == (80, 128)
        assert rows[1].values.shape[0] > 80 and rows[1].values.shape[0] % 80 == 0
        assert rows[0].values.dtype == torch.bfloat16
        assert torch.count_nonzero(rows[0].values[-1]) > 0
        assert torch.equal(rows[0].ids[:, 3], torch.arange(80))
        assert torch.equal(torch.get_rng_state(), rng)
        assert not any(p.requires_grad for p in encoder.model.parameters())
        assert not encoder.model.training
        assert not any("visual" in name for name, _ in encoder.model.named_modules())
        caption = encoder.encode_prompt("")
        assert caption.mask is None and torch.equal(caption.features[0], rows[0].values)
        values = load_file(tmp_path / "model.safetensors")
        for bad in (
            {k: v for k, v in values.items() if k != "embed_tokens.weight"},
            {**values, "embed_tokens.weight": torch.full_like(values["embed_tokens.weight"], float("nan"))},
            {**values, "unexpected.weight": torch.zeros(1)},
        ):
            save_file(bad, tmp_path / "model.safetensors")
            with pytest.raises(ValueError, match="FLUX3 text tensor"):
                load_flux3_text(
                    tmp_path, text_config=local_text_config(tmp_path), device=torch.device("cpu"), dtype=torch.float32
                )
        with pytest.raises(ValueError, match="nonempty"):
            encoder.encode([])
    finally:
        torch.set_num_threads(previous)
