# Released text encoder configurations

These JSON files contain topology only, no weights, tokenizer assets or executable code.
Retrieved on 2026-10-21 from these upstream configuration URLs:

- qwen3-4b.json: https://huggingface.co/Qwen/Qwen3-4B/resolve/main/config.json
- qwen3-8b.json: https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json
- t5-xxl.json: https://huggingface.co/google/t5-v1_1-xxl/resolve/main/config.json
- clip-l.json: https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json (text_config only, model_type set to clip_text_model)
- mistral3.json: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/resolve/main/config.json
- qwen2-5-vl.json: https://huggingface.co/Qwen/Qwen-Image/resolve/main/text_encoder/config.json
- qwen3-vl-4b.json: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/main/config.json
- qwen-image-2-1.json: https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/main/text_encoder/config.json

These vendored snapshots are used only for bare single-file artifacts. Local sidecars override them, and directory loading requires its own config.json. Updating a snapshot changes single-file cache identity.
