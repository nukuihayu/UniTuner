"""Stage cache publication and dispatch family-native production."""

import json
from pathlib import Path

from unituner.config.cache import CacheConfig
from unituner.config.text import TokenizerSourceConfig
from unituner.data.cache.feature_encoding import RepaCacheWriter
from unituner.data.cache.rebuild import fresh_cache_config
from unituner.execution.runtime import runtime_scope
from unituner.models.catalog import CACHE_PROVIDERS, IMAGE_FAMILIES, image_family_spec
from unituner.models.providers import load_provider
from unituner.models.text_assets import pin_tokenizer_sources
from unituner.progress import status


@runtime_scope
def run(config: CacheConfig, *, device: str) -> int:
    """Publish native caches and optional real encoder targets from identical target crops."""
    status(
        f"[cache] Starting {config.model.family} on {device}; batch_size={config.cache.batch_size}; "
        "mode=rebuild (existing records will not be reused)"
    )
    with fresh_cache_config(config) as staged:
        result = _produce(staged, device=device)
    status("[cache] Complete; all cache directories replaced successfully")
    print(json.dumps({"cached": result}), flush=True)
    return 0


def _produce(config: CacheConfig, *, device: str) -> int:
    """Resolve tokenizer assets, load encoders and populate unpublished directories."""
    status("[cache] Resolving tokenizer/processor assets")
    text_config = config.model.text
    h3 = config.model.minimax_h3
    if h3 is not None and h3.text_processor is not None and text_config.tokenizer.source is None:
        text_config = text_config.model_copy(
            update={"tokenizer": TokenizerSourceConfig(source=h3.text_processor, offline=True)}
        )
    if config.model.family in IMAGE_FAMILIES and image_family_spec(config.model.family).tokenizer_in_assets:
        if config.model.assets is None:
            raise ValueError(f"{config.model.family} cache requires model.assets with a local tokenizer directory")
        tokenizer = Path(config.model.assets) / "tokenizer"
        if not tokenizer.is_dir():
            raise ValueError(f"{config.model.family} cache requires a local model.assets/tokenizer directory")
        text_config = text_config.model_copy(
            update={"tokenizer": TokenizerSourceConfig(source=str(tokenizer), offline=True)}
        )
    pinned, tokenizer_digests = pin_tokenizer_sources(
        text_config,
        config.model.family,
        None if config.model.text_encoder is None else Path(config.model.text_encoder),
        version=config.model.version,
    )
    config = config.model_copy(update={"model": config.model.model_copy(update={"text": pinned})})
    repa = None if config.repa is None else RepaCacheWriter(config.repa)
    result = _run(config, device=device, repa=repa, tokenizer_digests=tokenizer_digests)
    if repa is not None:
        repa.finish()
    return result


def _run(config: CacheConfig, *, device: str, repa: RepaCacheWriter | None, tokenizer_digests: dict[str, str]) -> int:
    """Import only the selected native cache assembly layer."""
    provider = CACHE_PROVIDERS.get(config.model.family)
    if provider is None:
        raise ValueError(f"No cache producer for {config.model.family}")
    return load_provider(provider)(config, device=device, repa=repa, tokenizer_digests=tokenizer_digests)
