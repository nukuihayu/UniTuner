"""Stage cache publication and dispatch family-native production."""

import json
from pathlib import Path

from unitunner.config.cache import CacheConfig
from unitunner.config.text import TokenizerSourceConfig
from unitunner.data.cache.feature_encoding import RepaCacheWriter
from unitunner.data.cache.rebuild import fresh_cache_config
from unitunner.execution.runtime import runtime_scope
from unitunner.models.catalog import ModelFamily
from unitunner.models.text_assets import pin_tokenizer_sources
from unitunner.progress import status


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
    if config.model.family == ModelFamily.MINIMAX_H3:
        from unitunner.data.production.minimax_h3 import produce_minimax_h3

        return produce_minimax_h3(config, device=device, repa=repa, tokenizer_digests=tokenizer_digests)
    if config.model.family == ModelFamily.WAN:
        from unitunner.data.production.wan import produce_wan

        return produce_wan(config, device=device, repa=repa, tokenizer_digests=tokenizer_digests)
    from unitunner.data.production.images import produce_images

    return produce_images(config, device=device, repa=repa, tokenizer_digests=tokenizer_digests)
