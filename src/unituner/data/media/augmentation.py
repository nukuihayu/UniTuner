"""Choose reproducible sample-level spatial augmentation without consuming model RNG."""

import hashlib
import math

from pydantic import JsonValue


def sample_horizontal_flip(probability: float, seed: int, item_key: str) -> bool:
    """Share one cache-time decision across all target and condition pixels of a sample."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Horizontal flip probability must be finite and in [0,1]")
    digest = hashlib.sha256(f"horizontal-flip-v1\0{seed}\0{item_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") < probability * 2**64


def flip_identity(probability: float, seed: int) -> dict[str, JsonValue]:
    """Keep unaugmented cache identities stable; pin the policy and seed when enabled."""
    return (
        {"horizontal_flip": {"probability": probability, "seed": seed, "policy": "sample-sha256-v1"}}
        if probability
        else {}
    )
