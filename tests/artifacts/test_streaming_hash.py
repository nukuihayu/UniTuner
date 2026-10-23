"""Keep streamed dataset fingerprints compatible with canonical JSON identities."""

import math

import pytest

from unitunner.artifacts.hashing import canonical_sequence_sha256, canonical_sha256


@pytest.mark.parametrize("field", ("a", "entries", "末尾"))
@pytest.mark.parametrize("items", ([], [None, True, 1, -0.0, "雪"], [{"b": [2, 1], "a": "é"}] * 100))
def test_streamed_canonical_identity_matches_existing_format(field, items):
    context = {"z": "终", "degradation": {"scales": [2, 4, 6]}, "schema": "restoration-images-2"}
    assert canonical_sequence_sha256(iter(items), field=field, context=context) == canonical_sha256(
        {**context, field: items}
    )


def test_stream_rejects_duplicate_fields_and_nonfinite_items():
    with pytest.raises(ValueError, match="field"):
        canonical_sequence_sha256([], field="entries", context={"entries": []})
    with pytest.raises(ValueError, match="JSON"):
        canonical_sequence_sha256(iter([math.nan]), field="entries", context={})
