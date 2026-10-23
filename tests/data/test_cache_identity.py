"""Check source-content identity instead of basename-based cache collisions."""

from unitunner.artifacts.hashing import file_sha256
from unitunner.data.cache.schema import CacheIdentity


def test_same_basename_different_contents(tmp_path):
    a = tmp_path / "a" / "photo.png"
    b = tmp_path / "b" / "photo.png"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(b"first image")
    b.write_bytes(b"second image")

    def identity(path):
        return CacheIdentity(
            source_sha256=file_sha256(path), encoder_sha256="a" * 64, transform_sha256="b" * 64, layout="CHW"
        )

    assert identity(a).digest() != identity(b).digest()
    a.write_bytes(b.read_bytes())
    assert identity(a).digest() == identity(b).digest()
