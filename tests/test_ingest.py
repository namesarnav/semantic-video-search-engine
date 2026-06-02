from __future__ import annotations

import pytest

from sv_engine.ingest import content_hash


def test_same_content_same_hash(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"identical bytes")
    b.write_bytes(b"identical bytes")

    # Different names, same content -- must be one video, not two.
    assert content_hash(a) == content_hash(b)


def test_modified_content_changes_hash(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"original")
    before = content_hash(path)

    path.write_bytes(b"modified")
    assert content_hash(path) != before


def test_hash_is_stable_across_calls(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"stable")
    assert content_hash(path) == content_hash(path)


def test_hash_spans_chunk_boundaries(tmp_path):
    """The hash reads in 1MB chunks; a difference past the first chunk must
    still register."""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    payload = bytearray(b"\x00" * (3 * 1024 * 1024))
    a.write_bytes(bytes(payload))
    payload[2 * 1024 * 1024] = 1
    b.write_bytes(bytes(payload))

    assert content_hash(a) != content_hash(b)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        content_hash(tmp_path / "nope.bin")
