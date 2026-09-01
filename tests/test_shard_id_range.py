"""Shard overlap uses real FAISS/SQL ids (start at 1), not 0-based ntotal."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.database import (  # noqa: E402
    BaseVDB,
    _handle_id_bounds,
    _shard_overlaps_range,
)


def test_overlap_empty_and_swap():
    assert _shard_overlaps_range(10, 9, 0, 100) == "empty"
    assert _shard_overlaps_range(1, 10, 20, 30) == "before"
    assert _shard_overlaps_range(40, 50, 20, 30) == "after"
    assert _shard_overlaps_range(1, 10, 0, 0) == "overlap"


def test_tds_chunk_boundary_does_not_include_next_shard():
    # Production chunk shards around TDS last id 126911.
    shards = [
        (1, 100000),
        (100001, 126831),
        (126832, 126911),
        (126912, 226911),
    ]
    hi = 126911
    rels = [_shard_overlaps_range(lo, h, 0, hi) for lo, h in shards]
    assert rels == ["overlap", "overlap", "overlap", "after"]


def test_tds_node_boundary_does_not_include_next_shard():
    shards = [
        (1, 1500000),
        (1500001, 1551894),
        (1551895, 1651894),
    ]
    hi = 1551894
    rels = [_shard_overlaps_range(lo, h, 0, hi) for lo, h in shards]
    assert rels == ["overlap", "overlap", "after"]


def test_inclusive_hi_still_keeps_shard_that_contains_hi():
    assert _shard_overlaps_range(126832, 126911, 0, 126911) == "overlap"
    assert _shard_overlaps_range(126912, 226911, 0, 126911) == "after"
    # id == hi is in this shard, not the next one.
    assert _shard_overlaps_range(1, 10, 0, 10) == "overlap"
    assert _shard_overlaps_range(11, 20, 0, 10) == "after"


def test_zero_based_ntotal_fallback_was_the_bug():
    """Old pin: id_start starts at 0, next shard after 126911 vectors
    is assumed to cover [126911, ...], which overlaps hi=126911."""
    n_tds = 126911
    old_next_lo = 0 + n_tds  # cumulative ntotal
    old_next_hi = old_next_lo + 100000 - 1
    assert _shard_overlaps_range(old_next_lo, old_next_hi, 0, 126911) == "overlap"

    new_next_lo = 1 + n_tds  # SQL ids 1..n_tds, next starts at n_tds+1
    new_next_hi = new_next_lo + 100000 - 1
    assert _shard_overlaps_range(new_next_lo, new_next_hi, 0, 126911) == "after"


class _FakeHandle:
    def __init__(self, ntotal, bounds=None):
        self.ntotal = ntotal
        self._bounds = bounds

    def id_bounds(self):
        return self._bounds


def test_handle_id_bounds_prefers_id_map():
    h = _FakeHandle(100000, bounds=(126912, 226911))
    assert _handle_id_bounds(h, fallback_lo=126911) == (126912, 226911, 100000)


def test_handle_id_bounds_fallback_is_one_based():
    h = _FakeHandle(80, bounds=None)
    # After 126831 vectors (ids 1..126831), next shard has 80 vectors.
    assert _handle_id_bounds(h, fallback_lo=126832) == (126832, 126911, 80)
    # fallback_lo=0 must not recreate the off-by-one.
    assert _handle_id_bounds(h, fallback_lo=0) == (1, 80, 80)


def _vec(i, dim=32):
    v = np.zeros(dim, dtype=np.float32)
    v[0] = float(i)
    return v.tolist()


def _build_two_shards(tmp: Path, n_first=10, n_second=10, dim=32) -> BaseVDB:
    vdb = BaseVDB(
        str(tmp),
        "chunk",
        dim,
        shard_max_vectors=n_first,
        index_type="flat_l2",
        index_quant="none",
    )
    first = [{"id": i, "embedding": _vec(i, dim)} for i in range(1, n_first + 1)]
    vdb.add(first)
    vdb.save()
    assert vdb.seal_and_rotate(force=True)
    second = [
        {"id": i, "embedding": _vec(i, dim)}
        for i in range(n_first + 1, n_first + n_second + 1)
    ]
    vdb.add(second)
    vdb.save()
    assert vdb.seal_and_rotate(force=True)
    return vdb


def test_pin_at_sql_id_boundary_does_not_load_next_shard():
    with tempfile.TemporaryDirectory() as td:
        vdb = _build_two_shards(Path(td), n_first=10, n_second=10)
        stats = vdb.pin_shards(max_vectors=10)
        assert stats["shards"] == ["0.vdb"]
        assert stats["ntotal"] == 10
        assert stats["min_vectors"] == 0
        assert stats["max_vectors"] == 10
        vdb.unpin_shards()

        stats = vdb.pin_shards(min_vectors=11, max_vectors=20)
        assert stats["shards"] == ["1.vdb"]
        assert stats["ntotal"] == 10


def test_search_at_sql_id_boundary_does_not_scan_next_shard():
    with tempfile.TemporaryDirectory() as td:
        vdb = _build_two_shards(Path(td), n_first=10, n_second=10)
        hits = vdb.search(_vec(10), topk=5, max_vectors=10)
        st = vdb.last_search_stats
        assert st["ntotal_scanned"] == 10
        assert st["skipped_tail"] is True
        assert st["max_vectors"] == 10
        ids = {h["id"] for h in hits}
        assert ids.isdisjoint(set(range(11, 21)))
        assert 10 in ids or ids  # nearest to id=10 should be in 1..10
        for h in hits:
            assert 1 <= h["id"] <= 10


if __name__ == "__main__":
    tests = [fn for name, fn in list(globals().items()) if name.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(tests)} passed")
