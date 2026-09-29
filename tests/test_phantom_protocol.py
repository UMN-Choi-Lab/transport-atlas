"""Offline unit tests for the constraint-first phantom selection protocol.

Numpy-only, inline arrays (mirrors tests/test_authors.py style; no fixture
files).  The pure functions under test are the reference oracle for the GPU
path in scripts/07_phantom_eval.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from transport_atlas.process.phantom_protocol import (
    build_exclusion_columns,
    resolve_topk_buffer,
    select_constraint_first,
    select_pool_then_filter,
)


def _brute_force(sims_row, excl, k):
    """Naive oracle: sort the full row by (-sim, index), filter, truncate."""
    n = len(sims_row)
    order = sorted(range(n), key=lambda j: (-sims_row[j], j))
    excl = set(int(x) for x in excl)
    return [j for j in order if j not in excl][:k]


def test_constraint_first_matches_brute_force_oracle():
    # ~12-author fixture with hand-built exclusions and hand-set cosines.
    sims = np.array([0.99, 0.85, 0.85, 0.70, 0.65, 0.60, 0.55, 0.50,
                     0.45, 0.40, 0.35, 0.30])
    excl = np.array([0, 2, 5], dtype=np.int32)  # self=0 plus two near-train
    got = select_constraint_first(sims, excl, 5)
    assert got.tolist() == _brute_force(sims, excl, 5)
    # spot the hand-computed answer too: 1 (.85), 3 (.70), 4 (.65), 6 (.55), 7
    assert got.tolist() == [1, 3, 4, 6, 7]


def test_excluded_top_author_is_replaced_by_next_eligible():
    sims = np.array([0.1, 0.9, 0.8, 0.7, 0.6])
    # top author (idx 1) excluded -> slot goes to the next eligible (idx 2..)
    got = select_constraint_first(sims, np.array([0, 1]), 3)
    assert 1 not in got.tolist()
    assert got.tolist() == [2, 3, 4]


def test_legacy_pool_then_filter_truncates_below_k():
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    # pool_k=5 -> {0,1,2,3,4}; 2 of those excluded -> exactly 3 remain
    got = select_pool_then_filter(sims, np.array([1, 3]), 5, pool_k=5)
    assert got.tolist() == [0, 2, 4]
    assert len(got) == 3


def test_prefix_property_cf5_equals_cf20_prefix():
    rng = np.random.default_rng(7)
    sims = rng.normal(size=30)
    excl = np.array([0, 4, 9, 13], dtype=np.int32)
    top20 = select_constraint_first(sims, excl, 20)
    top5 = select_constraint_first(sims, excl, 5)
    assert top5.tolist() == top20[:5].tolist()


def test_tie_determinism_ascending_index():
    sims = np.array([0.5, 0.9, 0.9, 0.9, 0.1, 0.9])
    a = select_constraint_first(sims, np.array([0]), 3)
    b = select_constraint_first(sims, np.array([0]), 3)
    # duplicate values break by ascending index; identical across calls
    assert a.tolist() == [1, 2, 3]
    assert a.tolist() == b.tolist()


def test_legacy_is_prefix_subset_of_constraint_first():
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = 25
        sims = np.round(rng.normal(size=n), 2)  # rounding forces real ties
        excl_n = rng.integers(1, 6)
        excl = np.unique(rng.integers(0, n, size=excl_n)).astype(np.int32)
        for k in (3, 5, 10):
            leg = select_pool_then_filter(sims, excl, k, pool_k=k)
            cf = select_constraint_first(sims, excl, k)
            # legacy(K) is exactly the first len(leg) entries of cf(K)
            assert leg.tolist() == cf[:len(leg)].tolist()


def test_eligible_fewer_than_k_returns_all_eligible():
    sims = np.array([0.9, 0.8, 0.7, 0.6])
    got = select_constraint_first(sims, np.array([0, 2]), 5)
    # only 2 eligible; production asserts len == k and aborts — the reference
    # documents the shortfall by returning all eligible
    assert got.tolist() == [1, 3]


def test_build_exclusion_columns_boundary_semantics():
    keys = ["a", "b", "c", "d", "e"]
    near = {
        "a": {"b": 1, "c": 2, "d": 3},   # e unknown (beyond cutoff)
        "b": {"a": 1},
        "c": {"a": 2},
        "d": {"a": 3},
        "e": {},
    }
    cols = build_exclusion_columns(keys, near, min_hops=3)
    # self always included; d <= 2 excluded; d == 3 and unknown ELIGIBLE
    assert cols[0].tolist() == [0, 1, 2]      # a: self, b(d1), c(d2); NOT d, e
    assert cols[3].tolist() == [3]            # d: only self (a is at d=3)
    assert cols[4].tolist() == [4]            # e: only self
    # min_hops respected: min_hops=2 excludes only d <= 1
    cols2 = build_exclusion_columns(keys, near, min_hops=2)
    assert cols2[0].tolist() == [0, 1]        # a: self + b(d1) only
    assert all(c.dtype == np.int32 for c in cols)


def test_resolve_topk_buffer_certifies_or_flags():
    # Clean boundary: buffer-end value strictly below the k-th selected value.
    vals = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    idxs = np.array([10, 11, 12, 13, 14])
    v, ix, needs = resolve_topk_buffer(vals, idxs, 3)
    assert not needs
    assert ix.tolist() == [10, 11, 12]
    assert v.tolist() == [0.9, 0.8, 0.7]
    # Tie group spanning the buffer end -> cannot certify -> needs_exact.
    vals = np.array([0.9, 0.7, 0.7, 0.7, 0.7])
    idxs = np.array([3, 9, 7, 5, 2])
    v, ix, needs = resolve_topk_buffer(vals, idxs, 3)
    assert needs
    # Stable tiebreak still applied within the buffer: ascending index on ties.
    assert ix.tolist() == [3, 2, 5]


def test_resolve_topk_buffer_stable_tiebreak_deterministic():
    vals = np.array([0.5, 0.5, 0.5, 0.4, 0.3])
    idxs = np.array([42, 7, 19, 1, 2])
    v1, i1, _ = resolve_topk_buffer(vals, idxs, 2)
    v2, i2, _ = resolve_topk_buffer(vals, idxs, 2)
    assert i1.tolist() == [7, 19]  # ties by ascending index
    assert i1.tolist() == i2.tolist() and v1.tolist() == v2.tolist()


def test_torch_gpu_path_matches_numpy_reference():
    torch = pytest.importorskip("torch")  # skips on host; runs in docker image
    rng = np.random.default_rng(0)
    n, k, buf = 40, 5, 8
    sims = rng.normal(size=n).astype(np.float32)
    excl = np.array([0, 3, 17, 25], dtype=np.int32)
    t = torch.from_numpy(sims.copy())
    t[torch.from_numpy(excl.astype(np.int64))] = torch.finfo(torch.float32).min
    vals, idxs = torch.topk(t, k + buf)
    v, ix, needs = resolve_topk_buffer(vals.numpy(), idxs.numpy(), k)
    ref = select_constraint_first(sims.astype(np.float64), excl, k)
    if needs:
        ix = ref  # production falls back to the exact numpy recompute
    assert sorted(np.asarray(ix).tolist()) == sorted(ref.tolist())
    ref_vals = np.sort(sims[ref])[::-1]
    got_vals = np.sort(sims[np.asarray(ix)])[::-1]
    assert np.allclose(got_vals, ref_vals, atol=1e-5)
