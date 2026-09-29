"""Pure-numpy reference implementations of the phantom-eval selection protocol.

This module is the offline-testable oracle for scripts/07_phantom_eval.py:
it must stay torch-free so the host interpreter (no CUDA, no torch) can import
and unit-test it, and so the GPU path in 07 can be spot-checked against it at
runtime.

Protocol (constraint-first, the headline definition):
    For eval author i, the exclusion set excl(i) = {i} ∪ {j : d_train(i, j) <=
    min_hops - 1}.  Eligible(i) = all eval authors minus excl(i).  phantom_K(i)
    = the K highest-cosine members of Eligible(i); K is applied AFTER the
    distance constraint.  Ties are broken deterministically by (-similarity,
    ascending eval index) under a stable sort.

Legacy (as-submitted) semantics are kept for the bridge metrics:
    take the global top pool_k by similarity FIRST, then drop excluded members
    without replacement and truncate to K (can return fewer than K).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "build_exclusion_columns",
    "select_constraint_first",
    "select_pool_then_filter",
    "resolve_topk_buffer",
]


def build_exclusion_columns(
    eval_keys: list[str],
    near_neighbors: dict[str, dict[str, int]],
    min_hops: int,
) -> list[np.ndarray]:
    """Per-author exclusion sets as eval-column index arrays (int32).

    excl(i) always contains i itself, plus every eval author j whose train-graph
    distance d(i, j) satisfies d <= min_hops - 1.  Authors absent from
    ``near_neighbors[key_i]`` have unknown distance (> BFS cutoff or in a
    different component) and are therefore ELIGIBLE; a recorded d == min_hops is
    likewise eligible.

    Parameters
    ----------
    eval_keys : sorted list of eval author keys (defines column order).
    near_neighbors : {key: {other_key: distance}} from the train-graph BFS.
    min_hops : the protocol's PHANTOM_MIN_HOPS (candidates need d >= min_hops).
    """
    key_idx = {k: i for i, k in enumerate(eval_keys)}
    cols: list[np.ndarray] = []
    for i, k in enumerate(eval_keys):
        excl = [i]
        for kk, dd in near_neighbors.get(k, {}).items():
            if dd <= min_hops - 1:
                j = key_idx.get(kk)
                if j is not None and j != i:
                    excl.append(j)
        cols.append(np.array(sorted(set(excl)), dtype=np.int32))
    return cols


def select_constraint_first(
    sims_row: np.ndarray,
    excl_cols_row: np.ndarray,
    k: int,
) -> np.ndarray:
    """Reference constraint-first selection for one author.

    Mask the excluded columns, then stable-sort the FULL eligible pool by
    (-similarity, ascending index) and return the first ``k`` indices.  If
    fewer than ``k`` candidates are eligible, all of them are returned
    (production code in 07 asserts len == k and aborts otherwise).
    """
    row = np.asarray(sims_row, dtype=np.float64).copy()
    excl = np.asarray(excl_cols_row, dtype=np.int64)
    if excl.size:
        row[excl] = -np.inf
    order = np.argsort(-row, kind="stable")  # stable => index-ascending on ties
    eligible = order[np.isfinite(row[order])]
    return eligible[:k].astype(np.int64)


def select_pool_then_filter(
    sims_row: np.ndarray,
    excl_cols_row: np.ndarray,
    k: int,
    pool_k: int,
) -> np.ndarray:
    """Legacy (as-submitted) selection: global top ``pool_k`` FIRST, then drop
    excluded members without replacement and truncate to ``k``.

    Can return fewer than ``k`` indices — that is the historical behaviour the
    bridge metrics reproduce.
    """
    row = np.asarray(sims_row, dtype=np.float64)
    order = np.argsort(-row, kind="stable")[:pool_k]
    excl = set(int(x) for x in np.asarray(excl_cols_row).tolist())
    chosen = [int(j) for j in order if int(j) not in excl]
    return np.array(chosen[:k], dtype=np.int64)


def resolve_topk_buffer(
    vals_buf: np.ndarray,
    idxs_buf: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Deterministic tiebreak over a (k + buffer)-wide top candidate buffer.

    Stable-sorts the buffer by (-value, ascending index) and returns the first
    ``k`` (values, indices).  ``needs_exact`` is True when the buffer cannot
    CERTIFY the tie group at the K-boundary: i.e. the smallest value in the
    buffer is not strictly below the k-th selected value, so members of the
    boundary tie group may exist outside the buffer and the caller must
    recompute that row exactly from the full similarity matrix.
    """
    vals = np.asarray(vals_buf, dtype=np.float64)
    idxs = np.asarray(idxs_buf, dtype=np.int64)
    assert vals.shape == idxs.shape and vals.ndim == 1
    # stable sort by (-val, ascending idx): sort by idx first, then by -val stably
    order0 = np.argsort(idxs, kind="stable")
    order = order0[np.argsort(-vals[order0], kind="stable")]
    top_v = vals[order][:k]
    top_i = idxs[order][:k]
    if len(vals) <= k:
        # Buffer no larger than k: nothing beyond it to certify against.
        return top_v, top_i, False
    kth_val = top_v[-1] if len(top_v) == k else -np.inf
    buf_min = float(vals.min())
    needs_exact = bool(buf_min >= kth_val)
    return top_v, top_i, needs_exact
