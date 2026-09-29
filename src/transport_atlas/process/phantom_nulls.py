"""Torch-free graph baselines + configuration-model nulls for the phantom eval.

Used by scripts/07_phantom_eval.py (Reviewer 4, point 3).  Everything here is
pure numpy/scipy so the host interpreter (no torch) can import and unit-test
it offline; the scipy CSR personalized-PageRank implementation below is the
AUTHORITATIVE scorer (a torch backend in 07, if used, must match it).

Contents:
    csr_from_edges            symmetric unweighted CSR adjacency
    ppr_scores                personalized PageRank via chunked power iteration
    rank_top_k_eligible       deterministic eligible-pool ranking with
                              deterministic degree fill on shortfall
    degree_cdf                cumulative distribution for weighted sampling
    draw_from_cdf             rejection sampling from a CDF with exclusions
    chung_lu_expected_hits    analytic expected hits under the Chung-Lu /
                              configuration-model null on the test graph
    stub_match_rewire         Monte Carlo degree-preserving stub matching with
                              rejection + repair rounds (Fosdick et al. 2018)
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import scipy.sparse as sp

__all__ = [
    "csr_from_edges",
    "ppr_scores",
    "rank_top_k_eligible",
    "degree_cdf",
    "draw_from_cdf",
    "chung_lu_expected_hits",
    "stub_match_rewire",
]


def csr_from_edges(nodes: list[str], edges: Iterable[tuple[str, str]]) -> sp.csr_matrix:
    """Symmetric unweighted float32 CSR adjacency over ``nodes`` (index order)."""
    idx = {n: i for i, n in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    for a, b in edges:
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        rows.extend((ia, ib))
        cols.extend((ib, ia))
    n = len(nodes)
    m = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    m.data[:] = 1.0  # collapse any duplicate edge entries to weight 1
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def ppr_scores(
    csr_adj: sp.csr_matrix,
    anchor_cols: np.ndarray,
    alpha: float = 0.15,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """Personalized PageRank columns for a batch of anchors.

    X_{t+1} = (1 - alpha) * P^T X_t + alpha * E, with P = D^{-1} A row-stochastic
    and E the one-hot restart matrix for ``anchor_cols``.  Zero-degree anchors
    keep all mass on themselves (their row of P is all-zero, so X converges to
    alpha * sum_t (1-alpha)^t * e = e).  Returns an (N_g, B) float32 matrix.
    Stops when the max per-column L1 change drops below ``tol``.
    """
    n = csr_adj.shape[0]
    anchor_cols = np.asarray(anchor_cols, dtype=np.int64)
    b = len(anchor_cols)
    deg = np.asarray(csr_adj.sum(axis=1)).ravel().astype(np.float64)
    inv_deg = np.zeros(n, dtype=np.float64)
    nz = deg > 0
    inv_deg[nz] = 1.0 / deg[nz]
    # P^T = A^T D^{-1} = A D^{-1} (A symmetric): scale columns of A by inv_deg
    pt = csr_adj.astype(np.float64) @ sp.diags(inv_deg)
    e = np.zeros((n, b), dtype=np.float64)
    e[anchor_cols, np.arange(b)] = 1.0
    x = e.copy()
    for _ in range(max_iter):
        x_new = (1.0 - alpha) * (pt @ x) + alpha * e
        delta = np.abs(x_new - x).sum(axis=0).max()
        x = x_new
        if delta < tol:
            break
    return x.astype(np.float32)


def rank_top_k_eligible(
    scores: np.ndarray,
    candidate_idx: np.ndarray,
    excl: set[int],
    k: int,
    fill_order: list[int],
) -> tuple[list[int], int]:
    """Rank ``candidate_idx`` by (-score, position-in-candidate_idx) and return
    the top ``k`` eligible, filling any shortfall deterministically.

    - ``scores``: full score vector (indexed by the same space as candidate_idx).
    - ``candidate_idx``: the allowed candidate universe (e.g. eval authors'
      graph indices), in a caller-supplied deterministic order — ties in score
      are broken by this order, so pass it sorted by the desired tiebreak
      (e.g. author_key order).
    - ``excl``: indices to exclude (self + near-train).
    - Only strictly-positive scores are ranked; zero/negative-score candidates
      are treated as "no graph signal" and left to the fill stage.
    - ``fill_order``: deterministic fill list (e.g. eval indices sorted by
      (-train_degree, author_key)); consumed in order, skipping ``excl`` and
      already-chosen indices.

    Returns (chosen indices, n_filled).
    """
    cand = np.asarray(candidate_idx, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)[cand]
    order = np.argsort(-s, kind="stable")  # stable => candidate_idx order on ties
    chosen: list[int] = []
    for pos in order:
        j = int(cand[pos])
        if j in excl:
            continue
        if s[pos] <= 0.0:
            break  # stable sort => all remaining are also <= 0
        chosen.append(j)
        if len(chosen) >= k:
            break
    n_filled = 0
    if len(chosen) < k:
        have = set(chosen)
        for j in fill_order:
            if j in excl or j in have:
                continue
            chosen.append(int(j))
            have.add(j)
            n_filled += 1
            if len(chosen) >= k:
                break
    return chosen, n_filled


def degree_cdf(weights: np.ndarray) -> np.ndarray | None:
    """Cumulative distribution from non-negative weights; None if total == 0."""
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if total <= 0.0:
        return None
    return np.cumsum(w) / total


def draw_from_cdf(
    rng,
    cdf: np.ndarray,
    keys: list,
    excl: set,
    k_need: int,
    tries_mult: int = 40,
) -> list:
    """Draw ``k_need`` distinct keys ~ CDF weights, rejecting ``excl`` members.

    Zero-weight keys have zero probability mass (their CDF step is flat) and
    are never drawn.  Rejection-samples for at most ``k_need * tries_mult``
    tries; a shortfall is returned as-is (callers complete it via random fill).
    ``rng`` is a ``random.Random`` instance for reproducibility.
    """
    n = len(keys)
    out: list = []
    seen = set(excl)
    tries = 0
    while len(out) < k_need and tries < k_need * tries_mult:
        u = rng.random()
        j = int(np.searchsorted(cdf, u, side="right"))
        if j >= n:
            j = n - 1
        cand = keys[j]
        if cand not in seen:
            out.append(cand)
            seen.add(cand)
        tries += 1
    return out


def chung_lu_expected_hits(
    cand_lists: dict[str, list[str]],
    s: dict[str, int],
    m_star: int,
) -> float:
    """Analytic expected hits under the configuration-model (Chung-Lu) null.

    Holding fixed each author's number of distinct eligible test partners
    (degree sequence {s_a} of the realized eligible test graph, 2 * m_star
    stubs total) and destroying only the pairing, the probability a directed
    prediction (a, c) is realized is min(1, s_a * s_c / (2 m_star)).  Sums
    that over every prediction; the clamp matters for hub-hub pairs.
    Returns 0.0 when m_star == 0.
    """
    if m_star <= 0:
        return 0.0
    two_m = 2.0 * m_star
    exp_hits = 0.0
    for a, cands in cand_lists.items():
        s_a = s.get(a, 0)
        if s_a <= 0:
            continue
        for c in cands:
            s_c = s.get(c, 0)
            if s_c <= 0:
                continue
            exp_hits += min(1.0, (s_a * s_c) / two_m)
    return exp_hits


def stub_match_rewire(
    stubs: np.ndarray,
    is_ineligible: Callable[[int, int], bool],
    rng: np.random.Generator,
    max_repair: int = 50,
) -> tuple[set[tuple[int, int]], int]:
    """One degree-preserving rewire of the test graph via constrained stub
    matching with rejection + repair (Fosdick et al. 2018).

    ``stubs``: int array where node index a appears s_a times (length 2m).
    Shuffle, pair consecutive stubs; ACCEPT a pair iff it is not a self-loop,
    not a duplicate of an already-accepted pair, and not ``is_ineligible(a, b)``
    (used for near-train pairs — the null lives on the protocol's eligible
    simple-graph space).  Rejected pairs return their stubs to a leftover pool
    that is re-shuffled and re-paired for up to ``max_repair`` rounds; any
    final remainder is discarded.

    Returns (accepted canonical (min, max) pairs, n_discarded_stubs).
    """
    accepted: set[tuple[int, int]] = set()
    pool = np.asarray(stubs, dtype=np.int64).copy()
    for _ in range(max_repair + 1):
        if len(pool) < 2:
            break
        rng.shuffle(pool)
        if len(pool) % 2 == 1:
            # odd leftover stub can never pair this round; carry it forward
            carry = pool[-1:]
            pairs = pool[:-1].reshape(-1, 2)
        else:
            carry = pool[:0]
            pairs = pool.reshape(-1, 2)
        leftover: list[int] = list(carry)
        for a, b in pairs:
            a, b = int(a), int(b)
            key = (a, b) if a < b else (b, a)
            if a == b or key in accepted or is_ineligible(a, b):
                leftover.extend((a, b))
                continue
            accepted.add(key)
        pool = np.asarray(leftover, dtype=np.int64)
    return accepted, int(len(pool))
