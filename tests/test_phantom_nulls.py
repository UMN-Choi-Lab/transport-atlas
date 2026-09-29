"""Offline unit tests for the graph baselines + configuration-model nulls.

Tiny inline graphs, numpy/scipy only (test_authors.py style) — everything runs
under the host interpreter without torch.
"""
from __future__ import annotations

import random
from collections import Counter

import numpy as np

from transport_atlas.process.phantom_nulls import (
    chung_lu_expected_hits,
    csr_from_edges,
    degree_cdf,
    draw_from_cdf,
    ppr_scores,
    rank_top_k_eligible,
    stub_match_rewire,
)


# ----------------------------------------------------------------------
# degree_cdf / draw_from_cdf
# ----------------------------------------------------------------------
def test_degree_cdf_zero_total_is_none():
    assert degree_cdf(np.zeros(4)) is None


def test_draw_from_cdf_zero_weight_never_drawn_and_proportional():
    keys = ["a", "b", "c", "d"]
    cdf = degree_cdf(np.array([3.0, 1.0, 0.0, 4.0]))
    rng = random.Random(123)
    counts = Counter()
    for _ in range(10_000):
        got = draw_from_cdf(rng, cdf, keys, excl=set(), k_need=1)
        counts[got[0]] += 1
    assert counts["c"] == 0  # zero-weight: never drawn
    # statistical sanity: frequency roughly proportional to weight (3:1)
    ratio = counts["a"] / max(counts["b"], 1)
    assert 2.0 < ratio < 4.5, counts
    assert counts["d"] > counts["a"]


def test_draw_from_cdf_exclusions_exact_k_and_determinism():
    keys = list("abcdefgh")
    cdf = degree_cdf(np.ones(8))
    got1 = draw_from_cdf(random.Random(5), cdf, keys, {"a", "b"}, 4)
    got2 = draw_from_cdf(random.Random(5), cdf, keys, {"a", "b"}, 4)
    assert got1 == got2  # deterministic under a fixed seed
    assert len(got1) == 4 == len(set(got1))
    assert not {"a", "b"} & set(got1)


def test_draw_from_cdf_min_hops2_shaped_exclusion():
    # A min-hops-2 exclusion set excludes only d <= 1 (self + direct
    # coauthors) — the website's --phantom-min-hops 2 variant.
    keys = list("abcde")
    excl_h2 = {"a", "b"}          # self + one direct coauthor
    cdf = degree_cdf(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    got = draw_from_cdf(random.Random(1), cdf, keys, excl_h2, 3)
    assert sorted(got) == ["c", "d", "e"]


# ----------------------------------------------------------------------
# chung_lu_expected_hits
# ----------------------------------------------------------------------
def test_chung_lu_hand_computed_with_clamp():
    # 4 authors; test-graph degrees s = {a:3, b:3, c:1, d:1}; m* = 4 edges.
    s = {"a": 3, "b": 3, "c": 1, "d": 1}
    m_star = 4
    # predictions: a -> [b, c]; d -> [b]
    cand = {"a": ["b", "c"], "d": ["b"]}
    # E = min(1, 3*3/8) + min(1, 3*1/8) + min(1, 1*3/8) = 9/8->1 + 3/8 + 3/8
    exp = chung_lu_expected_hits(cand, s, m_star)
    assert abs(exp - (1.0 + 3 / 8 + 3 / 8)) < 1e-12
    # without the clamp it would be 9/8 + 3/8 + 3/8 — verify clamp bit
    assert exp < 9 / 8 + 3 / 8 + 3 / 8


def test_chung_lu_m_star_zero_guard_and_zero_degree():
    assert chung_lu_expected_hits({"a": ["b"]}, {"a": 1, "b": 1}, 0) == 0.0
    # zero-degree endpoints contribute nothing
    assert chung_lu_expected_hits({"a": ["b"]}, {"a": 0, "b": 5}, 3) == 0.0
    assert chung_lu_expected_hits({"a": ["b"]}, {"a": 5, "b": 0}, 3) == 0.0


# ----------------------------------------------------------------------
# stub_match_rewire
# ----------------------------------------------------------------------
def _stub_array(s: dict[int, int]) -> np.ndarray:
    return np.concatenate([np.full(v, k, dtype=np.int64)
                           for k, v in sorted(s.items())])


def test_stub_match_preserves_degrees_no_self_loops_no_dups():
    s = {0: 2, 1: 3, 2: 2, 3: 1, 4: 2}  # 10 stubs, m = 5
    stubs = _stub_array(s)
    rng = np.random.default_rng(11)
    accepted, n_disc = stub_match_rewire(stubs, lambda a, b: False, rng)
    deg = Counter()
    for a, b in accepted:
        assert a != b            # no self-loops
        assert a < b             # canonical ordering
        deg[a] += 1
        deg[b] += 1
    assert len(accepted) == len(set(accepted))  # no duplicate edges
    # degree accounting: accepted degree + discarded stub count == s
    disc = Counter()
    total_disc = 0
    # reconstruct discards from conservation: 2*|accepted| + n_disc == 2m
    assert 2 * len(accepted) + n_disc == len(stubs)
    for node, s_v in s.items():
        assert deg[node] <= s_v
        total_disc += s_v - deg[node]
    assert total_disc == n_disc
    del disc


def test_stub_match_respects_ineligible_pairs_and_determinism():
    s = {0: 1, 1: 1}
    stubs = _stub_array(s)
    # only possible pair (0, 1) is ineligible -> everything discarded
    accepted, n_disc = stub_match_rewire(
        stubs, lambda a, b: True, np.random.default_rng(3))
    assert accepted == set() and n_disc == 2
    # determinism under a fixed Generator seed
    s = {0: 2, 1: 2, 2: 2, 3: 2}
    stubs = _stub_array(s)
    a1, d1 = stub_match_rewire(stubs, lambda a, b: False,
                               np.random.default_rng(9))
    a2, d2 = stub_match_rewire(stubs, lambda a, b: False,
                               np.random.default_rng(9))
    assert a1 == a2 and d1 == d2


def test_stub_match_forced_rejection_discard_accounting():
    # Engineered to force rejections: two nodes with 2 stubs each can realize
    # the (0, 1) edge only once; the second (0, 1) pairing is a duplicate and
    # both stubs of it must be discarded after the repair rounds.
    s = {0: 2, 1: 2}
    stubs = _stub_array(s)
    accepted, n_disc = stub_match_rewire(stubs, lambda a, b: False,
                                         np.random.default_rng(2))
    assert accepted == {(0, 1)}
    assert n_disc == 2


# ----------------------------------------------------------------------
# ppr_scores
# ----------------------------------------------------------------------
def test_ppr_path_graph_ordering_and_distribution():
    nodes = ["n0", "n1", "n2", "n3", "n4"]
    edges = [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4")]
    csr = csr_from_edges(nodes, edges)
    x = ppr_scores(csr, np.array([0]), alpha=0.15, tol=1e-8, max_iter=200)
    col = x[:, 0].astype(np.float64)
    assert abs(col.sum() - 1.0) < 1e-4      # proper distribution
    # Nearer NON-anchor nodes score higher (the anchor itself is always
    # excluded by the protocol; a degree-1 anchor can score below its sole
    # neighbor because it pushes all its outflow there).
    assert col[1] > col[2] > col[3] > col[4]
    assert col[4] > 0.0                     # mass reaches beyond BFS-3 horizon


def test_ppr_converges_within_max_iter():
    nodes = ["a", "b", "c"]
    csr = csr_from_edges(nodes, [("a", "b"), ("b", "c")])
    x1 = ppr_scores(csr, np.array([0, 2]), max_iter=100)
    x2 = ppr_scores(csr, np.array([0, 2]), max_iter=300)
    assert np.allclose(x1, x2, atol=1e-4)
    assert x1.shape == (3, 2)


# ----------------------------------------------------------------------
# rank_top_k_eligible
# ----------------------------------------------------------------------
def test_rank_top_k_excludes_restricts_and_tiebreaks():
    scores = np.array([0.0, 0.9, 0.5, 0.5, 0.7, 0.3])
    cand = np.array([1, 2, 3, 4])          # 0 and 5 are not candidates
    chosen, nf = rank_top_k_eligible(scores, cand, excl={1}, k=3,
                                     fill_order=[])
    # 1 excluded; order: 4 (.7), then tie .5 broken by candidate order (2 < 3)
    assert chosen == [4, 2, 3]
    assert nf == 0


def test_rank_top_k_degree_fill_on_shortfall():
    scores = np.array([0.0, 0.9, 0.0, 0.0, 0.0])
    cand = np.arange(5)
    fill = [3, 2, 4, 0]                    # deterministic (-degree, key) order
    chosen, nf = rank_top_k_eligible(scores, cand, excl={0}, k=3,
                                     fill_order=fill)
    # only idx 1 has positive score; 2 slots filled from fill_order
    assert chosen == [1, 3, 2]
    assert nf == 2


def test_rank_top_k_min_hops2_exclusion_flows_through():
    # min-hops-2 exclusion (self + direct coauthors only): d==2 nodes are
    # eligible and must be rankable.
    scores = np.array([1.0, 0.8, 0.6, 0.4])
    excl_h2 = {0, 1}                       # self + one direct coauthor
    chosen, nf = rank_top_k_eligible(scores, np.arange(4), excl_h2, 2, [])
    assert chosen == [2, 3]
    assert nf == 0
