"""Offline unit tests for the pure metric helpers + grid enumeration of
paper/analysis/09_partition_robustness.py.

paper/analysis is not a package, so the module is loaded via importlib.
These tests are fixture-free and synthetic-only: they never read or write
anything under data/ (including the script's cache dir) and never touch the
network. sklearn is the reference implementation for NMI/AMI.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (adjusted_mutual_info_score,
                             normalized_mutual_info_score)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper" / "analysis" / "09_partition_robustness.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("partition_robustness_09",
                                                  SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- ECS ------------------------------------------------------------------

def test_ecs_identity_on_identical_partitions(mod):
    rng = np.random.default_rng(0)
    p = rng.integers(0, 9, 400)
    assert abs(mod.ecs_hard(p, p) - 1.0) <= 1e-12


def test_ecs_singletons_vs_one_cluster_is_one_over_n(mod):
    n = 97
    ecs = mod.ecs_hard(np.arange(n), np.zeros(n, dtype=int))
    assert abs(ecs - 1.0 / n) <= 1e-12


def test_ecs_closed_form_matches_brute_force_ppr(mod):
    rng = np.random.default_rng(1)
    for _ in range(4):
        a = rng.integers(0, int(rng.integers(2, 10)), 200)
        b = rng.integers(0, int(rng.integers(2, 10)), 200)
        closed = mod.ecs_hard(a, b)
        for alpha in (0.9, 0.5):  # alpha-free: PPR damping cancels
            assert abs(closed - mod.ecs_brute_ppr(a, b, alpha=alpha)) <= 1e-9


# --- fast NMI / VI --------------------------------------------------------

def test_fast_nmi_matches_sklearn_on_toy_partitions(mod):
    rng = np.random.default_rng(2)
    for _ in range(15):
        sz = int(rng.integers(30, 2000))
        a = rng.integers(0, int(rng.integers(1, 40)), sz)
        b = rng.integers(0, int(rng.integers(1, 40)), sz)
        assert abs(mod.fast_nmi(a, b)
                   - normalized_mutual_info_score(a, b)) <= 1e-9
    # degenerate cases must match sklearn's conventions
    p = rng.integers(0, 5, 100)
    const = np.zeros(100, dtype=int)
    for x, y in ((p, p), (p, const), (const, const)):
        assert abs(mod.fast_nmi(x, y)
                   - normalized_mutual_info_score(x, y)) <= 1e-9


def test_vi_bits_identity_and_symmetry(mod):
    rng = np.random.default_rng(3)
    a = rng.integers(0, 6, 300)
    b = rng.integers(0, 4, 300)
    assert abs(mod.vi_bits(a, a)) <= 1e-12
    assert abs(mod.vi_bits(a, b) - mod.vi_bits(b, a)) <= 1e-12
    # two independent fair coins: VI = H(a|b)+H(b|a) = 2 bits exactly
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    assert abs(mod.vi_bits(x, y) - 2.0) <= 1e-12


# --- AMI wrapper sanity ---------------------------------------------------

def test_ami_toy_partitions(mod):
    rng = np.random.default_rng(4)
    a = rng.integers(0, 8, 1500)
    assert abs(adjusted_mutual_info_score(a, a) - 1.0) <= 1e-12
    amis = [adjusted_mutual_info_score(a, mod.permuted_labels(a, rng))
            for _ in range(30)]
    assert abs(float(np.mean(amis))) < 0.02  # chance-corrected ~ 0


# --- permutation null: size-matched + deterministic -----------------------

def test_permuted_labels_are_size_matched(mod):
    rng = np.random.default_rng(5)
    b = rng.integers(0, 7, 500)
    perm = mod.permuted_labels(b, np.random.default_rng(11))
    assert np.array_equal(np.sort(perm), np.sort(b))  # sizes preserved
    assert not np.array_equal(perm, b)                # actually shuffled


def test_perm_null_deterministic_and_order_independent(mod):
    rng = np.random.default_rng(6)
    a = rng.integers(0, 12, 800)
    b = rng.integers(0, 9, 800)

    def null_for(idx):
        r = np.random.default_rng(np.random.SeedSequence([42, idx]))
        return mod.perm_null(a, b, 15, r)

    first = {i: null_for(i) for i in (0, 1, 2)}
    second = {i: null_for(i) for i in (2, 0, 1)}  # permuted execution order
    for i in (0, 1, 2):
        assert np.array_equal(first[i], second[i])
    assert not np.array_equal(first[0], first[1])  # distinct per-cell streams
    assert np.all(first[0] >= 0.0) and np.all(first[0] <= 1.0)


def test_null_stats_degenerate_sd_yields_none_and_strict_json(mod):
    nulls = np.full(20, 0.5)
    stats = mod.null_stats(0.5, 0.4, nulls)
    assert stats["z"] is None
    assert stats["m"] == 20
    # must serialize under the repo's allow_nan=False hard rule
    json.dumps(stats, allow_nan=False)


def test_null_stats_basic_fields(mod):
    rng = np.random.default_rng(7)
    a = rng.integers(0, 10, 2000)
    b = rng.integers(0, 10, 2000)
    obs = mod.fast_nmi(a, b)
    nulls = mod.perm_null(a, b, 50, np.random.default_rng(0))
    stats = mod.null_stats(obs, 0.0, nulls)
    assert 0.0 < stats["p_emp"] <= 1.0
    assert stats["mean"] >= 0.0 and stats["sd"] >= 0.0
    json.dumps(stats, allow_nan=False)


# --- grid enumeration determinism ----------------------------------------

def test_grid_enumeration_is_deterministic_and_static(mod):
    g1 = mod.enumerate_grid()
    g2 = mod.enumerate_grid()
    assert g1 == g2
    assert [c["cell_index"] for c in g1] == list(range(len(g1)))
    ids = [c["cell_id"] for c in g1]
    assert len(ids) == len(set(ids)), "cell_ids must be unique"


def test_grid_axis_counts_match_spec(mod):
    grid = mod.enumerate_grid()
    counts = {}
    for c in grid:
        counts[c["axis"]] = counts.get(c["axis"], 0) + 1
    # 6 headline; 5x5 A; (2x6)x(1+3)+1 calibration B; 28x2 C; 3x5 D
    assert counts == {"0": 6, "A": 25, "B": 49, "C": 56, "D": 15}
    assert len(grid) == 151


def test_grid_axis_filtering_preserves_cell_index(mod):
    grid = mod.enumerate_grid()
    subset = [c for c in grid if c["axis"] in {"0", "A"}]
    # cell_index survives subsetting untouched -> per-cell RNG streams are
    # independent of --axes selection and execution order
    assert [c["cell_index"] for c in subset] == \
           [c["cell_index"] for c in grid if c["axis"] in {"0", "A"}]
    assert all(grid[c["cell_index"]] == c for c in subset)


def test_contiguize(mod):
    lab = np.array([10, 10, 3, 7, 3])
    out = mod.contiguize(lab)
    assert out.tolist() == [2, 2, 0, 1, 0]
    assert out.dtype == np.int64
