#!/usr/bin/env python
"""§8 — Phantom-collaborator predictive test with temporal holdout.

Canonical protocol text (constraint-first; the manuscript mirrors this):

Train cutoff: year <= 2019.  Test window: 2020 <= year <= 2025.
(2026 is dropped because the corpus snapshot captures only a partial year.)

For each eval author A (train centroid + >= MIN_PAPERS_TRAIN train papers):

    phantom_K(A) := the K most cosine-similar authors AMONG candidates at
                    train-graph distance >= PHANTOM_MIN_HOPS (or disconnected /
                    beyond the BFS cutoff).  K is applied AFTER the distance
                    constraint: the FULL eligible pool is ranked and truncated
                    at K, so |phantom_K(A)| == K always and the K=5/10/20
                    lists are nested prefixes of one fixed ranking.

    realized(A)  := authors A actually coauthored with in the test window,
                    excluding near-train pairs (same exclusion rule).

    precision@K(A) := |candidates ∩ realized(A)| / K          (ALL methods)
    recall@K(A)    := |candidates ∩ realized(A)| / |realized(A)|

Exclusion set excl(A) = {A} ∪ {B : d_train(A,B) <= PHANTOM_MIN_HOPS - 1}; it is
built ONCE and shared byte-identically by the phantom ranker and every
baseline, so all methods emit exactly K candidates per author from the
identical eligible pool and micro precision = total_hits / (n_eval * K)
(micro == macro under exact-K; asserted).

Deterministic ties: candidates are ordered by (-cosine, ascending eval index)
under a stable sort.  The GPU top-(K + TIE_BUF) buffer is resolved on CPU; any
row whose K-boundary tie group cannot be certified inside the buffer is
recomputed exactly in numpy (no widen-and-retry heuristics).

Centroid weights: train papers are weighted w = 1 + log1p(c) where c is the
citation count AS OF THE TRAIN CUTOFF (current total minus post-cutoff
counts_by_year, fetched by scripts/fetch_citation_snapshots.py; see
--citations-asof).  Papers missing from the snapshot get c = 0 and are
counted in config.citation_weights.  Using current-snapshot counts would be
temporal leakage, so a missing snapshot file is a hard error.

Baselines / graph predictors (identical protocol, exactly K each):
    random          uniform over the eligible pool
    pref_attach     ∝ train-period paper count
    config_degree   ∝ train coauthor-graph degree.  With the anchor fixed, the
                    configuration-model conditional probability that one of
                    A's stubs attaches to author c is ∝ deg_train(c), so
                    per-anchor degree-proportional sampling restricted to the
                    eligible set IS the configuration-model conditional for
                    this per-anchor precision metric (an edge-swap MCMC over
                    the whole graph converges to the same conditional).
    same_venue      uniform over eligible venue-mates, random-padded (padding
                    is load-bearing for exact-K; instrumented via n_padded)
    same_community  uniform over eligible same-Leiden-community authors
                    (unweighted train graph, ModularityVertexPartition,
                    seed=SEED); enabled with --graph-baselines full
    graph_ppr       deterministic personalized-PageRank ranker on the FULL
                    train graph (alpha=0.15, scipy reference implementation),
                    tie-broken by (-score, author_key); shortfall filled
                    deterministically by (-train_degree, author_key) rank

Stochastic baselines are drawn --baseline-draws times with derived seeds
random.Random(f"{SEED}:{method}:{K}:{draw}") and reported POOLED (hits and
predictions accumulate over draws; predictions = n_draws * n_eval * K,
asserted), plus per-draw hit lists and the across-draw SD.  Every method also
gets an author-level cluster bootstrap CI (B = 1000, np.random.default_rng
(SEED)); it is approximate (realized edges shared between two anchors induce
cross-author dependence).

metrics_legacy_phantom reproduces the previously implemented pool-then-filter
variant IN THE SAME RUN (top-K cosine first, then drop near-train without
replacement; <= K candidates; rng-free), reporting both the as-submitted
hits/len(candidates) reading and the harsher hits/(n_eval*K) reading.  The
legacy list is a prefix-subset of the constraint-first list, so legacy hits <=
constraint-first hits per K (asserted).  Never compare against archived JSONs
from before the 2026-07-20 data refresh.

config_null is a degree-preserving configuration-model NULL on the realized
eligible test graph E* (analytic Chung-Lu expectation with the min(1, .)
clamp, plus Monte Carlo constrained stub matching that rejects self-loops,
duplicate edges, and near-train pairs, seeds PCG64(SEED + 777 + r)).  It
conditions on the realized test-window degree sequence and is reported as a
null distribution (expected precision, z, one-sided empirical p) for the
deterministic predictors — NEVER as a predictor row.

Outputs:
    data/processed/phantom_eval{SUFFIX}.json
    paper/analysis/_phantom_eval{SUFFIX}.json  (summary copy for prose quoting)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transport_atlas.process.authors import author_key as _raw_author_key
from transport_atlas.process.phantom_nulls import (
    chung_lu_expected_hits,
    csr_from_edges,
    degree_cdf,
    draw_from_cdf,
    ppr_scores,
    rank_top_k_eligible,
    stub_match_rewire,
)
from transport_atlas.process.phantom_protocol import (
    build_exclusion_columns,
    resolve_topk_buffer,
    select_constraint_first,
)

# ---- config ----
EMBED_DIR = Path(os.environ.get("EMBED_OUT", "/data2/chois/transport-atlas"))
TRAIN_CUTOFF_YEAR = 2019
TEST_YEARS = range(2020, 2026)  # 2020..2025 inclusive (exclude partial 2026)

MIN_PAPERS_TRAIN = 2
TOP_K = 20
EVAL_KS = (5, 10, 20)
PHANTOM_MIN_HOPS = 3
BFS_CUTOFF = 3
# When the script is run from CLI with --phantom-min-hops != default, output
# is suffixed (e.g., phantom_eval_k2.json) so the headline file is preserved
# for the paper's main figures/tables. Set non-empty by the __main__ block.
OUT_SUFFIX = ""
WHITEN_TOP_PC = 1
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TIE_BUF = 8              # GPU top-k buffer beyond TOP_K for tie certification
BASELINE_DRAWS = 10      # draws per stochastic baseline (CLI --baseline-draws)
BOOTSTRAP_B = 1000       # author-level cluster bootstrap resamples
NULL_REWIRES = 200       # Monte Carlo rewires for config_null (CLI)
GRAPH_BASELINES = "full"  # full | headline | off (CLI --graph-baselines)
SPOT_CHECK_N = 50        # runtime numpy spot-check sample size
PPR_ALPHA = 0.15
PPR_TOL = 1e-6
PPR_MAX_ITER = 100
PPR_CHUNK = 512
CITATIONS_ASOF: str | None = None  # CLI --citations-asof; default set in main()


# ---- alias map (copied from 06_author_similarity.py) ----
def _load_alias_map() -> dict[str, str]:
    import yaml
    repo = Path(__file__).resolve().parents[1]
    cfg = repo / "config" / "pipeline.yaml"
    pipe = yaml.safe_load(cfg.read_text()) or {}
    mp: dict[str, str] = {}
    for a in pipe.get("author_aliases", []) or []:
        ids = a.get("openalex_ids") or []
        if len(ids) < 2:
            continue
        target = ids[0].lower()
        for other in ids[1:]:
            mp[other.lower()] = target
    auto_path = repo / "data" / "interim" / "author_aliases_auto.json"
    if auto_path.exists():
        for k, v in json.loads(auto_path.read_text()).items():
            mp.setdefault(k, v)
    return mp


def author_key_with_alias(a: dict, alias_map: dict) -> str:
    k = _raw_author_key(a)
    return alias_map.get(k, k) if k else k


def _load_citation_map(path: Path) -> dict[str, int]:
    """paper_id -> citations as of the train cutoff (DuckDB read; >10k rows).

    Fails loudly if the snapshot is absent or was built for a different
    cutoff — silent fallback to current-snapshot counts would reintroduce
    the temporal leakage this fixes.
    """
    if not path.exists():
        print(
            f"[phantom] FATAL: citation snapshot not found: {path}\n"
            f"  The centroid weights require citations as of "
            f"{TRAIN_CUTOFF_YEAR} (not the current OpenAlex snapshot).\n"
            f"  Build it first:\n"
            f"    PYTHONPATH=src /usr/bin/python3 "
            f"scripts/fetch_citation_snapshots.py --cutoff-year "
            f"{TRAIN_CUTOFF_YEAR}\n"
            f"  or point --citations-asof at an existing snapshot parquet.",
            file=sys.stderr)
        raise SystemExit(2)
    import duckdb
    con = duckdb.connect()
    cutoffs = [r[0] for r in con.execute(
        "SELECT DISTINCT cutoff_year FROM read_parquet(?)",
        [str(path)]).fetchall()]
    if cutoffs != [TRAIN_CUTOFF_YEAR]:
        print(f"[phantom] FATAL: {path} was built for cutoff_year={cutoffs}, "
              f"need {TRAIN_CUTOFF_YEAR}. Re-run "
              f"scripts/fetch_citation_snapshots.py --cutoff-year "
              f"{TRAIN_CUTOFF_YEAR}.", file=sys.stderr)
        raise SystemExit(2)
    rows = con.execute(
        "SELECT paper_id, cited_asof_cutoff FROM read_parquet(?)",
        [str(path)]).fetchall()
    return {r[0]: int(r[1]) for r in rows}


def _bootstrap_ci95(per_author_hits: np.ndarray, preds_per_author: int,
                    b: int = BOOTSTRAP_B) -> list[float]:
    """Author-level cluster bootstrap CI for micro precision.

    Approximate: realized edges shared between two anchors induce
    cross-author dependence.  A fresh default_rng(SEED) per call makes the
    resample indices identical across methods (paired bootstrap).
    """
    n = len(per_author_hits)
    gen = np.random.default_rng(SEED)
    sums = np.empty(b, dtype=np.float64)
    for i in range(b):
        idx = gen.integers(0, n, n)
        sums[i] = per_author_hits[idx].sum()
    micro = sums / float(n * preds_per_author)
    return [float(np.percentile(micro, 2.5)), float(np.percentile(micro, 97.5))]


def main() -> int:  # noqa: C901
    assert PHANTOM_MIN_HOPS <= BFS_CUTOFF, (
        f"PHANTOM_MIN_HOPS={PHANTOM_MIN_HOPS} needs BFS distances up to "
        f"{PHANTOM_MIN_HOPS - 1} < BFS_CUTOFF={BFS_CUTOFF}; eligibility is "
        f"undefined beyond the BFS horizon")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    repo = Path(__file__).resolve().parents[1]
    print(f"[phantom] device={DEVICE}  cutoff<= {TRAIN_CUTOFF_YEAR}  "
          f"test={TEST_YEARS.start}..{TEST_YEARS.stop-1}  "
          f"min_hops={PHANTOM_MIN_HOPS}  draws={BASELINE_DRAWS}  "
          f"graph_baselines={GRAPH_BASELINES}", flush=True)

    # ----------------------------------------------------------------
    # As-of-cutoff citation snapshot (temporal-leakage fix). Hard requirement.
    # ----------------------------------------------------------------
    cites_path = Path(CITATIONS_ASOF) if CITATIONS_ASOF else (
        repo / "data" / "interim" / "citation_snapshots.parquet")
    cites_asof = _load_citation_map(cites_path)
    print(f"[phantom] citation snapshot: {len(cites_asof):,} train papers "
          f"(as of {TRAIN_CUTOFF_YEAR}) from {cites_path}", flush=True)

    # ----------------------------------------------------------------
    # Load paper embeddings + paper metadata + alias map
    # ----------------------------------------------------------------
    embed_path = EMBED_DIR / "paper_embeddings.parquet"
    if not embed_path.exists():
        print(f"[phantom] missing {embed_path}", file=sys.stderr)
        return 1
    emb_df = pd.read_parquet(embed_path)
    pid_to_row = {pid: i for i, pid in enumerate(emb_df["paper_id"].tolist())}
    E_raw = np.stack(emb_df["emb"].tolist()).astype(np.float32)
    print(f"[phantom] loaded {E_raw.shape[0]:,} paper embeddings", flush=True)

    papers = pd.read_parquet(repo / "data" / "interim" / "papers.parquet")
    authors_tbl = pd.read_parquet(repo / "data" / "interim" / "authors.parquet")
    alias_map = _load_alias_map()

    # Partition paper rows by year
    train_rows: list[int] = []
    for _, r in papers.iterrows():
        pid = r["paper_id"]
        row = pid_to_row.get(pid)
        if row is None:
            continue
        yr = r.get("year")
        if yr is None or pd.isna(yr):
            continue
        if int(yr) <= TRAIN_CUTOFF_YEAR:
            train_rows.append(row)
    print(f"[phantom] train papers with embeddings: {len(train_rows):,}",
          flush=True)

    # ----------------------------------------------------------------
    # Train-only whitening: fit mean + top-1 PC on the train subset,
    # then apply to all train embeddings. (Papers >=2020 never see this.)
    # ----------------------------------------------------------------
    t0 = time.time()
    E_train = E_raw[train_rows]
    E_t = torch.from_numpy(E_train).to(DEVICE).float()
    mu = E_t.mean(dim=0, keepdim=True)
    E_t = E_t - mu
    cov = (E_t.T @ E_t) / E_t.shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    top_dirs = evecs[:, -WHITEN_TOP_PC:]
    E_t = E_t - (E_t @ top_dirs) @ top_dirs.T
    std = E_t.std(dim=0, keepdim=True) + 1e-8
    E_t = E_t / std
    E_train_w = E_t.cpu().numpy().astype(np.float32)
    ev_ratio = float(evals[-1] / evals.sum())
    print(f"[phantom] train-only whitening in {time.time()-t0:.1f}s  "
          f"(top-1 PC explains {ev_ratio*100:.1f}%)", flush=True)
    del E_t, cov, evecs, top_dirs

    # Map from global paper-row to position in train_rows (so we can index E_train_w)
    train_row_of_global = {g: i for i, g in enumerate(train_rows)}

    # ----------------------------------------------------------------
    # Aggregate train embeddings to author centroids.
    # Also build train author→coauthors set for graph construction.
    # ----------------------------------------------------------------
    keep_keys = set(authors_tbl.loc[
        authors_tbl["n_papers"] >= 1, "author_key"])
    key_to_name = dict(zip(authors_tbl["author_key"],
                           authors_tbl["canonical_name"]))

    author_sum: dict[str, np.ndarray] = {}
    author_wsum: dict[str, float] = {}
    author_papers_train: dict[str, int] = defaultdict(int)
    author_venues_train: dict[str, set[str]] = defaultdict(set)
    train_edges: set[tuple[str, str]] = set()

    test_edges: set[tuple[str, str]] = set()
    test_active_keys: set[str] = set()

    n_weighted_papers = 0
    n_missing_cites = 0

    for _, r in tqdm(papers.iterrows(), total=len(papers), desc="[phantom] scan"):
        pid = r["paper_id"]
        yr = r.get("year")
        if yr is None or pd.isna(yr):
            continue
        year = int(yr)
        authors_list = r.get("authors")
        if authors_list is None or len(authors_list) == 0:
            continue

        # Resolve author keys once
        keys_here: list[str] = []
        for a in authors_list:
            if not isinstance(a, dict):
                continue
            k = author_key_with_alias(a, alias_map)
            if k and k in keep_keys:
                keys_here.append(k)
        if not keys_here:
            continue
        uniq_keys = list(dict.fromkeys(keys_here))  # preserve order, unique

        if year <= TRAIN_CUTOFF_YEAR:
            # Paper contributes to centroid, paper count, venue set, edges
            row = pid_to_row.get(pid)
            if row is not None:
                tr_row = train_row_of_global.get(row)
                if tr_row is not None:
                    v = E_train_w[tr_row]
                    # Temporal-leakage fix: weight by citations AS OF the
                    # train cutoff (never the current snapshot). Missing
                    # snapshot entries weigh as c=0 and are counted.
                    cites = cites_asof.get(pid)
                    if cites is None:
                        n_missing_cites += 1
                        cites = 0
                    n_weighted_papers += 1
                    w = 1.0 + np.log1p(cites)
                    for k in uniq_keys:
                        if k not in author_sum:
                            author_sum[k] = np.zeros(v.shape, dtype=np.float32)
                            author_wsum[k] = 0.0
                        author_sum[k] += w * v
                        author_wsum[k] += w
            # paper count + venue bookkeeping even without embedding
            venue = r.get("venue_slug") or r.get("venue")
            for k in uniq_keys:
                author_papers_train[k] += 1
                if isinstance(venue, str) and venue:
                    author_venues_train[k].add(venue)
            # train coauthor edges
            for i in range(len(uniq_keys)):
                for j in range(i + 1, len(uniq_keys)):
                    a, b = uniq_keys[i], uniq_keys[j]
                    if a == b:
                        continue
                    train_edges.add((a, b) if a < b else (b, a))
        elif year in TEST_YEARS:
            for k in uniq_keys:
                test_active_keys.add(k)
            for i in range(len(uniq_keys)):
                for j in range(i + 1, len(uniq_keys)):
                    a, b = uniq_keys[i], uniq_keys[j]
                    if a == b:
                        continue
                    test_edges.add((a, b) if a < b else (b, a))

    print(f"[phantom] train authors (>=1 paper): {len(author_papers_train):,}  "
          f"edges: {len(train_edges):,}", flush=True)
    print(f"[phantom] test  authors: {len(test_active_keys):,}  "
          f"edges: {len(test_edges):,}", flush=True)
    print(f"[phantom] centroid weights: {n_weighted_papers:,} train papers, "
          f"{n_missing_cites:,} missing from citation snapshot (weighted c=0)",
          flush=True)

    # Eligible eval authors: have a train embedding centroid AND >= MIN_PAPERS_TRAIN
    eval_keys = sorted(
        k for k in author_sum.keys()
        if author_papers_train[k] >= MIN_PAPERS_TRAIN
    )
    key_idx = {k: i for i, k in enumerate(eval_keys)}
    print(f"[phantom] eligible eval authors (>= {MIN_PAPERS_TRAIN} train papers + "
          f"centroid): {len(eval_keys):,}", flush=True)

    # Author centroid matrix (normalized)
    A = np.stack([
        author_sum[k] / max(author_wsum[k], 1e-8) for k in eval_keys
    ]).astype(np.float32)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    print(f"[phantom] author centroids: {A.shape}", flush=True)

    # ----------------------------------------------------------------
    # Build train coauthor graph as an adjacency dict; BFS up to 3 hops
    # from each eval author. Only count hops through nodes present in the graph.
    # ----------------------------------------------------------------
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in train_edges:
        adj[a].add(b)
        adj[b].add(a)
    print(f"[phantom] train graph: {len(adj):,} nodes (including non-eval)",
          flush=True)

    t0 = time.time()
    # For each eval key, compute BFS up to BFS_CUTOFF; store only eval-key→dist.
    eval_set = set(eval_keys)

    def bfs_distances(src: str, cutoff: int) -> dict[str, int]:
        dist: dict[str, int] = {src: 0}
        frontier = [src]
        for d in range(1, cutoff + 1):
            nxt: list[str] = []
            for u in frontier:
                for v in adj.get(u, ()):  # type: ignore[arg-type]
                    if v not in dist:
                        dist[v] = d
                        nxt.append(v)
            if not nxt:
                break
            frontier = nxt
        return dist

    near_neighbors: dict[str, dict[str, int]] = {}
    for k in tqdm(eval_keys, desc="[phantom] BFS"):
        d = bfs_distances(k, BFS_CUTOFF)
        # Keep only distances to other eval keys (and only if < cutoff — we
        # look up distance <= BFS_CUTOFF, anything else is treated as "far").
        near_neighbors[k] = {kk: dd for kk, dd in d.items()
                             if kk in eval_set and kk != k}
    print(f"[phantom] BFS done in {time.time()-t0:.1f}s", flush=True)

    n_eval = len(eval_keys)

    # ----------------------------------------------------------------
    # SHARED exclusion structure: excl(i) = self + train-dist <= min_hops - 1,
    # as eval-column index arrays. Everything (phantom masking, baseline
    # samplers, graph rankers) derives eligibility from this ONE object.
    # ----------------------------------------------------------------
    excl_cols = build_exclusion_columns(eval_keys, near_neighbors,
                                        PHANTOM_MIN_HOPS)
    excl_sets_int: list[set[int]] = [set(map(int, c)) for c in excl_cols]
    excl_of: dict[str, set[str]] = {
        eval_keys[i]: {eval_keys[j] for j in excl_cols[i]}
        for i in range(n_eval)
    }
    pool_sizes = np.array([n_eval - len(c) for c in excl_cols], dtype=np.int64)
    worst = int(pool_sizes.argmin())
    if pool_sizes[worst] <= TOP_K + TIE_BUF:
        print(f"[phantom] FATAL: eligible pool for author "
              f"{eval_keys[worst]!r} has only {pool_sizes[worst]} members "
              f"(need > {TOP_K + TIE_BUF}); exact-K protocol impossible.",
              file=sys.stderr)
        raise SystemExit(2)
    print(f"[phantom] eligible pools: min={int(pool_sizes.min()):,}  "
          f"p50={int(np.median(pool_sizes)):,}  "
          f"mean={float(pool_sizes.mean()):,.1f}", flush=True)

    # ----------------------------------------------------------------
    # Dual-branch chunked kNN over author centroids:
    #   legacy branch: top-(K+TIE_BUF) BEFORE exclusion masking (bridge)
    #   cf branch:     top-(K+TIE_BUF) AFTER masking excl(i) to float32 min
    # Exact masking — no top-M oversampling heuristic.
    # ----------------------------------------------------------------
    t0 = time.time()
    A_t = torch.from_numpy(A).to(DEVICE)
    CHUNK = 2048
    BUF = TOP_K + TIE_BUF
    NEG = torch.finfo(torch.float32).min
    buf_sim_leg = np.zeros((n_eval, BUF), dtype=np.float32)
    buf_idx_leg = np.zeros((n_eval, BUF), dtype=np.int64)
    buf_sim_cf = np.zeros((n_eval, BUF), dtype=np.float32)
    buf_idx_cf = np.zeros((n_eval, BUF), dtype=np.int64)
    with torch.no_grad():
        for i0 in range(0, n_eval, CHUNK):
            i1 = min(i0 + CHUNK, n_eval)
            q = A_t[i0:i1]
            sims = q @ A_t.T
            rloc = torch.arange(i1 - i0, device=sims.device)
            sims[rloc, rloc + i0] = NEG  # self-mask (both branches)
            # Legacy branch: pool-then-filter inputs (pre-exclusion top-k)
            vals, idxs = torch.topk(sims, BUF, dim=1)
            buf_sim_leg[i0:i1] = vals.cpu().numpy()
            buf_idx_leg[i0:i1] = idxs.cpu().numpy()
            # Exact exclusion masking, then constraint-first top-k
            lens = [len(excl_cols[i]) for i in range(i0, i1)]
            rows_np = np.repeat(np.arange(i1 - i0, dtype=np.int64), lens)
            cols_np = np.concatenate(
                [excl_cols[i] for i in range(i0, i1)]).astype(np.int64)
            rows_t = torch.from_numpy(rows_np).to(sims.device)
            cols_t = torch.from_numpy(cols_np).to(sims.device)
            sims[rows_t, cols_t] = NEG
            vals, idxs = torch.topk(sims, BUF, dim=1)
            buf_sim_cf[i0:i1] = vals.cpu().numpy()
            buf_idx_cf[i0:i1] = idxs.cpu().numpy()
    print(f"[phantom] dual top-{BUF} kNN in {time.time()-t0:.1f}s", flush=True)

    # Deterministic tie resolution on CPU; exact numpy recompute for any row
    # whose K-boundary tie group the buffer cannot certify.
    t0 = time.time()
    topk_sim_cf = np.zeros((n_eval, TOP_K), dtype=np.float32)
    topk_idx_cf = np.zeros((n_eval, TOP_K), dtype=np.int32)
    topk_sim_leg = np.zeros((n_eval, TOP_K), dtype=np.float32)
    topk_idx_leg = np.zeros((n_eval, TOP_K), dtype=np.int32)
    n_exact_cf = n_exact_leg = 0
    for i in range(n_eval):
        v, ix, needs = resolve_topk_buffer(buf_sim_cf[i], buf_idx_cf[i], TOP_K)
        if needs:
            n_exact_cf += 1
            row = (A @ A[i]).astype(np.float64)
            ix = select_constraint_first(row, excl_cols[i], TOP_K)
            v = row[ix]
        topk_sim_cf[i] = v.astype(np.float32)
        topk_idx_cf[i] = ix.astype(np.int32)
        v, ix, needs = resolve_topk_buffer(buf_sim_leg[i], buf_idx_leg[i],
                                           TOP_K)
        if needs:
            n_exact_leg += 1
            row = (A @ A[i]).astype(np.float64)
            row[i] = -np.inf
            order0 = np.argsort(-row, kind="stable")[:TOP_K]
            ix, v = order0, row[order0]
        topk_sim_leg[i] = v.astype(np.float32)
        topk_idx_leg[i] = ix.astype(np.int32)
    del buf_sim_leg, buf_idx_leg, buf_sim_cf, buf_idx_cf
    print(f"[phantom] tie resolution in {time.time()-t0:.1f}s "
          f"(exact recomputes: cf={n_exact_cf}, legacy={n_exact_leg})",
          flush=True)

    # No masked sentinel may leak into a selection (cosines are >= -1; a leak
    # means some author had < TOP_K eligible candidates despite the pool check).
    assert float(topk_sim_cf.min()) > -2.0, (
        "masked sentinel leaked into constraint-first selection")
    assert topk_idx_cf.shape == (n_eval, TOP_K)

    # ----------------------------------------------------------------
    # Helpers: did pair realize in test?  Is pair a train coauthor?
    # ----------------------------------------------------------------
    def _pair_key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def is_near_train(a: str, b: str, max_hops: int = PHANTOM_MIN_HOPS - 1) -> bool:
        """True if a and b are within max_hops in the train coauthor graph."""
        d = near_neighbors.get(a, {}).get(b)
        return d is not None and d <= max_hops

    # ----------------------------------------------------------------
    # Runtime spot-check: recompute SPOT_CHECK_N random authors exactly in
    # numpy and verify the GPU constraint-first selection (abort on failure).
    # ----------------------------------------------------------------
    t0 = time.time()
    sc_rng = np.random.default_rng(SEED)
    sc_idx = sc_rng.choice(n_eval, size=min(SPOT_CHECK_N, n_eval),
                           replace=False)
    for i in sc_idx:
        i = int(i)
        anchor = eval_keys[i]
        gpu_idx = topk_idx_cf[i]
        if len(set(gpu_idx.tolist())) != TOP_K:
            print(f"[phantom] SPOT-CHECK FAIL: {anchor} has "
                  f"{len(set(gpu_idx.tolist()))} != {TOP_K} candidates",
                  file=sys.stderr)
            raise SystemExit(2)
        for j in gpu_idx:
            nb = eval_keys[int(j)]
            if nb == anchor or is_near_train(anchor, nb):
                print(f"[phantom] SPOT-CHECK FAIL: {anchor} -> {nb} violates "
                      f"d >= {PHANTOM_MIN_HOPS}", file=sys.stderr)
                raise SystemExit(2)
        row = (A @ A[i]).astype(np.float64)
        ref_idx = select_constraint_first(row, excl_cols[i], TOP_K)
        ref_vals = np.sort(row[ref_idx])[::-1]
        gpu_vals = np.sort(topk_sim_cf[i].astype(np.float64))[::-1]
        if not np.allclose(gpu_vals, ref_vals, atol=1e-5):
            print(f"[phantom] SPOT-CHECK FAIL: {anchor} GPU sims deviate "
                  f">1e-5 from numpy reference", file=sys.stderr)
            raise SystemExit(2)
    spot_check = {"n_authors": int(len(sc_idx)), "passed": True}
    print(f"[phantom] spot-check passed ({len(sc_idx)} authors, "
          f"{time.time()-t0:.1f}s)", flush=True)

    # ----------------------------------------------------------------
    # Candidate lists.
    #   cf:      exactly TOP_K per author, straight from the cf arrays.
    #   legacy:  as-submitted pool-then-filter truncation (<= TOP_K).
    # legacy(K) is a prefix-subset of cf(K) (filtering a sorted prefix
    # preserves order), so legacy hits <= cf hits per K — asserted later.
    # ----------------------------------------------------------------
    phantom_cf: dict[str, list[str]] = {}
    for i, k in enumerate(eval_keys):
        phantom_cf[k] = [eval_keys[int(j)] for j in topk_idx_cf[i]]

    phantom_leg: dict[str, list[str]] = {}
    for i, k in enumerate(eval_keys):
        chosen: list[str] = []
        for j in range(TOP_K):
            nb_key = eval_keys[int(topk_idx_leg[i, j])]
            assert nb_key != k  # self was masked before the legacy topk
            if is_near_train(k, nb_key):
                continue  # already close; doesn't count as phantom
            chosen.append(nb_key)
            if len(chosen) >= TOP_K:
                break
        phantom_leg[k] = chosen
    total_leg_cands = sum(len(v) for v in phantom_leg.values())
    mean_excluded_legacy = (n_eval * TOP_K - total_leg_cands) / n_eval

    # Realized future partners (only pairs where A is active in train)
    realized_future: dict[str, set[str]] = defaultdict(set)
    for (a, b) in test_edges:
        if a in eval_set and b in eval_set:
            # exclude near-train links
            if not is_near_train(a, b):
                realized_future[a].add(b)
                realized_future[b].add(a)

    n_with_any_realized = sum(1 for r in realized_future.values() if r)
    print(f"[phantom] authors with >=1 realized phantom partner: "
          f"{n_with_any_realized:,} / {len(eval_keys):,}", flush=True)

    # ----------------------------------------------------------------
    # Graph structures shared by config_degree / graph_ppr / same_community
    # ----------------------------------------------------------------
    deg_eval = np.array([float(len(adj.get(k, ()))) for k in eval_keys],
                        dtype=np.float64)
    n_zero_degree_eval = int((deg_eval == 0).sum())
    graph_nodes = sorted(adj)
    gidx = {n: i for i, n in enumerate(graph_nodes)}

    ppr_enabled = GRAPH_BASELINES in ("full", "headline")
    ppr_lists: dict[str, list[str]] = {}
    ppr_nfill: dict[str, int] = {}
    ppr_cfg: dict = {}
    if ppr_enabled:
        t0 = time.time()
        csr = csr_from_edges(graph_nodes, train_edges)
        eval_gidx = np.array([gidx.get(k, -1) for k in eval_keys],
                             dtype=np.int64)
        in_graph = np.where(eval_gidx >= 0)[0]
        no_graph = np.where(eval_gidx < 0)[0]
        # Deterministic degree-rank fill: (-train_degree, author_key);
        # eval_keys is sorted, so a stable argsort of -degree gives the
        # author_key tiebreak for free. Graph-only information — steel-mans
        # the baseline instead of diluting it with random fill.
        fill_order = [int(x) for x in np.argsort(-deg_eval, kind="stable")]
        n_degree_filled = 0
        eval_rows_mask = eval_gidx >= 0
        eval_rows = eval_gidx[eval_rows_mask]
        print(f"[phantom] graph_ppr: {len(in_graph):,} anchors in graph, "
              f"{len(no_graph):,} fill-only ...", flush=True)
        zeros = np.zeros(n_eval, dtype=np.float64)
        for c0 in tqdm(range(0, len(in_graph), PPR_CHUNK),
                       desc="[phantom] PPR"):
            block = in_graph[c0:c0 + PPR_CHUNK]
            X = ppr_scores(csr, eval_gidx[block], alpha=PPR_ALPHA,
                           tol=PPR_TOL, max_iter=PPR_MAX_ITER)
            for bi, ei in enumerate(block):
                ei = int(ei)
                scores_eval = zeros.copy()
                scores_eval[eval_rows_mask] = X[eval_rows, bi]
                chosen, nf = rank_top_k_eligible(
                    scores_eval, np.arange(n_eval), excl_sets_int[ei],
                    TOP_K, fill_order)
                assert len(chosen) == TOP_K
                anchor = eval_keys[ei]
                ppr_lists[anchor] = [eval_keys[j] for j in chosen]
                ppr_nfill[anchor] = nf
                n_degree_filled += nf
        for ei in no_graph:
            ei = int(ei)
            chosen, nf = rank_top_k_eligible(
                zeros, np.arange(n_eval), excl_sets_int[ei], TOP_K,
                fill_order)
            assert len(chosen) == TOP_K
            anchor = eval_keys[ei]
            ppr_lists[anchor] = [eval_keys[j] for j in chosen]
            ppr_nfill[anchor] = nf
            n_degree_filled += nf
        ppr_cfg = {
            "alpha": PPR_ALPHA, "tol": PPR_TOL, "max_iter": PPR_MAX_ITER,
            "backend": "scipy", "fill": "train_degree_rank",
            "n_degree_filled": int(n_degree_filled),
            "fill_fraction": float(n_degree_filled / (n_eval * TOP_K)),
            "n_anchors_no_train_degree": int(len(no_graph)),
        }
        print(f"[phantom] graph_ppr ranked in {time.time()-t0:.1f}s "
              f"(fill_fraction={ppr_cfg['fill_fraction']:.4f})", flush=True)

    # same_community pools (flag-gated; graceful degradation without leidenalg)
    sc_enabled = False
    sc_pool: dict[str, list[str]] = {}
    sc_cfg: dict = {"enabled": False}
    if GRAPH_BASELINES == "full":
        try:
            import igraph as ig
            import leidenalg
            t0 = time.time()
            g = ig.Graph(n=len(graph_nodes),
                         edges=[(gidx[a], gidx[b]) for a, b in train_edges])
            part = leidenalg.find_partition(
                g, leidenalg.ModularityVertexPartition, seed=SEED)
            membership = part.membership
            comm_eval_members: dict[int, list[str]] = defaultdict(list)
            comm_of_eval: dict[str, int] = {}
            for k in eval_keys:
                gi = gidx.get(k)
                if gi is None:
                    continue
                cid = membership[gi]
                comm_of_eval[k] = cid
                comm_eval_members[cid].append(k)
            for k in eval_keys:
                cid = comm_of_eval.get(k)
                if cid is None:
                    sc_pool[k] = []
                    continue
                excl = excl_of[k]
                sc_pool[k] = [m for m in comm_eval_members[cid]
                              if m not in excl]
            sc_enabled = True
            sc_cfg = {"enabled": True,
                      "n_communities": int(len(part)),
                      "algorithm": f"leiden_modularity_seed{SEED}"}
            print(f"[phantom] same_community: {len(part):,} communities in "
                  f"{time.time()-t0:.1f}s", flush=True)
        except Exception as e:  # leidenalg/igraph missing or failed
            print(f"[phantom] same_community disabled ({e})", flush=True)
            sc_cfg = {"enabled": False}

    # ----------------------------------------------------------------
    # Stochastic samplers. All draw from the SAME eligible pool (excl_of) and
    # always return exactly k_need via the shared random completion.
    # ----------------------------------------------------------------
    pa_weights = np.array(
        [author_papers_train.get(k, 0) for k in eval_keys], dtype=np.float64)
    pa_cdf = degree_cdf(pa_weights)

    cd_cdf = degree_cdf(deg_eval)  # configuration-model conditional weights

    # Venue → authors index, then per-anchor pool (union of same-venue authors),
    # precomputed ONCE.
    venue_to_keys: dict[str, list[str]] = defaultdict(list)
    for k in eval_keys:
        for v in author_venues_train.get(k, set()):
            venue_to_keys[v].append(k)
    print(f"[phantom] precomputing same-venue pools for {len(eval_keys):,} "
          f"authors ...", flush=True)
    t0 = time.time()
    sv_pool: dict[str, list[str]] = {}
    for k in eval_keys:
        union: set[str] = set()
        for v in author_venues_train.get(k, set()):
            union.update(venue_to_keys.get(v, ()))
        union.difference_update(excl_of[k])  # shared exclusion (self included)
        sv_pool[k] = sorted(union)
    print(f"[phantom] same-venue pools built in {time.time()-t0:.1f}s "
          f"(median pool={int(np.median([len(v) for v in sv_pool.values()])):,})",
          flush=True)

    sv_padded = {"n": 0}  # pooled same_venue padding diagnostic (reset per K)

    def _fill_random(anchor: str, k_need: int, already, rng) -> list[str]:
        """Exactly k_need eligible authors: rejection sampling, then a
        deterministic completion from an rng-shuffled index list."""
        excl = excl_of[anchor]
        out: list[str] = []
        seen = set(excl)
        seen.update(already)
        tries = 0
        while len(out) < k_need and tries < k_need * 40:
            cand = eval_keys[rng.randrange(0, n_eval)]
            if cand not in seen:
                out.append(cand)
                seen.add(cand)
            tries += 1
        if len(out) < k_need:
            order = list(range(n_eval))
            rng.shuffle(order)
            for j in order:
                cand = eval_keys[j]
                if cand not in seen:
                    out.append(cand)
                    seen.add(cand)
                    if len(out) >= k_need:
                        break
        return out

    def draw_random(anchor: str, k_need: int, rng) -> list[str]:
        return _fill_random(anchor, k_need, (), rng)

    def draw_pref_attach(anchor: str, k_need: int, rng) -> list[str]:
        if pa_cdf is None:
            return draw_random(anchor, k_need, rng)
        out = draw_from_cdf(rng, pa_cdf, eval_keys, excl_of[anchor], k_need)
        if len(out) < k_need:
            out += _fill_random(anchor, k_need - len(out), out, rng)
        return out

    def draw_config_degree(anchor: str, k_need: int, rng) -> list[str]:
        """Degree-preserving configuration-model sampler: with the anchor
        fixed, P(stub attaches to c) ∝ deg_train(c); zero-degree authors have
        no stubs and are never drawn (weight 0)."""
        if cd_cdf is None:
            return draw_random(anchor, k_need, rng)
        out = draw_from_cdf(rng, cd_cdf, eval_keys, excl_of[anchor], k_need)
        if len(out) < k_need:
            out += _fill_random(anchor, k_need - len(out), out, rng)
        return out

    def draw_same_venue(anchor: str, k_need: int, rng) -> list[str]:
        pool = sv_pool.get(anchor, [])
        if len(pool) < k_need:
            extras = _fill_random(anchor, k_need - len(pool), pool, rng)
            sv_padded["n"] += len(extras)
            return list(pool) + extras
        return rng.sample(pool, k_need)

    def draw_same_community(anchor: str, k_need: int, rng) -> list[str]:
        pool = sc_pool.get(anchor, [])
        if len(pool) < k_need:
            extras = _fill_random(anchor, k_need - len(pool), pool, rng)
            return list(pool) + extras
        return rng.sample(pool, k_need)

    # ----------------------------------------------------------------
    # Metrics: deterministic rankers (single pass) + pooled stochastic draws.
    # ----------------------------------------------------------------
    det_lists: dict[str, dict[str, list[str]]] = {"phantom": phantom_cf}
    if ppr_enabled:
        det_lists["graph_ppr"] = ppr_lists
    drawers = {
        "random": draw_random,
        "pref_attach": draw_pref_attach,
        "config_degree": draw_config_degree,
        "same_venue": draw_same_venue,
    }
    if sc_enabled:
        drawers["same_community"] = draw_same_community
    method_seed_order = list(drawers.keys())

    EMPTY: set[str] = set()
    results: dict[str, dict] = {}
    metrics_legacy_phantom: dict[str, dict] = {}
    per_author_hits_by: dict[tuple[str, int], np.ndarray] = {}

    for K in EVAL_KS:
        print(f"[phantom] eval K={K} ...", flush=True)
        metrics_K: dict[str, dict] = {}

        for mname, lists in det_lists.items():
            pa_hits = np.zeros(n_eval, dtype=np.int64)
            recalls: list[float] = []
            g_hits = g_preds = 0  # graph_ppr non-filled ("graph only") slots
            for i, anchor in enumerate(eval_keys):
                cand = lists[anchor][:K]
                assert len(cand) == K, (mname, anchor, len(cand))
                realized_set = realized_future.get(anchor, EMPTY)
                h = len(set(cand) & realized_set)
                pa_hits[i] = h
                if realized_set:
                    recalls.append(h / len(realized_set))
                if mname == "graph_ppr":
                    n_scored = max(0, min(K, TOP_K - ppr_nfill[anchor]))
                    g_preds += n_scored
                    g_hits += len(set(cand[:n_scored]) & realized_set)
            hits = int(pa_hits.sum())
            preds = n_eval * K
            micro_p = hits / preds
            macro_p = float(np.mean(pa_hits / K))
            assert abs(micro_p - macro_p) < 1e-12, (mname, K)
            macro_r = float(np.mean(recalls)) if recalls else 0.0
            m = {
                "hits": hits, "predictions": preds,
                "micro_precision": micro_p,
                "macro_precision": macro_p,
                "macro_recall": macro_r,
                "n_authors_scored": n_eval,
                "n_draws": 1,
                "bootstrap_ci95": _bootstrap_ci95(pa_hits, K),
            }
            if mname == "graph_ppr":
                m["micro_precision_graph_only"] = (
                    g_hits / g_preds if g_preds else None)
                m["n_graph_only_predictions"] = int(g_preds)
            metrics_K[mname] = m
            per_author_hits_by[(mname, K)] = pa_hits
            print(f"  {mname:>14s}  hits={hits:>6,}  micro_P={micro_p:.5f}  "
                  f"macro_R={macro_r:.4f}", flush=True)

        # Legacy bridge (rng-free; as-submitted pool-then-filter semantics)
        lhits = lpreds = lscored = 0
        lcand_total = 0
        for anchor in eval_keys:
            cand = phantom_leg[anchor][:K]
            lcand_total += len(cand)
            if not cand:  # as-submitted: zero-candidate authors are skipped
                continue
            lscored += 1
            lpreds += len(cand)
            lhits += len(set(cand) & realized_future.get(anchor, EMPTY))
        assert lhits <= metrics_K["phantom"]["hits"], (
            "legacy hits exceed constraint-first hits — prefix property broken")
        metrics_legacy_phantom[f"K={K}"] = {
            "hits": int(lhits),
            "predictions": int(lpreds),
            "micro_precision": (lhits / lpreds) if lpreds else 0.0,
            "micro_precision_over_nK": lhits / (n_eval * K),
            "n_authors_scored": int(lscored),
            "mean_candidates_per_author": lcand_total / n_eval,
        }

        for mname, drawer in drawers.items():
            pooled = np.zeros(n_eval, dtype=np.int64)
            per_draw_hits: list[int] = []
            per_draw_micro: list[float] = []
            recalls = []
            if mname == "same_venue":
                sv_padded["n"] = 0
            for draw in range(BASELINE_DRAWS):
                drng = random.Random(f"{SEED}:{mname}:{K}:{draw}")
                dh = 0
                for i, anchor in enumerate(eval_keys):
                    cand = drawer(anchor, K, drng)
                    assert len(cand) == K, (mname, anchor, len(cand))
                    realized_set = realized_future.get(anchor, EMPTY)
                    h = len(set(cand) & realized_set)
                    pooled[i] += h
                    dh += h
                    if realized_set:
                        recalls.append(h / len(realized_set))
                per_draw_hits.append(int(dh))
                per_draw_micro.append(dh / (n_eval * K))
            hits = int(sum(per_draw_hits))
            preds = BASELINE_DRAWS * n_eval * K
            micro_p = hits / preds
            macro_p = float(np.mean(pooled / (BASELINE_DRAWS * K)))
            assert abs(micro_p - macro_p) < 1e-12, (mname, K)
            sd = (float(np.std(per_draw_micro, ddof=1))
                  if len(per_draw_micro) > 1 else 0.0)
            m = {
                "hits": hits, "predictions": preds,
                "micro_precision": micro_p,
                "macro_precision": macro_p,
                "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
                "n_authors_scored": n_eval,
                "n_draws": BASELINE_DRAWS,
                "predictions_per_draw": n_eval * K,
                "per_draw_hits": per_draw_hits,
                "hits_per_draw": per_draw_hits,           # alias (spec Task 3)
                "micro_precision_sd": sd,
                "micro_precision_std": sd,                # alias (spec Task 3)
                "hits_mean_per_draw": float(np.mean(per_draw_hits)),
                "bootstrap_ci95": _bootstrap_ci95(pooled, BASELINE_DRAWS * K),
            }
            if mname == "same_venue":
                m["n_padded"] = int(sv_padded["n"])
            metrics_K[mname] = m
            per_author_hits_by[(mname, K)] = pooled
            print(f"  {mname:>14s}  pooled_hits={hits:>6,}  "
                  f"micro_P={micro_p:.5f}  sd={sd:.5f}", flush=True)

        # Lift of phantom over EVERY non-phantom method (None when base == 0;
        # float('inf') would crash json.dumps(allow_nan=False)).
        phantom_m = metrics_K["phantom"]["micro_precision"]
        for mname in metrics_K:
            if mname == "phantom":
                continue
            base = metrics_K[mname]["micro_precision"]
            metrics_K[mname]["lift_phantom_vs"] = (
                phantom_m / base if base > 0 else None)
        results[f"K={K}"] = metrics_K

    # ----------------------------------------------------------------
    # Configuration-model NULL on the realized eligible test graph E*:
    # analytic Chung-Lu expectation + Monte Carlo constrained stub matching.
    # Reported per deterministic method/K as a null distribution, never as a
    # predictor row.
    # ----------------------------------------------------------------
    s_map = {a: len(v) for a, v in realized_future.items() if v}
    m_star = sum(s_map.values()) // 2
    print(f"[phantom] config null: m*={m_star:,} eligible test edges, "
          f"R={NULL_REWIRES}", flush=True)
    null_config = {
        "m_eligible_test_edges": int(m_star),
        "n_authors_with_realized": int(n_with_any_realized),
        "rewires": int(NULL_REWIRES),
        "stub_discard_rate_mean": None,
        "seed_scheme": "PCG64(SEED+777+r)",
    }
    if m_star > 0:
        # Pre-encode each deterministic method/K prediction list as canonical
        # pair codes (directed multiplicity preserved).
        def _codes(lists: dict[str, list[str]], K: int) -> np.ndarray:
            cs = []
            for anchor, lst in lists.items():
                ia = key_idx[anchor]
                for c in lst[:K]:
                    ic = key_idx[c]
                    lo, hi = (ia, ic) if ia < ic else (ic, ia)
                    cs.append(lo * n_eval + hi)
            return np.sort(np.array(cs, dtype=np.int64))

        method_codes = {(mn, K): _codes(lists, K)
                        for mn, lists in det_lists.items() for K in EVAL_KS}

        def _count_hits(codes: np.ndarray, edges_sorted: np.ndarray) -> int:
            if len(edges_sorted) == 0 or len(codes) == 0:
                return 0
            pos = np.searchsorted(edges_sorted, codes)
            ok = pos < len(edges_sorted)
            match = np.zeros(len(codes), dtype=bool)
            match[ok] = edges_sorted[pos[ok]] == codes[ok]
            return int(match.sum())

        # Analytic Chung-Lu expectation per method/K
        analytic = {
            (mn, K): chung_lu_expected_hits(
                {a: lst[:K] for a, lst in lists.items()}, s_map, m_star)
            for mn, lists in det_lists.items() for K in EVAL_KS
        }

        mc_hits: dict[tuple[str, int], list[int]] = defaultdict(list)
        discard_rates: list[float] = []
        if NULL_REWIRES > 0:
            stubs = np.concatenate([
                np.full(s, key_idx[a], dtype=np.int64)
                for a, s in sorted(s_map.items())])

            def _ineligible(ai: int, bi: int) -> bool:
                return is_near_train(eval_keys[ai], eval_keys[bi])

            t0 = time.time()
            for r_i in tqdm(range(NULL_REWIRES), desc="[phantom] null MC"):
                gen = np.random.Generator(np.random.PCG64(SEED + 777 + r_i))
                accepted, n_disc = stub_match_rewire(stubs, _ineligible, gen)
                discard_rates.append(n_disc / len(stubs))
                edge_codes = np.sort(np.array(
                    [lo * n_eval + hi for (lo, hi) in accepted],
                    dtype=np.int64))
                for key, codes in method_codes.items():
                    mc_hits[key].append(_count_hits(codes, edge_codes))
            disc_mean = float(np.mean(discard_rates))
            null_config["stub_discard_rate_mean"] = disc_mean
            if disc_mean > 0.01:
                print(f"[phantom] WARNING: stub discard rate "
                      f"{disc_mean:.3%} > 1% — consider double-edge-swap "
                      f"MCMC for the null", flush=True)
            print(f"[phantom] null MC done in {time.time()-t0:.1f}s "
                  f"(discard rate {disc_mean:.4%})", flush=True)

        for mn in det_lists:
            for K in EVAL_KS:
                n_pred = n_eval * K
                h_obs = results[f"K={K}"][mn]["hits"]
                exp_h = analytic[(mn, K)]
                blk: dict = {
                    "expected_hits_analytic": float(exp_h),
                    "expected_micro_precision_analytic": float(exp_h / n_pred),
                    "n_rewires": int(NULL_REWIRES),
                }
                hs = mc_hits.get((mn, K))
                if hs:
                    arr = np.array(hs, dtype=np.float64)
                    mean_h = float(arr.mean())
                    std_h = (float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)
                    mc_micro = mean_h / n_pred
                    blk.update({
                        "mc_mean_hits": mean_h,
                        "mc_std_hits": std_h,
                        "mc_mean_micro_precision": mc_micro,
                        "z_score": ((h_obs - mean_h) / std_h)
                        if std_h > 0 else None,
                        "p_one_sided": float(
                            (1 + int((arr >= h_obs).sum()))
                            / (len(arr) + 1)),
                        "lift_vs_null": (
                            (h_obs / n_pred) / mc_micro
                            if mc_micro > 0 else None),
                    })
                    se = std_h / max(np.sqrt(len(arr)), 1.0)
                    if se > 0 and abs(exp_h - mean_h) > 3 * se:
                        print(f"[phantom] WARNING: analytic E[hits]="
                              f"{exp_h:.1f} outside 3 SE of MC mean "
                              f"{mean_h:.1f} ({mn}, K={K})", flush=True)
                else:
                    blk.update({
                        "mc_mean_hits": None, "mc_std_hits": None,
                        "mc_mean_micro_precision": None, "z_score": None,
                        "p_one_sided": None, "lift_vs_null": None,
                    })
                results[f"K={K}"][mn]["config_null"] = blk

    # ----------------------------------------------------------------
    # Phantom vs graph_ppr overlap diagnostic: are semantic and structural
    # signals redundant or complementary?
    # ----------------------------------------------------------------
    phantom_graph_overlap: dict[str, dict] = {}
    if ppr_enabled:
        for K in EVAL_KS:
            jac: list[float] = []
            hit_overlap = 0
            for anchor in eval_keys:
                sp = set(phantom_cf[anchor][:K])
                sg = set(ppr_lists[anchor][:K])
                jac.append(len(sp & sg) / len(sp | sg))
                realized_set = realized_future.get(anchor, EMPTY)
                if realized_set:
                    hit_overlap += len((sp & realized_set)
                                       & (sg & realized_set))
            phantom_graph_overlap[f"K={K}"] = {
                "mean_jaccard": float(np.mean(jac)),
                "hit_overlap": int(hit_overlap),
            }

    # ----------------------------------------------------------------
    # Calibration: single source of truth = the constraint-first arrays.
    # Exactly n_eval * TOP_K pairs.
    # ----------------------------------------------------------------
    print("[phantom] building similarity-calibration ...", flush=True)
    all_pairs_sim: list[float] = []
    all_pairs_realized: list[int] = []
    for i, k in enumerate(eval_keys):
        for j in range(TOP_K):
            nb = eval_keys[int(topk_idx_cf[i, j])]
            assert nb != k and not is_near_train(k, nb), (
                "near-train pair leaked through the constraint-first mask")
            realized_pair = 1 if _pair_key(k, nb) in test_edges else 0
            all_pairs_sim.append(float(topk_sim_cf[i, j]))
            all_pairs_realized.append(realized_pair)
    sims_arr = np.asarray(all_pairs_sim, dtype=np.float32)
    rel_arr = np.asarray(all_pairs_realized, dtype=np.int32)
    assert len(sims_arr) == n_eval * TOP_K
    print(f"[phantom] calibration pairs: {len(sims_arr):,}  "
          f"positive: {int(rel_arr.sum()):,}", flush=True)

    # 10 equal-frequency quantiles
    N_BINS = 10
    if len(sims_arr) >= N_BINS:
        # argsort, bin by rank
        order = np.argsort(sims_arr)
        bin_size = len(sims_arr) / N_BINS
        calib_rows = []
        for b in range(N_BINS):
            lo = int(b * bin_size)
            hi = int((b + 1) * bin_size) if b < N_BINS - 1 else len(sims_arr)
            idx = order[lo:hi]
            if len(idx) == 0:
                continue
            calib_rows.append({
                "bucket":      b,
                "sim_lo":      float(sims_arr[idx].min()),
                "sim_hi":      float(sims_arr[idx].max()),
                "sim_median":  float(np.median(sims_arr[idx])),
                "n_pairs":     int(len(idx)),
                "n_realized":  int(rel_arr[idx].sum()),
                "realize_rate": float(rel_arr[idx].mean()),
            })
    else:
        calib_rows = []

    # ----------------------------------------------------------------
    # Case studies: highest-sim phantoms (constraint-first arrays) that
    # realized in test. Schema unchanged (train_dist: 3 or None = ">= 4").
    # ----------------------------------------------------------------
    cases: list[dict] = []
    for i, k in enumerate(eval_keys):
        for j in range(TOP_K):
            nb = eval_keys[int(topk_idx_cf[i, j])]
            assert nb != k and not is_near_train(k, nb)
            if _pair_key(k, nb) not in test_edges:
                continue
            cases.append({
                "a": k, "a_name": key_to_name.get(k) or k,
                "b": nb, "b_name": key_to_name.get(nb) or nb,
                "sim": float(topk_sim_cf[i, j]),
                "train_dist": near_neighbors.get(k, {}).get(nb, None),
            })
    cases.sort(key=lambda c: -c["sim"])
    cases = cases[:60]  # enough raw cases for ~20+ unique pairs after A/B dedup
    print(f"[phantom] found {len(cases)} realized-phantom cases (top-60 saved)",
          flush=True)

    # ----------------------------------------------------------------
    # Write output JSONs
    # ----------------------------------------------------------------
    out = {
        "config": {
            "train_cutoff_year":      TRAIN_CUTOFF_YEAR,
            "test_years":             [min(TEST_YEARS), max(TEST_YEARS)],
            "top_k":                  TOP_K,
            "phantom_min_hops":       PHANTOM_MIN_HOPS,
            "bfs_cutoff":             BFS_CUTOFF,
            "whiten_top_pc":          WHITEN_TOP_PC,
            "min_train_papers":       MIN_PAPERS_TRAIN,
            "seed":                   SEED,
            "n_eval_authors":         len(eval_keys),
            "n_train_edges":          len(train_edges),
            "n_test_edges":           len(test_edges),
            "n_authors_with_realized": n_with_any_realized,
            "train_whitening_pc_share": ev_ratio,
            # --- protocol descriptor (constraint-first correction) ---
            "protocol":                 "constraint_first",
            "k_applied_after_constraint": True,
            "exact_k_all_methods":      True,
            "matched_counts":           True,
            "n_anchors_skipped":        0,
            "legacy_included":          True,
            "n_baseline_draws":         BASELINE_DRAWS,
            "baseline_draws":           BASELINE_DRAWS,
            "bootstrap_B":              BOOTSTRAP_B,
            "tie_buffer":               TIE_BUF,
            "n_exact_tie_recomputes":   {"cf": int(n_exact_cf),
                                         "legacy": int(n_exact_leg)},
            "eligible_pool":            {"min": int(pool_sizes.min()),
                                         "p50": int(np.median(pool_sizes)),
                                         "mean": float(pool_sizes.mean())},
            "mean_excluded_in_top20_legacy": float(mean_excluded_legacy),
            "spot_check":               spot_check,
            "seed_scheme":  "random.Random(f'{SEED}:{method}:{K}:{draw}')",
            "method_seed_order":        method_seed_order,
            # --- graph baselines + null ---
            "graph_baselines":          GRAPH_BASELINES,
            "null_rewires":             NULL_REWIRES,
            "config_degree": {"weight": "train_coauthor_degree",
                              "n_zero_degree_eval": n_zero_degree_eval},
            "graph_ppr":                ppr_cfg if ppr_enabled else
                                        {"enabled": False},
            "same_community":           sc_cfg,
            # --- temporal-leakage fix (as-of-cutoff citation weights) ---
            "citation_weights": {
                "source": "openalex_counts_by_year_asof_cutoff",
                "path": str(cites_path),
                "cutoff_year": TRAIN_CUTOFF_YEAR,
                "n_weighted_papers": int(n_weighted_papers),
                "n_missing_snapshot": int(n_missing_cites),
            },
        },
        "metrics":       results,
        "metrics_legacy_phantom": metrics_legacy_phantom,
        "phantom_graph_overlap": phantom_graph_overlap,
        "null_config":   null_config,
        "calibration":   calib_rows,
        "cases":         cases,
    }
    out_dir = repo / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"phantom_eval{OUT_SUFFIX}.json"
    paper_name = f"_phantom_eval{OUT_SUFFIX}.json"
    (out_dir / out_name).write_text(
        json.dumps(out, indent=2, allow_nan=False))
    (repo / "paper" / "analysis" / paper_name).write_text(
        json.dumps(out, indent=2, allow_nan=False))
    print(f"[phantom] wrote data/processed/{out_name}  "
          f"and paper/analysis/{paper_name}", flush=True)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phantom-min-hops", type=int, default=PHANTOM_MIN_HOPS,
        help=f"Phantom = top-K semantic neighbor with train_dist >= K. "
             f"Default {PHANTOM_MIN_HOPS} (matches headline numbers). "
             f"Pass 2 for the website's looser definition.")
    parser.add_argument(
        "--out-suffix", type=str, default=None,
        help="Suffix for output JSON (e.g., '_k2'). Auto-derived from "
             "--phantom-min-hops when not given and value differs from default.")
    parser.add_argument(
        "--baseline-draws", type=int, default=BASELINE_DRAWS,
        help=f"Draws per stochastic baseline, pooled in the report "
             f"(default {BASELINE_DRAWS}).")
    parser.add_argument(
        "--null-rewires", type=int, default=NULL_REWIRES,
        help=f"Monte Carlo rewires for the configuration-model null "
             f"(default {NULL_REWIRES}; 0 disables the MC block, analytic "
             f"expectation is always reported).")
    parser.add_argument(
        "--graph-baselines", choices=("full", "headline", "off"),
        default=GRAPH_BASELINES,
        help="full = graph_ppr + same_community; headline = graph_ppr only; "
             "off = neither (drawers + null only).")
    parser.add_argument(
        "--citations-asof", type=str, default=None,
        help="Path to the as-of-cutoff citation snapshot parquet (default "
             "data/interim/citation_snapshots.parquet, built by "
             "scripts/fetch_citation_snapshots.py). REQUIRED to exist; "
             "there is no fallback to leaky current-snapshot counts.")
    args = parser.parse_args()
    PHANTOM_MIN_HOPS = args.phantom_min_hops
    if args.out_suffix is not None:
        OUT_SUFFIX = args.out_suffix
    elif args.phantom_min_hops != 3:
        OUT_SUFFIX = f"_k{args.phantom_min_hops}"
    BASELINE_DRAWS = args.baseline_draws
    NULL_REWIRES = args.null_rewires
    GRAPH_BASELINES = args.graph_baselines
    CITATIONS_ASOF = args.citations_asof
    sys.exit(main())
