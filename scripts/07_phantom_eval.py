#!/usr/bin/env python
"""§8 — Phantom-collaborator predictive test with temporal holdout.

Train cutoff: year <= 2019.  Test window: 2020 <= year <= 2025.
(2026 is dropped because the corpus snapshot captures only a partial year.)

For each author A active in the train period:
    phantom_K(A) := top-K semantic neighbors in the train-only embedding
                    whose train-period coauthor distance is >= PHANTOM_MIN_HOPS
                    (or infinite — i.e. different connected components).

    realized(A) := authors that A actually coauthored with in the test window.

    precision@K(A) := |phantom_K(A) ∩ realized(A)| / K
    recall@K(A)    := |phantom_K(A) ∩ realized(A)| / |realized(A) \\ train_coauthors(A)|

Baselines (same K):
    - random:         K authors drawn uniformly from train-period authors
    - pref-attach:    K authors drawn proportional to train-period paper count
    - same-venue:     K authors sharing >= 1 train-period venue with A, sampled
                      uniformly; falls back to random if A has no venue-mates.

All candidate sets exclude existing train coauthors (dist <= 2).

Outputs:
    data/processed/phantom_eval.json
    paper/analysis/_phantom_eval.json  (summary copy for prose quoting)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transport_atlas.process.authors import author_key as _raw_author_key

# ---- config ----
EMBED_DIR = Path(os.environ.get("EMBED_OUT", "/data2/chois/transport-atlas"))
TRAIN_CUTOFF_YEAR = 2019
TEST_YEARS = range(2020, 2026)  # 2020..2025 inclusive (exclude partial 2026)

MIN_PAPERS_TRAIN = 2
TOP_K = 20
EVAL_KS = (5, 10, 20)
PHANTOM_MIN_HOPS = 3
BFS_CUTOFF = 3
WHITEN_TOP_PC = 1
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def main() -> int:  # noqa: C901
    rng = random.Random(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    repo = Path(__file__).resolve().parents[1]
    print(f"[phantom] device={DEVICE}  cutoff<= {TRAIN_CUTOFF_YEAR}  "
          f"test={TEST_YEARS.start}..{TEST_YEARS.stop-1}", flush=True)

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
                    _c = r.get("cited_by_count")
                    cites = 0 if _c is None or (
                        isinstance(_c, float) and _c != _c) else int(_c or 0)
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

    # ----------------------------------------------------------------
    # Top-K semantic neighbors among eval authors.
    # ----------------------------------------------------------------
    t0 = time.time()
    A_t = torch.from_numpy(A).to(DEVICE)
    CHUNK = 2048
    topk_sim = np.zeros((len(eval_keys), TOP_K), dtype=np.float32)
    topk_idx = np.zeros((len(eval_keys), TOP_K), dtype=np.int32)
    with torch.no_grad():
        for i in range(0, len(eval_keys), CHUNK):
            q = A_t[i : i + CHUNK]
            sims = q @ A_t.T
            for r, gi in enumerate(range(i, min(i + CHUNK, len(eval_keys)))):
                sims[r, gi] = -1.0
            vals, idxs = torch.topk(sims, TOP_K, dim=1)
            topk_sim[i : i + CHUNK] = vals.cpu().numpy()
            topk_idx[i : i + CHUNK] = idxs.cpu().numpy()
    print(f"[phantom] top-{TOP_K} kNN in {time.time()-t0:.1f}s", flush=True)

    # ----------------------------------------------------------------
    # Helpers: did pair realize in test?  Is pair a train coauthor?
    # ----------------------------------------------------------------
    def _pair_key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def realized(a: str, b: str) -> bool:
        return _pair_key(a, b) in test_edges

    def is_near_train(a: str, b: str, max_hops: int = PHANTOM_MIN_HOPS - 1) -> bool:
        """True if a and b are within max_hops in the train coauthor graph."""
        d = near_neighbors.get(a, {}).get(b)
        return d is not None and d <= max_hops

    # ----------------------------------------------------------------
    # For each author, build:
    #   phantom_neighbors(K) — top-K semantic neighbors with train dist >= PHANTOM_MIN_HOPS
    #     (falls back to "unknown" distance, which means dist > BFS_CUTOFF; still a phantom)
    #   realized_future(a) — test-edge partners of a, excluding near-train coauthors
    # ----------------------------------------------------------------
    phantom_neighbors: dict[str, list[tuple[str, float]]] = {}
    for i, k in enumerate(eval_keys):
        chosen: list[tuple[str, float]] = []
        for j in range(TOP_K):
            nb_key = eval_keys[topk_idx[i, j]]
            if nb_key == k:
                continue
            if is_near_train(k, nb_key):
                continue  # already close; doesn't count as phantom
            chosen.append((nb_key, float(topk_sim[i, j])))
            if len(chosen) >= TOP_K:
                break
        phantom_neighbors[k] = chosen

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
    # Baselines: for each author, draw K candidate partners using:
    #   - random:      uniform over eval_keys, excluding near-train + self
    #   - pref-attach: weighted by train paper count
    #   - same-venue:  uniform over authors sharing >=1 venue, falling back
    #                  to random if empty.
    # Each baseline is a single random draw (seeded).
    # ----------------------------------------------------------------
    # Pre-compute pref-attach cumulative distribution once.  Sampling is then
    # O(log N) per draw via searchsorted instead of O(N) via np.random.choice.
    pa_weights = np.array(
        [author_papers_train.get(k, 0) for k in eval_keys], dtype=np.float64)
    pa_total = float(pa_weights.sum())
    pa_cdf = np.cumsum(pa_weights) / pa_total if pa_total > 0 else None

    # Venue → authors index, then per-anchor pool (union of same-venue authors),
    # precomputed ONCE as a numpy array.  Shuffling a fresh view per K-draw.
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
        union.discard(k)
        # drop near-train coauthors (distance <= PHANTOM_MIN_HOPS - 1)
        nn = near_neighbors.get(k, {})
        for kk, dd in nn.items():
            if dd <= PHANTOM_MIN_HOPS - 1:
                union.discard(kk)
        sv_pool[k] = list(union)
    print(f"[phantom] same-venue pools built in {time.time()-t0:.1f}s "
          f"(median pool={int(np.median([len(v) for v in sv_pool.values()])):,})",
          flush=True)

    # Precompute exclusion set (near-train coauthors) per anchor once.
    excl_of: dict[str, set[str]] = {}
    for k in eval_keys:
        excl = {k}
        nn = near_neighbors.get(k, {})
        for kk, dd in nn.items():
            if dd <= PHANTOM_MIN_HOPS - 1:
                excl.add(kk)
        excl_of[k] = excl

    n_eval = len(eval_keys)

    def draw_random(anchor: str, k_need: int) -> list[str]:
        excl = excl_of[anchor]
        out: list[str] = []
        seen = set(excl)
        tries = 0
        while len(out) < k_need and tries < k_need * 40:
            cand = eval_keys[rng.randrange(0, n_eval)]
            if cand not in seen:
                out.append(cand)
                seen.add(cand)
            tries += 1
        return out

    def draw_pref_attach(anchor: str, k_need: int) -> list[str]:
        if pa_cdf is None:
            return draw_random(anchor, k_need)
        excl = excl_of[anchor]
        out: list[str] = []
        seen = set(excl)
        tries = 0
        while len(out) < k_need and tries < k_need * 40:
            u = rng.random()
            cand_idx = int(np.searchsorted(pa_cdf, u, side="right"))
            if cand_idx >= n_eval:
                cand_idx = n_eval - 1
            cand = eval_keys[cand_idx]
            if cand not in seen:
                out.append(cand)
                seen.add(cand)
            tries += 1
        return out

    def draw_same_venue(anchor: str, k_need: int) -> list[str]:
        pool = sv_pool.get(anchor, [])
        if len(pool) < k_need:
            extras = draw_random(anchor, k_need - len(pool))
            return list(pool) + extras
        # reservoir-style pick: random sample without replacement via random.sample
        return rng.sample(pool, k_need)

    # ----------------------------------------------------------------
    # Compute precision@K, recall@K, lift for each K and each method.
    # ----------------------------------------------------------------
    results: dict[str, dict] = {}
    methods = {
        "phantom":    None,  # special
        "random":     draw_random,
        "pref_attach": draw_pref_attach,
        "same_venue": draw_same_venue,
    }

    for K in EVAL_KS:
        print(f"[phantom] eval K={K} ...", flush=True)
        metrics_K: dict[str, dict] = {}
        for mname, drawer in methods.items():
            hits = 0
            preds = 0
            per_author_p: list[float] = []
            per_author_r: list[float] = []
            for anchor in eval_keys:
                realized_set = realized_future.get(anchor, set())
                if mname == "phantom":
                    cand = [nb for nb, _s in phantom_neighbors.get(anchor, [])][:K]
                else:
                    cand = drawer(anchor, K) if drawer else []
                if not cand:
                    continue
                hit_set = set(cand) & realized_set
                hits += len(hit_set)
                preds += len(cand)
                per_author_p.append(len(hit_set) / len(cand))
                if realized_set:
                    per_author_r.append(len(hit_set) / len(realized_set))
            micro_p = hits / preds if preds else 0.0
            macro_p = float(np.mean(per_author_p)) if per_author_p else 0.0
            macro_r = float(np.mean(per_author_r)) if per_author_r else 0.0
            metrics_K[mname] = {
                "hits": int(hits), "predictions": int(preds),
                "micro_precision": micro_p,
                "macro_precision": macro_p,
                "macro_recall":    macro_r,
                "n_authors_scored": len(per_author_p),
            }
            print(f"  {mname:>12s}  hits={hits:>6,}  "
                  f"micro_P={micro_p:.4f}  macro_P={macro_p:.4f}  "
                  f"macro_R={macro_r:.4f}", flush=True)
        # Lift over each baseline
        phantom_m = metrics_K["phantom"]["micro_precision"]
        for mname in ("random", "pref_attach", "same_venue"):
            base = metrics_K[mname]["micro_precision"]
            metrics_K[mname]["lift_phantom_vs"] = (
                phantom_m / base if base > 0 else float("inf"))
        results[f"K={K}"] = metrics_K

    # ----------------------------------------------------------------
    # Calibration: for the top-20 semantic neighbors, bucket by similarity
    # and compute realized-rate per bucket.
    # ----------------------------------------------------------------
    print("[phantom] building similarity-calibration ...", flush=True)
    all_pairs_sim: list[float] = []
    all_pairs_realized: list[int] = []
    for i, k in enumerate(eval_keys):
        for j in range(TOP_K):
            nb = eval_keys[topk_idx[i, j]]
            if nb == k:
                continue
            if is_near_train(k, nb):
                continue
            realized_pair = 1 if _pair_key(k, nb) in test_edges else 0
            all_pairs_sim.append(float(topk_sim[i, j]))
            all_pairs_realized.append(realized_pair)
    sims_arr = np.asarray(all_pairs_sim, dtype=np.float32)
    rel_arr = np.asarray(all_pairs_realized, dtype=np.int32)
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
    # Case studies: 10 highest-sim phantoms that realized in test.
    # ----------------------------------------------------------------
    cases: list[dict] = []
    for i, k in enumerate(eval_keys):
        for j in range(TOP_K):
            nb = eval_keys[topk_idx[i, j]]
            if nb == k:
                continue
            if is_near_train(k, nb):
                continue
            if _pair_key(k, nb) not in test_edges:
                continue
            cases.append({
                "a": k, "a_name": key_to_name.get(k) or k,
                "b": nb, "b_name": key_to_name.get(nb) or nb,
                "sim": float(topk_sim[i, j]),
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
        },
        "metrics":       results,
        "calibration":   calib_rows,
        "cases":         cases,
    }
    out_dir = repo / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phantom_eval.json").write_text(
        json.dumps(out, indent=2, allow_nan=False))
    (repo / "paper" / "analysis" / "_phantom_eval.json").write_text(
        json.dumps(out, indent=2, allow_nan=False))
    print(f"[phantom] wrote data/processed/phantom_eval.json  "
          f"and paper/analysis/_phantom_eval.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
