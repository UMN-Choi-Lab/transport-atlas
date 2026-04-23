#!/usr/bin/env python
"""One-off: did the train-cutoff=2019 phantom predictor suggest
Choi (UMN) + Sun (MIT/McGill) as a future coauthor pair?

Rebuilds just enough of the scripts/07_phantom_eval.py logic to answer
the specific pair. Prints: train distance, pairwise similarity, rank of
Sun in Choi's top-20 phantom list (and vice versa), and whether they
coauthored in 2020-2025.
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from transport_atlas.process.authors import author_key as _raw_author_key  # noqa

EMBED_DIR = Path(os.environ.get("EMBED_OUT", "/data2/chois/transport-atlas"))
TRAIN_CUTOFF_YEAR = 2019
TEST_YEARS = range(2020, 2026)
WHITEN_TOP_PC = 1
TOP_K = 20
PHANTOM_MIN_HOPS = 3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_alias_map() -> dict:
    import yaml
    repo = Path(__file__).resolve().parents[2]
    pipe = yaml.safe_load((repo / "config" / "pipeline.yaml").read_text()) or {}
    mp = {}
    for a in pipe.get("author_aliases", []) or []:
        ids = a.get("openalex_ids") or []
        if len(ids) < 2:
            continue
        target = ids[0].lower()
        for other in ids[1:]:
            mp[other.lower()] = target
    auto = repo / "data" / "interim" / "author_aliases_auto.json"
    if auto.exists():
        for k, v in json.loads(auto.read_text()).items():
            mp.setdefault(k, v)
    return mp


def akey(a: dict, alias_map: dict) -> str:
    k = _raw_author_key(a)
    return alias_map.get(k, k) if k else k


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    np.random.seed(SEED); torch.manual_seed(SEED)

    authors_tbl = pd.read_parquet(repo / "data" / "interim" / "authors.parquet")

    # Find candidate keys for Choi Seongjin and Sun Lijun.
    def _match(name_sub: str) -> pd.DataFrame:
        m = authors_tbl["canonical_name"].str.contains(name_sub, na=False, case=False)
        return authors_tbl.loc[m, ["author_key", "canonical_name", "n_papers"]]\
                          .sort_values("n_papers", ascending=False)

    cand_choi = _match("choi, seongjin")
    cand_sun  = _match("sun, lijun")
    print("\n[candidates] Choi, Seongjin:")
    print(cand_choi.head(5).to_string(index=False))
    print("\n[candidates] Sun, Lijun:")
    print(cand_sun.head(5).to_string(index=False))

    if cand_choi.empty or cand_sun.empty:
        print("[!] could not locate one of the authors; aborting.")
        return 1
    choi_key = cand_choi.iloc[0]["author_key"]
    sun_key  = cand_sun.iloc[0]["author_key"]
    choi_name = cand_choi.iloc[0]["canonical_name"]
    sun_name  = cand_sun.iloc[0]["canonical_name"]
    print(f"\nChosen: choi={choi_key} ({choi_name}); sun={sun_key} ({sun_name})")

    # Load paper embeddings + papers table
    embd = pd.read_parquet(EMBED_DIR / "paper_embeddings.parquet")
    pid_to_row = {pid: i for i, pid in enumerate(embd["paper_id"].tolist())}
    E = np.stack(embd["emb"].tolist()).astype(np.float32)

    papers = pd.read_parquet(repo / "data" / "interim" / "papers.parquet")
    alias = _load_alias_map()

    # Train subset of rows
    train_rows = []
    for _, r in papers.iterrows():
        y = r.get("year")
        if y is None or pd.isna(y): continue
        if int(y) > TRAIN_CUTOFF_YEAR: continue
        row = pid_to_row.get(r["paper_id"])
        if row is not None:
            train_rows.append(row)
    E_train = E[train_rows]
    print(f"[embed] train papers: {len(train_rows):,}")

    # Whitening (train-only)
    Et = torch.from_numpy(E_train).to(DEVICE).float()
    mu = Et.mean(dim=0, keepdim=True); Et = Et - mu
    cov = (Et.T @ Et) / Et.shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    top = evecs[:, -WHITEN_TOP_PC:]
    Et = Et - (Et @ top) @ top.T
    std = Et.std(dim=0, keepdim=True) + 1e-8
    Et = Et / std
    E_train_w = Et.cpu().numpy().astype(np.float32)
    tr_of_global = {g: i for i, g in enumerate(train_rows)}

    # Accumulate train centroids for the two authors + active co-authorship sets
    sum_by = defaultdict(lambda: np.zeros(E.shape[1], dtype=np.float32))
    wsum_by = defaultdict(float)
    train_edges: set = set()
    test_edges: set = set()
    train_adj: dict = defaultdict(set)

    for _, r in papers.iterrows():
        y = r.get("year")
        if y is None or pd.isna(y): continue
        year = int(y)
        auths = r.get("authors")
        if auths is None or len(auths) == 0: continue
        keys = []
        for a in auths:
            if isinstance(a, dict):
                k = akey(a, alias)
                if k: keys.append(k)
        keys = list(dict.fromkeys(keys))
        if year <= TRAIN_CUTOFF_YEAR:
            row = pid_to_row.get(r["paper_id"])
            if row is not None:
                tr = tr_of_global.get(row)
                if tr is not None:
                    v = E_train_w[tr]
                    c = r.get("cited_by_count")
                    cites = 0 if c is None or (isinstance(c, float) and c != c) else int(c or 0)
                    w = 1.0 + np.log1p(cites)
                    for k in keys:
                        if k in (choi_key, sun_key):
                            sum_by[k] += w * v
                            wsum_by[k] += w
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    a, b = keys[i], keys[j]
                    if a == b: continue
                    train_adj[a].add(b); train_adj[b].add(a)
                    train_edges.add((a, b) if a < b else (b, a))
        elif year in TEST_YEARS:
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    a, b = keys[i], keys[j]
                    if a == b: continue
                    test_edges.add((a, b) if a < b else (b, a))

    if wsum_by[choi_key] == 0 or wsum_by[sun_key] == 0:
        print(f"[!] missing train centroid: "
              f"choi_wsum={wsum_by[choi_key]:.1f}  sun_wsum={wsum_by[sun_key]:.1f}")
        print("Choi may be too junior (few pre-2020 papers).")
        # Still proceed with test-edge check.

    def _norm(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    if wsum_by[choi_key] > 0 and wsum_by[sun_key] > 0:
        ca = _norm(sum_by[choi_key] / wsum_by[choi_key])
        sa = _norm(sum_by[sun_key] / wsum_by[sun_key])
        cos = float((ca * sa).sum())
        print(f"\n[sim] cos(Choi centroid, Sun centroid) = {cos:.4f}")
    else:
        cos = None

    # BFS distance in train coauthor graph up to 5 hops
    def bfs(src, cutoff=5):
        dist = {src: 0}
        frontier = [src]
        for d in range(1, cutoff+1):
            nxt = []
            for u in frontier:
                for v in train_adj.get(u, ()):
                    if v not in dist:
                        dist[v] = d; nxt.append(v)
            if not nxt: break
            frontier = nxt
        return dist

    d_choi = bfs(choi_key, cutoff=5)
    train_dist = d_choi.get(sun_key)
    print(f"[train-graph] distance(Choi, Sun) = "
          f"{'∞ (disconnected or >5 hops)' if train_dist is None else train_dist}")

    # Check test-edge
    pair = (choi_key, sun_key) if choi_key < sun_key else (sun_key, choi_key)
    realized = pair in test_edges
    print(f"[test-window 2020-2025] Choi-Sun coauthorship: {'YES' if realized else 'no'}")

    # Print Choi's top-20 semantic neighbors (among all authors with >=2 train papers)
    if wsum_by[choi_key] > 0:
        print("\n[rebuilding Choi's top-20 neighbors over all eligible authors...]")
        # Re-scan to collect centroids for all authors with >=2 train papers.
        npaps = defaultdict(int)
        sum2 = defaultdict(lambda: np.zeros(E.shape[1], dtype=np.float32))
        wsum2 = defaultdict(float)
        for _, r in papers.iterrows():
            y = r.get("year")
            if y is None or pd.isna(y): continue
            if int(y) > TRAIN_CUTOFF_YEAR: continue
            auths = r.get("authors")
            if auths is None or len(auths) == 0: continue
            row = pid_to_row.get(r["paper_id"])
            if row is None: continue
            tr = tr_of_global.get(row)
            if tr is None: continue
            v = E_train_w[tr]
            c = r.get("cited_by_count")
            cites = 0 if c is None or (isinstance(c, float) and c != c) else int(c or 0)
            w = 1.0 + np.log1p(cites)
            for a in auths:
                if not isinstance(a, dict): continue
                k = akey(a, alias)
                if not k: continue
                npaps[k] += 1
                sum2[k] += w * v
                wsum2[k] += w
        keep = [k for k, n in npaps.items() if n >= 2 and wsum2[k] > 0]
        if choi_key not in keep: keep.append(choi_key)
        A = np.stack([sum2[k] / wsum2[k] for k in keep]).astype(np.float32)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        idx_choi = keep.index(choi_key)
        sims = A @ A[idx_choi]
        sims[idx_choi] = -1
        topk = np.argsort(-sims)[:TOP_K]
        key_to_name = dict(zip(authors_tbl["author_key"],
                               authors_tbl["canonical_name"]))
        print(f"\n[top-{TOP_K} semantic neighbors of Choi, Seongjin (train≤2019)]")
        rank_of_sun = None
        for rnk, ji in enumerate(topk, 1):
            nb = keep[ji]
            nb_name = key_to_name.get(nb, nb)
            td = d_choi.get(nb)
            mark = " **<-- Sun, Lijun" if nb == sun_key else ""
            print(f"  {rnk:2d}. sim={sims[ji]:.4f}  train_d={td}  {nb_name}{mark}")
            if nb == sun_key:
                rank_of_sun = rnk
        # If Sun didn't make top-K, find Sun's rank in the full ranking
        if sun_key in keep and rank_of_sun is None:
            sun_idx = keep.index(sun_key)
            full_order = np.argsort(-sims)
            rank_all = int(np.where(full_order == sun_idx)[0][0]) + 1
            print(f"\n[!] Sun not in Choi's top-{TOP_K}; rank in full list = {rank_all} "
                  f"out of {len(keep):,}.")
        elif rank_of_sun is not None:
            print(f"\n[✓] Sun ranks #{rank_of_sun} in Choi's phantom list.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
