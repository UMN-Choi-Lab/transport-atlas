#!/usr/bin/env python
"""Experiment: 5-year trajectories with annual stride (sliding windows).

Reuses the production pipeline's whitened SPECTER2 embeddings + UMAP fit
but swaps the trajectory binning from calendar bins (year // 5 * 5) to a
1-year stride sliding 5-year window. Each paper now contributes to up to
5 bins.

Output: data/processed/author_trajectories.sliding.json
        (calendar version is preserved at author_trajectories.json — to
         actually swap, copy this output over after inspection)
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from transport_atlas.process.authors import author_key as _raw_author_key

EMBED_DIR = Path(os.environ.get("EMBED_OUT", "/data2/chois/transport-atlas"))
UMAP_SEED = 42
WINDOW = 5
STRIDE = 1
MIN_PAPERS_PER_BIN = 2
MIN_PAPERS_FOR_EMBED = 2
WHITEN_TOP_PC = 1

OUT = ROOT / "data" / "processed" / "author_trajectories.sliding.json"


def _load_alias_map() -> dict:
    cfg = ROOT / "config" / "pipeline.yaml"
    mp: dict[str, str] = {}
    try:
        import yaml
        cfg_data = yaml.safe_load(cfg.read_text()) or {}
        for ids in cfg_data.get("alias_groups", []) or []:
            if not ids or len(ids) < 2:
                continue
            target = ids[0].lower()
            for other in ids[1:]:
                mp[other.lower()] = target
    except Exception as e:
        print(f"[sliding] WARN: alias load failed: {e}", file=sys.stderr)
    auto_path = ROOT / "data" / "interim" / "author_aliases_auto.json"
    if auto_path.exists():
        for k, v in json.loads(auto_path.read_text()).items():
            mp.setdefault(k, v)
    return mp


def author_key_with_alias(a: dict, alias_map: dict) -> str:
    k = _raw_author_key(a)
    return alias_map.get(k, k) if k else k


def main() -> int:
    print(f"[sliding] backend=numpy", flush=True)

    # ------ Embeddings + whitening (mirror 06_author_similarity.py) -------
    embed_path = EMBED_DIR / "paper_embeddings.parquet"
    emb_df = pd.read_parquet(embed_path)
    pid_to_row = {pid: i for i, pid in enumerate(emb_df["paper_id"].tolist())}
    E = np.stack(emb_df["emb"].tolist()).astype(np.float32)
    print(f"[sliding] loaded {E.shape[0]:,} paper embeddings (dim {E.shape[1]})", flush=True)

    t0 = time.time()
    mu = E.mean(axis=0, keepdims=True)
    E = E - mu
    # Covariance is 768x768 — cheap on CPU
    cov = (E.T @ E) / E.shape[0]
    evals, evecs = np.linalg.eigh(cov)  # ascending
    top_dirs = evecs[:, -WHITEN_TOP_PC:]   # (768, k)
    proj = (E @ top_dirs) @ top_dirs.T
    E = E - proj
    std = E.std(axis=0, keepdims=True) + 1e-8
    E = E / std
    E = E.astype(np.float32)
    print(f"[sliding] whitening done in {time.time() - t0:.1f}s", flush=True)
    del cov, evecs, proj

    papers = pd.read_parquet(ROOT / "data" / "interim" / "papers.parquet")
    print(f"[sliding] papers: {len(papers):,}", flush=True)

    # ------ Coauthor-network keys -> node ids -----------------------------
    net = json.loads((ROOT / "data" / "processed" / "coauthor_network.json").read_text())
    atlas_key_to_nodeid = {n["key"]: n["id"] for n in net["nodes"] if n.get("key")}

    alias_map = _load_alias_map()

    # ------ Per-author aggregation + per-author year events ---------------
    author_sum = {}
    author_wsum = {}
    author_years = defaultdict(list)
    skipped = 0
    for r in papers.itertuples(index=False):
        pid = r.paper_id
        row = pid_to_row.get(pid)
        if row is None:
            skipped += 1
            continue
        v = E[row]
        cites = 0 if r.cited_by_count is None else int(r.cited_by_count or 0)
        w = 1.0 + np.log1p(cites)
        year = int(r.year) if r.year is not None and not (isinstance(r.year, float) and r.year != r.year) else None
        if r.authors is None or len(r.authors) == 0:
            continue
        keys = set()
        for a in r.authors:
            if not isinstance(a, dict):
                continue
            k = author_key_with_alias(a, alias_map)
            if k:
                keys.add(k)
        for k in keys:
            if k not in author_sum:
                author_sum[k] = np.zeros(E.shape[1], dtype=np.float32)
                author_wsum[k] = 0.0
            author_sum[k] += w * v
            author_wsum[k] += w
            if year is not None:
                author_years[k].append((year, w, row))
    print(f"[sliding] aggregated for {len(author_sum):,} authors "
          f"(skipped {skipped:,} papers without embeddings)", flush=True)

    keep_keys = {k for k, ws in author_wsum.items() if ws >= MIN_PAPERS_FOR_EMBED}
    print(f"[sliding] authors with >= {MIN_PAPERS_FOR_EMBED} papers: {len(keep_keys):,}", flush=True)
    keys = sorted(k for k in author_sum if k in keep_keys)
    A = np.stack([author_sum[k] / max(author_wsum[k], 1e-8) for k in keys])
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    print(f"[sliding] author vectors: {A.shape}", flush=True)

    # ------ UMAP fit on author centroids (matches production geometry) ---
    print("[sliding] UMAP 2D fit ...", flush=True)
    t0 = time.time()
    import umap
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=UMAP_SEED, verbose=False,
    )
    coords = reducer.fit_transform(A)
    umap_mean = coords.mean(axis=0)
    coords = coords - umap_mean
    umap_scale = 100 / max(np.abs(coords).max(), 1e-6)
    coords = coords * umap_scale
    print(f"[sliding] UMAP done in {time.time() - t0:.1f}s", flush=True)

    # ------ Sliding-window trajectory centroids ---------------------------
    print(f"[sliding] building W={WINDOW}y stride={STRIDE}y trajectories...", flush=True)
    t0 = time.time()
    pending: list[tuple[str, int, np.ndarray, int]] = []
    for k in keys:
        events = author_years.get(k, [])
        if not events:
            continue
        nid = atlas_key_to_nodeid.get(k)
        if nid is None:
            continue
        years_present = sorted({y for y, _, _ in events})
        ymin, ymax = years_present[0], years_present[-1]
        # Bin start year b means window [b, b+W-1]. Iterate every STRIDE
        # years from ymin..ymax. (Edge windows that extend past the corpus
        # are fine — they just include fewer years of data.)
        per_bin = []
        for bstart in range(ymin, ymax + 1, STRIDE):
            in_window = [(w, idx) for (y, w, idx) in events
                         if bstart <= y <= bstart + WINDOW - 1]
            if len(in_window) < MIN_PAPERS_PER_BIN:
                continue
            wsum = sum(w for w, _ in in_window)
            vec = np.zeros(E.shape[1], dtype=np.float32)
            for w, idx in in_window:
                vec += w * E[idx]
            vec /= max(wsum, 1e-8)
            vec /= (np.linalg.norm(vec) + 1e-8)
            per_bin.append((bstart, vec, len(in_window)))
        if len(per_bin) < 2:
            continue
        for bstart, vec, n in per_bin:
            pending.append((k, bstart, vec, n))
    print(f"[sliding] bin centroids to transform: {len(pending):,}  "
          f"in {time.time() - t0:.1f}s", flush=True)

    print("[sliding] batched UMAP.transform on centroids ...", flush=True)
    t0 = time.time()
    all_vecs = np.stack([p[2] for p in pending])
    all_xy = (reducer.transform(all_vecs) - umap_mean) * umap_scale
    print(f"[sliding] transform done in {time.time() - t0:.1f}s", flush=True)

    per_key: dict[str, list[tuple[int, float, float, int]]] = defaultdict(list)
    for i, (k, bstart, _, n) in enumerate(pending):
        per_key[k].append((bstart, float(all_xy[i, 0]), float(all_xy[i, 1]), n))

    trajectories: dict[str, list[dict]] = {}
    for k, items in per_key.items():
        nid = atlas_key_to_nodeid.get(k)
        if nid is None:
            continue
        items.sort(key=lambda t: t[0])
        trajectories[str(nid)] = [
            {"p": int(b), "x": round(x, 2), "y": round(y, 2), "n": int(nn)}
            for (b, x, y, nn) in items
        ]

    print(f"[sliding] trajectories: {len(trajectories):,} authors", flush=True)
    bins_per = [len(v) for v in trajectories.values()]
    if bins_per:
        bins_per.sort()
        print(f"[sliding]   bins per author: median={bins_per[len(bins_per)//2]}  "
              f"mean={sum(bins_per)/len(bins_per):.1f}  max={bins_per[-1]}")

    OUT.write_text(json.dumps(trajectories, allow_nan=False))
    print(f"[sliding] wrote {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
