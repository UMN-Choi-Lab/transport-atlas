#!/usr/bin/env python
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from transport_atlas.process.authors import author_key as _raw_author_key

OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.prop_cycle": plt.cycler("color", OKABE_ITO),
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper" / "manuscript" / "tables"
FIGURES = ROOT / "paper" / "manuscript" / "figures"
_JSON_BASE = ROOT / "paper" / "analysis" / "_ablation_embeddings"

EMBED_DIR = Path(os.environ["EMBED_OUT"]) if "EMBED_OUT" in os.environ else Path("/embed")
TRAIN_CUTOFF_YEAR = 2019
YEAR_MAX = 2025
TEST_YEARS = range(2020, YEAR_MAX + 1)
MIN_PAPERS_TRAIN = 2
TOP_K = 20
EVAL_KS = (5, 10, 20)
PHANTOM_MIN_HOPS = 3
BFS_CUTOFF = 3
WHITEN_TOP_PC = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
CHUNK = 2048
CONCEPT_DIM = 128


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    plt.close(fig)


def _tex_escape(s) -> str:
    if not isinstance(s, str):
        s = str(s)
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _load_alias_map() -> dict[str, str]:
    import yaml

    cfg = ROOT / "config" / "pipeline.yaml"
    pipe = yaml.safe_load(cfg.read_text()) or {}
    mp: dict[str, str] = {}
    for a in pipe.get("author_aliases", []) or []:
        ids = a.get("openalex_ids") or []
        if len(ids) < 2:
            continue
        target = ids[0].lower()
        for other in ids[1:]:
            mp[other.lower()] = target
    auto_path = ROOT / "data" / "interim" / "author_aliases_auto.json"
    if auto_path.exists():
        for k, v in json.loads(auto_path.read_text()).items():
            mp.setdefault(k, v)
    return mp


def author_key_with_alias(a: dict, alias_map: dict[str, str]) -> str:
    k = _raw_author_key(a)
    return alias_map.get(k, k) if k else k


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def whiten_train_only(
    e_raw: np.ndarray,
    train_rows: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    t0 = time.time()
    e_t = torch.from_numpy(e_raw[train_rows]).to(DEVICE).float()
    mu = e_t.mean(dim=0, keepdim=True)
    centered = e_t - mu
    cov = (centered.T @ centered) / centered.shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    top_dirs = evecs[:, -WHITEN_TOP_PC:]
    centered = centered - (centered @ top_dirs) @ top_dirs.T
    std = centered.std(dim=0, keepdim=True) + 1e-8

    full_t = torch.from_numpy(e_raw).to(DEVICE).float()
    full_t = full_t - mu
    full_t = full_t - (full_t @ top_dirs) @ top_dirs.T
    full_t = full_t / std
    out = full_t.cpu().numpy().astype(np.float32)
    stats = {
        "seconds": time.time() - t0,
        "top1_explained_variance_ratio": float(evals[-1] / evals.sum()),
    }
    del e_t, centered, cov, evecs, top_dirs, full_t
    return out, stats


def build_concept_docs(papers_scan: list[dict], n_rows: int) -> list[str]:
    docs = [""] * n_rows
    for rec in papers_scan:
        toks: list[str] = []
        for c in rec["concepts"]:
            if not isinstance(c, dict):
                continue
            lvl = c.get("level")
            nm = c.get("name")
            sc = c.get("score") or 0.0
            if lvl is None or nm is None or int(lvl) < 2:
                continue
            reps = max(1, int(round(float(sc) * 5)))
            tok = nm.lower().replace(" ", "_").replace("-", "_")
            toks.extend([tok] * reps)
        docs[rec["row"]] = " ".join(toks)
    return docs


def build_concept_features(
    train_docs: list[str],
    all_docs: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    t0 = time.time()
    vec = TfidfVectorizer(
        min_df=10,
        max_df=0.5,
        sublinear_tf=True,
        token_pattern=r"[a-z_]+",
    )
    c_train = vec.fit_transform(train_docs)
    n_components = min(CONCEPT_DIM, max(1, c_train.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(c_train)
    c_dense = svd.transform(vec.transform(all_docs)).astype(np.float32)
    c_dense = l2_normalize(c_dense)
    return c_dense, {
        "seconds": time.time() - t0,
        "vocab": int(len(vec.vocabulary_)),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "dim": int(c_dense.shape[1]),
    }


def build_venue_lda_features(
    whitened: np.ndarray,
    train_rows: np.ndarray,
    train_labels: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    t0 = time.time()
    lda = LinearDiscriminantAnalysis(n_components=None, solver="svd")
    lda.fit(whitened[train_rows], train_labels)
    dense = lda.transform(whitened).astype(np.float32)
    dense = l2_normalize(dense)
    return dense, {
        "seconds": time.time() - t0,
        "dim": int(dense.shape[1]),
        "classes": int(len(lda.classes_)),
    }


def topk_neighbors(a: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    a_t = torch.from_numpy(a).to(DEVICE)
    topk_sim = np.zeros((a.shape[0], top_k), dtype=np.float32)
    topk_idx = np.zeros((a.shape[0], top_k), dtype=np.int32)
    with torch.no_grad():
        for i in range(0, a.shape[0], CHUNK):
            q = a_t[i:i + CHUNK]
            sims = q @ a_t.T
            for r, gi in enumerate(range(i, min(i + CHUNK, a.shape[0]))):
                sims[r, gi] = -1.0
            vals, idxs = torch.topk(sims, top_k, dim=1)
            topk_sim[i:i + CHUNK] = vals.cpu().numpy()
            topk_idx[i:i + CHUNK] = idxs.cpu().numpy()
    return topk_sim, topk_idx


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def bfs_near_neighbors(
    eval_keys: list[str],
    adj: dict[str, set[str]],
) -> dict[str, dict[str, int]]:
    eval_set = set(eval_keys)

    def bfs_distances(src: str) -> dict[str, int]:
        dist = {src: 0}
        frontier = [src]
        for d in range(1, BFS_CUTOFF + 1):
            nxt: list[str] = []
            for u in frontier:
                for v in adj.get(u, ()):
                    if v not in dist:
                        dist[v] = d
                        nxt.append(v)
            if not nxt:
                break
            frontier = nxt
        return dist

    near = {}
    for k in eval_keys:
        dist = bfs_distances(k)
        near[k] = {
            kk: dd for kk, dd in dist.items()
            if kk in eval_set and kk != k
        }
    return near


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    wall_t0 = time.time()

    embed_file = os.environ.get("ABLATION_EMBED_FILE", "paper_embeddings.parquet")
    suffix = os.environ.get("ABLATION_SUFFIX", "")
    embed_path = EMBED_DIR / embed_file
    if not embed_path.exists():
        print(f"[ablation] missing {embed_path}", file=sys.stderr)
        return 1
    if not (ROOT / "scripts" / "07_phantom_eval.py").exists():
        print("[ablation] missing scripts/07_phantom_eval.py", file=sys.stderr)
        return 1
    if not (ROOT / "scripts" / "06_author_similarity.py").exists():
        print("[ablation] missing scripts/06_author_similarity.py", file=sys.stderr)
        return 1

    print(f"[ablation] device={DEVICE}", flush=True)
    emb_df = pd.read_parquet(embed_path)
    papers = pd.read_parquet(ROOT / "data" / "interim" / "papers.parquet")
    authors_tbl = pd.read_parquet(ROOT / "data" / "interim" / "authors.parquet")
    alias_map = _load_alias_map()

    pid_to_row = {pid: i for i, pid in enumerate(emb_df["paper_id"].tolist())}
    e_raw = np.stack(emb_df["emb"].tolist()).astype(np.float32)
    print(f"[ablation] embeddings={e_raw.shape[0]:,} dim={e_raw.shape[1]}", flush=True)

    keep_keys = set(authors_tbl.loc[authors_tbl["n_papers"] >= 1, "author_key"])
    papers_scan: list[dict] = []
    train_rows_list: list[int] = []
    train_labels: list[str] = []
    train_papers_for_author: list[dict] = []
    author_papers_train: dict[str, int] = defaultdict(int)
    author_venues_train: dict[str, set[str]] = defaultdict(set)
    author_has_train_embedding: dict[str, bool] = defaultdict(bool)
    adj: dict[str, set[str]] = defaultdict(set)
    test_edges: set[tuple[str, str]] = set()

    for _, r in papers.iterrows():
        yr = r.get("year")
        if yr is None or pd.isna(yr):
            continue
        year = int(yr)
        if year > YEAR_MAX:
            continue
        pid = r["paper_id"]
        row = pid_to_row.get(pid)
        authors_list = r.get("authors")
        if authors_list is None or len(authors_list) == 0:
            continue

        keys_here: list[str] = []
        for a in authors_list:
            if not isinstance(a, dict):
                continue
            k = author_key_with_alias(a, alias_map)
            if k and k in keep_keys:
                keys_here.append(k)
        if not keys_here:
            continue
        uniq_keys = list(dict.fromkeys(keys_here))
        venue = r.get("venue_slug") or "_unk"
        concepts = r.get("concepts")
        cites_val = r.get("cited_by_count")
        if cites_val is None or pd.isna(cites_val):
            cites = 0
        else:
            cites = int(cites_val)
        paper_rec = {
            "paper_id": pid,
            "row": row,
            "year": year,
            "authors": uniq_keys,
            "venue": venue,
            "concepts": list(concepts) if concepts is not None else [],
            "cites": cites,
        }
        if row is not None:
            papers_scan.append(paper_rec)

        if year <= TRAIN_CUTOFF_YEAR:
            for k in uniq_keys:
                author_papers_train[k] += 1
                author_venues_train[k].add(venue)
                if row is not None:
                    author_has_train_embedding[k] = True
            for i in range(len(uniq_keys)):
                for j in range(i + 1, len(uniq_keys)):
                    a = uniq_keys[i]
                    b = uniq_keys[j]
                    if a == b:
                        continue
                    adj[a].add(b)
                    adj[b].add(a)
            if row is not None:
                train_rows_list.append(row)
                train_labels.append(venue)
                train_papers_for_author.append({
                    "row": row,
                    "authors": uniq_keys,
                    "weight": 1.0 + math.log1p(max(0, paper_rec["cites"])),
                })
        elif year in TEST_YEARS:
            for i in range(len(uniq_keys)):
                for j in range(i + 1, len(uniq_keys)):
                    a = uniq_keys[i]
                    b = uniq_keys[j]
                    if a == b:
                        continue
                    test_edges.add(_pair_key(a, b))

    if not train_rows_list:
        print("[ablation] no train papers with embeddings", file=sys.stderr)
        return 1

    train_rows = np.array(train_rows_list, dtype=np.int64)
    eval_keys = sorted(
        k for k, n_train in author_papers_train.items()
        if n_train >= MIN_PAPERS_TRAIN and author_has_train_embedding.get(k, False)
    )
    if not eval_keys:
        print("[ablation] no eligible authors", file=sys.stderr)
        return 1
    key_idx = {k: i for i, k in enumerate(eval_keys)}

    print(
        f"[ablation] train_rows={len(train_rows):,} eval_authors={len(eval_keys):,} "
        f"test_edges={len(test_edges):,}",
        flush=True,
    )

    t0 = time.time()
    near_neighbors = bfs_near_neighbors(eval_keys, adj)
    print(f"[ablation] bfs_seconds={time.time() - t0:.1f}", flush=True)

    realized_future: dict[str, set[str]] = defaultdict(set)
    for a, b in test_edges:
        d_ab = near_neighbors.get(a, {}).get(b)
        if a in key_idx and b in key_idx and (d_ab is None or d_ab >= PHANTOM_MIN_HOPS):
            realized_future[a].add(b)
            realized_future[b].add(a)

    excl_of: dict[str, set[str]] = {}
    venue_to_keys: dict[str, list[str]] = defaultdict(list)
    for k in eval_keys:
        for v in author_venues_train.get(k, set()):
            venue_to_keys[v].append(k)
    sv_pool: dict[str, list[str]] = {}
    for k in eval_keys:
        excl = {k}
        for kk, dd in near_neighbors.get(k, {}).items():
            if dd <= PHANTOM_MIN_HOPS - 1:
                excl.add(kk)
        excl_of[k] = excl
        union: set[str] = set()
        for v in author_venues_train.get(k, set()):
            union.update(venue_to_keys.get(v, ()))
        union.difference_update(excl)
        sv_pool[k] = list(union)

    pa_weights = np.array([author_papers_train.get(k, 0) for k in eval_keys], dtype=np.float64)
    pa_total = float(pa_weights.sum())
    pa_cdf = np.cumsum(pa_weights) / pa_total if pa_total > 0 else None
    n_eval = len(eval_keys)

    draw_rngs = {
        "random": random.Random(SEED),
        "popularity": random.Random(SEED),
        "same_venue": random.Random(SEED),
    }

    def draw_random(anchor: str, k_need: int) -> list[str]:
        rng = draw_rngs["random"]
        seen = set(excl_of[anchor])
        out: list[str] = []
        tries = 0
        while len(out) < k_need and tries < k_need * 50:
            cand = eval_keys[rng.randrange(0, n_eval)]
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
            tries += 1
        return out

    def draw_popularity(anchor: str, k_need: int) -> list[str]:
        if pa_cdf is None:
            return draw_random(anchor, k_need)
        rng = draw_rngs["popularity"]
        seen = set(excl_of[anchor])
        out: list[str] = []
        tries = 0
        while len(out) < k_need and tries < k_need * 50:
            u = rng.random()
            cand_idx = int(np.searchsorted(pa_cdf, u, side="right"))
            if cand_idx >= n_eval:
                cand_idx = n_eval - 1
            cand = eval_keys[cand_idx]
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
            tries += 1
        return out

    def draw_same_venue(anchor: str, k_need: int) -> list[str]:
        rng = draw_rngs["same_venue"]
        pool = sv_pool.get(anchor, [])
        if len(pool) >= k_need:
            return rng.sample(pool, k_need)
        extras: list[str] = []
        pool_set = set(pool)
        for cand in draw_random(anchor, k_need * 2):
            if cand not in pool_set:
                extras.append(cand)
                pool_set.add(cand)
            if len(pool) + len(extras) >= k_need:
                break
        return list(pool) + extras[: max(0, k_need - len(pool))]

    baseline_drawers = {
        "random": draw_random,
        "popularity": draw_popularity,
        "same_venue": draw_same_venue,
    }
    baseline_candidates: dict[str, dict[int, dict[str, list[str]]]] = {
        method: {k_val: {} for k_val in EVAL_KS}
        for method in baseline_drawers
    }
    for method_name, drawer in baseline_drawers.items():
        for k_val in EVAL_KS:
            for anchor in eval_keys:
                baseline_candidates[method_name][k_val][anchor] = drawer(anchor, k_val)

    concept_docs = build_concept_docs([rec for rec in papers_scan if rec["row"] is not None], e_raw.shape[0])
    train_docs = [concept_docs[row] for row in train_rows]

    whitened, whiten_stats = whiten_train_only(e_raw, train_rows)
    concept_dense, concept_stats = build_concept_features(train_docs, concept_docs)
    lda_dense, lda_stats = build_venue_lda_features(whitened, train_rows, train_labels)
    whitened_norm = l2_normalize(whitened)

    prefix = os.environ.get("ABLATION_CONFIG_PREFIX", "")
    configs: list[tuple[str, np.ndarray]] = [
        (f"{prefix}raw_specter2", e_raw),
        (f"{prefix}whitened", whitened),
        (
            f"{prefix}whitened+concept",
            np.hstack([
                np.sqrt(0.65) * whitened_norm,
                np.sqrt(0.35) * concept_dense,
            ]).astype(np.float32),
        ),
        (
            f"{prefix}hybrid_full",
            np.hstack([
                np.sqrt(0.55) * whitened_norm,
                np.sqrt(0.30) * concept_dense,
                np.sqrt(0.15) * lda_dense,
            ]).astype(np.float32),
        ),
    ]

    results: dict[str, dict] = {
        "config": {
            "device": DEVICE,
            "train_cutoff_year": TRAIN_CUTOFF_YEAR,
            "year_max": YEAR_MAX,
            "test_years": [TEST_YEARS.start, TEST_YEARS.stop - 1],
            "n_eval_authors": len(eval_keys),
            "n_train_papers": int(len(train_rows)),
            "n_realized_authors": int(sum(1 for v in realized_future.values() if v)),
            "precompute": {
                "whitening": whiten_stats,
                "concept": concept_stats,
                "venue_lda": lda_stats,
            },
        },
        "configs": {},
    }

    precision20_lines: list[str] = []
    summary_rows: list[dict] = []

    for config_name, paper_matrix in configs:
        t0 = time.time()
        author_sum: dict[str, np.ndarray] = {}
        author_wsum: dict[str, float] = {}
        for rec in train_papers_for_author:
            v = paper_matrix[rec["row"]]
            for k in rec["authors"]:
                if k not in key_idx:
                    continue
                if k not in author_sum:
                    author_sum[k] = np.zeros(paper_matrix.shape[1], dtype=np.float32)
                    author_wsum[k] = 0.0
                author_sum[k] += rec["weight"] * v
                author_wsum[k] += rec["weight"]

        missing = [k for k in eval_keys if k not in author_sum]
        if missing:
            print(f"[ablation] config={config_name} failed missing centroids={len(missing)}", file=sys.stderr)
            return 1

        a = np.stack([
            author_sum[k] / max(author_wsum[k], 1e-8) for k in eval_keys
        ]).astype(np.float32)
        norms = np.linalg.norm(a, axis=1)
        if np.any(norms <= 1e-12):
            zero_keys = [eval_keys[i] for i, n in enumerate(norms) if n <= 1e-12]
            print(f"[ablation] config={config_name} failed zero centroids={len(zero_keys)}", file=sys.stderr)
            return 1
        a = l2_normalize(a)

        topk_sim, topk_idx = topk_neighbors(a, TOP_K)

        phantom_neighbors: dict[str, list[str]] = {}
        for i, anchor in enumerate(eval_keys):
            chosen: list[str] = []
            for j in range(TOP_K):
                nb = eval_keys[topk_idx[i, j]]
                if nb in excl_of[anchor]:
                    continue
                chosen.append(nb)
                if len(chosen) >= TOP_K:
                    break
            phantom_neighbors[anchor] = chosen

        config_metrics: dict[str, dict] = {}
        for k_val in EVAL_KS:
            methods_metrics: dict[str, dict] = {}
            for method_name in ("phantom", "random", "popularity", "same_venue"):
                hits = 0
                preds = 0
                per_author_p: list[float] = []
                per_author_r: list[float] = []
                for anchor in eval_keys:
                    realized_set = realized_future.get(anchor, set())
                    if method_name == "phantom":
                        cand = phantom_neighbors[anchor][:k_val]
                    else:
                        cand = baseline_candidates[method_name][k_val][anchor]
                    if not cand:
                        continue
                    hit_set = set(cand) & realized_set
                    hits += len(hit_set)
                    preds += len(cand)
                    per_author_p.append(len(hit_set) / len(cand))
                    if realized_set:
                        per_author_r.append(len(hit_set) / len(realized_set))
                methods_metrics[method_name] = {
                    "hits": int(hits),
                    "predictions": int(preds),
                    "micro_precision": float(hits / preds if preds else 0.0),
                    "macro_precision": float(np.mean(per_author_p)) if per_author_p else 0.0,
                    "macro_recall": float(np.mean(per_author_r)) if per_author_r else 0.0,
                    "n_authors_scored": int(len(per_author_p)),
                }
            phantom_p = methods_metrics["phantom"]["micro_precision"]
            for baseline_name in ("random", "popularity", "same_venue"):
                base_p = methods_metrics[baseline_name]["micro_precision"]
                lift = float(phantom_p / base_p) if base_p > 0 else float("inf")
                methods_metrics["phantom"][f"lift_vs_{baseline_name}"] = lift
                methods_metrics[baseline_name]["phantom_lift_over_method"] = lift
            config_metrics[f"K={k_val}"] = methods_metrics

        elapsed = time.time() - t0
        p20 = config_metrics["K=20"]["phantom"]["micro_precision"]
        lr = config_metrics["K=20"]["phantom"]["lift_vs_random"]
        lp = config_metrics["K=20"]["phantom"]["lift_vs_popularity"]
        lv = config_metrics["K=20"]["phantom"]["lift_vs_same_venue"]
        print(
            f"[ablation] config={config_name} precision@20={p20*100:.1f}% "
            f"lift_vs_random={lr:.2f}x lift_vs_pop={lp:.2f}x lift_vs_venue={lv:.2f}x "
            f"seconds={elapsed:.1f}",
            flush=True,
        )
        precision20_lines.append(
            f"[ablation] config={config_name} precision@20={p20*100:.1f}% "
            f"lift_vs_random={lr:.2f}x lift_vs_pop={lp:.2f}x lift_vs_venue={lv:.2f}x"
        )
        summary_rows.append({
            "config": config_name,
            "p5": config_metrics["K=5"]["phantom"]["micro_precision"],
            "p10": config_metrics["K=10"]["phantom"]["micro_precision"],
            "p20": p20,
            "lift_random_20": lr,
            "lift_popularity_20": lp,
            "lift_venue_20": lv,
            "seconds": elapsed,
        })
        results["configs"][config_name] = {
            "paper_dim": int(paper_matrix.shape[1]),
            "metrics": config_metrics,
            "runtime_seconds": elapsed,
        }

    best_p5 = max(r["p5"] for r in summary_rows)
    best_p10 = max(r["p10"] for r in summary_rows)
    best_p20 = max(r["p20"] for r in summary_rows)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Config} & \textbf{P@5 (\%)} & \textbf{P@10 (\%)} & "
            r"\textbf{P@20 (\%)} & \textbf{Lift vs random @20} & "
            r"\textbf{Lift vs popularity @20} & \textbf{Lift vs same-venue @20} \\"
        ),
        r"\midrule",
    ]
    for row in summary_rows:
        p5_txt = f"{row['p5']*100:.2f}"
        p10_txt = f"{row['p10']*100:.2f}"
        p20_txt = f"{row['p20']*100:.2f}"
        if row["p5"] == best_p5:
            p5_txt = r"\textbf{" + p5_txt + "}"
        if row["p10"] == best_p10:
            p10_txt = r"\textbf{" + p10_txt + "}"
        if row["p20"] == best_p20:
            p20_txt = r"\textbf{" + p20_txt + "}"
        lines.append(
            f"{_tex_escape(row['config'])} & {p5_txt} & {p10_txt} & {p20_txt} & "
            f"{row['lift_random_20']:.2f}$\\times$ & {row['lift_popularity_20']:.2f}$\\times$ & "
            f"{row['lift_venue_20']:.2f}$\\times$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / f"08_ablation_embeddings{suffix}.tex").write_text("\n".join(lines) + "\n")

    ks = np.array(EVAL_KS)
    x = np.arange(len(ks), dtype=float)
    width = 0.18
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, row in enumerate(summary_rows):
        ys = np.array([row["p5"], row["p10"], row["p20"]]) * 100
        offset = (i - (len(summary_rows) - 1) / 2) * width
        ax.bar(x + offset, ys, width=width, label=row["config"], color=OKABE_ITO[i])
    ax.set_xticks(x, [str(k) for k in ks])
    ax.set_xlabel("$K$")
    ax.set_ylabel(r"Micro-precision (\%)")
    ax.set_title("Phantom collaborator precision under embedding ablations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _save(fig, f"08_ablation_precision_at_k{suffix}")

    json_out = _JSON_BASE.with_name(_JSON_BASE.name + f"{suffix}.json")
    json_out.write_text(json.dumps(results, indent=2))

    print(f"[ablation] wrote {TABLES / f'08_ablation_embeddings{suffix}.tex'}", flush=True)
    print(f"[ablation] wrote {FIGURES / f'08_ablation_precision_at_k{suffix}.pdf'}", flush=True)
    print(f"[ablation] wrote {json_out}", flush=True)
    for line in precision20_lines:
        print(line, flush=True)
    print(f"[ablation] total_wall_clock={time.time() - wall_t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
