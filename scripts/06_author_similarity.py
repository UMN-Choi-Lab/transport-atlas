#!/usr/bin/env python
"""Aggregate paper embeddings to authors + compute similarity, UMAP, trajectories.

Inputs:
    /data2/chois/transport-atlas/paper_embeddings.parquet  (paper_id, emb[768])
    data/interim/papers.parquet                            (paper_id, authors, year, cited_by_count)
    data/interim/authors.parquet                           (author_key, canonical_name, n_papers, ...)
    data/processed/coauthor_network.json                   (for Leiden community mapping)

Outputs:
    data/processed/author_similar.json      {author_key: [{id, name, sim}, ...top-10]}
    data/processed/topic_coords.json        {author_key: {x, y, community, name, papers}}
    data/processed/author_trajectories.json {author_key: [{period, x, y, n}, ...]}
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
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transport_atlas.process.authors import author_key as _raw_author_key

# Config
EMBED_DIR = Path(os.environ.get("EMBED_OUT", "/data2/chois/transport-atlas"))
MIN_PAPERS_FOR_EMBED = 2
TOP_K_SIMILAR = 20
TRAJECTORY_BIN_YEARS = 5
TRAJECTORY_MIN_PAPERS_PER_BIN = 2
UMAP_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Semantic-community detection knobs
SEM_KNN_K = 20              # kNN graph used for semantic Leiden
SEM_LEIDEN_RESOLUTION = 1.0
SEM_LEIDEN_SEED = 42
PHANTOM_MIN_HOPS = 2        # semantic neighbor is a "phantom" if coauthor_dist >= this
                            # (i.e., never directly coauthored — d=1 is the only non-phantom)
BFS_CUTOFF = 3              # BFS max hops when labeling coauthor_dist


def _load_alias_map() -> dict[str, str]:
    """Manual aliases from pipeline.yaml + auto-detected ORCID splits from dedupe."""
    import yaml
    repo = Path(__file__).resolve().parents[1]
    cfg = repo / "config" / "pipeline.yaml"
    pipe = yaml.safe_load(cfg.read_text()) or {}
    mp = {}
    for a in pipe.get("author_aliases", []) or []:
        ids = a.get("openalex_ids") or []
        if len(ids) < 2:
            continue
        target = ids[0].lower()
        for other in ids[1:]:
            mp[other.lower()] = target
    auto_path = repo / "data" / "interim" / "author_aliases_auto.json"
    if auto_path.exists():
        auto_map = json.loads(auto_path.read_text())
        for k, v in auto_map.items():
            mp.setdefault(k, v)  # manual entries win
    return mp


def author_key_with_alias(a: dict, alias_map: dict) -> str:
    k = _raw_author_key(a)
    return alias_map.get(k, k) if k else k


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    print(f"[sim] device={DEVICE}", flush=True)

    # Load paper embeddings
    embed_path = EMBED_DIR / "paper_embeddings.parquet"
    if not embed_path.exists():
        print(f"[sim] missing {embed_path}; run 05_embed_papers.py first", file=sys.stderr)
        return 1
    emb_df = pd.read_parquet(embed_path)
    pid_to_row = {pid: i for i, pid in enumerate(emb_df["paper_id"].tolist())}
    E = np.stack(emb_df["emb"].tolist()).astype(np.float32)  # (n_papers, 768)
    print(f"[sim] loaded {E.shape[0]:,} paper embeddings (dim {E.shape[1]})", flush=True)

    # ——— Whitening: amplify within-domain structure (Arora et al. 2017, "all-but-the-top") ———
    # SPECTER2 on a mono-domain corpus (all transportation) produces embeddings that share
    # a strong "this is transportation" direction + uneven per-dim variance. Median pairwise
    # cosine ~0.98 is what that looks like. Two steps here:
    #   A. Common-component removal — subtract mean and project out the top-1 principal direction.
    #   B. Per-dim z-score — equalize per-dim variance so noisy dims don't dominate cosine.
    # Together these shift median pairwise sim from ~0.98 toward 0.7-0.8, revealing sub-field
    # structure (pavement vs flow vs CAV vs behavior) that was previously compressed.
    WHITEN_TOP_PC = 1
    t0 = time.time()
    E_t = torch.from_numpy(E).to(DEVICE).float()
    mu = E_t.mean(dim=0, keepdim=True)
    E_t = E_t - mu
    # Top-k principal directions via 768×768 covariance eigendecomp (cheap)
    cov = (E_t.T @ E_t) / E_t.shape[0]
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    top_dirs = evecs[:, -WHITEN_TOP_PC:]   # (768, k)
    proj = (E_t @ top_dirs) @ top_dirs.T   # (n, 768)
    E_t = E_t - proj
    # Per-dim z-score
    std = E_t.std(dim=0, keepdim=True) + 1e-8
    E_t = E_t / std
    E = E_t.cpu().numpy().astype(np.float32)
    ev_ratio = float(evals[-1] / evals.sum())
    print(f"[sim] whitening done in {time.time() - t0:.1f}s  "
          f"(top-1 PC accounted for {ev_ratio*100:.1f}% of variance)", flush=True)
    del E_t, cov, evecs, proj

    # Load papers w/ authors + year + cites
    papers = pd.read_parquet(repo / "data" / "interim" / "papers.parquet")
    authors_tbl = pd.read_parquet(repo / "data" / "interim" / "authors.parquet")
    print(f"[sim] papers: {len(papers):,}  authors: {len(authors_tbl):,}", flush=True)

    # ——— C: Concept TF-IDF concat ———
    # OpenAlex concepts (L2+) provide sub-field labels that SPECTER2 embeddings blur.
    # Build paper × concept sparse matrix, reduce via TruncatedSVD, L2-normalize.
    print("[sim] building concept TF-IDF features...", flush=True)
    t0 = time.time()
    paper_ids_ordered = emb_df["paper_id"].tolist()
    pid_to_paper_row = {pid: i for i, pid in enumerate(paper_ids_ordered)}
    n_papers_emb = len(paper_ids_ordered)
    paper_concepts_docs: list[str] = [""] * n_papers_emb
    paper_venue_labels: list[str] = ["_unk"] * n_papers_emb
    for _, r in papers.iterrows():
        row_idx = pid_to_paper_row.get(r["paper_id"])
        if row_idx is None:
            continue
        concepts = r.get("concepts")
        try:
            c_list = [] if concepts is None or len(concepts) == 0 else list(concepts)
        except TypeError:
            c_list = []
        toks = []
        for c in c_list:
            if not isinstance(c, dict):
                continue
            lvl = c.get("level")
            nm = c.get("name")
            sc = c.get("score") or 0
            if lvl is None or nm is None or int(lvl) < 2:
                continue
            reps = max(1, int(round(float(sc) * 5)))
            toks.extend([nm.lower().replace(" ", "_").replace("-", "_")] * reps)
        paper_concepts_docs[row_idx] = " ".join(toks)
        paper_venue_labels[row_idx] = r.get("venue_slug") or "_unk"

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    vec = TfidfVectorizer(min_df=10, max_df=0.5, sublinear_tf=True, token_pattern=r"[a-z_]+")
    C_sparse = vec.fit_transform(paper_concepts_docs)
    C_SVD_DIM = 128
    svd = TruncatedSVD(n_components=min(C_SVD_DIM, C_sparse.shape[1] - 1), random_state=42)
    C_dense = svd.fit_transform(C_sparse).astype(np.float32)
    C_dense /= (np.linalg.norm(C_dense, axis=1, keepdims=True) + 1e-8)
    print(f"[sim] concept: sparse={C_sparse.shape} → SVD {C_dense.shape}  "
          f"explained_var={svd.explained_variance_ratio_.sum():.3f}  in {time.time() - t0:.1f}s",
          flush=True)

    # ——— D: Venue-LDA projection ———
    # Supervised projection with venue as weak label: max between-class variance.
    # 29 venues → up to 28 LDA components. Prevents circularity with coauthor communities.
    print("[sim] fitting venue LDA projection...", flush=True)
    t0 = time.time()
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=None, solver="svd")
    L_dense = lda.fit_transform(E, paper_venue_labels).astype(np.float32)
    L_dense /= (np.linalg.norm(L_dense, axis=1, keepdims=True) + 1e-8)
    print(f"[sim] LDA: {E.shape} → {L_dense.shape}  in {time.time() - t0:.1f}s", flush=True)

    # ——— Hybrid concat with sqrt weights ———
    # cos(H_a, H_b) = α·cos_specter + β·cos_concept + γ·cos_lda  when weights sum to 1.
    ALPHA_E, ALPHA_C, ALPHA_L = 0.55, 0.30, 0.15
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    H = np.hstack([
        np.sqrt(ALPHA_E) * E_norm,
        np.sqrt(ALPHA_C) * C_dense,
        np.sqrt(ALPHA_L) * L_dense,
    ]).astype(np.float32)
    print(f"[sim] hybrid embedding: {H.shape}  "
          f"(α_E={ALPHA_E:.2f}, α_C={ALPHA_C:.2f}, α_L={ALPHA_L:.2f})", flush=True)
    E = H
    del H, E_norm, C_dense, L_dense, C_sparse

    # Which authors make the cut?
    keep_keys = set(authors_tbl.loc[authors_tbl["n_papers"] >= MIN_PAPERS_FOR_EMBED, "author_key"])
    key_to_name = dict(zip(authors_tbl["author_key"], authors_tbl["canonical_name"]))
    print(f"[sim] authors with >= {MIN_PAPERS_FOR_EMBED} papers: {len(keep_keys):,}", flush=True)

    # Atlas node id mapping — use the "key" field written by coauthor_graph.py
    # so we match each node by its exact author_key (avoids collisions when multiple
    # researchers share a romanized canonical name, e.g. "sun, lijun").
    net = json.loads((repo / "data" / "processed" / "coauthor_network.json").read_text())
    community_by_id = {n["id"]: n.get("community") for n in net["nodes"]}
    atlas_key_to_nodeid: dict[str, int] = {}
    missing_key_field = 0
    for n in net["nodes"]:
        k = n.get("key")
        if k:
            atlas_key_to_nodeid[k] = n["id"]
        else:
            missing_key_field += 1
    if missing_key_field:
        # Fallback: old graph JSON without "key" — match by canonical name (loses precision
        # on homonyms). Rerun coauthor_graph.py to restore exact matching.
        print(f"[sim] WARN: {missing_key_field} nodes missing 'key' field — falling back to name matching",
              flush=True)
        canon_to_keys: dict[str, list[str]] = defaultdict(list)
        for kk, nm in key_to_name.items():
            if nm:
                canon_to_keys[nm].append(kk)
        for n in net["nodes"]:
            if n.get("key"):
                continue
            for kk in canon_to_keys.get(n["label"], []):
                atlas_key_to_nodeid.setdefault(kk, n["id"])
    print(f"[sim] mapped {len(atlas_key_to_nodeid):,} author_keys to atlas node ids", flush=True)

    # Apply the same alias map as dedupe/graph so we merge multiple OpenAlex IDs.
    alias_map = _load_alias_map()

    # Aggregate paper embeddings per author (citation-weighted mean).
    # Also collect per-year buckets for trajectories.
    author_sum = {}       # key -> np.array(768) running weighted sum
    author_wsum = {}      # key -> scalar running weight sum
    author_years = defaultdict(list)  # key -> [(year, weight, idx)]

    skipped_missing_emb = 0
    for _, r in tqdm(papers.iterrows(), total=len(papers), desc="aggregate"):
        pid = r["paper_id"]
        row = pid_to_row.get(pid)
        if row is None:
            skipped_missing_emb += 1
            continue
        v = E[row]
        _c = r.get("cited_by_count")
        cites = 0 if _c is None or (isinstance(_c, float) and _c != _c) else int(_c or 0)
        w = 1.0 + np.log1p(cites)   # dampen cite inflation
        year = int(r["year"]) if pd.notna(r.get("year")) else None
        authors_list = r.get("authors")
        if authors_list is None or len(authors_list) == 0:
            continue
        keys = set()
        for a in authors_list:
            if not isinstance(a, dict):
                continue
            k = author_key_with_alias(a, alias_map)
            if k and k in keep_keys:
                keys.add(k)
        for k in keys:
            if k not in author_sum:
                author_sum[k] = np.zeros(E.shape[1], dtype=np.float32)
                author_wsum[k] = 0.0
            author_sum[k] += w * v
            author_wsum[k] += w
            if year is not None:
                author_years[k].append((year, w, row))
    if skipped_missing_emb:
        print(f"[sim] skipped {skipped_missing_emb:,} papers without embeddings", flush=True)

    # Author vectors (normalized for cosine)
    keys = sorted(author_sum.keys())
    A = np.stack([author_sum[k] / max(author_wsum[k], 1e-8) for k in keys])
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    print(f"[sim] author vectors: {A.shape}", flush=True)

    # Nearest neighbors via GPU matmul
    print("[sim] computing top-K nearest neighbors...", flush=True)
    t0 = time.time()
    A_t = torch.from_numpy(A).to(DEVICE)
    # Chunked matmul so we don't alloc a 27k×27k float32 matrix all at once
    CHUNK = 2048
    topk_sim = np.zeros((len(keys), TOP_K_SIMILAR), dtype=np.float32)
    topk_idx = np.zeros((len(keys), TOP_K_SIMILAR), dtype=np.int32)
    with torch.no_grad():
        for i in range(0, len(keys), CHUNK):
            q = A_t[i : i + CHUNK]                     # (c, d)
            sims = q @ A_t.T                           # (c, N)
            # mask self
            for r, global_i in enumerate(range(i, min(i + CHUNK, len(keys)))):
                sims[r, global_i] = -1.0
            vals, idxs = torch.topk(sims, TOP_K_SIMILAR, dim=1)
            topk_sim[i : i + CHUNK] = vals.cpu().numpy()
            topk_idx[i : i + CHUNK] = idxs.cpu().numpy()
    print(f"[sim] kNN done in {time.time() - t0:.1f}s", flush=True)

    # Build coauthor-graph adjacency from the already-exported network JSON,
    # then BFS up to BFS_CUTOFF hops from each atlas node so we can label each
    # semantic neighbor pair with coauthor_dist ∈ {1, 2, 3, null}.
    import networkx as nx
    G = nx.Graph()
    for n in net["nodes"]:
        G.add_node(n["id"])
    for e in net["edges"]:
        G.add_edge(e["source"], e["target"])
    print(f"[sim] coauthor graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges — computing BFS distances (cutoff {BFS_CUTOFF})...",
          flush=True)
    t0 = time.time()
    dist_of: dict[int, dict[int, int]] = {}
    for nid in G.nodes():
        dist_of[nid] = nx.single_source_shortest_path_length(G, nid, cutoff=BFS_CUTOFF)
    print(f"[sim] BFS done in {time.time() - t0:.1f}s", flush=True)

    def coauthor_dist(src_nid: int, dst_nid: int) -> int | None:
        d = dist_of.get(src_nid, {}).get(dst_nid)
        return int(d) if d is not None else None

    # Build author_similar.json — annotate each semantic neighbor with coauthor_dist.
    # Also carry ORCID + OpenAlex id so the UI can render profile links directly.
    key_to_orcid = dict(zip(authors_tbl["author_key"], authors_tbl["orcid"]))
    def _orcid_from_key(kk: str) -> str | None:
        orc = key_to_orcid.get(kk)
        # NaN-safe: pandas/parquet yields float NaN for missing ORCIDs, which is truthy
        if isinstance(orc, str) and orc:
            return orc
        # author_key fallback: ORCID-shaped keys (0000-0000-0000-000X)
        if isinstance(kk, str) and len(kk) == 19 and kk[4] == "-":
            return kk
        return None
    def _openalex_from_key(kk: str) -> str | None:
        return kk if isinstance(kk, str) and kk.startswith("a") else None

    author_similar = {}
    phantom_counts = []
    for i, k in enumerate(keys):
        nid = atlas_key_to_nodeid.get(k)
        if nid is None:
            continue
        entries = []
        phantoms = 0
        for j in range(TOP_K_SIMILAR):
            other_k = keys[topk_idx[i, j]]
            other_nid = atlas_key_to_nodeid.get(other_k)
            if other_nid is None:
                continue
            d = coauthor_dist(nid, other_nid)
            # "phantom": never directly coauthored — d is None (disconnected/>=4) or d >= PHANTOM_MIN_HOPS (=2)
            is_phantom = (d is None) or (d >= PHANTOM_MIN_HOPS)
            entries.append({
                "id": other_nid,
                "orcid": _orcid_from_key(other_k),
                "oa": _openalex_from_key(other_k),
                "name": key_to_name.get(other_k) or other_k,
                "sim": round(float(topk_sim[i, j]), 4),
                "d": d,               # coauthor-graph distance (1..3) or null (≥4 / disconnected)
                "phantom": is_phantom,
            })
            if is_phantom:
                phantoms += 1
        if entries:
            author_similar[str(nid)] = entries[:TOP_K_SIMILAR]
            phantom_counts.append(phantoms)
    if phantom_counts:
        avg_p = sum(phantom_counts) / len(phantom_counts)
        print(f"[sim] phantoms per author: mean={avg_p:.1f}/{TOP_K_SIMILAR} "
              f"(fraction={avg_p/TOP_K_SIMILAR:.1%})", flush=True)
    print(f"[sim] similar entries: {len(author_similar):,}", flush=True)

    # ==== Semantic Leiden community detection ====
    # Build a weighted kNN graph in embedding space and run Leiden on it.
    # This partitions authors by semantic similarity independent of who they co-author.
    print(f"[sim] building semantic kNN graph (k={SEM_KNN_K}) ...", flush=True)
    t0 = time.time()
    K = SEM_KNN_K
    sem_sim = np.zeros((len(keys), K), dtype=np.float32)
    sem_idx = np.zeros((len(keys), K), dtype=np.int32)
    with torch.no_grad():
        for i in range(0, len(keys), CHUNK):
            q = A_t[i : i + CHUNK]
            sims = q @ A_t.T
            for r, global_i in enumerate(range(i, min(i + CHUNK, len(keys)))):
                sims[r, global_i] = -1.0
            vals, idxs = torch.topk(sims, K, dim=1)
            sem_sim[i : i + CHUNK] = vals.cpu().numpy()
            sem_idx[i : i + CHUNK] = idxs.cpu().numpy()
    print(f"[sim] kNN({K}) matmul done in {time.time() - t0:.1f}s", flush=True)

    try:
        import igraph as ig
        import leidenalg
        t0 = time.time()
        edges_sem = []
        weights_sem = []
        for i in range(len(keys)):
            for jj in range(K):
                j = int(sem_idx[i, jj])
                s = float(sem_sim[i, jj])
                if s <= 0:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if a == b:
                    continue
                edges_sem.append((a, b))
                weights_sem.append(s)
        sem_g = ig.Graph(n=len(keys), edges=edges_sem, edge_attrs={"weight": weights_sem},
                         directed=False)
        sem_g.simplify(combine_edges={"weight": "sum"})
        part = leidenalg.find_partition(
            sem_g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=SEM_LEIDEN_RESOLUTION,
            seed=SEM_LEIDEN_SEED,
        )
        sem_comm = part.membership
        print(f"[sim] semantic Leiden: {len(set(sem_comm))} communities (Q={part.modularity:.4f}) "
              f"in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[sim] semantic Leiden failed ({e}); filling zeros", flush=True)
        sem_comm = [0] * len(keys)

    # ==== Combined Leiden: multiplex coauthor + semantic ====
    # Merge two graphs into one weighted graph. Coauthor edge weight = log(1+n_collabs);
    # semantic edge weight = max(0, sim-threshold). Alpha blends the two: ALPHA_COMB=0.5
    # means roughly equal say. Leiden on the sum gives communities that respect both
    # social structure and topical alignment.
    print("[sim] building combined multiplex graph...", flush=True)
    t0 = time.time()
    ALPHA_COMB = 0.5           # weight for coauthor layer (1-ALPHA_COMB for semantic)
    SEM_THRESH_COMB = 0.6      # drop weak semantic links to keep backbone
    SEM_K_COMB = 5             # narrower mutual-kNN for combined layer (not 20)
    # Map coauthor-graph node ids → author key → index in `keys`
    nid_to_key = {n["id"]: n.get("key") for n in net["nodes"]}
    idx_of_key = {k: i for i, k in enumerate(keys)}

    edges_comb: list[tuple[int, int]] = []
    weights_comb: list[float] = []
    # Layer 1: coauthor edges (log-scaled weight)
    n_coauth_added = 0
    for e in net["edges"]:
        s_key = nid_to_key.get(e["source"])
        t_key = nid_to_key.get(e["target"])
        if s_key is None or t_key is None:
            continue
        if s_key not in idx_of_key or t_key not in idx_of_key:
            continue
        a, b = idx_of_key[s_key], idx_of_key[t_key]
        if a == b:
            continue
        w = float(np.log1p(e.get("weight", 1)))
        edges_comb.append((min(a, b), max(a, b)))
        weights_comb.append(ALPHA_COMB * w)
        n_coauth_added += 1
    # Layer 2: semantic edges — narrower mutual-top-K + higher threshold so the semantic
    # layer doesn't swamp the coauthor backbone (which has ~68k edges).
    k_comb = min(SEM_K_COMB, K)
    top_mutual_sets_c = [set(int(sem_idx[i, jj]) for jj in range(k_comb)) for i in range(len(keys))]
    n_sem_added = 0
    for i in range(len(keys)):
        for jj in range(k_comb):
            j = int(sem_idx[i, jj])
            if i == j:
                continue
            s = float(sem_sim[i, jj])
            if s < SEM_THRESH_COMB:
                continue
            if i not in top_mutual_sets_c[j]:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges_comb.append((a, b))
            weights_comb.append((1 - ALPHA_COMB) * (s - SEM_THRESH_COMB) / (1 - SEM_THRESH_COMB))
            n_sem_added += 1
    print(f"[sim] combined graph: {n_coauth_added:,} coauthor + {n_sem_added:,} semantic edges "
          f"(α_coauth={ALPHA_COMB:.2f}) in {time.time() - t0:.1f}s", flush=True)

    try:
        t0 = time.time()
        comb_g = ig.Graph(n=len(keys), edges=edges_comb,
                          edge_attrs={"weight": weights_comb}, directed=False)
        comb_g.simplify(combine_edges={"weight": "sum"})
        # Lower resolution than semantic-only Leiden because the multiplex graph is
        # denser (coauthor ≥2 ties + semantic top-20 mutual). 0.5 yields ~100-200
        # communities — comparable to the coauthor Leiden (145).
        comb_part = leidenalg.find_partition(
            comb_g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=0.5,
            seed=42,
        )
        comb_comm = list(comb_part.membership)
        raw_q = comb_part.modularity
        # Collapse small communities (< MIN_SIZE) into a single "misc" bucket — mirrors
        # coauthor_graph.py's island treatment. Keeps ~100-300 meaningful comms + 1 misc.
        MIN_COMB_SIZE = 10
        from collections import Counter as _CntrLocal
        _size = _CntrLocal(comb_comm)
        misc_ids = {c for c, n in _size.items() if n < MIN_COMB_SIZE}
        if misc_ids:
            # Find a free id for misc (max + 1)
            misc_new = max(comb_comm) + 1
            comb_comm = [misc_new if c in misc_ids else c for c in comb_comm]
        print(f"[sim] combined Leiden: {len({c for c in comb_comm})} communities "
              f"(Q={raw_q:.4f}) — collapsed {len(misc_ids)} islands <{MIN_COMB_SIZE} into misc "
              f"in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[sim] combined Leiden failed ({e}); filling zeros", flush=True)
        comb_comm = [0] * len(keys)
    # Renumber combined communities by size
    size_by_cc = _Cntr_tmp(comb_comm) if False else None  # placeholder to avoid import
    from collections import Counter as _CntrCC
    size_by_cc = _CntrCC(comb_comm)
    ordered_cc = [c for c, _ in size_by_cc.most_common()]
    cc_remap = {old: new for new, old in enumerate(ordered_cc)}
    comb_comm = [cc_remap[c] for c in comb_comm]

    # Rank semantic communities by size, renumber desc
    from collections import Counter as _Cntr
    size_by_sc = _Cntr(sem_comm)
    ordered_sc = [c for c, _ in size_by_sc.most_common()]
    sc_remap = {old: new for new, old in enumerate(ordered_sc)}
    sem_comm = [sc_remap[c] for c in sem_comm]

    # Per-semantic-community top authors + title-tf-idf labels (reuse bag from coauthor graph? no —
    # compute here from author_titles so labels reflect semantic clustering).
    print(f"[sim] building semantic community labels...", flush=True)
    # For label text, use titles of all papers authored by community members.
    from sklearn.feature_extraction.text import TfidfVectorizer
    n_sc = max(sem_comm) + 1 if sem_comm else 0
    member_titles: list[list[str]] = [[] for _ in range(n_sc)]
    for i, k in enumerate(keys):
        for row_idx in [idx for (_yr, _w, idx) in author_years.get(k, [])]:
            # row_idx -> paper. We lost the title at that index because emb_df doesn't carry it.
            # Use papers.parquet via paper_id -> title.
            pass
    # Simpler: use papers directly, per-community aggregation via author→sem_comm map.
    key_to_sc = {k: sem_comm[i] for i, k in enumerate(keys)}
    docs_sc = [[] for _ in range(n_sc)]
    for _, r in papers.iterrows():
        authors_list = r.get("authors")
        if authors_list is None or len(authors_list) == 0:
            continue
        title = (r.get("title") or "").strip().lower()
        if not title:
            continue
        visited_scs: set = set()
        for a in authors_list:
            if not isinstance(a, dict):
                continue
            k = author_key_with_alias(a, alias_map)
            sc = key_to_sc.get(k)
            if sc is not None and sc not in visited_scs:
                docs_sc[sc].append(title)
                visited_scs.add(sc)

    _STOP_SEM = {
        "a","an","and","or","the","of","on","in","for","to","with","via","using","based",
        "new","novel","approach","method","methods","model","models","system","systems",
        "analysis","study","review","paper","effect","effects","framework","survey",
        "application","applications","performance","toward","towards","between","under",
        "over","this","that","these","those","we","its","their","their","our","as","from",
        "by","at","is","are","be","into","such","has","have","had","been","can","will",
        "transportation","transport","traffic","travel","road","roads","vehicle","vehicles",
        "driver","drivers","driving","vehicular","network","networks","control","design",
        "data","time","high","low","non","multi","ieee","society","international","paper",
        "paper:","paper,","—","–","·",
    }
    try:
        vec = TfidfVectorizer(max_df=0.5, min_df=3, stop_words=sorted(_STOP_SEM),
                              token_pattern=r"[A-Za-z][A-Za-z\-]{2,}",
                              ngram_range=(1, 2))
        joined = [" ".join(d) for d in docs_sc]
        # Drop empty docs for fitting
        nonempty_idx = [i for i, d in enumerate(joined) if d]
        if nonempty_idx:
            X = vec.fit_transform([joined[i] for i in nonempty_idx])
            vocab = vec.get_feature_names_out()
            sc_labels = [[] for _ in range(n_sc)]
            for row, orig_i in enumerate(nonempty_idx):
                arr = X[row].toarray().flatten()
                top = np.argsort(-arr)[:12]
                picked, seen = [], set()
                for j in top:
                    if arr[j] <= 0: break
                    w = vocab[j]
                    root = w.split()[0]
                    if root in seen and " " not in w: continue
                    picked.append(w); seen.add(root)
                    if len(picked) >= 6: break
                sc_labels[orig_i] = picked
        else:
            sc_labels = [[] for _ in range(n_sc)]
    except Exception as e:
        print(f"[sim] sem-label tfidf failed ({e})", flush=True)
        sc_labels = [[] for _ in range(n_sc)]

    # Per-sc top authors — by weighted degree in the sem kNN graph
    sc_to_members: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(sem_comm):
        sc_to_members[c].append(i)
    sc_meta = []
    for c in range(n_sc):
        members = sc_to_members.get(c, [])
        if not members: continue
        # Top by paper count within members (simplest)
        ranked = sorted(members, key=lambda i: -int(
            authors_tbl.loc[authors_tbl["author_key"] == keys[i], "n_papers"].iloc[0]
            if (authors_tbl["author_key"] == keys[i]).any() else 0
        ))[:5]
        sc_meta.append({
            "id": c,
            "size": len(members),
            "top_authors": [key_to_name.get(keys[i]) or keys[i] for i in ranked],
            "label_words": sc_labels[c] if c < len(sc_labels) else [],
        })
    print(f"[sim] semantic communities: {len(sc_meta)} (mainland+tail)", flush=True)
    print("[sim] top-5 semantic communities:", flush=True)
    for m in sorted(sc_meta, key=lambda x: -x["size"])[:5]:
        hubs = ", ".join(m["top_authors"][:3])
        kw = " · ".join(m["label_words"][:5])
        print(f"  #{m['id']} n={m['size']:>5}  hubs: {hubs} | kw: {kw}", flush=True)

    # ——— Combined-community labels (mirror semantic-community labeling) ———
    print(f"[sim] building combined community labels...", flush=True)
    n_cc = max(comb_comm) + 1 if comb_comm else 0
    key_to_cc = {k: comb_comm[i] for i, k in enumerate(keys)}
    docs_cc = [[] for _ in range(n_cc)]
    for _, r in papers.iterrows():
        authors_list = r.get("authors")
        try:
            if authors_list is None or len(authors_list) == 0:
                continue
        except TypeError:
            continue
        title = (r.get("title") or "").strip().lower()
        if not title:
            continue
        visited_ccs: set = set()
        for a in authors_list:
            if not isinstance(a, dict):
                continue
            k = author_key_with_alias(a, alias_map)
            cc = key_to_cc.get(k)
            if cc is not None and cc not in visited_ccs:
                docs_cc[cc].append(title)
                visited_ccs.add(cc)
    try:
        vec2 = TfidfVectorizer(max_df=0.5, min_df=3, stop_words=sorted(_STOP_SEM),
                               token_pattern=r"[A-Za-z][A-Za-z\-]{2,}",
                               ngram_range=(1, 2))
        joined = [" ".join(d) for d in docs_cc]
        nonempty_idx = [i for i, d in enumerate(joined) if d]
        cc_labels = [[] for _ in range(n_cc)]
        if nonempty_idx:
            X2 = vec2.fit_transform([joined[i] for i in nonempty_idx])
            vocab2 = vec2.get_feature_names_out()
            for row, orig_i in enumerate(nonempty_idx):
                arr = X2[row].toarray().flatten()
                top = np.argsort(-arr)[:12]
                picked, seen = [], set()
                for j in top:
                    if arr[j] <= 0: break
                    w = vocab2[j]
                    root = w.split()[0]
                    if root in seen and " " not in w: continue
                    picked.append(w); seen.add(root)
                    if len(picked) >= 6: break
                cc_labels[orig_i] = picked
    except Exception as e:
        print(f"[sim] combined TF-IDF failed ({e})", flush=True)
        cc_labels = [[] for _ in range(n_cc)]
    cc_to_members: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(comb_comm):
        cc_to_members[c].append(i)
    cc_meta = []
    n_papers_by_key = authors_tbl.set_index("author_key")["n_papers"].to_dict()
    for c in range(n_cc):
        members = cc_to_members.get(c, [])
        if not members: continue
        ranked = sorted(members, key=lambda i: -int(n_papers_by_key.get(keys[i], 0)))[:5]
        cc_meta.append({
            "id": c,
            "size": len(members),
            "top_authors": [key_to_name.get(keys[i]) or keys[i] for i in ranked],
            "label_words": cc_labels[c] if c < len(cc_labels) else [],
        })
    print(f"[sim] combined communities: {len(cc_meta)}", flush=True)
    for m in sorted(cc_meta, key=lambda x: -x["size"])[:5]:
        hubs = ", ".join(m["top_authors"][:3])
        kw = " · ".join(m["label_words"][:5])
        print(f"  #{m['id']} n={m['size']:>5}  hubs: {hubs} | kw: {kw}", flush=True)

    # UMAP to 2D
    print("[sim] UMAP 2D ...", flush=True)
    t0 = time.time()
    import umap
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=UMAP_SEED, verbose=False,
    )
    coords = reducer.fit_transform(A)
    # Normalize to roughly [-100, 100] for nice rendering. SAVE mean+scale so
    # trajectory coords go through the identical post-fit transform.
    umap_mean = coords.mean(axis=0)
    coords = coords - umap_mean
    umap_scale = 100 / max(np.abs(coords).max(), 1e-6)
    coords = coords * umap_scale
    print(f"[sim] UMAP done in {time.time() - t0:.1f}s", flush=True)

    topic_coords = {}
    for i, k in enumerate(keys):
        nid = atlas_key_to_nodeid.get(k)
        if nid is None:
            continue
        topic_coords[str(nid)] = {
            "x": round(float(coords[i, 0]), 2),
            "y": round(float(coords[i, 1]), 2),
            "c": community_by_id.get(nid),                         # coauthor Leiden
            "sc": int(sem_comm[i]) if i < len(sem_comm) else None, # semantic Leiden
            "cc": int(comb_comm[i]) if i < len(comb_comm) else None,  # combined multiplex
            "n": key_to_name.get(k) or k,
            "p": int(n_papers_by_key.get(k, 0)),
            "o": _orcid_from_key(k),                               # normalized ORCID or null
        }

    # Trajectories — per year-bin centroid in UMAP space.
    # Batched: collect ALL (author, bin_start, vec) tuples, stack into one tensor,
    # call reducer.transform ONCE. (Per-author transform was O(k) slow on numba.)
    print("[sim] building trajectories (batched)...", flush=True)
    t0 = time.time()
    pending: list[tuple[str, int, np.ndarray, int]] = []  # (key, bstart, vec, n)
    for k in keys:
        events = author_years.get(k, [])
        if not events:
            continue
        nid = atlas_key_to_nodeid.get(k)
        if nid is None:
            continue
        bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for year, w, idx in events:
            bin_start = (year // TRAJECTORY_BIN_YEARS) * TRAJECTORY_BIN_YEARS
            bins[bin_start].append((w, idx))
        per_bin = []
        for bstart in sorted(bins):
            items = bins[bstart]
            if len(items) < TRAJECTORY_MIN_PAPERS_PER_BIN:
                continue
            wsum = sum(w for w, _ in items)
            vec = np.zeros(E.shape[1], dtype=np.float32)
            for w, idx in items:
                vec += w * E[idx]
            vec /= max(wsum, 1e-8)
            vec /= (np.linalg.norm(vec) + 1e-8)
            per_bin.append((bstart, vec, len(items)))
        if len(per_bin) < 2:
            continue
        for bstart, vec, n in per_bin:
            pending.append((k, bstart, vec, n))
    print(f"[sim] trajectory bins to transform: {len(pending):,}", flush=True)

    trajectories: dict[str, list[dict]] = {}
    if pending:
        all_vecs = np.stack([p[2] for p in pending])
        t1 = time.time()
        # Apply the same (x - umap_mean) * umap_scale transform used on author coords
        # so trajectory points live in the same frame as topic_coords.
        all_xy = (reducer.transform(all_vecs) - umap_mean) * umap_scale
        print(f"[sim] batched UMAP.transform: {time.time() - t1:.1f}s", flush=True)
        per_key: dict[str, list[tuple[int, float, float, int]]] = defaultdict(list)
        for i, (k, bstart, _vec, n) in enumerate(pending):
            per_key[k].append((bstart, float(all_xy[i, 0]), float(all_xy[i, 1]), n))
        for k, items in per_key.items():
            nid = atlas_key_to_nodeid.get(k)
            if nid is None:
                continue
            items.sort(key=lambda t: t[0])
            trajectories[str(nid)] = [
                {"p": int(b), "x": round(x, 2), "y": round(y, 2), "n": int(nn)}
                for (b, x, y, nn) in items
            ]
    print(f"[sim] trajectories: {len(trajectories):,} in {time.time() - t0:.1f}s", flush=True)

    # Also emit a semantic_communities.json with the HSL palette matching the existing one.
    def _golden_hsl(cid: int) -> str:
        hue = (cid * 137.508) % 360
        sat = 70 if cid % 2 == 0 else 55
        light = 58 if cid % 3 == 0 else 50
        return f"hsl({hue:.1f}, {sat}%, {light}%)"

    semantic_communities = [{
        **m,
        "color": _golden_hsl(m["id"]),
    } for m in sc_meta]
    combined_communities = []
    for m in cc_meta:
        is_misc = m["size"] > (len(keys) * 0.15)  # if > 15% of authors, it's the misc bucket
        combined_communities.append({
            **m,
            "color": "hsl(0, 0%, 45%)" if is_misc else _golden_hsl(m["id"] + 11),
            "misc": is_misc,
        })

    # Write outputs
    out_dir = repo / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    # allow_nan=False forces strict JSON — NaN/Infinity raise instead of emitting
    # invalid tokens that the browser's JSON.parse rejects.
    (out_dir / "author_similar.json").write_text(json.dumps(author_similar, allow_nan=False))
    (out_dir / "topic_coords.json").write_text(json.dumps(topic_coords, allow_nan=False))
    (out_dir / "author_trajectories.json").write_text(json.dumps(trajectories, allow_nan=False))
    (out_dir / "semantic_communities.json").write_text(json.dumps(semantic_communities, allow_nan=False))
    (out_dir / "combined_communities.json").write_text(json.dumps(combined_communities, allow_nan=False))
    print(f"[sim] wrote author_similar.json ({len(author_similar):,}), "
          f"topic_coords.json ({len(topic_coords):,}), "
          f"author_trajectories.json ({len(trajectories):,}), "
          f"semantic_communities.json ({len(semantic_communities):,}), "
          f"combined_communities.json ({len(combined_communities):,})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
