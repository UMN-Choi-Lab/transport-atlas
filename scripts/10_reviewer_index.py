#!/usr/bin/env python
"""Build the reviewer-finder index for the static atlas (SPECTER2 path).

Aggregates the existing per-paper SPECTER2 embeddings into per-author vectors,
quantizes to int8, and writes a small metadata JSON plus a flat binary.

Inputs:
    /data2/chois/transport-atlas/paper_embeddings.parquet   (paper_id, emb[768] float16)
    data/interim/papers.parquet                              (paper_id, year, concepts, authors, cited_by_count)
    data/interim/authors.parquet                             (author_key, canonical_name, n_papers, last_year, orcid, venues)
    data/processed/coauthor_network.json                     (atlas node ids, coauthor edges)

Outputs:
    data/processed/reviewer_authors.bin            (n_authors * 768 bytes int8, row-major)
    data/processed/reviewer_index.json             (metadata + per-author entries + COI adjacency)

Schema of reviewer_index.json:
    {
      "version": 2,
      "build_time": "...",
      "model": "allenai/specter2_base",
      "dim": 768,
      "n_authors": <int>,
      "scales": [<float>, ...],          # per-author abs-max, parallel to authors[].
                                          # browser cosine: dot(q, int8_row) * scale_a / 127 / |q|
      "vocab_concepts": ["traffic flow", ...],
      "concept_idf": [<float>, ...],
      "authors": [
        {
          "i": <atlas_node_id>, "k": <author_key>, "n": <name>, "o": <orcid|null>,
          "p": <n_papers_in_corpus>, "y": <last_year>, "v": ["tr-c", "t-its"],
          "ci": [<concept_idx>, ...],     # top concepts for explanation (not ranking)
          "cw": [<float>, ...]
        }, ...
      ],
      "orcid_to_idx": {<orcid>: <author_array_idx>},
      "name_to_idxs": {<lower_canonical_name>: [<idx>, ...]},
      "coi_adj": [[<idx>, ...], ...]
    }

Run inside the project Docker image:
    ./docker/run_embed.sh reviewer-index
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transport_atlas.process.authors import author_key as _raw_author_key

# --- knobs --------------------------------------------------------------------
MIN_PAPERS_FOR_INDEX = 2
TOP_CONCEPTS_PER_AUTHOR = 12              # explanation only, not ranking
MIN_CONCEPT_LEVEL = 2
RECENCY_HALF_LIFE_YEARS = 6.0             # author-vector recency weight
PAPERS_PER_AUTHOR_CAP = 30                # candidate papers shipped per author for ask B
THIS_YEAR = datetime.now().year
EMBED_DIR = Path(os.environ.get("EMBED_OUT", "/data2/chois/transport-atlas"))
EMBED_DIM = 768
PAPER_ROW_BYTES = 4 + EMBED_DIM           # [scale: f32][int8 vec × 768]
MODEL_ID = "allenai/specter2_base"


def _load_alias_map() -> dict[str, str]:
    """Same alias map as 06_author_similarity.py — keep keying consistent."""
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


def _author_key_with_alias(a: dict, alias_map: dict) -> str:
    k = _raw_author_key(a)
    return alias_map.get(k, k) if k else k


def _norm_concept_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def _orcid_from_key(kk: str, key_to_orcid: dict) -> str | None:
    orc = key_to_orcid.get(kk)
    if isinstance(orc, str) and orc:
        return orc
    if isinstance(kk, str) and len(kk) == 19 and kk[4] == "-":
        return kk
    return None


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "data" / "processed"
    print(f"[rev] reading inputs from {repo}", flush=True)

    embed_path = EMBED_DIR / "paper_embeddings.parquet"
    if not embed_path.exists():
        print(f"[rev] missing {embed_path}; run 05_embed_papers.py first",
              file=sys.stderr)
        return 1
    emb_df = pd.read_parquet(embed_path)
    pid_to_row = {pid: i for i, pid in enumerate(emb_df["paper_id"].tolist())}
    E = np.stack(emb_df["emb"].tolist()).astype(np.float32)   # (n_papers, 768)
    if E.shape[1] != EMBED_DIM:
        print(f"[rev] WARN: embedding dim {E.shape[1]} != expected {EMBED_DIM}", flush=True)
    print(f"[rev] paper embeddings: {E.shape[0]:,} x {E.shape[1]}", flush=True)

    papers = pd.read_parquet(repo / "data" / "interim" / "papers.parquet")
    authors_tbl = pd.read_parquet(repo / "data" / "interim" / "authors.parquet")
    net = json.loads((repo / "data" / "processed" / "coauthor_network.json").read_text())
    print(f"[rev] papers={len(papers):,}  authors={len(authors_tbl):,}  "
          f"net.nodes={len(net['nodes']):,}  net.edges={len(net['edges']):,}", flush=True)

    alias_map = _load_alias_map()

    # --- author universe -----------------------------------------------------
    keep_keys = set(
        authors_tbl.loc[authors_tbl["n_papers"] >= MIN_PAPERS_FOR_INDEX, "author_key"]
    )
    key_to_name = dict(zip(authors_tbl["author_key"], authors_tbl["canonical_name"]))
    key_to_orcid = dict(zip(authors_tbl["author_key"], authors_tbl["orcid"]))
    key_to_lastyear = dict(zip(authors_tbl["author_key"], authors_tbl["last_year"]))
    key_to_npapers = dict(zip(authors_tbl["author_key"], authors_tbl["n_papers"]))
    key_to_venues = dict(zip(authors_tbl["author_key"], authors_tbl["venues"]))

    atlas_key_to_nodeid: dict[str, int] = {}
    for n in net["nodes"]:
        k = n.get("key")
        if k:
            atlas_key_to_nodeid[k] = n["id"]
    print(f"[rev] atlas nodes mapped by key: {len(atlas_key_to_nodeid):,}", flush=True)
    keep_keys = {k for k in keep_keys if k in atlas_key_to_nodeid}
    print(f"[rev] eligible authors: {len(keep_keys):,}", flush=True)

    # --- concept matrix (auxiliary; for explanation, not ranking) ------------
    print("[rev] building concept matrix ...", flush=True)
    t0 = time.time()
    concept_index: dict[str, int] = {}
    rows_c, cols_c, vals_c = [], [], []
    for i, raw_concepts in enumerate(papers["concepts"].tolist()):
        if raw_concepts is None:
            continue
        try:
            iter_c = list(raw_concepts)
        except TypeError:
            continue
        for c in iter_c:
            if not isinstance(c, dict):
                continue
            lvl = c.get("level")
            nm = c.get("name")
            sc = c.get("score") or 0
            if lvl is None or nm is None or int(lvl) < MIN_CONCEPT_LEVEL:
                continue
            key = _norm_concept_name(nm)
            if not key:
                continue
            cidx = concept_index.get(key)
            if cidx is None:
                cidx = len(concept_index)
                concept_index[key] = cidx
            rows_c.append(i); cols_c.append(cidx); vals_c.append(float(sc))
    n_concepts = len(concept_index)
    C = sparse.csr_matrix(
        (vals_c, (rows_c, cols_c)),
        shape=(len(papers), max(n_concepts, 1)),
        dtype=np.float32,
    )
    df_c = np.asarray((C > 0).sum(axis=0)).flatten().astype(np.float32)
    concept_idf = (np.log((1 + len(papers)) / (1 + df_c)) + 1.0).astype(np.float32)
    vocab_concepts = [None] * n_concepts
    for nm, idx in concept_index.items():
        vocab_concepts[idx] = nm
    C = C.multiply(concept_idf).tocsr()
    print(f"[rev] concepts: vocab={n_concepts:,}  nnz={C.nnz:,}  "
          f"in {time.time()-t0:.1f}s", flush=True)

    # --- per-author aggregation: weighted mean of SPECTER2 paper vectors ---
    print("[rev] aggregating SPECTER2 author vectors ...", flush=True)
    t0 = time.time()
    paper_pid_to_pidx = {pid: i for i, pid in enumerate(papers["paper_id"].tolist())}
    # Map: author_key -> list of (paper_emb_row, paper_concept_row, weight)
    author_rows: dict[str, list[tuple[int, int, float]]] = defaultdict(list)

    skipped_no_emb = 0
    for _, r in papers.iterrows():
        pid = r["paper_id"]
        emb_row = pid_to_row.get(pid)
        if emb_row is None:
            skipped_no_emb += 1
            continue
        concept_row = paper_pid_to_pidx.get(pid)
        cites = r.get("cited_by_count")
        try:
            cites = 0 if cites is None or (isinstance(cites, float) and cites != cites) else int(cites)
        except (TypeError, ValueError):
            cites = 0
        year = r.get("year")
        if pd.isna(year):
            year = THIS_YEAR
        else:
            year = int(year)
        recency = 0.5 ** (max(0, (THIS_YEAR - year)) / RECENCY_HALF_LIFE_YEARS)
        cite_w = 1.0 + np.log1p(cites)
        authors_list = r.get("authors")
        if authors_list is None or len(authors_list) == 0:
            continue
        keys_in = set()
        for a in authors_list:
            if not isinstance(a, dict):
                continue
            k = _author_key_with_alias(a, alias_map)
            if k and k in keep_keys:
                keys_in.add(k)
        if not keys_in:
            continue
        share = 1.0 / len(keys_in)
        weight = float(cite_w * recency * share)
        for k in keys_in:
            author_rows[k].append((emb_row, concept_row, weight))
    if skipped_no_emb:
        print(f"[rev] {skipped_no_emb:,} papers missing embeddings (skipped)", flush=True)

    keys_sorted = sorted(author_rows.keys())
    n_authors = len(keys_sorted)
    print(f"[rev] computing author vectors for {n_authors:,} authors ...", flush=True)
    A = np.zeros((n_authors, EMBED_DIM), dtype=np.float32)
    by_a_concept: dict[int, np.ndarray] = {}
    for ai, k in enumerate(keys_sorted):
        if ai and ai % 5000 == 0:
            print(f"[rev]   {ai:,}/{n_authors:,}", flush=True)
        items = author_rows[k]
        emb_idx = np.fromiter((i for i, _, _ in items), dtype=np.int64, count=len(items))
        wts = np.fromiter((w for _, _, w in items), dtype=np.float32, count=len(items))
        wsum = float(wts.sum())
        if wsum <= 0:
            continue
        # Weighted mean of paper embeddings.
        A[ai] = (E[emb_idx] * wts[:, None]).sum(axis=0) / wsum
        # Concept aggregation (auxiliary)
        if n_concepts:
            concept_idx = np.fromiter((c for _, c, _ in items if c is not None),
                                      dtype=np.int64, count=sum(1 for _, c, _ in items if c is not None))
            if concept_idx.size:
                # rebuild aligned weights for concept rows
                cw_local = np.fromiter(
                    (w for _, c, w in items if c is not None),
                    dtype=np.float32, count=concept_idx.size,
                )
                aggc = (C[concept_idx].multiply(cw_local.reshape(-1, 1))).sum(axis=0).A1
                if aggc.size and aggc.max() > 0:
                    by_a_concept[ai] = aggc

    # L2-normalize author vectors (cosine = dot of L2-normalized vectors).
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    A = A / norms
    print(f"[rev] author aggregation done in {time.time()-t0:.1f}s  "
          f"(A shape={A.shape})", flush=True)

    # --- int8 quantization (per-vector abs-max scale) ------------------------
    print("[rev] quantizing to int8 ...", flush=True)
    t0 = time.time()
    abs_max = np.abs(A).max(axis=1)                                # (n_authors,)
    abs_max = np.where(abs_max < 1e-8, 1e-8, abs_max)              # avoid div-by-zero
    A_q = np.round(A / abs_max[:, None] * 127.0).clip(-127, 127).astype(np.int8)
    scales = abs_max.astype(np.float32)                            # browser multiplies back
    print(f"[rev] quantize done in {time.time()-t0:.1f}s  "
          f"(int8 shape={A_q.shape}, mean abs_max={float(abs_max.mean()):.4f})",
          flush=True)

    # --- per-author top-paper rows (candidates for ask B "evidence") ---------
    # For each kept author, take the top-N (emb_row, weight) tuples by weight.
    # weight = cite_w * recency * (1/n_authors) — already encodes the ranking we
    # want (most-typical paper for this author). We keep the original E-row index
    # here; the global kept_paper_idx mapping is built below once we know the
    # union across all kept authors.
    author_top_emb_rows: dict[int, list[int]] = {}
    for ai, k in enumerate(keys_sorted):
        if atlas_key_to_nodeid.get(k) is None:
            continue
        items = author_rows[k]
        if not items:
            author_top_emb_rows[ai] = []
            continue
        # sort by weight desc; tie-break stable
        items_sorted = sorted(items, key=lambda t: -t[2])
        top = items_sorted[:PAPERS_PER_AUTHOR_CAP]
        # dedupe emb_row preserving order (a paper can appear once per author anyway)
        seen = set()
        rows: list[int] = []
        for emb_row, _, _ in top:
            if emb_row in seen:
                continue
            seen.add(emb_row)
            rows.append(int(emb_row))
        author_top_emb_rows[ai] = rows

    # --- per-author entries --------------------------------------------------
    authors_out = []
    orcid_to_idx: dict[str, int] = {}
    name_to_idxs: dict[str, list[int]] = defaultdict(list)
    keep_indices: list[int] = []      # original ai → array index in authors_out
    for ai, k in enumerate(keys_sorted):
        nid = atlas_key_to_nodeid.get(k)
        if nid is None:
            continue
        # top concepts for explanation
        ci, cw = [], []
        aggc = by_a_concept.get(ai)
        if aggc is not None:
            nzc = np.nonzero(aggc)[0]
            if nzc.size > TOP_CONCEPTS_PER_AUTHOR:
                topkc = nzc[np.argpartition(-aggc[nzc], TOP_CONCEPTS_PER_AUTHOR)[:TOP_CONCEPTS_PER_AUTHOR]]
            else:
                topkc = nzc
            valsc = aggc[topkc]
            normc = float(np.linalg.norm(valsc))
            if normc > 0:
                valsc = valsc / normc
            order = np.argsort(-valsc)
            ci = [int(topkc[j]) for j in order]
            cw = [round(float(valsc[j]), 4) for j in order]

        venues = key_to_venues.get(k)
        try:
            v_list = [str(x) for x in (venues if venues is not None else [])][:5]
        except TypeError:
            v_list = []
        last_year = key_to_lastyear.get(k)
        try:
            ly = int(last_year) if last_year is not None and not (
                isinstance(last_year, float) and last_year != last_year
            ) else None
        except (TypeError, ValueError):
            ly = None
        np_count = key_to_npapers.get(k, 0)
        try:
            np_count = int(np_count) if np_count is not None else 0
        except (TypeError, ValueError):
            np_count = 0
        name = key_to_name.get(k) or k
        orcid = _orcid_from_key(k, key_to_orcid)
        idx_in_array = len(authors_out)
        authors_out.append({
            "i": int(nid),
            "k": k,
            "n": name,
            "o": orcid,
            "p": np_count,
            "y": ly,
            "v": v_list,
            "ci": ci,
            "cw": cw,
        })
        keep_indices.append(ai)
        if orcid:
            orcid_to_idx[orcid] = idx_in_array
        if isinstance(name, str) and name:
            name_to_idxs[name.lower()].append(idx_in_array)
    print(f"[rev] author entries: {len(authors_out):,}", flush=True)

    # Re-slice the int8 matrix and scales to match the kept authors order.
    A_q_kept = A_q[keep_indices]
    scales_kept = scales[keep_indices]

    # --- per-paper index (ask A: refs blend, ask B: evidence rerank) ---------
    print("[rev] building per-paper index ...", flush=True)
    t0 = time.time()
    # Union all top-N paper rows across kept authors. Preserve sort order (smallest
    # E-row index first) so the bin layout is deterministic build-to-build.
    kept_emb_rows_set: set[int] = set()
    for ai_orig in keep_indices:
        kept_emb_rows_set.update(author_top_emb_rows.get(ai_orig, []))
    kept_emb_rows = sorted(kept_emb_rows_set)
    n_kept_papers = len(kept_emb_rows)
    emb_row_to_pidx: dict[int, int] = {r: i for i, r in enumerate(kept_emb_rows)}

    # Attach per-author paper_idxs (`pi`) — indices into the global kept_emb_rows.
    for ai_keep, ai_orig in enumerate(keep_indices):
        rows = author_top_emb_rows.get(ai_orig, [])
        authors_out[ai_keep]["pi"] = [emb_row_to_pidx[r] for r in rows if r in emb_row_to_pidx]

    # Quantize the kept papers' SPECTER2 vectors (per-row abs-max, int8).
    P = E[np.asarray(kept_emb_rows, dtype=np.int64)].copy()        # (n_kept, 768)
    p_norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    P /= p_norms                                                    # L2-normalize
    p_abs_max = np.abs(P).max(axis=1)
    p_abs_max = np.where(p_abs_max < 1e-8, 1e-8, p_abs_max)
    P_q = np.round(P / p_abs_max[:, None] * 127.0).clip(-127, 127).astype(np.int8)
    p_scales = p_abs_max.astype(np.float32)

    # Build paper metadata + DOI lookup.
    # `pid_to_row` maps paper_id → row in E. We need the inverse for kept rows.
    row_to_pid: dict[int, str] = {row: pid for pid, row in pid_to_row.items()}
    paper_meta: list[dict] = []
    doi_to_idx: dict[str, int] = {}
    titles = papers["title"].tolist()
    years = papers["year"].tolist()
    venues = papers["venue_slug"].tolist()
    dois = papers["doi"].tolist()
    pid_to_papers_row = paper_pid_to_pidx
    skipped_no_meta = 0
    for new_idx, e_row in enumerate(kept_emb_rows):
        pid = row_to_pid.get(e_row)
        if pid is None:
            skipped_no_meta += 1
            paper_meta.append({"t": "", "y": None, "v": "", "d": None})
            continue
        p_pidx = pid_to_papers_row.get(pid)
        if p_pidx is None:
            skipped_no_meta += 1
            paper_meta.append({"t": "", "y": None, "v": "", "d": None})
            continue
        title = titles[p_pidx]
        if not isinstance(title, str):
            title = ""
        title = title.strip()[:240]
        yr_raw = years[p_pidx]
        try:
            yr = int(yr_raw) if yr_raw is not None and not (
                isinstance(yr_raw, float) and yr_raw != yr_raw
            ) else None
        except (TypeError, ValueError):
            yr = None
        vn = venues[p_pidx]
        vn = vn if isinstance(vn, str) else ""
        doi = dois[p_pidx]
        if isinstance(doi, str) and doi:
            doi_clean = doi.strip().lower()
            doi_to_idx[doi_clean] = new_idx
        else:
            doi_clean = None
        paper_meta.append({"t": title, "y": yr, "v": vn, "d": doi_clean})
    print(f"[rev] kept papers: {n_kept_papers:,}  "
          f"(top-{PAPERS_PER_AUTHOR_CAP}/author, dedup'd; "
          f"{len(doi_to_idx):,} DOIs in lookup; "
          f"{skipped_no_meta:,} missing meta) in {time.time()-t0:.1f}s",
          flush=True)

    # --- COI adjacency -------------------------------------------------------
    node_to_arr_idx = {a["i"]: idx for idx, a in enumerate(authors_out)}
    coi_adj: list[list[int]] = [[] for _ in authors_out]
    n_edges = 0
    for e in net["edges"]:
        s = node_to_arr_idx.get(e["source"])
        t = node_to_arr_idx.get(e["target"])
        if s is None or t is None or s == t:
            continue
        coi_adj[s].append(t); coi_adj[t].append(s); n_edges += 1
    for n in net["nodes"]:
        s = node_to_arr_idx.get(n["id"])
        if s is None:
            continue
        for cid, _cnt in (n.get("ac") or []):
            t = node_to_arr_idx.get(cid)
            if t is None or t == s:
                continue
            coi_adj[s].append(t); n_edges += 1
    coi_adj = [sorted(set(neigh)) for neigh in coi_adj]
    print(f"[rev] COI adjacency: {n_edges:,} edge-endpoints used", flush=True)

    # --- write binary + JSON --------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "reviewer_authors.bin"
    A_q_kept.tofile(bin_path)
    print(f"[rev] wrote {bin_path}  "
          f"({bin_path.stat().st_size/(1024*1024):.1f} MB; "
          f"{A_q_kept.shape[0]} x {A_q_kept.shape[1]} int8)", flush=True)

    out = {
        "version": 2,
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": MODEL_ID,
        "dim": EMBED_DIM,
        "n_authors": len(authors_out),
        "n_concepts": len(vocab_concepts),
        "this_year": THIS_YEAR,
        "recency_half_life_years": RECENCY_HALF_LIFE_YEARS,
        "scales": [round(float(x), 6) for x in scales_kept.tolist()],
        "vocab_concepts": vocab_concepts,
        "concept_idf": [round(float(x), 4) for x in concept_idf.tolist()],
        "authors": authors_out,
        "orcid_to_idx": orcid_to_idx,
        "name_to_idxs": dict(name_to_idxs),
        "coi_adj": coi_adj,
    }
    json_path = out_dir / "reviewer_index.json"
    json_path.write_text(json.dumps(out, allow_nan=False, separators=(",", ":")))
    print(f"[rev] wrote {json_path}  "
          f"({json_path.stat().st_size/(1024*1024):.1f} MB)", flush=True)

    # --- per-paper bin + JSON (Range-fetched from the browser) --------------
    # Layout: each row is exactly PAPER_ROW_BYTES bytes, so the browser computes
    # `bytes=[i*ROW, i*ROW+ROW-1]` to fetch row i.
    paper_bin_path = out_dir / "paper_emb.bin"
    with open(paper_bin_path, "wb") as f:
        for i in range(n_kept_papers):
            f.write(p_scales[i].tobytes())            # 4 bytes float32 LE
            f.write(P_q[i].tobytes())                 # 768 bytes int8
    print(f"[rev] wrote {paper_bin_path}  "
          f"({paper_bin_path.stat().st_size/(1024*1024):.1f} MB; "
          f"{n_kept_papers} rows × {PAPER_ROW_BYTES} bytes)", flush=True)

    paper_idx_doc = {
        "version": 1,
        "build_time": out["build_time"],
        "row_bytes": PAPER_ROW_BYTES,
        "dim": EMBED_DIM,
        "n_papers": n_kept_papers,
        "papers_per_author_cap": PAPERS_PER_AUTHOR_CAP,
        "doi_to_idx": doi_to_idx,
        "papers": paper_meta,
    }
    paper_idx_path = out_dir / "paper_index.json"
    paper_idx_path.write_text(json.dumps(paper_idx_doc, allow_nan=False, separators=(",", ":")))
    print(f"[rev] wrote {paper_idx_path}  "
          f"({paper_idx_path.stat().st_size/(1024*1024):.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
