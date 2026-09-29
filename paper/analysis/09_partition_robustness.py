#!/usr/bin/env python
"""§7 — Partition-alignment robustness: NMI/AMI/ECS sweeps + granularity controls.

Reviewer 4, point 4: is the low coauthor-vs-semantic NMI a granularity artifact?
This script sweeps the four reviewer-named knobs (semantic kNN k, Leiden
resolutions, coauthor graph threshold, multiplex mixing weight + semantic
threshold) around the published operating point, reports three agreement
families (NMI raw, AMI chance-corrected, element-centric ECS), and anchors
every cell against a size-matched permutation null.

Emits (all stems prefixed 07_ — §7 of the manuscript; NEVER 09_, which is
taken by trajectory-taxonomy outputs):
    paper/analysis/_partition_robustness{tag}.json   (snapshot; untagged by default)
    <outdir>/07_partition_robustness{tag}.tex
    <outdir>/07_partition_robustness{tag}.pdf  (+ .png)
where <outdir> = paper/analysis/_preview/ by DEFAULT (untracked scratch).
Only with the explicit --emit-manuscript flag do the .tex/figures go to
paper/manuscript/tables|figures/ — that flag belongs to the LATER manuscript
workflow (paper/manuscript/ is a separate Overleaf-synced git repo).

Also writes a resumable cache (author matrix + kNN) under
data/interim/_partition_robustness_cache/ (gitignored; resume by default,
rebuild on input-meta mismatch or --force-cache).

Constants mirrored VERBATIM from elsewhere in the repo — if those files are
retuned, this script must be updated to match:
  scripts/06_author_similarity.py:
    WHITEN_TOP_PC=1; per-dim z-score (torch .std => ddof=1); concept TF-IDF
    min_df=10/max_df=0.5/sublinear_tf/token [a-z_]+; TruncatedSVD 128 rs=42;
    venue LDA (solver=svd, n_components=None, "_unk" for missing venue);
    hybrid sqrt-concat ALPHA_E/C/L=0.55/0.30/0.15; paper weight
    w=1+log1p(cited_by_count) (NaN->0); per-paper author-key dedup via set;
    MIN_PAPERS_FOR_EMBED=2; keys=sorted; SEM_KNN_K=20;
    SEM_LEIDEN_RESOLUTION=1.0; SEM_LEIDEN_SEED=42; semantic kNN edges skip
    sim<=0 and self, simplify(combine_edges={"weight":"sum"}); combined
    multiplex ALPHA_COMB=0.5, SEM_THRESH_COMB=0.6, SEM_K_COMB=5 (mutual),
    coauthor layer weight alpha*log1p(w), semantic layer
    (1-alpha)*(sim-tau)/(1-tau), Leiden RB resolution 0.5 seed 42;
    _load_alias_map / author_key_with_alias (L50-74).
  src/transport_atlas/process/coauthor_graph.py:
    GIANT_THRESHOLD=5; BASE_THRESHOLD=2; ISLAND_MIN_SIZE=10; LEIDEN_SEED=42;
    ModularityVertexPartition == RBConfigurationVertexPartition @ gamma=1.0
    (the calibration cell); _propagate_and_islands (L240-291, copied).
  paper/analysis/03_partition_alignment.py:
    population filter (c/sc/cc non-null, sc/cc not misc-flagged; coauthor
    misc NOT filtered); _variation_of_information (log2 -> bits, copied as
    vi_bits); _tex_escape; OKABE_ITO + rcParams block; EMBED_PATH pattern.

Deliberate deviations (documented per spec):
  - matplotlib + rcParams live INSIDE the figure writer so the pure metric
    helpers import headless for offline pytest (tests/test_partition_
    robustness_metrics.py loads this module via importlib).
  - transport_atlas import is lazy (inside author_key_with_alias) for the
    same reason.
  - 06's <10-member combined-misc collapse is SKIPPED: it is presentation-
    layer relabeling; metrics need raw memberships.
  - Under --sample (smoke) the parity gate degrades to warn-only: a kNN
    partition of a subsample cannot reproduce the full-corpus partition.
  - --tag suffixes the JSON snapshot too (mirrors 05_trajectory_taxonomy.py)
    so tagged variant runs never clobber the headline snapshot.

Run (host or docker; no torch, no GPU, no network):
    PYTHONPATH=src /usr/bin/python3 paper/analysis/09_partition_robustness.py --self-test
    PYTHONPATH=src /usr/bin/python3 paper/analysis/09_partition_robustness.py \
        --sample 3000 --perms 10 --axes 0,A      # smoke
    ./docker/run_embed.sh analysis paper/analysis/09_partition_robustness.py  # full
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             normalized_mutual_info_score)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# --- paths (03_partition_alignment.py EMBED_PATH pattern, L41-44) ----------
EMBED_PATH = (
    Path(os.environ.get("EMBED_OUT", "/data2/chois/transport-atlas"))
    / "paper_embeddings.parquet"
)
PAPERS_PATH = ROOT / "data" / "interim" / "papers.parquet"
AUTHORS_PATH = ROOT / "data" / "interim" / "authors.parquet"
PROCESSED = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "data" / "interim" / "_partition_robustness_cache"
PREVIEW_DIR = ROOT / "paper" / "analysis" / "_preview"
PRIOR_03_SNAPSHOT = ROOT / "paper" / "analysis" / "_partition_alignment.json"

OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]

# --- constants mirrored verbatim from scripts/06_author_similarity.py ------
WHITEN_TOP_PC = 1
ALPHA_E, ALPHA_C, ALPHA_L = 0.55, 0.30, 0.15
MIN_PAPERS_FOR_EMBED = 2
C_SVD_DIM = 128
SEM_KNN_K = 20
SEM_LEIDEN_RESOLUTION = 1.0
SEM_LEIDEN_SEED = 42
ALPHA_COMB = 0.5
SEM_THRESH_COMB = 0.6
SEM_K_COMB = 5
COMB_LEIDEN_RESOLUTION = 0.5
CHUNK = 2048
# --- constants mirrored from src/transport_atlas/process/coauthor_graph.py -
GIANT_THRESHOLD = 5
BASE_THRESHOLD = 2
ISLAND_MIN_SIZE = 10
LEIDEN_SEED = 42

# --- sweep grids (spec §7) --------------------------------------------------
AXIS_A_KS = (10, 15, 20, 30, 50)
AXIS_A_GAMMAS = (0.3, 0.5, 1.0, 2.0, 4.0)
AXIS_B_THRS = (2, 5)
AXIS_B_GAMMAS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
AXIS_B_SEM_REFS = (0.3, 1.0, 4.0)          # rebuilt semantic refs at k=20
AXIS_C_ALPHAS = (0.25, 0.4, 0.5, 0.6, 0.75)
AXIS_C_TAUS = (0.5, 0.55, 0.6, 0.65, 0.7)
AXIS_C_GAMMA_OAT = (0.3, 0.8, 1.0)         # one-at-a-time at (alpha,tau)=(0.5,0.6)
AXIS_D_SEEDS = (1, 7, 13, 42, 99)
PARITY_GATE_HARD = 0.70
PARITY_GATE_WARN = 0.90
DRIFT_WARN_NMI = 0.03
SAMPLE_STREAM = 424242                      # SeedSequence stream id for --sample
SMOKE_BANNER = "SMOKE — NOT PUBLICATION NUMBERS"


def _plog(msg: str) -> None:
    print(f"[robust] {msg}", flush=True)


# ===========================================================================
# Pure metric helpers — data-free, importable headless (tests use importlib).
# ===========================================================================

def contiguize(labels) -> np.ndarray:
    """Map arbitrary hashable/int labels to contiguous 0..K-1 int64 codes."""
    _, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv.astype(np.int64)


def _contingency(ai: np.ndarray, bi: np.ndarray):
    """Sparse contingency of two contiguized label vectors.

    Returns (rows, cols, counts, na, nb) over nonzero cells only.
    """
    na = int(ai.max()) + 1 if ai.size else 0
    nb = int(bi.max()) + 1 if bi.size else 0
    fused = ai.astype(np.int64) * nb + bi
    if na * nb <= 2_000_000:
        cont = np.bincount(fused, minlength=na * nb)
        nz = np.nonzero(cont)[0]
        return nz // nb, nz % nb, cont[nz], na, nb
    uniq, counts = np.unique(fused, return_counts=True)
    return uniq // nb, uniq % nb, counts, na, nb


def fast_nmi(a, b) -> float:
    """NMI with arithmetic normalization; numerically matches sklearn
    normalized_mutual_info_score (validated at startup to <=1e-9)."""
    ai = contiguize(a)
    bi = contiguize(b)
    n = ai.size
    if n == 0:
        return 1.0
    rows, cols, counts, na, nb = _contingency(ai, bi)
    if na == 1 and nb == 1:
        return 1.0                       # sklearn special case: both trivial
    pa = np.bincount(ai, minlength=na).astype(np.float64)
    pb = np.bincount(bi, minlength=nb).astype(np.float64)
    nz = counts.astype(np.float64)
    outer = pa[rows] * pb[cols]
    mi_terms = (nz / n) * (np.log(nz) - np.log(n) + np.log(float(n) * n) - np.log(outer))
    mi = max(float(mi_terms.sum()), 0.0)
    pa_n = pa[pa > 0] / n
    pb_n = pb[pb > 0] / n
    ha = float(-np.sum(pa_n * np.log(pa_n)))
    hb = float(-np.sum(pb_n * np.log(pb_n)))
    norm = max((ha + hb) / 2.0, float(np.finfo(np.float64).eps))
    return float(mi / norm)


def vi_bits(a, b) -> float:
    """VI(U, V) = H(U|V) + H(V|U) in bits. Lower is more agreement.

    Verbatim copy of 03_partition_alignment.py::_variation_of_information.
    """
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    pairs, counts = np.unique(np.stack([a, b], axis=1), axis=0, return_counts=True)
    pa = {}; pb = {}; pab = {}
    for (ai, bi), c in zip(pairs, counts):
        pa[ai] = pa.get(ai, 0) + c
        pb[bi] = pb.get(bi, 0) + c
        pab[(ai, bi)] = pab.get((ai, bi), 0) + c
    h_ab = 0.0
    for (ai, bi), c in pab.items():
        p_ij = c / n
        p_i = pa[ai] / n
        h_ab -= p_ij * np.log2(p_ij / p_i)  # = H(V|U)
    h_ba = 0.0
    for (ai, bi), c in pab.items():
        p_ij = c / n
        p_j = pb[bi] / n
        h_ba -= p_ij * np.log2(p_ij / p_j)  # = H(U|V)
    return h_ab + h_ba


def ecs_hard(a, b) -> float:
    """Element-centric similarity (Gates et al. 2019), closed form for hard
    partitions. Alpha-free: the PPR damping parameter cancels exactly (the
    brute-force equivalence is enforced in --self-test / pytest).

    For each nonzero contingency cell with cluster sizes n_a, n_b, overlap o:
        S_cell = 1 - 0.5*(o*|1/n_a - 1/n_b| + (n_a-o)/n_a + (n_b-o)/n_b)
        ECS    = (1/N) * sum_cells o * S_cell
    Identities: ECS(P,P)=1; ECS(singletons, one-cluster)=1/N.
    """
    ai = contiguize(a)
    bi = contiguize(b)
    n = ai.size
    if n == 0:
        return 1.0
    rows, cols, counts, na, nb = _contingency(ai, bi)
    size_a = np.bincount(ai, minlength=na).astype(np.float64)
    size_b = np.bincount(bi, minlength=nb).astype(np.float64)
    n_a = size_a[rows]
    n_b = size_b[cols]
    o = counts.astype(np.float64)
    s_cell = 1.0 - 0.5 * (o * np.abs(1.0 / n_a - 1.0 / n_b)
                          + (n_a - o) / n_a + (n_b - o) / n_b)
    return float(np.sum(o * s_cell) / n)


def ecs_brute_ppr(a, b, alpha: float = 0.9) -> float:
    """Brute-force ECS via dense personalized-PageRank affinity L1 distance.

    O(n^2) — self-test / unit-test reference only (n<=~500)."""
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)

    def affinity(labels: np.ndarray) -> np.ndarray:
        m = np.zeros((n, n), dtype=np.float64)
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            m[np.ix_(idx, idx)] = alpha / len(idx)
        m[np.diag_indices(n)] += (1.0 - alpha)
        return m

    l1 = np.abs(affinity(a) - affinity(b)).sum(axis=1)
    return float(np.mean(1.0 - l1 / (2.0 * alpha)))


def permuted_labels(b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One size-matched shuffle: a full permutation of the label vector
    preserves BOTH community-size distributions exactly and destroys only
    element-wise correspondence."""
    return rng.permutation(b)


def perm_null(a, b, m: int, rng: np.random.Generator) -> np.ndarray:
    """m size-matched permutation-null NMI draws (permute ONE vector: b)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.array([fast_nmi(a, permuted_labels(b, rng)) for _ in range(m)],
                    dtype=np.float64)


def null_stats(obs_nmi: float, ami: float, nulls: np.ndarray) -> dict:
    """Null summary. Degenerate z / perm_adjusted are emitted as None (never
    inf/NaN — the snapshot is written with allow_nan=False)."""
    m = int(len(nulls))
    mean = float(np.mean(nulls)) if m else 0.0
    sd = float(np.std(nulls, ddof=1)) if m > 1 else 0.0
    z = float((obs_nmi - mean) / sd) if sd >= 1e-12 else None
    p_emp = float((1 + int(np.sum(nulls >= obs_nmi))) / (m + 1)) if m else None
    excess = float(obs_nmi - mean)
    denom = 1.0 - mean
    perm_adjusted = float((obs_nmi - mean) / denom) if denom >= 1e-12 else None
    gap = float(perm_adjusted - ami) if perm_adjusted is not None else None
    return {"mean": mean, "sd": sd, "z": z, "p_emp": p_emp, "m": m,
            "excess": excess, "perm_adjusted": perm_adjusted,
            "ami_agreement_gap": gap}


# ===========================================================================
# Static grid enumeration — data-free, fully deterministic. cell_index is
# assigned BEFORE any compute and is independent of --axes subsetting, so
# per-cell RNG SeedSequence([base_seed, cell_index]) makes results identical
# across execution orders and axis subsets.
# ===========================================================================

def enumerate_grid() -> list[dict]:
    cells: list[dict] = []

    def add(axis, cell_id, spec_a, spec_b, population="pstar",
            perms_mult=1, params=None):
        cells.append({
            "cell_index": len(cells), "axis": axis, "cell_id": cell_id,
            "A": spec_a, "B": spec_b, "population": population,
            "perms_mult": perms_mult, "params": params or {},
        })

    # Cell 0 — headline continuity on shipped labels (03's alignment_dict
    # schema), on BOTH 03's exact population and P*; 10x null draws.
    for pop in ("pop03", "pstar"):
        for a, b in (("c", "sc"), ("c", "cc"), ("sc", "cc")):
            add("0", f"0:{a}-vs-{b}@{pop}", ("shipped", a), ("shipped", b),
                population=pop, perms_mult=10,
                params={"pair": [a, b], "population": pop})
    # Axis A — semantic k x resolution, full cross, vs shipped coauthor c.
    for k in AXIS_A_KS:
        for g in AXIS_A_GAMMAS:
            add("A", f"A:k={k},gs={g}", ("sem", k, g, LEIDEN_SEED),
                ("shipped", "c"), params={"k": k, "gamma_sem": g,
                                          "seed": LEIDEN_SEED})
    # Axis B — coauthor granularity + graph-threshold control, vs shipped sc
    # AND vs rebuilt semantic at k=20, gamma in AXIS_B_SEM_REFS.
    for thr in AXIS_B_THRS:
        for g in AXIS_B_GAMMAS:
            add("B", f"B:thr={thr},gc={g}|sc", ("co", thr, g, LEIDEN_SEED),
                ("shipped", "sc"),
                params={"thr": thr, "gamma_co": g, "ref": "shipped_sc",
                        "seed": LEIDEN_SEED})
            for gref in AXIS_B_SEM_REFS:
                add("B", f"B:thr={thr},gc={g}|sem(k=20,gs={gref})",
                    ("co", thr, g, LEIDEN_SEED),
                    ("sem", SEM_KNN_K, gref, LEIDEN_SEED),
                    params={"thr": thr, "gamma_co": g,
                            "ref": f"sem_k20_gs{gref}", "seed": LEIDEN_SEED})
    # Calibration cell: RB@1.0 == shipped ModularityVertexPartition method.
    add("B", "B:cal:thr=5,gc=1.0|c",
        ("co", GIANT_THRESHOLD, 1.0, LEIDEN_SEED), ("shipped", "c"),
        params={"thr": GIANT_THRESHOLD, "gamma_co": 1.0, "ref": "shipped_c",
                "calibration": True, "seed": LEIDEN_SEED})
    # Axis C — multiplex knobs, each vs BOTH shipped c and shipped sc.
    combos = [(al, ta, COMB_LEIDEN_RESOLUTION)
              for al in AXIS_C_ALPHAS for ta in AXIS_C_TAUS]
    combos += [(ALPHA_COMB, SEM_THRESH_COMB, gc) for gc in AXIS_C_GAMMA_OAT]
    for al, ta, gc in combos:
        for ref in ("c", "sc"):
            add("C", f"C:a={al},t={ta},gcomb={gc}|{ref}",
                ("comb", al, ta, gc, LEIDEN_SEED), ("shipped", ref),
                params={"alpha": al, "tau": ta, "gamma_comb": gc,
                        "ref": f"shipped_{ref}", "seed": LEIDEN_SEED})
    # Axis D — optimizer-seed stability of the three published configs.
    for seed in AXIS_D_SEEDS:
        add("D", f"D:sem*@seed={seed}",
            ("sem", SEM_KNN_K, SEM_LEIDEN_RESOLUTION, seed), ("shipped", "sc"),
            params={"config": "semantic", "seed": seed})
    for seed in AXIS_D_SEEDS:
        add("D", f"D:co*@seed={seed}",
            ("co", GIANT_THRESHOLD, 1.0, seed), ("shipped", "c"),
            params={"config": "coauthor_control", "seed": seed})
    for seed in AXIS_D_SEEDS:
        add("D", f"D:comb*@seed={seed}",
            ("comb", ALPHA_COMB, SEM_THRESH_COMB, COMB_LEIDEN_RESOLUTION, seed),
            ("shipped", "cc"), params={"config": "combined", "seed": seed})
    return cells


# ===========================================================================
# Stage 1 — hybrid author matrix (numpy CPU translation of scripts/
# 06_author_similarity.py), cached under data/interim/.
# ===========================================================================

def _load_alias_map() -> dict[str, str]:
    """Manual aliases from pipeline.yaml + auto-detected ORCID splits from
    dedupe. Verbatim copy of scripts/06_author_similarity.py L50-69
    (repo -> ROOT)."""
    import yaml
    cfg = ROOT / "config" / "pipeline.yaml"
    pipe = yaml.safe_load(cfg.read_text()) or {}
    mp = {}
    for a in pipe.get("author_aliases", []) or []:
        ids = a.get("openalex_ids") or []
        if len(ids) < 2:
            continue
        target = ids[0].lower()
        for other in ids[1:]:
            mp[other.lower()] = target
    auto_path = ROOT / "data" / "interim" / "author_aliases_auto.json"
    if auto_path.exists():
        auto_map = json.loads(auto_path.read_text())
        for k, v in auto_map.items():
            mp.setdefault(k, v)  # manual entries win
    return mp


_AUTHOR_KEY_FN = None


def author_key_with_alias(a: dict, alias_map: dict) -> str:
    """Verbatim semantics of scripts/06_author_similarity.py L72-74; the
    transport_atlas import is lazy so the module stays importable by the
    offline metric tests without touching src/ dependencies."""
    global _AUTHOR_KEY_FN
    if _AUTHOR_KEY_FN is None:
        from transport_atlas.process.authors import author_key as _ak
        _AUTHOR_KEY_FN = _ak
    k = _AUTHOR_KEY_FN(a)
    return alias_map.get(k, k) if k else k


def _whiten_all(e_raw: np.ndarray) -> tuple[np.ndarray, float]:
    """06's whitening (L99-113) in numpy: mean-center, remove top-1 PC via
    768x768 eigh, per-dim z-score. torch .std() is unbiased -> ddof=1."""
    mu = e_raw.mean(axis=0, keepdims=True)
    ec = (e_raw - mu).astype(np.float32)
    cov = (ec.T.astype(np.float64) @ ec.astype(np.float64)) / ec.shape[0]
    evals, evecs = np.linalg.eigh(cov)  # ascending
    top_dirs = evecs[:, -WHITEN_TOP_PC:].astype(np.float32)
    ec = ec - (ec @ top_dirs) @ top_dirs.T
    std = ec.std(axis=0, ddof=1, keepdims=True) + 1e-8
    top1_share = float(evals[-1] / evals.sum())
    return (ec / std).astype(np.float32), top1_share


def _l2_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def build_author_matrix(sample: int, base_seed: int) -> tuple[list[str], np.ndarray, dict]:
    """Rebuild the hybrid author matrix exactly as 06_author_similarity.py
    (torch -> numpy). Returns (keys, A, build_stats)."""
    import duckdb  # DuckDB for all >10k-row parquet reads (repo rule)

    t0 = time.time()
    con = duckdb.connect()
    # ORDER BY pins row order (DuckDB parallel scans are order-unstable);
    # affects only float summation order, not any label semantics.
    emb_df = con.execute(
        "SELECT paper_id, emb FROM read_parquet(?) ORDER BY paper_id",
        [str(EMBED_PATH)]).df()
    pid_to_row = {pid: i for i, pid in enumerate(emb_df["paper_id"].tolist())}
    e_raw = np.stack(emb_df["emb"].tolist()).astype(np.float32)
    n_rows = e_raw.shape[0]
    del emb_df
    _plog(f"embeddings loaded: {e_raw.shape} in {time.time() - t0:.1f}s")

    papers_df = con.execute(
        "SELECT paper_id, venue_slug, cited_by_count, concepts, authors "
        "FROM read_parquet(?)", [str(PAPERS_PATH)]).df()
    authors_df = con.execute(
        "SELECT author_key, n_papers FROM read_parquet(?)",
        [str(AUTHORS_PATH)]).df()
    con.close()
    keep_keys = set(
        authors_df.loc[authors_df["n_papers"] >= MIN_PAPERS_FOR_EMBED,
                       "author_key"])
    alias_map = _load_alias_map()

    # Single pass over papers: concept docs + venue labels (06 L135-158) and
    # per-paper (row, weight, author-keys) records (06 L241-269). Column-zip
    # iteration — never iterrows.
    t0 = time.time()
    concept_docs: list[str] = [""] * n_rows
    venue_labels: list[str] = ["_unk"] * n_rows
    paper_author_recs: list[tuple[int, float, tuple[str, ...]]] = []
    skipped_missing_emb = 0
    cols = [papers_df[c].to_numpy()
            for c in ("paper_id", "venue_slug", "cited_by_count",
                      "concepts", "authors")]
    for pid, venue, cites, concepts, authors_l in zip(*cols):
        row = pid_to_row.get(pid)
        if row is None:
            skipped_missing_emb += 1
            continue
        try:
            c_list = [] if concepts is None or len(concepts) == 0 else list(concepts)
        except TypeError:
            c_list = []
        toks: list[str] = []
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
        concept_docs[row] = " ".join(toks)
        venue_labels[row] = venue if isinstance(venue, str) and venue else "_unk"
        cites_i = (0 if cites is None or (isinstance(cites, float) and cites != cites)
                   else int(cites or 0))
        w = 1.0 + math.log1p(cites_i)
        try:
            a_list = [] if authors_l is None or len(authors_l) == 0 else list(authors_l)
        except TypeError:
            a_list = []
        keys_here: set[str] = set()
        for a in a_list:
            if not isinstance(a, dict):
                continue
            k = author_key_with_alias(a, alias_map)
            if k and k in keep_keys:
                keys_here.add(k)
        if keys_here:
            paper_author_recs.append((row, w, tuple(sorted(keys_here))))
    del papers_df, authors_df
    _plog(f"papers pass: {len(paper_author_recs):,} papers with authors+emb, "
          f"{skipped_missing_emb:,} without embeddings, "
          f"in {time.time() - t0:.1f}s")

    # Features (06 L99-195): whiten -> concept TF-IDF/SVD -> venue LDA ->
    # sqrt-weighted hybrid concat. All fit on the full corpus, as 06 does.
    from sklearn.decomposition import TruncatedSVD
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.feature_extraction.text import TfidfVectorizer

    t0 = time.time()
    white, top1_share = _whiten_all(e_raw)
    _plog(f"whitening done in {time.time() - t0:.1f}s "
          f"(top-1 PC share {top1_share * 100:.1f}%)")

    t0 = time.time()
    vec = TfidfVectorizer(min_df=10, max_df=0.5, sublinear_tf=True,
                          token_pattern=r"[a-z_]+")
    c_sparse = vec.fit_transform(concept_docs)
    svd = TruncatedSVD(n_components=min(C_SVD_DIM, c_sparse.shape[1] - 1),
                       random_state=42)
    c_dense = _l2_rows(svd.fit_transform(c_sparse).astype(np.float32))
    _plog(f"concept TF-IDF: sparse={c_sparse.shape} -> SVD {c_dense.shape} "
          f"evr={svd.explained_variance_ratio_.sum():.3f} "
          f"in {time.time() - t0:.1f}s")

    t0 = time.time()
    lda = LinearDiscriminantAnalysis(n_components=None, solver="svd")
    l_dense = _l2_rows(lda.fit_transform(white, venue_labels).astype(np.float32))
    _plog(f"venue LDA: {white.shape} -> {l_dense.shape} "
          f"({len(lda.classes_)} venues) in {time.time() - t0:.1f}s")

    e_norm = _l2_rows(white)
    h = np.hstack([np.sqrt(ALPHA_E) * e_norm,
                   np.sqrt(ALPHA_C) * c_dense,
                   np.sqrt(ALPHA_L) * l_dense]).astype(np.float32)
    build_stats = {
        "top1_pc_share": round(top1_share, 4),
        "concept_vocab": int(len(vec.vocabulary_)),
        "concept_evr": round(float(svd.explained_variance_ratio_.sum()), 4),
        "lda_classes": int(len(lda.classes_)),
        "hybrid_dim": int(h.shape[1]),
        "n_emb_rows": int(n_rows),
        "skipped_missing_emb": int(skipped_missing_emb),
    }
    del e_raw, white, e_norm, c_dense, l_dense, c_sparse

    # Author aggregation (06 L234-276): citation-weighted mean, L2-normalized.
    t0 = time.time()
    author_sum: dict[str, np.ndarray] = {}
    author_wsum: dict[str, float] = {}
    dim = h.shape[1]
    for row, w, ks in paper_author_recs:
        v = h[row]
        for k in ks:
            if k not in author_sum:
                author_sum[k] = np.zeros(dim, dtype=np.float32)
                author_wsum[k] = 0.0
            author_sum[k] += w * v
            author_wsum[k] += w
    keys = sorted(author_sum)
    a_mat = np.stack([author_sum[k] / max(author_wsum[k], 1e-8) for k in keys])
    a_mat = _l2_rows(a_mat).astype(np.float32)
    assert np.isfinite(a_mat).all(), "non-finite author vectors"
    del h, author_sum, author_wsum
    _plog(f"author vectors: {a_mat.shape} in {time.time() - t0:.1f}s")

    if sample and sample < len(keys):
        rng = np.random.default_rng(
            np.random.SeedSequence([base_seed, SAMPLE_STREAM]))
        sel = np.sort(rng.choice(len(keys), size=sample, replace=False))
        keys = [keys[int(i)] for i in sel]
        a_mat = a_mat[sel]
        _plog(f"--sample: subsampled author universe to {len(keys):,}")
    build_stats["n_keys"] = int(len(keys))
    return keys, a_mat, build_stats


def knn_topk(a_mat: np.ndarray, kmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Chunked float32 matmul kNN at depth kmax; every swept k <= kmax is a
    prefix slice. Self is masked to -1 (mirrors 06 L287-296)."""
    n = a_mat.shape[0]
    k_eff = min(kmax, n - 1)
    knn_idx = np.zeros((n, k_eff), dtype=np.int32)
    knn_sim = np.zeros((n, k_eff), dtype=np.float32)
    t0 = time.time()
    for i0 in range(0, n, CHUNK):
        s = a_mat[i0:i0 + CHUNK] @ a_mat.T
        for r in range(s.shape[0]):
            s[r, i0 + r] = -1.0
        part = np.argpartition(-s, k_eff - 1, axis=1)[:, :k_eff]
        vals = np.take_along_axis(s, part, axis=1)
        order = np.argsort(-vals, axis=1, kind="stable")
        knn_idx[i0:i0 + CHUNK] = np.take_along_axis(part, order, axis=1)
        knn_sim[i0:i0 + CHUNK] = np.take_along_axis(vals, order, axis=1)
    _plog(f"kNN(k={k_eff}) over {n:,} authors in {time.time() - t0:.1f}s")
    return knn_idx, knn_sim


def _cache_input_meta(kmax: int, sample: int, seed: int) -> dict:
    def fstat(p: Path) -> dict:
        st = p.stat()
        return {"path": str(p), "mtime": int(st.st_mtime), "size": int(st.st_size)}

    inputs = [fstat(EMBED_PATH), fstat(PAPERS_PATH), fstat(AUTHORS_PATH),
              fstat(ROOT / "config" / "pipeline.yaml")]
    auto = ROOT / "data" / "interim" / "author_aliases_auto.json"
    if auto.exists():
        inputs.append(fstat(auto))
    return {
        "inputs": inputs, "kmax": int(kmax), "sample": int(sample),
        "seed": int(seed), "version": 1,
        "mirrored": {"WHITEN_TOP_PC": WHITEN_TOP_PC, "ALPHA_E": ALPHA_E,
                     "ALPHA_C": ALPHA_C, "ALPHA_L": ALPHA_L,
                     "MIN_PAPERS_FOR_EMBED": MIN_PAPERS_FOR_EMBED,
                     "C_SVD_DIM": C_SVD_DIM},
    }


def load_or_build_cache(kmax: int, sample: int, seed: int,
                        force: bool) -> tuple[list[str], np.ndarray,
                                              np.ndarray, np.ndarray, dict]:
    """Resume-by-default cache of (keys, A, knn_idx, knn_sim) keyed on input
    mtimes + params. Mismatch -> print reason and rebuild; --force-cache
    rebuilds unconditionally."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE_DIR / "cache_meta.json"
    mat_path = CACHE_DIR / "author_matrix.npz"
    knn_path = CACHE_DIR / f"knn_k{kmax}.npz"
    want = _cache_input_meta(kmax, sample, seed)
    if not force and meta_path.exists() and mat_path.exists() and knn_path.exists():
        try:
            have = json.loads(meta_path.read_text())
            have_cmp = {k: have.get(k) for k in want}
            if have_cmp == want:
                t0 = time.time()
                with np.load(mat_path, allow_pickle=False) as z:
                    keys = [str(k) for k in z["keys"]]
                    a_mat = z["a_mat"]
                with np.load(knn_path, allow_pickle=False) as z:
                    knn_idx = z["knn_idx"]
                    knn_sim = z["knn_sim"]
                _plog(f"cache HIT: {len(keys):,} authors, kNN {knn_idx.shape} "
                      f"loaded in {time.time() - t0:.1f}s from {CACHE_DIR}")
                return keys, a_mat, knn_idx, knn_sim, have
            _plog("cache STALE (inputs/params changed) — rebuilding:")
            for k in want:
                if have_cmp.get(k) != want[k]:
                    _plog(f"  mismatch in {k!r}")
        except Exception as e:  # corrupt cache -> rebuild
            _plog(f"cache unreadable ({e}) — rebuilding")
    elif force:
        _plog("--force-cache: rebuilding author matrix + kNN")
    else:
        _plog("cache MISS — building author matrix + kNN")

    keys, a_mat, build_stats = build_author_matrix(sample, seed)
    knn_idx, knn_sim = knn_topk(a_mat, kmax)
    np.savez(mat_path, keys=np.array(keys), a_mat=a_mat)
    np.savez(knn_path, knn_idx=knn_idx, knn_sim=knn_sim)
    meta = dict(want)
    meta["build_stats"] = build_stats
    meta_path.write_text(json.dumps(meta, indent=2, allow_nan=False))
    _plog(f"cache written to {CACHE_DIR}")
    return keys, a_mat, knn_idx, knn_sim, meta


# ===========================================================================
# Shipped artifacts + fixed evaluation populations (pop03 and P*).
# ===========================================================================

def load_shipped() -> dict:
    _plog("loading shipped artifacts ...")
    net = json.loads((PROCESSED / "coauthor_network.json").read_text())
    tc = json.loads((PROCESSED / "topic_coords.json").read_text())
    sem_comms = json.loads((PROCESSED / "semantic_communities.json").read_text())
    comb_comms = json.loads((PROCESSED / "combined_communities.json").read_text())
    co_comms = net["meta"]["communities"]
    return {
        "net": net, "topic_coords": tc,
        "sem_comms": sem_comms, "comb_comms": comb_comms, "co_comms": co_comms,
        "sem_misc": {c["id"] for c in sem_comms if c.get("misc")},
        "comb_misc": {c["id"] for c in comb_comms if c.get("misc")},
        # dynamic counts — never hardcode (cf. the "All 36 venues" bug class)
        "n_comms_published": {
            "coauthor": len(co_comms),
            "coauthor_misc": sum(1 for c in co_comms if c.get("misc")),
            "semantic": len(sem_comms),
            "semantic_misc": sum(1 for c in sem_comms if c.get("misc")),
            "combined": len(comb_comms),
            "combined_misc": sum(1 for c in comb_comms if c.get("misc")),
        },
    }


def _giant_component_roots(net: dict) -> tuple[np.ndarray, int]:
    """Union-find giant component of the base (weight>=BASE_THRESHOLD, i.e.
    all exported) coauthor edges. Returns (root per node id, giant_root)."""
    n = max(nd["id"] for nd in net["nodes"]) + 1
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for e in net["edges"]:
        ra, rb = find(int(e["source"])), find(int(e["target"]))
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    sizes = Counter(roots[[nd["id"] for nd in net["nodes"]]].tolist())
    giant_root = max(sizes, key=lambda r: sizes[r])
    return roots, int(giant_root)


def build_populations(shipped: dict, key_to_idx: dict[str, int]) -> dict:
    """pop03 = 03_partition_alignment.py's exact published filter (c/sc/cc
    non-null; sc not semantic-misc; cc not combined-misc; coauthor misc NOT
    filtered — mirrored exactly for headline continuity).
    P* = pop03 INTERSECT (key in rebuilt author matrix) INTERSECT (node in
    giant component of the base coauthor graph). Computed ONCE from shipped
    artifacts and held fixed across every grid cell."""
    tc = shipped["topic_coords"]
    net = shipped["net"]
    sem_misc = shipped["sem_misc"]
    comb_misc = shipped["comb_misc"]
    nid_to_key = {nd["id"]: nd.get("key") for nd in net["nodes"]}
    roots, giant_root = _giant_component_roots(net)

    excl = Counter()
    p03 = {"c": [], "sc": [], "cc": [], "nids": []}
    ps = {"c": [], "sc": [], "cc": [], "nids": [], "midx": [], "keys": []}
    for nid_str in sorted(tc, key=int):
        v = tc[nid_str]
        nid = int(nid_str)
        co, sc, cc = v.get("c"), v.get("sc"), v.get("cc")
        if co is None or sc is None or cc is None:
            excl["missing_label"] += 1
            continue
        if sc in sem_misc or cc in comb_misc:
            excl["misc_flagged"] += 1
            continue
        p03["c"].append(co); p03["sc"].append(sc); p03["cc"].append(cc)
        p03["nids"].append(nid)
        key = nid_to_key.get(nid)
        midx = key_to_idx.get(key) if key else None
        if midx is None:
            excl["not_in_author_matrix"] += 1
            continue
        if nid >= len(roots) or roots[nid] != giant_root:
            excl["not_in_giant_component"] += 1
            continue
        ps["c"].append(co); ps["sc"].append(sc); ps["cc"].append(cc)
        ps["nids"].append(nid); ps["midx"].append(midx); ps["keys"].append(key)

    pops = {
        "pop03": {k: np.asarray(p03[k]) for k in ("c", "sc", "cc", "nids")},
        "pstar": {k: np.asarray(ps[k]) for k in ("c", "sc", "cc", "nids", "midx")},
        "exclusion_counts": dict(excl),
        "n_topic_coords": len(tc),
        "n_pop03": len(p03["nids"]),
        "n_eval": len(ps["nids"]),
    }
    _plog(f"populations: topic_coords={len(tc):,}  pop03={pops['n_pop03']:,}  "
          f"P*={pops['n_eval']:,}  exclusions={dict(excl)}")
    return pops


# ===========================================================================
# Stage 2 — partition builders (igraph + leidenalg, fully seeded).
# ===========================================================================

def _leiden_rb(graph, gamma: float, seed: int):
    import leidenalg
    return leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=gamma, seed=seed)


def build_sem_partition(knn_idx: np.ndarray, knn_sim: np.ndarray, k: int,
                        gamma: float, seed: int) -> tuple[np.ndarray, float]:
    """Mirrors 06 L389-431: k-prefix of the cached kNN, skip sim<=0 and self,
    undirected (min,max) edges with weight=sim, simplify(sum) so mutual pairs
    double-weight, Leiden RBConfiguration at resolution gamma."""
    import igraph as ig
    n = knn_idx.shape[0]
    k_eff = min(k, knn_idx.shape[1])
    # Vectorized but identical (row-major) edge order to 06's double loop.
    ii = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    jj = knn_idx[:, :k_eff].astype(np.int64).ravel()
    ss = knn_sim[:, :k_eff].astype(np.float64).ravel()
    mask = (ss > 0) & (ii != jj)
    a = np.minimum(ii[mask], jj[mask])
    b = np.maximum(ii[mask], jj[mask])
    g = ig.Graph(n=n, edges=list(zip(a.tolist(), b.tolist())),
                 edge_attrs={"weight": ss[mask].tolist()}, directed=False)
    g.simplify(combine_edges={"weight": "sum"})
    part = _leiden_rb(g, gamma, seed)
    return np.asarray(part.membership, dtype=np.int64), float(part.modularity)


def _propagate_and_islands(node_to_cid: dict, edges_raw) -> tuple[dict, int, int | None]:
    """Propagate labels along base-threshold edges; detect islands; return
    (cids, n_island, misc_cid).

    Verbatim copy of src/transport_atlas/process/coauthor_graph.py L240-291
    (log.info -> _plog; operates on int node ids here, the code is
    id-type-agnostic). Guarantees EVERY base-graph node ends labeled, so the
    evaluation population never shrinks per cell."""
    import networkx as nx
    adj: dict = defaultdict(list)
    for a, b, w in edges_raw:
        adj[a].append((b, w))
        adj[b].append((a, w))

    # Label propagation
    n_before = len(node_to_cid)
    for _ in range(20):
        changed = False
        for nid in list(adj.keys()):
            if nid in node_to_cid:
                continue
            votes: dict = defaultdict(int)
            for nb, w in adj[nid]:
                if nb in node_to_cid:
                    votes[node_to_cid[nb]] += w
            if votes:
                node_to_cid[nid] = max(votes.items(), key=lambda kv: kv[1])[0]
                changed = True
        if not changed:
            break
    _plog(f"  propagation attached {len(node_to_cid) - n_before:,} fringe nodes")

    # Islands (disconnected from mainland)
    unassigned = {nid for nid in adj if nid not in node_to_cid}
    G_un = nx.Graph()
    G_un.add_nodes_from(unassigned)
    for a, b, _ in edges_raw:
        if a in unassigned and b in unassigned:
            G_un.add_edge(a, b)
    island_comps = sorted(nx.connected_components(G_un), key=len, reverse=True)
    next_cid = (max(node_to_cid.values()) + 1) if node_to_cid else 0
    n_island = 0
    misc: list = []
    for comp in island_comps:
        if len(comp) >= ISLAND_MIN_SIZE:
            for m in comp:
                node_to_cid[m] = next_cid
            next_cid += 1
            n_island += 1
        else:
            misc.extend(comp)
    misc_cid = next_cid if misc else None
    if misc_cid is not None:
        for m in misc:
            node_to_cid[m] = misc_cid
        next_cid += 1
    _plog(f"  islands: {n_island} communities + {len(misc)} in misc")
    return node_to_cid, n_island, misc_cid


def build_coauthor_control(base_edges: list[tuple[int, int, int]], thr: int,
                           gamma: float, seed: int) -> tuple[dict, float]:
    """ANALYSIS-ONLY granularity/threshold control replicating coauthor_graph
    .py's pipeline: (i) mainland = largest CC of the weight>=thr subgraph;
    (ii) Leiden RBConfiguration (== shipped ModularityVertexPartition at
    gamma=1.0) on the mainland; (iii) _propagate_and_islands over ALL base
    edges so every base-graph node gets a label. NOT the published atlas
    partition unless (thr=GIANT_THRESHOLD, gamma=1.0)."""
    import igraph as ig
    strong = [(u, v, w) for (u, v, w) in base_edges if w >= thr]
    nodes = sorted({u for u, _v, _w in strong} | {v for _u, v, _w in strong})
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes),
                 edges=[(idx[u], idx[v]) for u, v, _ in strong],
                 edge_attrs={"weight": [float(w) for _, _, w in strong]},
                 directed=False)
    comps = g.connected_components()
    giant = max(range(len(comps)), key=lambda c: len(comps[c]))
    sub_nodes = [i for i, m in enumerate(comps.membership) if m == giant]
    sub = g.induced_subgraph(sub_nodes)
    part = _leiden_rb(sub, gamma, seed)
    node_to_cid = {nodes[sub_nodes[i]]: int(cid)
                   for i, cid in enumerate(part.membership)}
    _plog(f"  coauthor_control(thr={thr},gamma={gamma}): mainland "
          f"{sub.vcount():,}n/{sub.ecount():,}e, {len(part):,} communities "
          f"(Q={part.modularity:.4f})")
    node_to_cid, _n_island, _misc = _propagate_and_islands(node_to_cid, base_edges)
    return node_to_cid, float(part.modularity)


def build_combined_partition(base_edges: list[tuple[int, int, int]],
                             nid_to_key: dict[int, str],
                             key_to_idx: dict[str, int], n_keys: int,
                             knn_idx: np.ndarray, knn_sim: np.ndarray,
                             alpha: float, tau: float, gamma_comb: float,
                             seed: int) -> tuple[np.ndarray, float]:
    """Mirrors 06 L444-508: coauthor layer alpha*log1p(w); semantic layer =
    mutual top-SEM_K_COMB (prefix of the cached kNN) with sim>=tau, weight
    (1-alpha)*(sim-tau)/(1-tau); simplify(sum); Leiden RB at gamma_comb.
    06's <10-member misc collapse is SKIPPED (presentation-layer relabeling;
    raw memberships are what the metrics need)."""
    import igraph as ig
    edges_comb: list[tuple[int, int]] = []
    weights_comb: list[float] = []
    for u, v, w in base_edges:
        s_key = nid_to_key.get(u)
        t_key = nid_to_key.get(v)
        if s_key is None or t_key is None:
            continue
        ia = key_to_idx.get(s_key)
        ib = key_to_idx.get(t_key)
        if ia is None or ib is None or ia == ib:
            continue
        edges_comb.append((min(ia, ib), max(ia, ib)))
        weights_comb.append(alpha * float(np.log1p(w)))
    n_coauth = len(edges_comb)
    k_comb = min(SEM_K_COMB, knn_idx.shape[1])
    top_mutual = [set(int(knn_idx[i, jj]) for jj in range(k_comb))
                  for i in range(n_keys)]
    n_sem = 0
    for i in range(n_keys):
        for jj in range(k_comb):
            j = int(knn_idx[i, jj])
            if i == j:
                continue
            s = float(knn_sim[i, jj])
            if s < tau:
                continue
            if i not in top_mutual[j]:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges_comb.append((a, b))
            weights_comb.append((1 - alpha) * (s - tau) / (1 - tau))
            n_sem += 1
    g = ig.Graph(n=n_keys, edges=edges_comb,
                 edge_attrs={"weight": weights_comb}, directed=False)
    g.simplify(combine_edges={"weight": "sum"})
    part = _leiden_rb(g, gamma_comb, seed)
    _plog(f"  combined(a={alpha},t={tau},g={gamma_comb}): {n_coauth:,} coauth "
          f"+ {n_sem:,} sem edges, {len(part):,} communities "
          f"(Q={part.modularity:.4f})")
    return np.asarray(part.membership, dtype=np.int64), float(part.modularity)


class PartitionFactory:
    """Lazy, memoized resolver from a partition spec tuple to labels on an
    evaluation population. Specs:
        ("shipped", "c"|"sc"|"cc")
        ("sem", k, gamma, seed)          — semantic kNN Leiden
        ("co", thr, gamma, seed)         — coauthor granularity control
        ("comb", alpha, tau, gamma, seed) — multiplex
    """

    def __init__(self, shipped: dict, pops: dict, keys: list[str],
                 knn_idx: np.ndarray, knn_sim: np.ndarray):
        self.pops = pops
        self.knn_idx = knn_idx
        self.knn_sim = knn_sim
        self.n_keys = len(keys)
        self.key_to_idx = {k: i for i, k in enumerate(keys)}
        net = shipped["net"]
        self.nid_to_key = {nd["id"]: nd.get("key") for nd in net["nodes"]}
        # base (weight>=BASE_THRESHOLD) edges in file order — deterministic
        self.base_edges = [(int(e["source"]), int(e["target"]),
                            int(e["weight"])) for e in net["edges"]]
        self._cache: dict[tuple, dict] = {}

    def _build(self, spec: tuple) -> dict:
        if spec in self._cache:
            return self._cache[spec]
        kind = spec[0]
        t0 = time.time()
        if kind == "sem":
            _, k, gamma, seed = spec
            labels, mod = build_sem_partition(self.knn_idx, self.knn_sim,
                                              int(k), float(gamma), int(seed))
            entry = {"kind": "idx", "labels": labels, "modularity": mod}
        elif kind == "co":
            _, thr, gamma, seed = spec
            node_map, mod = build_coauthor_control(self.base_edges, int(thr),
                                                   float(gamma), int(seed))
            entry = {"kind": "nid", "map": node_map, "modularity": mod}
        elif kind == "comb":
            _, alpha, tau, gamma, seed = spec
            labels, mod = build_combined_partition(
                self.base_edges, self.nid_to_key, self.key_to_idx,
                self.n_keys, self.knn_idx, self.knn_sim,
                float(alpha), float(tau), float(gamma), int(seed))
            entry = {"kind": "idx", "labels": labels, "modularity": mod}
        else:
            raise ValueError(f"unknown spec {spec!r}")
        entry["seconds"] = time.time() - t0
        self._cache[spec] = entry
        return entry

    def eval_labels(self, spec: tuple, population: str) -> np.ndarray:
        pop = self.pops[population]
        if spec[0] == "shipped":
            return np.asarray(pop[spec[1]])
        if population != "pstar":
            raise ValueError("built partitions are only evaluated on P*")
        entry = self._build(spec)
        if entry["kind"] == "idx":
            return entry["labels"][pop["midx"]]
        labels = np.array([entry["map"].get(int(nid), -1)
                           for nid in pop["nids"]], dtype=np.int64)
        assert (labels >= 0).all(), \
            "P* member missing a propagated coauthor-control label"
        return labels

    def modularity(self, spec: tuple):
        if spec[0] == "shipped":
            return None
        return self._build(spec)["modularity"]


# ===========================================================================
# Stage 4 — metrics per cell + summary assembly.
# ===========================================================================

def evaluate_cell(cell: dict, factory: PartitionFactory, perms: int,
                  base_seed: int) -> dict:
    a_lab = factory.eval_labels(cell["A"], cell["population"])
    b_lab = factory.eval_labels(cell["B"], cell["population"])
    a = contiguize(a_lab)
    b = contiguize(b_lab)
    obs_nmi = float(normalized_mutual_info_score(a, b))
    ami = float(adjusted_mutual_info_score(a, b, average_method="arithmetic"))
    ari = float(adjusted_rand_score(a, b))
    vi = float(vi_bits(a, b))
    ecs = float(ecs_hard(a, b))
    m = int(perms * cell["perms_mult"])
    rng = np.random.default_rng(
        np.random.SeedSequence([base_seed, cell["cell_index"]]))
    nulls = perm_null(a, b, m, rng)
    return {
        "cell_id": cell["cell_id"], "axis": cell["axis"],
        "cell_index": cell["cell_index"], "params": cell["params"],
        "population": cell["population"], "n": int(a.size),
        "n_clusters_A": int(a.max()) + 1 if a.size else 0,
        "n_clusters_B": int(b.max()) + 1 if b.size else 0,
        "modularity": factory.modularity(cell["A"]),
        "nmi": obs_nmi, "ami": ami, "ari": ari, "vi": vi, "ecs": ecs,
        "null": null_stats(obs_nmi, ami, nulls),
    }


def _rnd(x, nd=4):
    if isinstance(x, float):
        return round(x, nd)
    return x


def serialize_cell(r: dict) -> dict:
    null = dict(r["null"])
    for k in ("mean", "sd", "p_emp", "excess", "perm_adjusted",
              "ami_agreement_gap"):
        if null.get(k) is not None:
            null[k] = _rnd(float(null[k]))
    null["z"] = _rnd(float(null["z"]), 2) if null.get("z") is not None else None
    out = {
        "cell_id": r["cell_id"], "axis": r["axis"],
        "cell_index": int(r["cell_index"]), "params": r["params"],
        "population": r["population"], "n": int(r["n"]),
        "n_clusters_A": int(r["n_clusters_A"]),
        "n_clusters_B": int(r["n_clusters_B"]),
        "modularity": (_rnd(float(r["modularity"]))
                       if r["modularity"] is not None else None),
        "nmi": _rnd(float(r["nmi"])), "ami": _rnd(float(r["ami"])),
        "ari": _rnd(float(r["ari"])), "vi": _rnd(float(r["vi"])),
        "ecs": _rnd(float(r["ecs"])), "null": null,
    }
    return out


def _minmax(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return _rnd(float(min(vals))), _rnd(float(max(vals)))


def _matched_cell(b_cells: list[dict]) -> dict | None:
    """Granularity-matched selection: argmin |log(n_A/n_B)| over Axis B cells
    vs shipped sc. The match is selected, not guaranteed — the achieved count
    ratio is always reported."""
    pool = [r for r in b_cells if r["params"].get("ref") == "shipped_sc"
            and r["n_clusters_A"] > 0 and r["n_clusters_B"] > 0]
    if not pool:
        return None
    best = min(pool, key=lambda r: abs(math.log(r["n_clusters_A"]
                                                / r["n_clusters_B"])))
    return {
        "cell_id": best["cell_id"], "thr": best["params"].get("thr"),
        "gamma_co": best["params"].get("gamma_co"),
        "n_clusters_A": best["n_clusters_A"],
        "n_clusters_B": best["n_clusters_B"],
        "count_ratio": _rnd(best["n_clusters_A"] / best["n_clusters_B"]),
        "nmi": _rnd(best["nmi"]), "ami": _rnd(best["ami"]),
        "ecs": _rnd(best["ecs"]),
        "z": (_rnd(best["null"]["z"], 2)
              if best["null"]["z"] is not None else None),
    }


def build_summary(results: list[dict]) -> dict:
    by_axis = defaultdict(list)
    for r in results:
        by_axis[r["axis"]].append(r)
    summary: dict = {}
    # Cross-family (coauthor vs semantic) cells: Axis A (rebuilt semantic vs
    # shipped c), Axis B (coauthor control vs semantic refs), and the
    # headline P* c-vs-sc cell.
    cross = list(by_axis.get("A", [])) + [
        r for r in by_axis.get("B", []) if not r["params"].get("calibration")]
    cross += [r for r in by_axis.get("0", [])
              if r["params"].get("pair") == ["c", "sc"]
              and r["population"] == "pstar"]
    if cross:
        nmi_lo, nmi_hi = _minmax([r["nmi"] for r in cross])
        ami_lo, ami_hi = _minmax([r["ami"] for r in cross])
        ecs_lo, ecs_hi = _minmax([r["ecs"] for r in cross])
        zs = [r["null"]["z"] for r in cross if r["null"]["z"] is not None]
        summary["coauthor_vs_semantic"] = {
            "n_cells": len(cross),
            "nmi_min": nmi_lo, "nmi_max": nmi_hi,
            "ami_min": ami_lo, "ami_max": ami_hi,
            "ecs_min": ecs_lo, "ecs_max": ecs_hi,
            "null_nmi_max": _rnd(float(max(r["null"]["mean"] for r in cross))),
            "z_min": _rnd(float(min(zs)), 2) if zs else None,
            "matched_granularity": _matched_cell(by_axis.get("B", [])),
        }
    if by_axis.get("C"):
        cvs = {}
        for ref, name in (("shipped_c", "vs_coauthor"),
                          ("shipped_sc", "vs_semantic")):
            sub = [r for r in by_axis["C"] if r["params"].get("ref") == ref]
            if sub:
                nmi_lo, nmi_hi = _minmax([r["nmi"] for r in sub])
                ami_lo, ami_hi = _minmax([r["ami"] for r in sub])
                cvs[name] = {"n_cells": len(sub),
                             "nmi_min": nmi_lo, "nmi_max": nmi_hi,
                             "ami_min": ami_lo, "ami_max": ami_hi}
        summary["combined_intermediate"] = cvs
    return summary


def compute_seed_stability(factory: PartitionFactory,
                           results: list[dict]) -> dict:
    """Axis D: per-seed metrics vs the published counterpart come from the D
    cells; here we add same-config mean pairwise NMI/AMI among seed
    replicates (partition self-stability) on P*."""
    configs = {
        "semantic": [("sem", SEM_KNN_K, SEM_LEIDEN_RESOLUTION, s)
                     for s in AXIS_D_SEEDS],
        "coauthor_control": [("co", GIANT_THRESHOLD, 1.0, s)
                             for s in AXIS_D_SEEDS],
        "combined": [("comb", ALPHA_COMB, SEM_THRESH_COMB,
                      COMB_LEIDEN_RESOLUTION, s) for s in AXIS_D_SEEDS],
    }
    d_cells = [r for r in results if r["axis"] == "D"]
    out = {}
    for name, specs in configs.items():
        labs = [contiguize(factory.eval_labels(sp, "pstar")) for sp in specs]
        pair_nmi, pair_ami = [], []
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                pair_nmi.append(float(normalized_mutual_info_score(labs[i], labs[j])))
                pair_ami.append(float(adjusted_mutual_info_score(
                    labs[i], labs[j], average_method="arithmetic")))
        per_seed = [r for r in d_cells if r["params"].get("config") == name]
        per_seed.sort(key=lambda r: r["params"]["seed"])
        out[name] = {
            "seeds": list(AXIS_D_SEEDS),
            "per_seed_nmi": [_rnd(r["nmi"]) for r in per_seed],
            "per_seed_ami": [_rnd(r["ami"]) for r in per_seed],
            "self_nmi_mean": _rnd(float(np.mean(pair_nmi))),
            "self_nmi_sd": _rnd(float(np.std(pair_nmi, ddof=1))),
            "self_ami_mean": _rnd(float(np.mean(pair_ami))),
        }
    return out


# ===========================================================================
# Outputs — TeX table, 2x2 figure, JSON snapshot, manuscript sentence.
# ===========================================================================

def _tex_escape(s):
    """Verbatim copy of 03_partition_alignment.py L56-62."""
    if not isinstance(s, str):
        s = str(s)
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _table_row(setting: str, r: dict) -> str:
    null = r["null"]
    z_txt = f"{null['z']:.1f}" if null["z"] is not None else "--"
    return (f"{_tex_escape(setting)} & {r['n_clusters_A']} & "
            f"{r['n_clusters_B']} & {r['nmi']:.3f} & {r['ami']:.3f} & "
            f"{r['ecs']:.3f} & {null['mean']:.3f} $\\pm$ {null['sd']:.3f} & "
            f"{z_txt} \\\\")


def write_table(results: list[dict], seed_stab: dict | None, parity_nmi,
                tables_dir: Path, tag: str, smoke: bool,
                snapshot_date: str) -> Path:
    by_id = {r["cell_id"]: r for r in results}
    by_axis = defaultdict(list)
    for r in results:
        by_axis[r["axis"]].append(r)

    lines = []
    if smoke:
        lines.append(f"% {SMOKE_BANNER}")
    lines += [
        "% Generated by paper/analysis/09_partition_robustness.py — do not edit.",
        "% (i) Resolution-parameterized coauthor partitions are analysis-only",
        "%     granularity controls, not the published atlas partition.",
        "% (ii) NMI/AMI use arithmetic normalization; null = size-matched",
        "%     permutation (community-size distributions preserved exactly).",
        f"% (iii) Data snapshot: {snapshot_date}."
        + (f"  Parity NMI (rebuilt vs shipped semantic): {parity_nmi:.3f}."
           if parity_nmi is not None else ""),
        r"\begin{tabular}{lrrrrrcr}",
        r"\toprule",
        (r"\textbf{Setting} & $|C_A|$ & $|C_B|$ & \textbf{NMI} & "
         r"\textbf{AMI} & \textbf{ECS} & \textbf{Null NMI} & $z$ \\"),
        r"\midrule",
    ]

    def row_if(cell_id, setting):
        if cell_id in by_id:
            lines.append(_table_row(setting, by_id[cell_id]))

    row_if("0:c-vs-sc@pop03", "Shipped coauthor vs semantic (03 pop.)")
    row_if("0:c-vs-sc@pstar", "Shipped coauthor vs semantic (P*)")
    if parity_nmi is not None:
        lines.append(f"Parity: rebuilt semantic vs shipped & "
                     f"\\multicolumn{{2}}{{c}}{{--}} & {parity_nmi:.3f} & "
                     f"\\multicolumn{{4}}{{c}}{{--}} \\\\")
    row_if("B:cal:thr=5,gc=1.0|c", "Calibration: RB@1.0 vs shipped coauthor")
    if by_axis.get("A"):
        row_if("A:k=10,gs=1.0", "Semantic k=10 (gamma=1) vs coauthor")
        row_if("A:k=50,gs=1.0", "Semantic k=50 (gamma=1) vs coauthor")
        row_if("A:k=20,gs=0.3", "Semantic gamma=0.3 (k=20) vs coauthor")
        row_if("A:k=20,gs=4.0", "Semantic gamma=4 (k=20) vs coauthor")
        a_min = min(by_axis["A"], key=lambda r: r["nmi"])
        a_max = max(by_axis["A"], key=lambda r: r["nmi"])
        lines.append(_table_row(f"Axis-A min ({a_min['cell_id'][2:]})", a_min))
        if a_max["cell_id"] != a_min["cell_id"]:
            lines.append(_table_row(f"Axis-A max ({a_max['cell_id'][2:]})", a_max))
    for thr in AXIS_B_THRS:
        pool = [r for r in by_axis.get("B", [])
                if r["params"].get("ref") == "shipped_sc"
                and r["params"].get("thr") == thr]
        if pool:
            best = min(pool, key=lambda r: abs(math.log(
                max(r["n_clusters_A"], 1) / max(r["n_clusters_B"], 1))))
            lines.append(_table_row(
                f"Matched granularity thr={thr} "
                f"(gc={best['params']['gamma_co']})", best))
    if by_axis.get("C"):
        for cid, setting in (
                ("C:a=0.25,t=0.6,gcomb=0.5|c", "Combined alpha=0.25 vs coauthor"),
                ("C:a=0.75,t=0.6,gcomb=0.5|c", "Combined alpha=0.75 vs coauthor"),
                ("C:a=0.5,t=0.5,gcomb=0.5|c", "Combined tau=0.5 vs coauthor"),
                ("C:a=0.5,t=0.7,gcomb=0.5|c", "Combined tau=0.7 vs coauthor")):
            row_if(cid, setting)
    if seed_stab:
        for name, s in seed_stab.items():
            if s["per_seed_nmi"]:
                mu = float(np.mean(s["per_seed_nmi"]))
                sd = float(np.std(s["per_seed_nmi"], ddof=1))
                amu = float(np.mean(s["per_seed_ami"]))
                lines.append(
                    f"Seeds 1--99, {_tex_escape(name)} (self-NMI "
                    f"{s['self_nmi_mean']:.3f}) & \\multicolumn{{2}}{{c}}{{--}} & "
                    f"{mu:.3f} $\\pm$ {sd:.3f} & {amu:.3f} & "
                    f"\\multicolumn{{3}}{{c}}{{--}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = tables_dir / f"07_partition_robustness{tag}.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_figure(results: list[dict], figures_dir: Path, tag: str,
                 smoke: bool) -> Path:
    """2x2 OKABE_ITO panel. matplotlib + rcParams live INSIDE this function
    so the pure metric helpers stay importable headless (documented deviation
    from the module-top convention)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.prop_cycle": plt.cycler("color", OKABE_ITO),
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    def _save(fig, stem):  # parameterized copy of 03's _save (L50-54)
        for ext in ("pdf", "png"):
            fig.savefig(figures_dir / f"{stem}.{ext}")
        plt.close(fig)

    by_axis = defaultdict(list)
    for r in results:
        by_axis[r["axis"]].append(r)
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 7.2))

    # (a) NMI/AMI vs k, one line per gamma_sem, null band shaded.
    ax = axs[0, 0]
    a_cells = by_axis.get("A", [])
    if a_cells:
        lo = min(r["null"]["mean"] - 3 * r["null"]["sd"] for r in a_cells)
        hi = max(r["null"]["mean"] + 3 * r["null"]["sd"] for r in a_cells)
        ax.axhspan(max(lo, 0), hi, color="0.85", zorder=0,
                   label="null $\\pm 3\\sigma$")
        for gi, g in enumerate(AXIS_A_GAMMAS):
            sub = sorted((r for r in a_cells if r["params"]["gamma_sem"] == g),
                         key=lambda r: r["params"]["k"])
            if not sub:
                continue
            ks = [r["params"]["k"] for r in sub]
            ax.plot(ks, [r["nmi"] for r in sub], "o-",
                    color=OKABE_ITO[gi % 8], label=f"$\\gamma_s$={g}")
            ax.plot(ks, [r["ami"] for r in sub], "s--",
                    color=OKABE_ITO[gi % 8], alpha=0.6)
        ax.set_xlabel("$k$ (semantic kNN)")
        ax.set_ylabel("NMI (solid) / AMI (dashed)")
        ax.legend(frameon=False, ncol=2)
    else:
        ax.text(0.5, 0.5, "Axis A not run", ha="center", va="center")
    ax.set_title("(a) Semantic $k \\times \\gamma$ vs shipped coauthor")
    ax.grid(alpha=0.25)

    # (b) NMI/AMI vs gamma_co per thr; community counts on twin axis.
    ax = axs[0, 1]
    b_sc = [r for r in by_axis.get("B", [])
            if r["params"].get("ref") == "shipped_sc"]
    if b_sc:
        ax2 = ax.twinx()
        for ti, thr in enumerate(AXIS_B_THRS):
            sub = sorted((r for r in b_sc if r["params"]["thr"] == thr),
                         key=lambda r: r["params"]["gamma_co"])
            if not sub:
                continue
            gs = [r["params"]["gamma_co"] for r in sub]
            ax.plot(gs, [r["nmi"] for r in sub], "o-",
                    color=OKABE_ITO[ti], label=f"NMI thr={thr}")
            ax.plot(gs, [r["ami"] for r in sub], "s--",
                    color=OKABE_ITO[ti], alpha=0.6)
            ax2.plot(gs, [r["n_clusters_A"] for r in sub], ":",
                     color=OKABE_ITO[ti + 4], alpha=0.8)
            best = min(sub, key=lambda r: abs(math.log(
                max(r["n_clusters_A"], 1) / max(r["n_clusters_B"], 1))))
            ax.plot([best["params"]["gamma_co"]], [best["nmi"]], "*",
                    color=OKABE_ITO[ti], markersize=13)
        ax2.axhline(b_sc[0]["n_clusters_B"], color="0.4", lw=0.8, ls="-.")
        ax2.set_yscale("log")
        ax2.set_ylabel("# coauthor communities (dotted)")
        ax.set_xscale("log")
        ax.set_xlabel("$\\gamma_{co}$")
        ax.set_ylabel("NMI / AMI vs shipped semantic")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "Axis B not run", ha="center", va="center")
    ax.set_title("(b) Coauthor granularity control ($\\star$ = matched)")
    ax.grid(alpha=0.25)

    # (c) AMI heatmap over alpha x tau (combined vs coauthor).
    ax = axs[1, 0]
    c_cells = [r for r in by_axis.get("C", [])
               if r["params"].get("ref") == "shipped_c"
               and r["params"]["gamma_comb"] == COMB_LEIDEN_RESOLUTION]
    if c_cells:
        mat = np.full((len(AXIS_C_ALPHAS), len(AXIS_C_TAUS)), np.nan)
        cnt = {}
        for r in c_cells:
            i = AXIS_C_ALPHAS.index(r["params"]["alpha"])
            j = AXIS_C_TAUS.index(r["params"]["tau"])
            mat[i, j] = r["ami"]
            cnt[(i, j)] = r["n_clusters_A"]
        im = ax.imshow(mat, cmap="YlOrBr", aspect="auto")
        for (i, j), c in cnt.items():
            ax.text(j, i, str(c), ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(AXIS_C_TAUS)),
                      [str(t) for t in AXIS_C_TAUS])
        ax.set_yticks(range(len(AXIS_C_ALPHAS)),
                      [str(a) for a in AXIS_C_ALPHAS])
        ax.set_xlabel("$\\tau$ (semantic threshold)")
        ax.set_ylabel("$\\alpha$ (coauthor weight)")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="AMI vs coauthor")
    else:
        ax.text(0.5, 0.5, "Axis C not run", ha="center", va="center")
    ax.set_title("(c) Multiplex knobs (cells = $|C|$)")

    # (d) observed NMI vs null band across ALL cells, ordered by geometric
    # mean of the two community counts.
    ax = axs[1, 1]
    ordered = sorted(results, key=lambda r: math.sqrt(
        max(r["n_clusters_A"], 1) * max(r["n_clusters_B"], 1)))
    if ordered:
        x = np.arange(len(ordered))
        nm = np.array([r["null"]["mean"] for r in ordered])
        ns = np.array([r["null"]["sd"] for r in ordered])
        ax.fill_between(x, np.maximum(nm - 3 * ns, 0), nm + 3 * ns,
                        color="0.8", label="null $\\pm 3\\sigma$")
        ax.plot(x, [r["nmi"] for r in ordered], ".", color=OKABE_ITO[4],
                markersize=4, label="observed NMI")
        with_z = [(i, r) for i, r in enumerate(ordered)
                  if r["null"]["z"] is not None]
        if with_z:
            for i, r in (min(with_z, key=lambda t: t[1]["null"]["z"]),
                         max(with_z, key=lambda t: t[1]["null"]["z"])):
                ax.annotate(f"z={r['null']['z']:.0f}", (i, r["nmi"]),
                            fontsize=7, textcoords="offset points",
                            xytext=(0, 6), ha="center")
        ax.set_xlabel("cells ordered by $\\sqrt{|C_A|\\,|C_B|}$")
        ax.set_ylabel("NMI")
        ax.legend(frameon=False)
    ax.set_title("(d) Observed vs size-matched null (all cells)")
    ax.grid(alpha=0.25)

    if smoke:
        fig.suptitle(SMOKE_BANNER, color="#D55E00", fontsize=12)
    fig.tight_layout()
    _save(fig, f"07_partition_robustness{tag}")
    return figures_dir / f"07_partition_robustness{tag}.pdf"


def print_sentence(summary: dict) -> None:
    cs = summary.get("coauthor_vs_semantic")
    if not cs:
        _plog("summary sentence: cross-family axes (A/B) not run — skipped")
        return
    mg = cs.get("matched_granularity")
    parts = [
        f"Across {cs['n_cells']} configurations "
        f"(k in [{min(AXIS_A_KS)},{max(AXIS_A_KS)}], "
        f"gamma_sem in [{min(AXIS_A_GAMMAS)},{max(AXIS_A_GAMMAS)}], "
        f"gamma_co in [{min(AXIS_B_GAMMAS)},{max(AXIS_B_GAMMAS)}], "
        f"thr in {{{','.join(str(t) for t in AXIS_B_THRS)}}}, "
        f"alpha in [{min(AXIS_C_ALPHAS)},{max(AXIS_C_ALPHAS)}], "
        f"tau in [{min(AXIS_C_TAUS)},{max(AXIS_C_TAUS)}]), "
        f"coauthor-semantic NMI stays in [{cs['nmi_min']:.3f}, "
        f"{cs['nmi_max']:.3f}] (AMI [{cs['ami_min']:.3f}, {cs['ami_max']:.3f}])",
        f"size-matched random partitions yield NMI <= {cs['null_nmi_max']:.4f}",
    ]
    if cs.get("z_min") is not None:
        parts.append(f"so observed alignment exceeds the granularity-artifact "
                     f"floor by z >= {cs['z_min']:.0f} sigma in every cell")
    if mg:
        parts.append(f"at matched granularity (|C_co| = {mg['n_clusters_A']} "
                     f"vs |C_sem| = {mg['n_clusters_B']}) AMI = {mg['ami']:.3f}")
    print("\n[robust] MANUSCRIPT SENTENCE:\n  " + "; ".join(parts) + ".\n",
          flush=True)


# ===========================================================================
# Self-test — offline, synthetic, no data files, no network.
# ===========================================================================

def run_self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        _plog(f"self-test {'PASS' if ok else 'FAIL'}: {name} {detail}")
        if not ok:
            failures.append(name)

    rng = np.random.default_rng(42)
    # (1) ECS closed form == brute-force PPR; identities.
    max_d = 0.0
    for _ in range(5):
        a = rng.integers(0, int(rng.integers(2, 12)), 200)
        b = rng.integers(0, int(rng.integers(2, 12)), 200)
        max_d = max(max_d, abs(ecs_hard(a, b) - ecs_brute_ppr(a, b, alpha=0.9)))
    check("ECS closed form == brute-force PPR (alpha cancels)",
          max_d <= 1e-9, f"max|d|={max_d:.2e}")
    p = rng.integers(0, 7, 300)
    check("ECS(P,P)=1", abs(ecs_hard(p, p) - 1.0) <= 1e-12)
    n = 128
    check("ECS(singletons, one-cluster)=1/N",
          abs(ecs_hard(np.arange(n), np.zeros(n, dtype=int)) - 1.0 / n) <= 1e-12)
    # (2) fast NMI == sklearn.
    max_d = 0.0
    for _ in range(20):
        sz = int(rng.integers(50, 3000))
        a = rng.integers(0, int(rng.integers(1, 50)), sz)
        b = rng.integers(0, int(rng.integers(1, 50)), sz)
        max_d = max(max_d, abs(fast_nmi(a, b)
                               - normalized_mutual_info_score(a, b)))
    for a, b in ((p, p), (p, np.zeros_like(p)),
                 (np.zeros_like(p), np.zeros_like(p))):
        max_d = max(max_d, abs(fast_nmi(a, b)
                               - normalized_mutual_info_score(a, b)))
    check("fast NMI == sklearn (incl. degenerate)", max_d <= 1e-9,
          f"max|d|={max_d:.2e}")
    # (3) mean AMI over size-matched shuffles ~ 0.
    base = rng.integers(0, 10, 2000)
    amis = [adjusted_mutual_info_score(base, permuted_labels(base, rng))
            for _ in range(50)]
    check("mean AMI under shuffles ~ 0", abs(float(np.mean(amis))) < 0.01,
          f"mean={np.mean(amis):+.4f}")
    # (4) per-cell SeedSequence determinism + execution-order independence.
    a = rng.integers(0, 12, 1000)
    b = rng.integers(0, 8, 1000)

    def null_for(idx: int) -> np.ndarray:
        r = np.random.default_rng(np.random.SeedSequence([42, idx]))
        return perm_null(a, b, 20, r)

    r1 = {i: null_for(i) for i in (0, 1, 2)}
    r2 = {i: null_for(i) for i in (2, 0, 1)}
    check("SeedSequence([seed, cell_index]) reproducible + order-independent",
          all(np.array_equal(r1[i], r2[i]) for i in (0, 1, 2)))
    # (5) NMI/AMI monotone under planted hierarchical merges.
    lab16 = np.repeat(np.arange(16), 100)
    seq_nmi = [normalized_mutual_info_score(lab16, lab16 // d)
               for d in (1, 2, 4, 8)]
    seq_ami = [adjusted_mutual_info_score(lab16, lab16 // d)
               for d in (1, 2, 4, 8)]
    check("NMI strictly decreasing under merges",
          all(x > y for x, y in zip(seq_nmi, seq_nmi[1:])))
    check("AMI strictly decreasing under merges",
          all(x > y for x, y in zip(seq_ami, seq_ami[1:])))
    _plog(f"self-test: {'ALL PASS' if not failures else f'{len(failures)} FAILURES'}")
    return 0 if not failures else 1


# ===========================================================================
# CLI + main.
# ===========================================================================

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--perms", type=int, default=200,
                    help="permutation-null draws per cell (Cell 0 uses 10x)")
    ap.add_argument("--kmax", type=int, default=50,
                    help="kNN depth computed once; swept k are prefix slices")
    ap.add_argument("--seed", type=int, default=42, help="base seed")
    ap.add_argument("--sample", type=int, default=0,
                    help="SMOKE: seeded subsample of the author universe")
    ap.add_argument("--axes", default="0,A,B,C,D",
                    help="comma subset of {0,A,B,C,D}")
    ap.add_argument("--tag", default="",
                    help="suffix for every output stem (mirrors 05 --tag)")
    ap.add_argument("--emit-manuscript", action="store_true",
                    help="write .tex/figures to paper/manuscript/ "
                         "(LATER manuscript workflow ONLY; default is "
                         "paper/analysis/_preview/)")
    ap.add_argument("--force-cache", action="store_true",
                    help="rebuild the author-matrix/kNN cache unconditionally")
    ap.add_argument("--self-test", action="store_true",
                    help="run offline synthetic self-checks and exit")
    return ap.parse_args(argv)


def parity_gate(factory: PartitionFactory, pops: dict, smoke: bool):
    """Checked precondition: the rebuilt baseline semantic partition
    (k=20, gamma=1.0, seed=42) must reproduce the shipped sc labels on P*
    (NMI >= 0.70 hard, >= 0.90 clean). Under --sample this degrades to a
    warning: a subsample kNN partition cannot match the full-corpus one."""
    spec = ("sem", SEM_KNN_K, SEM_LEIDEN_RESOLUTION, SEM_LEIDEN_SEED)
    rebuilt = contiguize(factory.eval_labels(spec, "pstar"))
    shipped = contiguize(pops["pstar"]["sc"])
    nmi = float(normalized_mutual_info_score(rebuilt, shipped))
    _plog(f"parity gate: rebuilt baseline semantic vs shipped sc on P*: "
          f"NMI={nmi:.4f} (rebuilt {rebuilt.max() + 1} vs shipped "
          f"{shipped.max() + 1} communities, n={rebuilt.size:,})")
    if nmi < PARITY_GATE_HARD:
        rows, cols, counts, _, _ = _contingency(rebuilt, shipped)
        top = np.argsort(-counts)[:5]
        _plog("parity DIAGNOSTICS — top contingency cells "
              "(rebuilt_id, shipped_id, count):")
        for t in top:
            _plog(f"  ({int(rows[t])}, {int(cols[t])}, {int(counts[t])})")
        if smoke:
            _plog(f"parity WARNING (smoke mode): NMI {nmi:.3f} < "
                  f"{PARITY_GATE_HARD} is EXPECTED under --sample; "
                  "proceeding — smoke numbers are not publication numbers")
            return nmi, True
        _plog(f"parity gate FAILED: NMI {nmi:.3f} < {PARITY_GATE_HARD} — the "
              "sweep would not pivot around the published operating point. "
              "Check alias-map drift, kNN tie-breaks, sklearn version skew.")
        return nmi, False
    if nmi < PARITY_GATE_WARN:
        _plog(f"parity WARNING: NMI {nmi:.3f} in [{PARITY_GATE_HARD}, "
              f"{PARITY_GATE_WARN}) — float-path/tie-break drift at cluster "
              "boundaries; proceeding")
    return nmi, True


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    axes = {s.strip() for s in args.axes.split(",") if s.strip()}
    bad = axes - {"0", "A", "B", "C", "D"}
    if bad:
        _plog(f"unknown axes {sorted(bad)} (valid: 0,A,B,C,D)")
        return 1
    smoke = bool(args.sample)
    tag = args.tag
    if args.emit_manuscript:
        tables_dir = ROOT / "paper" / "manuscript" / "tables"
        figures_dir = ROOT / "paper" / "manuscript" / "figures"
        _plog("WARNING: --emit-manuscript writes into the Overleaf-synced "
              "paper/manuscript/ repo — later manuscript workflow only.")
    else:
        tables_dir = figures_dir = PREVIEW_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        print("=" * 66 + f"\n=== {SMOKE_BANNER} (--sample {args.sample}) ===\n"
              + "=" * 66, flush=True)

    wall0 = time.time()
    grid = enumerate_grid()
    to_run = [c for c in grid if c["axis"] in axes]
    _plog(f"grid: {len(grid)} cells enumerated, {len(to_run)} selected "
          f"(axes={sorted(axes)})")

    shipped = load_shipped()
    keys, _a_mat, knn_idx, knn_sim, cache_meta = load_or_build_cache(
        args.kmax, args.sample, args.seed, args.force_cache)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    pops = build_populations(shipped, key_to_idx)
    if pops["n_eval"] == 0:
        _plog("empty evaluation population P* — aborting")
        return 1

    # Startup validation: fast null-NMI implementation vs sklearn on the
    # observed headline pair.
    a0 = contiguize(pops["pop03"]["c"])
    b0 = contiguize(pops["pop03"]["sc"])
    d = abs(fast_nmi(a0, b0) - normalized_mutual_info_score(a0, b0))
    if d > 1e-9:
        _plog(f"fast-NMI validation FAILED: |fast - sklearn| = {d:.2e}")
        return 1
    _plog(f"fast-NMI validated vs sklearn on headline pair (|d|={d:.1e})")

    factory = PartitionFactory(shipped, pops, keys, knn_idx, knn_sim)
    parity_nmi, ok = parity_gate(factory, pops, smoke)
    if not ok:
        return 1

    results = []
    for i, cell in enumerate(to_run):
        t0 = time.time()
        r = evaluate_cell(cell, factory, args.perms, args.seed)
        results.append(r)
        _plog(f"cell {i + 1}/{len(to_run)} {r['cell_id']}: "
              f"NMI={r['nmi']:.3f} AMI={r['ami']:.3f} ECS={r['ecs']:.3f} "
              f"|A|={r['n_clusters_A']} |B|={r['n_clusters_B']} "
              f"z={r['null']['z'] if r['null']['z'] is None else round(r['null']['z'], 1)} "
              f"({time.time() - t0:.1f}s)")

    seed_stab = (compute_seed_stability(factory, results)
                 if "D" in axes else None)

    # Reference block: recomputed headline (Cell 0, 03 population) vs the
    # STALE prior snapshot — warn-only, never assert.
    reference: dict = {"prior_03_snapshot": None, "recomputed_headline": None,
                       "drift_warning": False}
    if PRIOR_03_SNAPSHOT.exists():
        reference["prior_03_snapshot"] = json.loads(
            PRIOR_03_SNAPSHOT.read_text())
    if "0" in axes:
        head = {}
        pair_name = {("c", "sc"): "Coauthor-vs-Semantic",
                     ("c", "cc"): "Coauthor-vs-Combined",
                     ("sc", "cc"): "Semantic-vs-Combined"}
        for r in results:
            if r["axis"] == "0" and r["population"] == "pop03":
                nm = pair_name[tuple(r["params"]["pair"])]
                head[nm] = {"NMI": _rnd(r["nmi"]), "AMI": _rnd(r["ami"]),
                            "ARI": _rnd(r["ari"]), "VI": _rnd(r["vi"]),
                            "ECS": _rnd(r["ecs"]),
                            "n_clusters_A": r["n_clusters_A"],
                            "n_clusters_B": r["n_clusters_B"]}
        reference["recomputed_headline"] = head
        prior = (reference["prior_03_snapshot"] or {}).get(
            "Coauthor-vs-Semantic", {})
        if prior.get("NMI") is not None and head.get("Coauthor-vs-Semantic"):
            drift = abs(head["Coauthor-vs-Semantic"]["NMI"] - prior["NMI"])
            reference["drift_warning"] = bool(drift > DRIFT_WARN_NMI)
            if reference["drift_warning"]:
                _plog(f"DRIFT WARNING: recomputed headline NMI "
                      f"{head['Coauthor-vs-Semantic']['NMI']:.4f} vs stale "
                      f"_partition_alignment.json {prior['NMI']:.4f} "
                      f"(|d|={drift:.4f} > {DRIFT_WARN_NMI}); the 03 snapshot "
                      "predates the current data refresh — regenerate 03 in "
                      "the manuscript workflow")

    summary = build_summary(results)
    snapshot_date = time.strftime(
        "%Y-%m-%d", time.localtime(PAPERS_PATH.stat().st_mtime))
    payload = {
        "config": {
            "grids": {"axis_A_k": list(AXIS_A_KS),
                      "axis_A_gamma_sem": list(AXIS_A_GAMMAS),
                      "axis_B_thr": list(AXIS_B_THRS),
                      "axis_B_gamma_co": list(AXIS_B_GAMMAS),
                      "axis_B_sem_refs": list(AXIS_B_SEM_REFS),
                      "axis_C_alpha": list(AXIS_C_ALPHAS),
                      "axis_C_tau": list(AXIS_C_TAUS),
                      "axis_C_gamma_oat": list(AXIS_C_GAMMA_OAT),
                      "axis_D_seeds": list(AXIS_D_SEEDS),
                      "baseline": {"k": SEM_KNN_K,
                                   "gamma_sem": SEM_LEIDEN_RESOLUTION,
                                   "alpha": ALPHA_COMB,
                                   "tau": SEM_THRESH_COMB,
                                   "gamma_comb": COMB_LEIDEN_RESOLUTION,
                                   "seed": LEIDEN_SEED}},
            "kmax": args.kmax, "perms": args.perms, "seed": args.seed,
            "sample": args.sample, "axes": sorted(axes), "tag": tag,
            "smoke": smoke,
            "smoke_banner": SMOKE_BANNER if smoke else None,
            "parity_nmi": _rnd(float(parity_nmi)),
            "cache_meta": cache_meta,
            "population": {
                "n_eval": pops["n_eval"], "n_pop03": pops["n_pop03"],
                "n_keys": len(keys),
                "n_atlas_nodes": len(shipped["net"]["nodes"]),
                "n_topic_coords": pops["n_topic_coords"],
                "exclusion_counts": pops["exclusion_counts"],
                "n_comms_published": shipped["n_comms_published"],
            },
            "data_snapshot_date": snapshot_date,
            "wall_seconds": round(time.time() - wall0, 1),
        },
        "reference": reference,
        "cells": [serialize_cell(r) for r in results],
        "seed_stability": seed_stab,
        "summary": summary,
    }
    snap_path = ROOT / "paper" / "analysis" / f"_partition_robustness{tag}.json"
    # allow_nan=False per repo hard rule — degenerate z is None, never NaN/inf.
    snap_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    _plog(f"wrote {snap_path.relative_to(ROOT)}")

    tex = write_table(results, seed_stab, parity_nmi, tables_dir, tag, smoke,
                      snapshot_date)
    _plog(f"wrote {tex.relative_to(ROOT)}")
    figp = write_figure(results, figures_dir, tag, smoke)
    _plog(f"wrote {figp.relative_to(ROOT)} (+ .png)")
    print_sentence(summary)
    _plog(f"done in {time.time() - wall0:.1f}s "
          f"({len(results)} cells{' — SMOKE' if smoke else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
