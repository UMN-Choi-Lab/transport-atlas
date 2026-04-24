#!/usr/bin/env python
"""§6 + §7 — Semantic structure + multiplex partition alignment.

Emits:
    tables/06_semantic_communities.tex
    tables/07_partition_alignment.tex
    figures/06_whitening_impact.pdf  (+ .png)
    figures/07_partition_cooccurrence.pdf  (+ .png)

Reads topic_coords.json, semantic_communities.json,
combined_communities.json, coauthor_network.json, and (for the whitening
plot) the cached SPECTER2 embeddings at
/data2/chois/transport-atlas/paper_embeddings.parquet.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score)

OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
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
EMBED_PATH = (
    Path(os.environ.get("EMBED_OUT", "/data2/chois/transport-atlas"))
    / "paper_embeddings.parquet"
)

SEED = 42
np.random.seed(SEED)


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    plt.close(fig)


def _tex_escape(s):
    if not isinstance(s, str): s = str(s)
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    for k, v in repl.items(): s = s.replace(k, v)
    return s


def _titlecase_label(s):
    if "," in s:
        last, rest = s.split(",", 1)
        parts = [p.strip() for p in rest.split()]
        return last.strip().title() + ", " + " ".join(p.title() for p in parts)
    return s.title()


def _variation_of_information(a, b):
    """VI(U, V) = H(U|V) + H(V|U). Lower is more agreement."""
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    # Joint distribution
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


def main() -> int:
    print("[partition] loading …")
    tc = json.loads((ROOT / "data" / "processed" / "topic_coords.json").read_text())
    sem_comms = json.loads(
        (ROOT / "data" / "processed" / "semantic_communities.json").read_text())
    comb_comms = json.loads(
        (ROOT / "data" / "processed" / "combined_communities.json").read_text())

    sem_misc = {c["id"] for c in sem_comms if c.get("misc")}
    comb_misc = {c["id"] for c in comb_comms if c.get("misc")}

    # Build per-author partition labels from topic_coords
    # Keep only authors with all three labels non-null and non-misc (so that
    # NMI / ARI compare the real structure, not the catch-all).
    rows = []
    for nid, v in tc.items():
        co = v.get("c")
        sc = v.get("sc")
        cc = v.get("cc")
        if co is None or sc is None or cc is None:
            continue
        if sc in sem_misc or cc in comb_misc:
            continue
        rows.append((int(nid), co, sc, cc))
    df = pd.DataFrame(rows, columns=["nid", "coauth", "semantic", "combined"])
    print(f"[partition]   {len(df):,} authors in non-misc intersection")

    # ------------------------------------------------------------------
    # Table 7 — semantic communities (22) with labels + exemplars
    # ------------------------------------------------------------------
    sem_sorted = sorted(
        [c for c in sem_comms if not c.get("misc")],
        key=lambda c: -c["size"])
    lines = [
        r"\begin{tabular}{rlp{0.32\linewidth}p{0.32\linewidth}}",
        r"\toprule",
        r"\textbf{\#} & \textbf{Size} & \textbf{Keyword label} & \textbf{Exemplar authors} \\",
        r"\midrule",
    ]
    for c in sem_sorted:
        kws = ", ".join((c.get("label_words") or [])[:6])
        ex = ", ".join(_titlecase_label(a) for a in (c.get("top_authors") or [])[:5])
        lines.append(
            f"{c['id']} & {c['size']:,} & {_tex_escape(kws)} & {_tex_escape(ex)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "06_semantic_communities.tex").write_text("\n".join(lines) + "\n")
    print("  ✓ tables/06_semantic_communities.tex")

    # ------------------------------------------------------------------
    # Table 8 — partition alignment (NMI / VI / ARI)
    # ------------------------------------------------------------------
    pairs = [
        ("Coauthor", "Semantic", df["coauth"].values, df["semantic"].values),
        ("Coauthor", "Combined", df["coauth"].values, df["combined"].values),
        ("Semantic", "Combined", df["semantic"].values, df["combined"].values),
    ]
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        (r"\textbf{A} & \textbf{B} & \textbf{NMI} & \textbf{ARI} & "
         r"\textbf{VI (bits)} & \textbf{$|A|$} & \textbf{$|B|$} \\"),
        r"\midrule",
    ]
    alignment_dict = {}
    for a_name, b_name, a, b in pairs:
        nmi = normalized_mutual_info_score(a, b)
        ari = adjusted_rand_score(a, b)
        vi = _variation_of_information(a, b)
        lines.append(
            f"{a_name} & {b_name} & {nmi:.3f} & {ari:.3f} & "
            f"{vi:.2f} & {len(np.unique(a))} & {len(np.unique(b))} \\\\"
        )
        alignment_dict[f"{a_name}-vs-{b_name}"] = {
            "NMI": float(nmi), "ARI": float(ari), "VI": float(vi),
            "n_clusters_A": int(len(np.unique(a))),
            "n_clusters_B": int(len(np.unique(b))),
        }
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "07_partition_alignment.tex").write_text("\n".join(lines) + "\n")
    (ROOT / "paper" / "analysis" / "_partition_alignment.json").write_text(
        json.dumps(alignment_dict, indent=2)
    )
    print("  ✓ tables/07_partition_alignment.tex")
    for k, v in alignment_dict.items():
        print(f"    {k}: NMI={v['NMI']:.3f}  ARI={v['ARI']:.3f}  VI={v['VI']:.2f}")

    # ------------------------------------------------------------------
    # Fig 8 — semantic × coauthor co-occurrence heatmap (re-orders rows
    # & cols so that matched communities sit on the diagonal band).
    # ------------------------------------------------------------------
    sem_sizes = df.groupby("semantic").size().sort_values(ascending=False)
    top_sem = sem_sizes.index.tolist()         # all non-misc semantic

    # Exclude misc-flagged coauthor communities: the misc bucket collapses
    # hundreds of small islands and would dominate the "columns" without
    # carrying a meaningful label.
    coauth_comms_meta_pre = json.loads(
        (ROOT / "data" / "processed" / "coauthor_network.json").read_text()
    )["meta"]["communities"]
    co_misc_ids = {c["id"] for c in coauth_comms_meta_pre if c.get("misc")}
    co_sizes = df.groupby("coauth").size().sort_values(ascending=False)
    co_nonmisc = [c for c in co_sizes.index if c not in co_misc_ids]
    top_co = co_nonmisc[:22]

    mat = np.zeros((len(top_sem), len(top_co)), dtype=float)
    sem_idx = {s: i for i, s in enumerate(top_sem)}
    co_idx = {c: j for j, c in enumerate(top_co)}
    for sc, co in zip(df["semantic"].values, df["coauth"].values):
        if sc in sem_idx and co in co_idx:
            mat[sem_idx[sc], co_idx[co]] += 1
    # Normalise rows to proportion of that semantic community
    row_totals = mat.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    mat_norm = mat / row_totals

    # Permute columns via greedy matching on row argmax
    col_perm = []
    remaining = list(range(mat_norm.shape[1]))
    for i in range(mat_norm.shape[0]):
        if not remaining: break
        best_j = max(remaining, key=lambda j: mat_norm[i, j])
        col_perm.append(best_j)
        remaining.remove(best_j)
    col_perm += remaining
    mat_perm = mat_norm[:, col_perm]

    # Labels
    sem_label = {c["id"]: (c.get("label_words") or ["?"])[0]
                 for c in sem_comms}
    coauth_comms_meta = json.loads(
        (ROOT / "data" / "processed" / "coauthor_network.json").read_text()
    )["meta"]["communities"]
    co_label = {c["id"]: (c.get("label_words") or ["?"])[0]
                for c in coauth_comms_meta}
    y_labels = [f"sc{sc} · {sem_label.get(sc, '?')}" for sc in top_sem]
    x_labels = [f"c{top_co[j]} · {co_label.get(top_co[j], '?')}" for j in col_perm]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat_perm, cmap="YlOrBr", aspect="auto",
                   vmin=0, vmax=max(mat_perm.max(), 0.1))
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=75, ha="right", fontsize=6.5)
    ax.set_yticklabels(y_labels, fontsize=6.5)
    ax.set_title(
        "Semantic × coauthor community co-occurrence\n"
        "(row-normalised — each row sums to 1 within top-22 coauthor comms)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Share of semantic community", fontsize=8)
    _save(fig, "07_partition_cooccurrence")
    print("  ✓ figures/07_partition_cooccurrence.pdf")

    # ------------------------------------------------------------------
    # Fig — whitening impact on pairwise cosine distribution.
    # Sample 5000 papers; compute pairwise cosines before vs after
    # whitening (mean-center + top-1 PC removal + z-score).
    # ------------------------------------------------------------------
    if EMBED_PATH.exists():
        print("[partition] whitening impact …")
        emb_df = pd.read_parquet(EMBED_PATH).sample(5000, random_state=SEED)
        E = np.stack(emb_df["emb"].tolist()).astype(np.float32)

        def _cos_sample(X, n=200_000):
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            m = Xn.shape[0]
            idx_i = np.random.randint(0, m, n)
            idx_j = np.random.randint(0, m, n)
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
            return (Xn[idx_i] * Xn[idx_j]).sum(axis=1)

        cos_raw = _cos_sample(E)

        # Apply the same whitening as scripts/06_author_similarity.py,
        # but with numpy since torch isn't in this environment. The math
        # is identical (768×768 symmetric eigh is trivial on CPU).
        Ec = E - E.mean(axis=0, keepdims=True)
        cov = (Ec.T @ Ec) / Ec.shape[0]
        evals, evecs = np.linalg.eigh(cov)  # ascending
        top_dirs = evecs[:, -1:]
        proj = (Ec @ top_dirs) @ top_dirs.T
        Ew = Ec - proj
        Ew = Ew / (Ew.std(axis=0, keepdims=True) + 1e-8)
        cos_white = _cos_sample(Ew)
        top1_var_share = float(evals[-1] / evals.sum())

        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.hist(cos_raw, bins=60, alpha=0.55, color=OKABE_ITO[1],
                label=f"Raw SPECTER2  (median {np.median(cos_raw):.3f})",
                density=True)
        ax.hist(cos_white, bins=60, alpha=0.55, color=OKABE_ITO[4],
                label=f"After whitening (median {np.median(cos_white):.3f})",
                density=True)
        ax.set_xlabel("Pairwise cosine similarity (5000-paper sample)")
        ax.set_ylabel("Density")
        ax.set_title(
            "Whitening compresses the pairwise-cosine distribution "
            f"(top-1 PC accounts for {top1_var_share*100:.1f}\\% of variance)"
        )
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
        _save(fig, "06_whitening_impact")
        (ROOT / "paper" / "analysis" / "_whitening_impact.json").write_text(
            json.dumps({"raw_median": float(np.median(cos_raw)),
                        "white_median": float(np.median(cos_white)),
                        "top1_pc_variance_share": top1_var_share,
                        "sample_pairs": int(len(cos_raw)),
                        "sample_papers": int(E.shape[0])}, indent=2)
        )
        print(f"  ✓ figures/06_whitening_impact.pdf  "
              f"(median {np.median(cos_raw):.3f} → {np.median(cos_white):.3f})")
    else:
        print(f"[partition] paper embeddings not found at {EMBED_PATH}; "
              "skipping whitening-impact figure.")

    print("[partition] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
