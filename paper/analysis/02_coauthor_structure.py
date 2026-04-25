#!/usr/bin/env python
"""§5 — Coauthor network structure.

Emits to paper/manuscript/{tables,figures}/:
    tables/05_top_communities.tex
    tables/05_top_centrality.tex
    figures/05_degree_distribution.pdf  (+ .png)
    figures/05_giant_component_over_time.pdf  (+ .png)
    figures/05_bridges.pdf  (+ .png)

Reads coauthor_network.json directly (no recomputation needed).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- shared plot settings (copied from 01_descriptive_tables.py) ----
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


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    plt.close(fig)


def _tex_escape(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    for k, v in repl.items(): s = s.replace(k, v)
    return s


def _titlecase_label(s: str) -> str:
    """Prettify lowercase 'last, first' author labels for tables."""
    if "," in s:
        last, rest = s.split(",", 1)
        parts = [p.strip() for p in rest.split()]
        return (last.strip().title() + ", " +
                " ".join(p.title() for p in parts))
    return s.title()


def main() -> int:
    print("[coauthor] loading coauthor_network.json …")
    cn = json.loads((ROOT / "data" / "processed" / "coauthor_network.json").read_text())
    nodes = cn["nodes"]
    edges = cn["edges"]
    meta = cn["meta"]
    comms = meta["communities"]
    print(f"[coauthor]   nodes={len(nodes):,}  edges={len(edges):,}  "
          f"communities={len(comms)}")

    # ------------------------------------------------------------------
    # Table 5 — top-20 coauthor communities
    # ------------------------------------------------------------------
    non_misc = [c for c in comms if not c.get("misc")]
    top20 = sorted(non_misc, key=lambda c: -c["size"])[:20]
    lines = [
        r"\begin{tabular}{rlp{0.32\linewidth}p{0.32\linewidth}}",
        r"\toprule",
        (r"\textbf{\#} & \textbf{Size} & \textbf{Keyword label} "
         r"& \textbf{Exemplar authors} \\"),
        r"\midrule",
    ]
    for c in top20:
        kws = ", ".join((c.get("label_words") or [])[:6])
        exemplars = ", ".join(
            _titlecase_label(a) for a in (c.get("top_authors") or [])[:5]
        )
        lines.append(
            f"{c['id']} & {c['size']:,} & {_tex_escape(kws)} & "
            f"{_tex_escape(exemplars)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "05_top_communities.tex").write_text("\n".join(lines) + "\n")
    print("  ✓ tables/05_top_communities.tex")

    # ------------------------------------------------------------------
    # Table 6 — top-30 authors by centrality (combined rank)
    # Score: rank-averaged across strength (s), betweenness (bc),
    # and weighted PageRank (prw). Lower rank sum = more central.
    # ------------------------------------------------------------------
    df = pd.DataFrame([
        {"label": n["label"], "papers": n["papers"], "s": n["s"],
         "bc": n["bc"], "prw": n["prw"], "c": n["c"]}
        for n in nodes
    ])
    df["rank_s"] = df["s"].rank(ascending=False, method="min")
    df["rank_bc"] = df["bc"].rank(ascending=False, method="min")
    df["rank_prw"] = df["prw"].rank(ascending=False, method="min")
    df["avg_rank"] = (df["rank_s"] + df["rank_bc"] + df["rank_prw"]) / 3
    top30 = df.sort_values("avg_rank").head(30)

    lines = [
        r"\begin{tabular}{rlrrrrr}",
        r"\toprule",
        (r"\textbf{\#} & \textbf{Author} & \textbf{Papers} & "
         r"\textbf{Strength} & \textbf{Betweenness} & "
         r"\textbf{PageRank$_{\!w}$} & \textbf{Citations} \\"),
        r"\midrule",
    ]
    for i, (_, r) in enumerate(top30.iterrows(), 1):
        lines.append(
            f"{i} & {_tex_escape(_titlecase_label(r['label']))} & "
            f"{int(r['papers'])} & {int(r['s'])} & "
            f"{r['bc']:.2f} & {r['prw']:.2f} & "
            f"{int(r['c']):,} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "05_top_centrality.tex").write_text("\n".join(lines) + "\n")
    print("  ✓ tables/05_top_centrality.tex")

    # ------------------------------------------------------------------
    # Fig 5 — degree & strength distributions (log-log, two panels)
    # ------------------------------------------------------------------
    degs = np.array([n["d"] for n in nodes])
    strs = np.array([n["s"] for n in nodes])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, vals, xlabel, color in zip(
        axes,
        [degs, strs],
        ["Degree $d$ (# unique coauthors)", "Strength $s$ (total collaborations)"],
        [OKABE_ITO[1], OKABE_ITO[4]],
    ):
        c = Counter(vals.astype(int).tolist())
        ks = np.array(sorted(c.keys()))
        ys = np.array([c[k] for k in ks])
        mask = (ks >= 1) & (ys > 0)
        ax.loglog(ks[mask], ys[mask], "o", color=color, markersize=3.5, alpha=0.75)
        # Fit α
        lx, ly = np.log(ks[mask].astype(float)), np.log(ys[mask].astype(float))
        slope, intercept = np.polyfit(lx, ly, 1)
        ref_k = ks[mask].astype(float)
        fit = np.exp(intercept) * ref_k ** slope
        ax.loglog(ref_k, fit, color="#444", linewidth=1.0, linestyle="--",
                  label=fr"OLS fit: $\alpha={-slope:.2f}$")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.legend(frameon=False, fontsize=7.5)
        ax.grid(which="both", alpha=0.2)
    fig.tight_layout()
    _save(fig, "05_degree_distribution")
    print("  ✓ figures/05_degree_distribution.pdf")

    # ------------------------------------------------------------------
    # Fig 5b — largest-component fraction vs year window.
    # Rebuild the graph at each cutoff by including only edges with
    # any year <= cutoff. Compute connected-component fractions via
    # union-find (no networkx dep).
    # ------------------------------------------------------------------
    def _giant_component_fraction(ids_in_edges: set[int], edge_pairs):
        if not ids_in_edges:
            return 0.0
        parent = {i: i for i in ids_in_edges}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for u, v in edge_pairs:
            union(u, v)
        sizes = Counter(find(i) for i in ids_in_edges).values()
        return max(sizes) / sum(sizes)

    cutoffs = list(range(1975, 2027, 5))
    giant_frac = []
    for y in cutoffs:
        valid_edges = [(e["source"], e["target"]) for e in edges
                       if e.get("years") and min(e["years"]) <= y]
        used = set()
        for u, v in valid_edges:
            used.add(u); used.add(v)
        giant_frac.append(_giant_component_fraction(used, valid_edges))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(cutoffs, [f * 100 for f in giant_frac],
            marker="o", color=OKABE_ITO[2])
    ax.axhline(58, color=OKABE_ITO[5], linestyle=":", linewidth=1,
               label="Sun & Rahwan (2017): 58%")
    ax.set_xlabel("Cutoff year (edges with first collaboration ≤ $y$)")
    ax.set_ylabel("Largest-component fraction (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 100)
    _save(fig, "05_giant_component_over_time")
    # Save the series for prose quoting
    (ROOT / "paper" / "analysis" / "_giant_fraction.json").write_text(
        json.dumps({"cutoffs": cutoffs,
                    "giant_fraction_pct": [round(f * 100, 2) for f in giant_frac]},
                   indent=2)
    )
    print(f"  ✓ figures/05_giant_component_over_time.pdf  (final: {giant_frac[-1]*100:.1f}%)")

    # ------------------------------------------------------------------
    # Fig 6 — bridge edges (top-100 by edge betweenness that cross
    # communities) shown as a small bipartite-ish matrix of community
    # pairs with edge count.
    # ------------------------------------------------------------------
    node_by_id = {n["id"]: n for n in nodes}
    cross_edges = []
    for e in edges:
        ca = node_by_id.get(e["source"], {}).get("community")
        cb = node_by_id.get(e["target"], {}).get("community")
        if ca is None or cb is None or ca == cb:
            continue
        cross_edges.append((min(ca, cb), max(ca, cb), e.get("eb", 0)))
    # Top-100 bridges by edge betweenness
    cross_edges.sort(key=lambda x: -x[2])
    top_bridges = cross_edges[:100]

    # Count community-pair frequencies in the top-100 bridges
    pair_counts = Counter((a, b) for a, b, _ in top_bridges)
    # Shortlist communities that appear in top-100
    active_comms = sorted({c for pair in pair_counts for c in pair})
    comm_label = {c["id"]: (c.get("label_words") or ["?"])[0] for c in comms}

    # Matrix of pair counts
    idx = {cid: i for i, cid in enumerate(active_comms)}
    M = np.zeros((len(active_comms), len(active_comms)))
    for (a, b), n in pair_counts.items():
        M[idx[a], idx[b]] = n
        M[idx[b], idx[a]] = n

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto")
    labels = [f"{c} · {comm_label[c]}" for c in active_comms]
    ax.set_xticks(range(len(active_comms)))
    ax.set_yticks(range(len(active_comms)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6.5)
    ax.set_yticklabels(labels, fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Bridge-edge count", fontsize=8)
    _save(fig, "05_bridges")
    print(f"  ✓ figures/05_bridges.pdf  "
          f"({len(active_comms)} communities involved in top-100 bridges)")

    print("[coauthor] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
