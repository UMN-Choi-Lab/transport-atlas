#!/usr/bin/env python
"""Sun and Rahwan comparables for direct figure and table alignment.

Emits to paper/manuscript/{tables,figures}/
    figures/04_authors_per_venue_bar.pdf  plus .png
        Mean authors per paper by venue like S and R Fig 2C
    figures/05_citation_distribution.pdf  plus .png
        Citation total distribution by author like S and R Fig 5C
    figures/05_shortest_path_distribution.pdf  plus .png
        Shortest path length distribution in the LCC like S and R Fig 6C
    tables/05_centrality_correlations.tex
        Kendall tau centrality correlation matrix like S and R Table 5

Reads papers.parquet for the venue bar.
Reads coauthor_network.json directly for the graph derived outputs.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

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
sys.path.insert(0, str(ROOT / "src"))
from transport_atlas.utils import config as _cfg  # noqa: E402

TABLES = ROOT / "paper" / "manuscript" / "tables"
FIGURES = ROOT / "paper" / "manuscript" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

YEAR_MAX = 2025
SEED = 42
TAU_SUBSAMPLE_MAX = 10_000
TAU_SOFT_LIMIT_SECONDS = 45.0


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


def _load_inputs():
    papers = pd.read_parquet(ROOT / "data" / "interim" / "papers.parquet")
    papers = papers[papers["year"] <= YEAR_MAX].copy()
    venues_cfg = _cfg.load_venues()
    cn = json.loads((ROOT / "data" / "processed" / "coauthor_network.json").read_text())
    return papers, venues_cfg, cn


def figure_authors_per_venue(papers: pd.DataFrame, venues_cfg) -> None:
    print("[sr] building authors per venue bar chart")
    papers = papers.copy()
    papers["n_authors"] = papers["authors"].apply(len)

    stats = (papers.groupby("venue_slug", as_index=False)["n_authors"]
             .agg(mean="mean", sem="sem", n="size")
             .sort_values("mean", ascending=False)
             .reset_index(drop=True))
    stats["sem"] = stats["sem"].fillna(0.0)

    slug_to_short = {v["slug"]: v.get("short", v["slug"]) for v in venues_cfg}
    labels = [slug_to_short.get(s, s) for s in stats["venue_slug"]]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(stats))
    ax.bar(
        x,
        stats["mean"],
        yerr=stats["sem"],
        color=OKABE_ITO[1],
        edgecolor="none",
        ecolor="#444444",
        capsize=1.5,
        width=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=5.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Mean authors per paper", fontsize=9)
    ax.set_title("By venue", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "04_authors_per_venue_bar")
    print("  ✓ figures/04_authors_per_venue_bar.pdf")


def figure_citation_distribution(nodes: list[dict]) -> None:
    print("[sr] building citation distribution")
    # The graph JSON already reflects the coauthor corpus.
    # That keeps this aligned with the single author exclusion rule.
    citations = np.array([int(n.get("c", 0)) for n in nodes], dtype=int)
    citations = citations[citations > 0]
    counts = Counter(citations.tolist())
    xs = np.array(sorted(counts.keys()), dtype=float)
    ys = np.array([counts[int(x)] for x in xs], dtype=float)

    lx = np.log(xs)
    ly = np.log(ys)
    slope, intercept = np.polyfit(lx, ly, 1)
    alpha = -slope
    fit = np.exp(intercept) * xs ** slope

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.loglog(xs, ys, "o", color=OKABE_ITO[0], markersize=3.5, alpha=0.75)
    ax.loglog(xs, fit, linestyle="--", color="#666666", linewidth=1.1)
    ax.set_xlabel("Citation total per author")
    ax.set_ylabel("Frequency")
    ax.set_title(fr"Author citation distribution  $\alpha={alpha:.2f}$")
    ax.grid(which="both", alpha=0.2)
    _save(fig, "05_citation_distribution")
    print("  ✓ figures/05_citation_distribution.pdf")


def figure_shortest_path_distribution(nodes: list[dict], edges: list[dict]) -> None:
    print("[sr] building shortest path distribution")
    g = nx.Graph()
    g.add_nodes_from(int(n["id"]) for n in nodes)
    g.add_edges_from((int(e["source"]), int(e["target"])) for e in edges)
    lcc = max(nx.connected_components(g), key=len)
    h = g.subgraph(lcc).copy()

    rng = np.random.default_rng(SEED)
    sample_n = min(2000, h.number_of_nodes())
    sources = rng.choice(np.array(list(h.nodes()), dtype=int), size=sample_n, replace=False)

    path_counter = Counter()
    total_len = 0
    total_pairs = 0
    for i, source in enumerate(sources, 1):
        dists = nx.single_source_shortest_path_length(h, int(source))
        for target, dist in dists.items():
            if target == source:
                continue
            path_counter[int(dist)] += 1
            total_len += int(dist)
            total_pairs += 1
        if i % 250 == 0 or i == sample_n:
            print(f"    sampled {i:,} / {sample_n:,} sources")

    xs = np.array(sorted(path_counter.keys()), dtype=int)
    ys = np.array([path_counter[int(x)] for x in xs], dtype=int)
    mean_path = total_len / total_pairs

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(xs, ys, width=0.8, color=OKABE_ITO[2], edgecolor="none")
    ax.set_xlabel("Shortest path length")
    ax.set_ylabel("Count")
    ax.set_xticks(xs)
    ax.set_title(f"Shortest path lengths in LCC  mean={mean_path:.2f}")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "05_shortest_path_distribution")
    print("  ✓ figures/05_shortest_path_distribution.pdf")


def _kendall_tau_with_fallback(df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, bool, int]:
    start = time.perf_counter()
    n = len(df)
    corr = np.eye(len(labels), dtype=float)
    for i, a in enumerate(labels):
        for j in range(i + 1, len(labels)):
            b = labels[j]
            tau = kendalltau(df[a], df[b], nan_policy="omit").statistic
            corr[i, j] = tau
            corr[j, i] = tau
    elapsed = time.perf_counter() - start
    if elapsed <= TAU_SOFT_LIMIT_SECONDS or n <= TAU_SUBSAMPLE_MAX:
        return corr, False, n

    print(f"[sr] kendall tau full run took {elapsed:.1f}s  rerunning on {TAU_SUBSAMPLE_MAX:,} authors")
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(df.index.to_numpy(), size=TAU_SUBSAMPLE_MAX, replace=False)
    sdf = df.loc[sample_idx].reset_index(drop=True)
    corr = np.eye(len(labels), dtype=float)
    for i, a in enumerate(labels):
        for j in range(i + 1, len(labels)):
            b = labels[j]
            tau = kendalltau(sdf[a], sdf[b], nan_policy="omit").statistic
            corr[i, j] = tau
            corr[j, i] = tau
    return corr, True, len(sdf)


def table_centrality_correlations(nodes: list[dict]) -> None:
    print("[sr] building centrality correlation table")
    records = []
    for n in nodes:
        row = {
            "d": n.get("d", 0),
            "s": n.get("s", 0),
            "c": n.get("c", 0),
            "bc": n.get("bc", 0.0),
            "pr": n.get("pr", 0.0),
            "prw": n.get("prw", 0.0),
        }
        # Our JSON uses bcw for weighted betweenness.
        if "bcw" in n:
            row["bcw"] = n.get("bcw", 0.0)
        records.append(row)
    df = pd.DataFrame.from_records(records)

    labels = ["d", "s", "c", "bc"]
    if "bcw" in df.columns:
        labels.append("bcw")
    labels += ["pr", "prw"]

    corr, subsampled, n_used = _kendall_tau_with_fallback(df[labels], labels)
    display = {
        "d": r"Degree $d$",
        "s": r"Strength $s$",
        "c": r"Citations $c$",
        "bc": r"Betweenness $bc$",
        "bcw": r"Weighted betweenness $bc_w$",
        "pr": r"PageRank $pr$",
        "prw": r"Weighted PageRank $pr_w$",
    }

    row_max = []
    for i in range(len(labels)):
        vals = [corr[i, j] for j in range(len(labels)) if j != i]
        row_max.append(max(vals))

    lines = []
    if subsampled:
        lines.append(f"% Kendall tau computed on a random {n_used:,} author subsample")
    else:
        lines.append(f"% Kendall tau computed on all {n_used:,} authors")
    lines += [
        r"\begin{tabular}{" + "l" + "r" * len(labels) + "}",
        r"\toprule",
        " & " + " & ".join(display[k] for k in labels) + r" \\",
        r"\midrule",
    ]
    for i, key in enumerate(labels):
        cells = []
        for j in range(len(labels)):
            val = corr[i, j]
            text = f"{val:.3f}"
            if i != j and np.isclose(val, row_max[i]):
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        lines.append(display[key] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "05_centrality_correlations.tex").write_text("\n".join(lines) + "\n")
    print("  ✓ tables/05_centrality_correlations.tex")

    # Heatmap companion figure
    short_labels = {
        "d": r"$d$", "s": r"$s$", "c": r"$c$",
        "bc": r"$bc$", "bcw": r"$bc_w$",
        "pr": r"$pr$", "prw": r"$pr_w$",
    }
    tick_labels = [short_labels[k] for k in labels]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr[i, j]
            txt_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label(r"Kendall $\tau$", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_centrality_heatmap.pdf")
    fig.savefig(FIGURES / "05_centrality_heatmap.png", dpi=300)
    plt.close(fig)
    print("  ✓ figures/05_centrality_heatmap.pdf")


def main() -> int:
    print("[sr] loading inputs")
    papers, venues_cfg, cn = _load_inputs()
    nodes = cn["nodes"]
    edges = cn["edges"]
    print(f"[sr]   papers={len(papers):,}  nodes={len(nodes):,}  edges={len(edges):,}")

    figure_authors_per_venue(papers, venues_cfg)
    figure_citation_distribution(nodes)
    figure_shortest_path_distribution(nodes, edges)
    table_centrality_correlations(nodes)
    print("[sr] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
