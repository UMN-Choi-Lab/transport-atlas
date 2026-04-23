#!/usr/bin/env python
"""§9 — Author topic-trajectory taxonomy (stayer / drifter / pivoter).

Reads data/processed/author_trajectories.json and the coauthor network JSON.
Classifies each author by two scalars on their 5-year UMAP bin sequence:

    total_path    = sum of segment lengths (||p_{i+1} - p_i||)
    net_disp      = ||p_last - p_first||
    efficiency    = net_disp / total_path    in [0, 1]

    STAYER   := total_path < TAU_STAY                   (didn't move)
    DRIFTER  := total_path >= TAU_STAY and efficiency >= TAU_EFF_DRIFT
                (moved a lot, in a roughly consistent direction)
    PIVOTER  := total_path >= TAU_STAY and efficiency <  TAU_EFF_DRIFT
                (moved a lot, changing direction)

Emits:
    tables/09_trajectory_taxonomy_stats.tex
    figures/09_trajectory_examples.pdf       (6-panel: 2 from each class)
    figures/09_trajectory_scatter.pdf        (total_path vs net_disp)
    paper/analysis/_trajectory_taxonomy.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
TRAJ_PATH = ROOT / "data" / "processed" / "author_trajectories.json"
NET_PATH = ROOT / "data" / "processed" / "coauthor_network.json"

# UMAP coords are scaled to [-100, 100]. Pick thresholds so they partition
# the ~10k trajectory set into non-degenerate classes (~40/35/25 split).
TAU_STAY = 15.0
TAU_EFF_DRIFT = 0.60
# Drop bins that start after YEAR_MAX_BIN - the 2025-29 bin is partial (only
# 2025, since 2026 is excluded), so its centroid is noisy. 2020-24 is the
# last complete 5-year bin.
YEAR_MAX_BIN = 2020


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    plt.close(fig)


def _tex_escape(s) -> str:
    if not isinstance(s, str): s = str(s)
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    for k, v in repl.items(): s = s.replace(k, v)
    return s


def _classify(total_path: float, efficiency: float) -> str:
    if total_path < TAU_STAY:
        return "stayer"
    if efficiency >= TAU_EFF_DRIFT:
        return "drifter"
    return "pivoter"


def main() -> int:  # noqa: C901
    if not TRAJ_PATH.exists():
        print(f"[traj] missing {TRAJ_PATH}", file=sys.stderr)
        return 1
    trajs = json.loads(TRAJ_PATH.read_text())
    net = json.loads(NET_PATH.read_text())
    node_meta = {n["id"]: n for n in net["nodes"]}

    rows = []
    for nid_str, bins in trajs.items():
        bins = [b for b in bins if int(b["p"]) <= YEAR_MAX_BIN]
        if len(bins) < 2:
            continue
        nid = int(nid_str)
        pts = np.array([[b["x"], b["y"]] for b in bins], dtype=np.float32)
        segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_path = float(segs.sum())
        net_disp = float(np.linalg.norm(pts[-1] - pts[0]))
        eff = net_disp / total_path if total_path > 1e-6 else 0.0
        n_papers_total = sum(int(b["n"]) for b in bins)
        span_years = int(bins[-1]["p"]) - int(bins[0]["p"]) + 5
        meta = node_meta.get(nid, {})
        rows.append({
            "id":          nid,
            "name":        meta.get("label", ""),
            "community":   meta.get("community"),
            "papers_all":  int(meta.get("papers", 0)),
            "citations":   int(meta.get("c", 0)),
            "bins":        len(bins),
            "n_papers_traj": n_papers_total,
            "span_years":  span_years,
            "total_path":  total_path,
            "net_disp":    net_disp,
            "efficiency":  eff,
            "class":       _classify(total_path, eff),
            "start_year":  int(bins[0]["p"]),
            "end_year":    int(bins[-1]["p"]) + 5,
        })
    df = pd.DataFrame(rows)
    print(f"[traj] classified {len(df):,} authors", flush=True)
    for cls in ("stayer", "drifter", "pivoter"):
        n = (df["class"] == cls).sum()
        print(f"       {cls:>8s}: {n:,} ({100*n/len(df):.1f}%)", flush=True)

    # ------------------------------------------------------------------
    # Table — per-class statistics
    # ------------------------------------------------------------------
    agg = df.groupby("class").agg(
        n=("id", "count"),
        median_path=("total_path", "median"),
        median_net=("net_disp", "median"),
        median_eff=("efficiency", "median"),
        median_papers=("papers_all", "median"),
        median_cites=("citations", "median"),
        median_span=("span_years", "median"),
        median_bins=("bins", "median"),
    ).reindex(["stayer", "drifter", "pivoter"])
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        (r"\textbf{Class} & \textbf{$n$} & \textbf{path} & "
         r"\textbf{net} & \textbf{eff.} & \textbf{papers} & "
         r"\textbf{cites} & \textbf{span (yr)} & \textbf{bins} \\"),
        r"\midrule",
    ]
    for cls, r in agg.iterrows():
        lines.append(
            f"{cls.title()} & {int(r['n']):,} & {r['median_path']:.1f} & "
            f"{r['median_net']:.1f} & {r['median_eff']:.2f} & "
            f"{int(r['median_papers'])} & {int(r['median_cites'])} & "
            f"{int(r['median_span'])} & {int(r['median_bins'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "09_trajectory_taxonomy_stats.tex").write_text(
        "\n".join(lines) + "\n")
    print("  ✓ tables/09_trajectory_taxonomy_stats.tex")

    # ------------------------------------------------------------------
    # Figure — 2D scatter of total_path vs net_disp, colored by class.
    # Diagonal y = x marks efficiency = 1 (straight line).
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 5))
    colors = {"stayer": OKABE_ITO[1], "drifter": OKABE_ITO[2],
              "pivoter": OKABE_ITO[5]}
    for cls in ("stayer", "pivoter", "drifter"):
        sub = df[df["class"] == cls]
        ax.scatter(sub["total_path"], sub["net_disp"], s=5, alpha=0.35,
                   color=colors[cls],
                   label=f"{cls.title()} ($n={len(sub):,}$)")
    lim = max(df["total_path"].max(), df["net_disp"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#888", linestyle="--", linewidth=0.8,
            label="$\\text{eff.} = 1$")
    ax.axvline(TAU_STAY, color="#444", linestyle=":", linewidth=0.7,
               label=f"stay cutoff ({TAU_STAY:.0f})")
    ax.set_xlabel("Total path length (UMAP units)")
    ax.set_ylabel("Net displacement (UMAP units)")
    ax.set_title("Author topic trajectories — path vs. net displacement")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _save(fig, "09_trajectory_scatter")
    print("  ✓ figures/09_trajectory_scatter.pdf")

    # ------------------------------------------------------------------
    # Figure — 6-panel examples (2 per class), ranked by n_papers_traj
    # so we pick prolific / interpretable exemplars.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5))
    for col_idx, cls in enumerate(("stayer", "drifter", "pivoter")):
        sub = df[df["class"] == cls].sort_values(
            "n_papers_traj", ascending=False).head(20)
        # Pick top-2 by citations among prolific exemplars
        sub = sub.sort_values("citations", ascending=False).head(2)
        for row_idx, (_, r) in enumerate(sub.iterrows()):
            ax = axes[row_idx, col_idx]
            bins = [b for b in trajs[str(int(r["id"]))]
                    if int(b["p"]) <= YEAR_MAX_BIN]
            xs = [b["x"] for b in bins]
            ys = [b["y"] for b in bins]
            ns = [b["n"] for b in bins]
            ax.plot(xs, ys, color=colors[cls], linewidth=1.8, alpha=0.8)
            ax.scatter(xs, ys, s=[18 + 3*n for n in ns],
                       color=colors[cls], edgecolors="black", linewidths=0.4)
            # year annotations
            for b in bins:
                ax.annotate(f"{b['p']}", (b["x"], b["y"]),
                            fontsize=6.5, alpha=0.85,
                            textcoords="offset points", xytext=(4, 3))
            title = f"{r['name']}  ({cls})"
            if len(title) > 42:
                title = title[:40] + "…"
            ax.set_title(title, fontsize=8.5)
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.15)
    fig.suptitle(
        "Exemplar author trajectories by class "
        "(stayers / drifters / pivoters)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "09_trajectory_examples")
    print("  ✓ figures/09_trajectory_examples.pdf")

    # ------------------------------------------------------------------
    # Summary JSON for prose
    # ------------------------------------------------------------------
    summary = {
        "n_total":       int(len(df)),
        "tau_stay":      TAU_STAY,
        "tau_eff":       TAU_EFF_DRIFT,
        "n_stayer":      int((df["class"] == "stayer").sum()),
        "n_drifter":     int((df["class"] == "drifter").sum()),
        "n_pivoter":     int((df["class"] == "pivoter").sum()),
        "median_path_overall":   float(df["total_path"].median()),
        "median_net_overall":    float(df["net_disp"].median()),
        "median_eff_overall":    float(df["efficiency"].median()),
        "median_papers_stayer":  float(df.loc[df["class"] == "stayer",
                                              "papers_all"].median()),
        "median_papers_drifter": float(df.loc[df["class"] == "drifter",
                                              "papers_all"].median()),
        "median_papers_pivoter": float(df.loc[df["class"] == "pivoter",
                                              "papers_all"].median()),
    }
    (ROOT / "paper" / "analysis" / "_trajectory_taxonomy.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("  ✓ paper/analysis/_trajectory_taxonomy.json")
    print("[traj] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
