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
# the trajectory set into non-degenerate classes.
TAU_STAY = 15.0
TAU_EFF_DRIFT = 0.60
# Restrict to authors with ≥ MIN_BINS five-year bins. With only 2 bins
# η is forced to 1 (single segment, path = net), so 2-bin trajectories
# can never register a non-monotone shape and would inflate the stayer
# and drifter classes with degenerate cases. ≥4 bins corresponds to a
# ≥20-year active publication window in the corpus and matches the
# median bin count of returners and switchers.
MIN_BINS = 4
# Inside the low-η group, distinguish *round-trip* (path closes back on
# itself, small net displacement) from *re-oriented* (one or two big
# turns but net displacement is large — e.g., mid-career topic switch).
# Using the same scale as TAU_STAY: a "stayed near home" trajectory and
# a "settled in a new home" trajectory both have net_disp roughly ≤
# TAU_STAY around their respective center; anything beyond that is a
# genuine re-orientation.
TAU_NET = 15.0
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


def _classify(total_path: float, efficiency: float, net_disp: float) -> str:
    if total_path < TAU_STAY:
        return "stayer"
    if efficiency >= TAU_EFF_DRIFT:
        return "drifter"
    if net_disp < TAU_NET:
        return "returner"   # round-trip pivoter: path closes back near origin
    return "switcher"        # re-oriented: low η but ended far from start


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
            "class":       _classify(total_path, eff, net_disp),
            "start_year":  int(bins[0]["p"]),
            "end_year":    int(bins[-1]["p"]) + 5,
        })
    df_all = pd.DataFrame(rows)
    n_total = len(df_all)
    df = df_all[df_all["bins"] >= MIN_BINS].copy()
    n_dropped = n_total - len(df)
    print(f"[traj] {n_total:,} authors with ≥2 bins; "
          f"restricted to {len(df):,} with ≥{MIN_BINS} bins "
          f"(dropped {n_dropped:,} short-trajectory authors).",
          flush=True)
    print(f"[traj] classified {len(df):,} shape-rich authors", flush=True)
    CLASS_ORDER = ("stayer", "drifter", "returner", "switcher")
    for cls in CLASS_ORDER:
        n = (df["class"] == cls).sum()
        print(f"       {cls:>9s}: {n:,} ({100*n/len(df):.1f}%)", flush=True)

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
    ).reindex(list(CLASS_ORDER))
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
    colors = {"stayer":   OKABE_ITO[1],
              "drifter":  OKABE_ITO[2],
              "returner": OKABE_ITO[5],
              "switcher": OKABE_ITO[3]}
    # Plot order: drifters last so they sit on top in dense regions.
    for cls in ("stayer", "switcher", "returner", "drifter"):
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
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _save(fig, "09_trajectory_scatter")
    print("  ✓ figures/09_trajectory_scatter.pdf")

    # ------------------------------------------------------------------
    # Figure — 4-panel examples (1 per class). Hand-picked exemplars
    # whose trajectories cleanly illustrate each class, falling back to
    # the top-cited author of that class if a named exemplar is missing
    # or has been reclassified by a future regen.
    # ------------------------------------------------------------------
    EXEMPLARS = {
        "stayer":   "bhat, chandra",        # huge career, one topic neighborhood
        "drifter":  "kockelman, kara",      # efficient one-way trajectory
        "returner": "mahmassani, hani",     # round-trip via 1990 excursion
        "switcher": "davis, gary",          # one big mid-career re-orientation
    }

    def _pick_for(cls: str) -> pd.Series | None:
        target = EXEMPLARS.get(cls, "")
        if target:
            matches = df[(df["class"] == cls)
                         & df["name"].str.lower().str.startswith(target)]
            if not matches.empty:
                return matches.sort_values("citations",
                                            ascending=False).iloc[0]
            print(f"[traj] WARNING: exemplar {target!r} not in class {cls!r}; "
                  f"falling back to top-cited", flush=True)
        cls_df = df[df["class"] == cls]
        if cls_df.empty:
            return None
        return (cls_df.sort_values("n_papers_traj", ascending=False)
                       .head(20)
                       .sort_values("citations", ascending=False)
                       .iloc[0])

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0))
    for col_idx, cls in enumerate(CLASS_ORDER):
        ax = axes[col_idx]
        r = _pick_for(cls)
        if r is None:
            ax.set_axis_off()
            continue
        bins = [b for b in trajs[str(int(r["id"]))]
                if int(b["p"]) <= YEAR_MAX_BIN]
        xs = [b["x"] for b in bins]
        ys = [b["y"] for b in bins]
        ns = [b["n"] for b in bins]
        ax.plot(xs, ys, color=colors[cls], linewidth=1.4, alpha=0.7)
        sizes = [min(14 + 4.0 * np.sqrt(n), 55) for n in ns]
        ax.scatter(xs, ys, s=sizes,
                   color=colors[cls], edgecolors="black",
                   linewidths=0.4, alpha=0.7, zorder=3)
        for b in bins:
            ax.annotate(f"{b['p']}", (b["x"], b["y"]),
                        fontsize=6.0, alpha=0.9,
                        textcoords="offset points", xytext=(6, 4))
        raw_name = str(r["name"] or "")
        if "," in raw_name:
            last, rest = raw_name.split(",", 1)
            pretty = (last.strip().title() + ", " +
                      " ".join(p.strip().title() for p in rest.split()))
        else:
            pretty = raw_name.title()
        title = f"{pretty}\n({cls}, $\\eta = {r['efficiency']:.2f}$)"
        ax.set_title(title, fontsize=8.5)
        if xs:
            x_mid = (max(xs) + min(xs)) / 2
            y_mid = (max(ys) + min(ys)) / 2
            span = max(max(xs) - min(xs), max(ys) - min(ys), 20.0)
            half = span * 0.65
            ax.set_xlim(x_mid - half, x_mid + half)
            ax.set_ylim(y_mid - half, y_mid + half)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    _save(fig, "09_trajectory_examples")
    print("  ✓ figures/09_trajectory_examples.pdf")

    # ------------------------------------------------------------------
    # Summary JSON for prose
    # ------------------------------------------------------------------
    summary = {
        "n_total":           int(n_total),
        "n_shape_rich":      int(len(df)),
        "min_bins":          MIN_BINS,
        "tau_stay":          TAU_STAY,
        "tau_eff":           TAU_EFF_DRIFT,
        "tau_net":           TAU_NET,
        "n_stayer":      int((df["class"] == "stayer").sum()),
        "n_drifter":     int((df["class"] == "drifter").sum()),
        "n_returner":    int((df["class"] == "returner").sum()),
        "n_switcher":    int((df["class"] == "switcher").sum()),
        "median_path_overall":   float(df["total_path"].median()),
        "median_net_overall":    float(df["net_disp"].median()),
        "median_eff_overall":    float(df["efficiency"].median()),
        "median_papers_stayer":  float(df.loc[df["class"] == "stayer",
                                              "papers_all"].median()),
        "median_papers_drifter": float(df.loc[df["class"] == "drifter",
                                              "papers_all"].median()),
        "median_papers_returner": float(df.loc[df["class"] == "returner",
                                                "papers_all"].median()),
        "median_papers_switcher": float(df.loc[df["class"] == "switcher",
                                                "papers_all"].median()),
    }
    (ROOT / "paper" / "analysis" / "_trajectory_taxonomy.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("  ✓ paper/analysis/_trajectory_taxonomy.json")

    # ------------------------------------------------------------------
    # Per-author classification, written to data/processed/ for the site
    # to consume (Trajectories tab). Keeps only the shape-rich subset.
    # ------------------------------------------------------------------
    EXEMPLAR_NAMES = {
        "bhat, chandra":     "stayer",
        "kockelman, kara":   "drifter",
        "mahmassani, hani":  "returner",
        "davis, gary":       "switcher",
    }
    site_rows = []
    for _, r in df.iterrows():
        nm = str(r["name"] or "").lower()
        is_ex = any(nm.startswith(k) and r["class"] == v
                    for k, v in EXEMPLAR_NAMES.items())
        site_rows.append({
            "id":         int(r["id"]),
            "name":       r["name"],
            "class":      r["class"],
            "eta":        round(float(r["efficiency"]), 3),
            "total_path": round(float(r["total_path"]), 2),
            "net_disp":   round(float(r["net_disp"]), 2),
            "papers":     int(r["papers_all"]),
            "citations":  int(r["citations"]),
            "n_bins":     int(r["bins"]),
            "span_years": int(r["span_years"]),
            "exemplar":   is_ex,
        })
    site_payload = {
        "config": {
            "tau_stay":   TAU_STAY,
            "tau_eff":    TAU_EFF_DRIFT,
            "tau_net":    TAU_NET,
            "min_bins":   MIN_BINS,
            "n_total":    int(n_total),
            "n_shape_rich": int(len(df)),
        },
        "authors": site_rows,
    }
    site_path = ROOT / "data" / "processed" / "trajectory_taxonomy.json"
    site_path.write_text(json.dumps(site_payload, allow_nan=False))
    print(f"  ✓ data/processed/trajectory_taxonomy.json "
          f"({len(site_rows):,} authors)")

    print("[traj] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
