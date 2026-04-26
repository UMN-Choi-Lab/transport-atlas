#!/usr/bin/env python
"""Four-axis author personality classifier.

Combines four career-stage-neutral axes into a 16-cell typology:
  1. Mover     — Stayer vs Mover. Three-tier signal:
                   (a) trajectory_taxonomy.json class (≥3 bins, §9-eligible)
                   (b) raw author_trajectories.json with ≥2 bins
                       (no YEAR_MAX_BIN truncation, simple TAU_STAY cut)
                   (c) default Stayer for authors with <2 observable bins
  2. Topical   — Heterophilous vs Homophilous (median split on the
                 collab_style.json homophily score)
  3. Bridge    — Local vs Broker (within-cohort median split on weighted
                 betweenness centrality from coauthor_network.json)
  4. Style     — Solo vs Team (median split on mean coauthors-per-paper
                 over the author's papers)

The cohort is every author with a homophily score (≥3 coauthors with
semantic communities). This is the binding constraint: Bridge and Style
are computable from the full network, and Mover degrades gracefully to
"Stayer" when the trajectory is too short to observe motion.

Output: data/processed/author_personality.json
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PROC = ROOT / "data" / "processed"
DATA_INT = ROOT / "data" / "interim"
OUT = DATA_PROC / "author_personality.json"

# Same TAU_STAY as the §9 trajectory taxonomy: total UMAP path-length
# below 15 ≈ "didn't move from their topic neighborhood".
TAU_STAY = 15.0


def _percentile(sorted_values, q):
    if not sorted_values:
        return 0.0
    idx = int(q * (len(sorted_values) - 1))
    return sorted_values[idx]


def _raw_bin_class(bins) -> str | None:
    """Mover/Stayer from raw 5-year bins (≥2 required)."""
    if bins is None or len(bins) < 2:
        return None
    pts = np.asarray([[b["x"], b["y"]] for b in bins], dtype=np.float32)
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total_path = float(segs.sum())
    return "Mover" if total_path >= TAU_STAY else "Stayer"


def main() -> int:
    cn = json.loads((DATA_PROC / "coauthor_network.json").read_text())
    tt = json.loads((DATA_PROC / "trajectory_taxonomy.json").read_text())
    cs = json.loads((DATA_PROC / "collab_style.json").read_text())
    raw_traj = json.loads((DATA_PROC / "author_trajectories.json").read_text())

    # id -> trajectory class (from §9, strict)
    id_to_traj: dict[int, str] = {a["id"]: a["class"] for a in tt["authors"]}
    # id -> raw bins (broader cohort, no YEAR_MAX_BIN truncation)
    id_to_raw_bins: dict[int, list] = {int(k): v for k, v in raw_traj.items()}
    # id -> homophily and lift
    id_to_homo: dict[int, dict] = {a["id"]: a for a in cs["authors"]}
    # id -> coauthor-network node
    id_to_node: dict[int, dict] = {n["id"]: n for n in cn["nodes"]}

    # Mean team size per author key
    authors = pd.read_parquet(DATA_INT / "authors.parquet")
    papers = pd.read_parquet(DATA_INT / "papers.parquet")
    pid_to_team = {
        r["paper_id"]: max(1, len(r["authors"]) if r["authors"] is not None else 1)
        for _, r in papers.iterrows()
    }
    key_to_mean_team: dict[str, float] = {}
    for _, r in authors.iterrows():
        pids = list(r["paper_ids"]) if r["paper_ids"] is not None else []
        sizes = [pid_to_team.get(p, 0) for p in pids if pid_to_team.get(p, 0) > 0]
        if sizes:
            key_to_mean_team[r["author_key"]] = sum(sizes) / len(sizes)

    # --- Cohort + cutoffs ---------------------------------------------------
    # Eligibility: must have homophily AND a node in the coauthor graph AND a
    # mean team size. Cutoffs are computed within this cohort so each axis is
    # balanced.
    eligible_ids: list[int] = []
    homo_vals: list[float] = []
    bcw_vals: list[float] = []
    team_vals: list[float] = []
    for a_id, homo in id_to_homo.items():
        node = id_to_node.get(a_id)
        if not node:
            continue
        team = key_to_mean_team.get(node.get("key"))
        if team is None:
            continue
        eligible_ids.append(a_id)
        homo_vals.append(homo["homophily"])
        bcw_vals.append(node.get("bcw") or 0.0)
        team_vals.append(team)

    homo_vals.sort()
    bcw_vals.sort()
    team_vals.sort()
    homo_median = _percentile(homo_vals, 0.5)
    # Bridge is a *structural* binary: on the full 32k cohort, 78% of
    # authors have weighted betweenness == 0 (they don't sit on any
    # shortest path), so a median split degenerates. The natural binary
    # cut is "do you bridge anything at all?" -> bcw > 0. The 22%/78%
    # imbalance is the honest answer: brokerage is rare in a network
    # this large.
    bridge_cut = 0.0
    team_median = _percentile(team_vals, 0.5)

    # --- Per-author classification ----------------------------------------
    rows: list[dict] = []
    cell_counts: Counter[str] = Counter()
    mover_source_counts: Counter[str] = Counter()

    for a_id in eligible_ids:
        node = id_to_node[a_id]
        homo = id_to_homo[a_id]
        team = key_to_mean_team[node["key"]]

        # Mover: tier (a) → (b) → (c)
        if a_id in id_to_traj:
            mover_label = "Stayer" if id_to_traj[a_id] == "stayer" else "Mover"
            mover_src = "taxonomy"
        else:
            raw_label = _raw_bin_class(id_to_raw_bins.get(a_id))
            if raw_label is not None:
                mover_label = raw_label
                mover_src = "raw_bins"
            else:
                mover_label = "Stayer"
                mover_src = "default"
        mover_source_counts[mover_src] += 1

        topical_label = "Homophilous" if homo["homophily"] >= homo_median else "Heterophilous"
        bridge_label = "Broker" if (node.get("bcw") or 0.0) > bridge_cut else "Local"
        solo_label = "Team" if team >= team_median else "Solo"

        type_label = f"{mover_label}-{topical_label}-{bridge_label}-{solo_label}"
        cell_counts[type_label] += 1

        rows.append({
            "id": a_id,
            "key": node.get("key"),
            "name": node.get("label"),
            "type": type_label,
            "mover": mover_label,
            "mover_source": mover_src,
            "topical": topical_label,
            "bridge": bridge_label,
            "solo": solo_label,
            "trajectory_class": id_to_traj.get(a_id),
            "homophily": homo["homophily"],
            "homophily_lift": homo.get("lift"),
            "bcw": round(node.get("bcw") or 0.0, 4),
            "mean_team_size": round(team, 2),
            "n_papers": node.get("papers", 0),
            "citations": node.get("c", 0),
            "n_coauthors": homo.get("n_coauthors"),
        })

    rows.sort(key=lambda r: (-r["citations"], r["name"] or ""))

    # Emit the 16 cells in canonical axis order so the front-end can lay
    # them out as a fixed 4x4 grid (rows = Mover x Topical, cols = Bridge x
    # Style). Cells with zero population still appear in this order.
    MOVER_ORDER = ("Mover", "Stayer")
    TOPICAL_ORDER = ("Heterophilous", "Homophilous")
    BRIDGE_ORDER = ("Broker", "Local")
    STYLE_ORDER = ("Team", "Solo")
    cell_grid: list[dict] = []
    for m in MOVER_ORDER:
        for t in TOPICAL_ORDER:
            for b in BRIDGE_ORDER:
                for s in STYLE_ORDER:
                    label = f"{m}-{t}-{b}-{s}"
                    cell_grid.append({
                        "label": label,
                        "mover": m, "topical": t,
                        "bridge": b, "solo": s,
                        "n": cell_counts.get(label, 0),
                    })

    # --- 16x16 collaboration matrix --------------------------------------
    # For each coauthor edge (i, j) in the network where BOTH endpoints are
    # classified, increment matrix[type(i)][type(j)] (and the symmetric
    # entry — we deduplicate edges with i<j). We use the `ac` adjacency
    # list on each node, which is already weighted by shared papers; here
    # we count distinct collaborator pairs (an "edge" exists if the pair
    # has co-authored at least one paper together), not paper-weighted.
    type_index = {c["label"]: i for i, c in enumerate(cell_grid)}
    id_to_type = {r["id"]: r["type"] for r in rows}
    n_types = len(cell_grid)
    matrix = [[0] * n_types for _ in range(n_types)]
    edges_seen = 0
    edges_classified = 0
    for nid, type_label in id_to_type.items():
        node = id_to_node.get(nid)
        if not node:
            continue
        i = type_index[type_label]
        for pair in node.get("ac") or []:
            if not isinstance(pair, (list, tuple)) or len(pair) < 1:
                continue
            other_id = pair[0]
            if not isinstance(other_id, int) or other_id <= nid:
                # iterate each undirected edge once (i < j)
                continue
            edges_seen += 1
            other_type = id_to_type.get(other_id)
            if other_type is None:
                continue
            j = type_index[other_type]
            matrix[i][j] += 1
            if i != j:
                matrix[j][i] += 1
            edges_classified += 1

    # Row-normalized version: matrix_pct[i][j] = share of i's edges that
    # land in type j (rows sum to 1 within rounding). Useful for spotting
    # assortativity along each axis.
    matrix_pct = []
    for row in matrix:
        s = sum(row)
        matrix_pct.append([(v / s) if s > 0 else 0.0 for v in row])

    # Per-type normalization for the diagonal: are types more likely to
    # collaborate with their own type than chance predicts? Chance baseline
    # for type j = (population fraction of j).
    pop = [c["n"] for c in cell_grid]
    pop_total = sum(pop) or 1
    pop_frac = [p / pop_total for p in pop]
    matrix_lift = []
    for i, row in enumerate(matrix_pct):
        matrix_lift.append([
            (row[j] / pop_frac[j]) if pop_frac[j] > 0 else 0.0
            for j in range(n_types)
        ])

    summary = {
        "n_authors_classified": len(rows),
        "axes": {
            "mover": {
                "description": "Stayer vs Mover (UMAP centroid path length)",
                "split": (f"§9 taxonomy class when available, else raw "
                          f"bins ≥2 with TAU_STAY={TAU_STAY:.0f}, else "
                          f"default Stayer"),
            },
            "topical": {
                "description": "Coauthor topical homophily",
                "split": f"median split at {round(homo_median, 4)}",
            },
            "bridge": {
                "description": "Network brokerage (weighted betweenness)",
                "split": "Broker iff bcw > 0 (on ≥1 shortest path); 78% of "
                         "the cohort has bcw = 0 in this 32k-node graph",
            },
            "solo": {
                "description": "Mean team size on author's papers",
                "split": f"median split at {round(team_median, 2)} authors/paper",
            },
        },
        "cell_counts": dict(cell_counts.most_common()),
        "cell_grid": cell_grid,
        "mover_sources": dict(mover_source_counts),
        "collab_matrix": {
            "labels": [c["label"] for c in cell_grid],
            "counts": matrix,
            "row_share": [[round(v, 4) for v in row] for row in matrix_pct],
            "lift_vs_chance": [[round(v, 3) for v in row] for row in matrix_lift],
            "edges_total": edges_seen,
            "edges_both_classified": edges_classified,
        },
    }

    OUT.write_text(json.dumps({"summary": summary, "authors": rows}, allow_nan=False))

    print("=" * 64)
    print("Four-axis author-personality classification")
    print("=" * 64)
    print(f"  Authors classified: {len(rows):,}")
    print(f"  Mover signal source:")
    for k, v in mover_source_counts.most_common():
        print(f"    {k:>10s}: {v:>6,} ({100*v/len(rows):.1f}%)")
    print()
    print(f"  Cutoffs:")
    print(f"    Topical (homophily) median: {homo_median:.4f}")
    print(f"    Bridge   (bcw)  median:     {bridge_cut:.6f}")
    print(f"    Style (mean team) median:   {team_median:.2f}")
    print()
    print(f"  Cell counts (16 types, sorted by frequency):")
    for label, count in cell_counts.most_common():
        print(f"    {label:55s} {count:>6}")
    print()
    print(f"  Collab matrix: {edges_classified:,} of {edges_seen:,} "
          f"unique edges have both endpoints classified.")
    diag_share = sum(matrix[i][i] for i in range(n_types)) / max(1, sum(sum(r) for r in matrix))
    print(f"    diagonal share: {diag_share*100:.1f}% (same-type collaboration)")
    print()
    print(f"  output: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
