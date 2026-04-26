#!/usr/bin/env python
"""Four-axis author personality classifier.

Combines four career-stage-neutral axes into a 16-cell typology:
  1. Mover     — Stayer vs Mover (drifter / returner / switcher pooled)
                 from trajectory_taxonomy.json
  2. Topical   — Heterophilous vs Homophilous (median split on the
                 collab_style.json homophily score)
  3. Bridge    — Local vs Broker (top-decile split on weighted
                 betweenness centrality from coauthor_network.json)
  4. Solo      — Solo vs Team (median split on mean coauthors-per-paper
                 over the author's papers)

The classification is restricted to authors who have all four axes
defined, i.e. trajectory-eligible (>=3 bins) AND in the homophily-scored
set (>=3 coauthors with semantic communities).

Output: data/processed/author_personality.json
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PROC = ROOT / "data" / "processed"
DATA_INT = ROOT / "data" / "interim"
OUT = DATA_PROC / "author_personality.json"

# Bridge cutoff: median split within the classified subset (balanced
# axis). The classified set is already an elite slice of the full graph
# — most of them sit in the global top decile of betweenness — so a
# within-subset median is the right binary cut for typology.
BRIDGE_WITHIN_SUBSET_QUANTILE = 0.50


def _percentile(sorted_values, q):
    if not sorted_values:
        return 0.0
    idx = int(q * (len(sorted_values) - 1))
    return sorted_values[idx]


def main() -> int:
    cn = json.loads((DATA_PROC / "coauthor_network.json").read_text())
    tt = json.loads((DATA_PROC / "trajectory_taxonomy.json").read_text())
    cs = json.loads((DATA_PROC / "collab_style.json").read_text())

    # id -> trajectory class
    id_to_traj: dict[int, str] = {a["id"]: a["class"] for a in tt["authors"]}
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

    # Cutoffs
    homo_values = sorted(a["homophily"] for a in cs["authors"])
    homo_median = _percentile(homo_values, 0.5)

    eligible_bcw: list[float] = []
    eligible_team: list[float] = []
    for traj in tt["authors"]:
        node = id_to_node.get(traj["id"])
        if not node or traj["id"] not in id_to_homo:
            continue
        team = key_to_mean_team.get(node.get("key"))
        if team is None:
            continue
        eligible_bcw.append(node.get("bcw") or 0.0)
        eligible_team.append(team)
    eligible_bcw.sort()
    eligible_team.sort()
    bridge_cut = _percentile(eligible_bcw, BRIDGE_WITHIN_SUBSET_QUANTILE)
    team_median = _percentile(eligible_team, 0.5)

    rows: list[dict] = []
    cell_counts: Counter[str] = Counter()

    for traj in tt["authors"]:
        a_id = traj["id"]
        node = id_to_node.get(a_id)
        if not node:
            continue
        homo = id_to_homo.get(a_id)
        if not homo:
            continue
        team = key_to_mean_team.get(node.get("key"))
        if team is None:
            continue

        mover_label = "Stayer" if traj["class"] == "stayer" else "Mover"
        topical_label = "Homophilous" if homo["homophily"] >= homo_median else "Heterophilous"
        bridge_label = "Broker" if (node.get("bcw") or 0.0) >= bridge_cut else "Local"
        solo_label = "Team" if team >= team_median else "Solo"

        type_label = f"{mover_label}-{topical_label}-{bridge_label}-{solo_label}"
        cell_counts[type_label] += 1

        rows.append({
            "id": a_id,
            "key": node.get("key"),
            "name": node.get("label"),
            "type": type_label,
            "mover": mover_label,
            "topical": topical_label,
            "bridge": bridge_label,
            "solo": solo_label,
            "trajectory_class": traj["class"],
            "homophily": homo["homophily"],
            "homophily_lift": homo.get("lift"),
            "bcw": round(node.get("bcw") or 0.0, 4),
            "mean_team_size": round(team, 2),
            "n_papers": node.get("papers", 0),
            "citations": node.get("c", 0),
            "n_coauthors": homo.get("n_coauthors"),
        })

    rows.sort(key=lambda r: (-r["citations"], r["name"] or ""))

    summary = {
        "n_authors_classified": len(rows),
        "axes": {
            "mover": {"description": "Stayer vs Mover (taxonomy of Sec. 9)",
                      "split": "stayer vs {drifter, returner, switcher}"},
            "topical": {"description": "Coauthor topical homophily",
                        "split": f"median split at {round(homo_median, 4)}"},
            "bridge": {"description": "Network brokerage (weighted betweenness)",
                       "split": f"within-subset median split at {round(bridge_cut, 4)}"},
            "solo": {"description": "Mean team size on author's papers",
                     "split": f"median split at {round(team_median, 2)} authors/paper"},
        },
        "cell_counts": dict(cell_counts.most_common()),
    }

    OUT.write_text(json.dumps({"summary": summary, "authors": rows}, allow_nan=False))

    print("=" * 64)
    print("Four-axis author-personality classification")
    print("=" * 64)
    print(f"  Authors classified: {len(rows)}")
    print()
    print(f"  Cutoffs:")
    print(f"    Topical (homophily) median: {homo_median:.4f}")
    print(f"    Bridge   (bcw)  top-10%:    {bridge_cut:.4f}")
    print(f"    Solo (mean team) median:    {team_median:.2f}")
    print()
    print(f"  Cell counts (16 types, sorted by frequency):")
    for label, count in cell_counts.most_common():
        print(f"    {label:55s} {count:>5}")
    print()
    print(f"  output: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
