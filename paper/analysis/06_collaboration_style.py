#!/usr/bin/env python
"""Per-author collaboration-style (homophily) score.

For every author with at least MIN_COAUTHORS coauthors, computes the
fraction of their coauthors whose semantic-Leiden community matches
their own. The result is a per-author "topical homophily" score in
[0, 1] where 1 = every coauthor sits in the same semantic community
and 0 = no coauthors do.

Inputs (data/processed/):
  - coauthor_network.json   : nodes with full coauthor adjacency (ac)
                              and integer ids
  - topic_coords.json       : per-author UMAP coords + semantic
                              community (sc field)

Output (data/processed/):
  - collab_style.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = DATA / "collab_style.json"

MIN_COAUTHORS = 3  # need at least this many neighbours for a stable rate


def main() -> int:
    cn = json.loads((DATA / "coauthor_network.json").read_text())
    tc = json.loads((DATA / "topic_coords.json").read_text())

    nodes = cn["nodes"]
    # id -> semantic community
    id_to_sem: dict[int, int | None] = {}
    for k, v in tc.items():
        try:
            id_to_sem[int(k)] = v.get("sc")
        except (ValueError, AttributeError):
            continue

    # Build per-id node record
    by_id = {n["id"]: n for n in nodes}

    # Global baseline: sample-derived rate at which two random author keys
    # share a semantic community. Computed from the semantic-community
    # size distribution as sum_c (n_c / N)^2 ≈ Simpson concentration.
    from collections import Counter
    sem_sizes = Counter(s for s in id_to_sem.values() if s is not None)
    N = sum(sem_sizes.values())
    baseline = sum((c / N) ** 2 for c in sem_sizes.values()) if N else 0.0

    out_authors: list[dict] = []
    skipped_no_sem = 0
    skipped_few_coauth = 0

    for n in nodes:
        a_id = n["id"]
        a_sem = id_to_sem.get(a_id)
        if a_sem is None:
            skipped_no_sem += 1
            continue
        coauths = n.get("ac") or []
        # ac is a list of [coauthor_id, edge_weight] pairs
        coauth_ids = [pair[0] for pair in coauths
                      if isinstance(pair, (list, tuple)) and len(pair) >= 1
                      and isinstance(pair[0], int)]
        if len(coauth_ids) < MIN_COAUTHORS:
            skipped_few_coauth += 1
            continue
        same = 0
        scored = 0
        for cid in coauth_ids:
            csem = id_to_sem.get(cid)
            if csem is None:
                continue
            scored += 1
            if csem == a_sem:
                same += 1
        if scored < MIN_COAUTHORS:
            skipped_few_coauth += 1
            continue
        h = same / scored
        out_authors.append({
            "id": a_id,
            "key": n.get("key"),
            "name": n.get("label"),
            "n_coauthors": scored,
            "n_same_community": same,
            "homophily": round(h, 4),
            "lift": round(h / baseline, 3) if baseline > 0 else None,
            "n_papers": n.get("papers", 0),
            "citations": n.get("c", 0),
            "semantic_community": a_sem,
            "coauthor_community": n.get("community"),
            "degree": n.get("d"),
        })

    # Sort by homophily desc for default display
    out_authors.sort(key=lambda r: -r["homophily"])

    hs = [a["homophily"] for a in out_authors]
    lifts = [a["lift"] for a in out_authors if a["lift"] is not None]

    summary = {
        "n_authors_scored": len(out_authors),
        "n_authors_skipped_no_sem": skipped_no_sem,
        "n_authors_skipped_few_coauthors": skipped_few_coauth,
        "min_coauthors": MIN_COAUTHORS,
        "global_baseline": round(baseline, 4),
        "median_homophily": round(statistics.median(hs), 4) if hs else None,
        "mean_homophily": round(statistics.mean(hs), 4) if hs else None,
        "median_lift": round(statistics.median(lifts), 3) if lifts else None,
        "fraction_above_baseline": round(
            sum(1 for h in hs if h > baseline) / len(hs), 4
        ) if hs else None,
    }

    payload = {"summary": summary, "authors": out_authors}
    OUT.write_text(json.dumps(payload, allow_nan=False))

    print("=" * 60)
    print("Collaboration-style (semantic-community homophily) summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:>40} : {v}")
    print(f"  output                                  : {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
