#!/usr/bin/env python
"""Count the full borderline-pair population for the LLM disambiguation audit.

Replicates the population definition of
``paper/analysis/build_alias_candidates.py`` but without the MAX_PAIRS=200
cap, and reports counts only (no metadata gathering, no candidate file).

Population:
  - Same surname + first-initial.
  - Both author keys have n_papers >= 2.
  - Pair not already in author_aliases_auto.json.
  - Coauthor overlap >= 2 in the deduplicated corpus.
  - ORCIDs do not contradict (skip if both have ORCIDs that differ).

Reports total count and the overlap-count distribution at several
thresholds, so we can size a Gemma full-population audit before launching it.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from transport_atlas.process.authors import author_key  # noqa: E402


def canonical_key(name: str) -> tuple[str, str] | None:
    if not isinstance(name, str):
        return None
    parts = name.split(",", 1)
    if len(parts) < 2:
        return None
    last = parts[0].strip().lower()
    first_tok = parts[1].strip().split()
    fi = first_tok[0][0].lower() if first_tok and first_tok[0] else ""
    if not last or not fi:
        return None
    return (last, fi)


def main() -> int:
    authors = pd.read_parquet(ROOT / "data" / "interim" / "authors.parquet")
    papers = pd.read_parquet(ROOT / "data" / "interim" / "papers.parquet")
    auto = json.loads(
        (ROOT / "data" / "interim" / "author_aliases_auto.json").read_text()
    )
    already = set(auto.keys()) | set(auto.values())

    # author_key -> set(coauthor_keys)
    coauthors: dict[str, set[str]] = defaultdict(set)
    for _, r in papers.iterrows():
        auths = r.get("authors")
        if auths is None or len(auths) == 0:
            continue
        keys: list[str] = []
        for a in auths:
            if isinstance(a, dict):
                k = author_key(a)
                if k and k not in keys:
                    keys.append(k)
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                if i != j:
                    coauthors[a].add(b)

    authors = authors[authors["n_papers"] >= 2].copy()
    authors["sfi"] = authors["canonical_name"].map(canonical_key)
    authors = authors[authors["sfi"].notna()]
    authors = authors[~authors["author_key"].isin(already)]

    # All same-surname + same-first-initial pairs are candidates before
    # filtering. Count along the way to also report the pre-filter pool.
    raw_pairs = 0
    overlap_counts: list[int] = []
    contradicting_orcid = 0

    for (_, _), g in authors.groupby("sfi"):
        if len(g) < 2:
            continue
        keys = list(g["author_key"])
        orcids = list(g["orcid"])
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                raw_pairs += 1
                a, b = keys[i], keys[j]
                oa, ob = orcids[i], orcids[j]
                if isinstance(oa, str) and isinstance(ob, str) and oa != ob:
                    contradicting_orcid += 1
                    continue
                ov = len(coauthors[a] & coauthors[b])
                if ov >= 2:
                    overlap_counts.append(ov)

    overlap_counts.sort(reverse=True)
    total = len(overlap_counts)

    def at_least(t: int) -> int:
        return sum(1 for v in overlap_counts if v >= t)

    print("=" * 60)
    print("LLM-audit borderline-pair population (full corpus)")
    print("=" * 60)
    print(f"All same-surname + first-initial pairs (pre-filter):  {raw_pairs:,}")
    print(f"  Skipped: contradicting non-null ORCIDs:             {contradicting_orcid:,}")
    print(f"  Skipped: in auto-merge map (excluded upstream)")
    print(f"Borderline pairs with shared-coauthor count >= 2:     {total:,}")
    print()
    print("Overlap-count distribution (cumulative, descending):")
    for t in (2, 3, 5, 10, 20, 50):
        print(f"  shared coauthors >= {t:>3}: {at_least(t):>6,}")
    if overlap_counts:
        p = overlap_counts
        print()
        print(f"Overlap quantiles: max={p[0]}, p50={p[len(p)//2]}, "
              f"p95={p[max(0, int(len(p)*0.05))]}, "
              f"p99={p[max(0, int(len(p)*0.01))]}, "
              f"min={p[-1]}")

    out = ROOT / "data" / "interim" / "alias_candidate_population.json"
    out.write_text(json.dumps({
        "raw_same_surname_pairs": raw_pairs,
        "skipped_contradicting_orcid": contradicting_orcid,
        "borderline_pairs_overlap_ge_2": total,
        "cumulative_at_threshold": {str(t): at_least(t) for t in (2, 3, 5, 10, 20, 50)},
    }, indent=2))
    print()
    print(f"Wrote summary to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
