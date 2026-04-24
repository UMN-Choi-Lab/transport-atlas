#!/usr/bin/env python
"""Build candidate ambiguous author pairs for LLM adjudication.

Finds pairs where:
  - Same surname + first-initial (potential name collision)
  - Distinct author keys not already in author_aliases_auto.json
  - Coauthor-overlap >= 1 (some reason to suspect they are the same person)
  - Both n_papers >= 2 so we have metadata to adjudicate on

For each pair, gathers name variants, top venues, top 3 paper titles, and
up to 5 shared coauthor surnames. Writes a compact JSON suitable for
batch adjudication by a small LM.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "interim" / "alias_candidates.json"
MAX_PAIRS = 200
MIN_COAUTHOR_OVERLAP = 2


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
    auto = json.loads((ROOT / "data" / "interim" / "author_aliases_auto.json").read_text())
    already = set(auto.keys()) | set(auto.values())

    # Per-paper canonical key list so we can compute coauthor overlap.
    sys_import = __import__("sys"); sys_import.path.insert(0, str(ROOT / "src"))
    from transport_atlas.process.authors import author_key

    pap_authors: dict[str, list[str]] = {}
    for _, r in papers.iterrows():
        keys = []
        auths = r.get("authors")
        if auths is None or len(auths) == 0:
            continue
        for a in auths:
            if isinstance(a, dict):
                k = author_key(a)
                if k and k not in keys:
                    keys.append(k)
        if keys:
            pap_authors[r["paper_id"]] = keys

    # author_key -> set(coauthor_keys)
    coauthors: dict[str, set[str]] = defaultdict(set)
    for keys in pap_authors.values():
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                if i != j:
                    coauthors[a].add(b)

    authors = authors[authors["n_papers"] >= 2].copy()
    authors["sfi"] = authors["canonical_name"].map(canonical_key)
    authors = authors[authors["sfi"].notna()]
    authors = authors[~authors["author_key"].isin(already)]

    pid_to_venue = dict(zip(papers["paper_id"], papers["venue_slug"]))
    pid_to_title = dict(zip(papers["paper_id"], papers["title"]))
    pid_to_year = dict(zip(papers["paper_id"], papers["year"]))

    key_to_name = dict(zip(authors["author_key"], authors["canonical_name"]))

    candidates: list[dict] = []
    groups = authors.groupby("sfi")
    for (last, fi), g in groups:
        if len(g) < 2:
            continue
        keys = list(g["author_key"])
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                # Check coauthor overlap
                overlap = coauthors[a] & coauthors[b]
                if len(overlap) < MIN_COAUTHOR_OVERLAP:
                    continue
                # Check ORCID: skip if both have ORCIDs that differ (strong signal they are different people)
                oa = g.loc[g["author_key"] == a, "orcid"].iloc[0]
                ob = g.loc[g["author_key"] == b, "orcid"].iloc[0]
                if isinstance(oa, str) and isinstance(ob, str) and oa != ob:
                    continue
                # Gather metadata for each side
                def side(k):
                    row = g.loc[g["author_key"] == k].iloc[0]
                    pids = list(row["paper_ids"])[:3]
                    titles = [pid_to_title.get(p, "") for p in pids]
                    titles = [t for t in titles if isinstance(t, str) and t][:3]
                    venues = list(row["venues"])[:5]
                    return {
                        "key": k,
                        "name": row["canonical_name"],
                        "n_papers": int(row["n_papers"]),
                        "last_year": int(row["last_year"]) if pd.notna(row["last_year"]) else None,
                        "orcid": row["orcid"] if isinstance(row["orcid"], str) else None,
                        "venues": venues,
                        "titles": titles,
                    }
                ca = coauthors[a] - {a, b}
                cb = coauthors[b] - {a, b}
                shared_ck = list(overlap)[:5]
                shared_names = [key_to_name.get(ck, ck) for ck in shared_ck]
                candidates.append({
                    "surname": last,
                    "first_initial": fi,
                    "overlap_count": len(overlap),
                    "shared_coauthors": shared_names,
                    "a": side(a),
                    "b": side(b),
                })

    # Sort by overlap count descending, take top MAX_PAIRS
    candidates.sort(key=lambda c: -c["overlap_count"])
    candidates = candidates[:MAX_PAIRS]
    OUT.write_text(json.dumps(candidates, indent=2, default=str))
    print(f"[cands] wrote {OUT} ({len(candidates)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
