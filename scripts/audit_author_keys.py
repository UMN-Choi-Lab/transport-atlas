#!/usr/bin/env python
"""Audit author_keys that are name-only (no OpenAlex ID, no ORCID).

Sun & Rahwan (2017) flagged common Asian surnames (Chen, Kim, Lee, Li, Liu,
Park, Wang, Yang, Zhang) as the worst cases for false merges — initials collapse
different people, full names split one person. Our pipeline dodges most of
this by keying on OpenAlex author IDs, but some records come in without an ID,
and those fall through to the normalized-name key.

This audit prints:
  1. A summary of name-only vs ID/ORCID keys (total + per-venue).
  2. Name-only keys with suspiciously high paper counts (likely false merges —
     multiple people squashed together under a short name).
  3. Authors with OpenAlex IDs on some papers and name-only keys on others
     (likely false splits — the same person appears under ≥2 keys).

Output goes to stdout; nothing is modified. To resolve a flagged case,
add an entry under `author_aliases:` in config/pipeline.yaml or fill in a
missing OpenAlex ID at the ingest layer.

Usage:
  python scripts/audit_author_keys.py
  python scripts/audit_author_keys.py --top 50
  python scripts/audit_author_keys.py --surname kim
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from transport_atlas.utils import config

ID_RE = re.compile(r"^a\d{8,}$")
ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dx]$")

# Known disambiguation-hard surnames (Sun & Rahwan 2017 §2.2).
COMMON_SURNAMES = ["chen", "kim", "lee", "li", "liu", "park", "wang", "yang", "zhang"]


def key_kind(k: str) -> str:
    if not isinstance(k, str) or not k:
        return "empty"
    if ID_RE.match(k):
        return "openalex_id"
    if ORCID_RE.match(k):
        return "orcid"
    return "name"


def _surname(canonical: str) -> str:
    if not isinstance(canonical, str) or not canonical:
        return ""
    return canonical.split(",", 1)[0].strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30,
                    help="show top-N name-only keys by paper count")
    ap.add_argument("--surname", default=None,
                    help="restrict false-split scan to this surname (e.g. 'kim')")
    args = ap.parse_args()

    interim = config.data_dir("interim")
    a = pd.read_parquet(interim / "authors.parquet")
    a["kind"] = a["author_key"].map(key_kind)

    # ---- Section 1: key-kind summary
    kind_summary = a.groupby("kind").agg(
        n_authors=("author_key", "nunique"),
        n_papers=("n_papers", "sum"),
        max_papers=("n_papers", "max"),
    )
    print("=" * 72)
    print("Section 1: author_key composition")
    print("=" * 72)
    print(kind_summary.to_string())
    print()

    total_authors = int(kind_summary["n_authors"].sum())
    total_papers = int(kind_summary["n_papers"].sum())
    name_only = kind_summary.loc["name"] if "name" in kind_summary.index else None
    if name_only is not None:
        pct_auth = 100 * name_only["n_authors"] / total_authors
        pct_pap = 100 * name_only["n_papers"] / total_papers
        print(f"Name-only tail: {name_only['n_authors']:,} authors ({pct_auth:.1f}%), "
              f"{name_only['n_papers']:,} papers ({pct_pap:.1f}%).")
    print()

    # ---- Section 2: name-only keys with high paper counts (false-merge suspects)
    print("=" * 72)
    print(f"Section 2: top {args.top} name-only keys by paper count (false-merge suspects)")
    print("=" * 72)
    name_keys = a[a["kind"] == "name"].copy()
    if args.surname:
        name_keys = name_keys[name_keys["canonical_name"].str.startswith(args.surname.lower() + ",")]
    name_keys = name_keys.sort_values("n_papers", ascending=False).head(args.top)
    for _, r in name_keys.iterrows():
        venues = list(r["venues"]) if r["venues"] is not None else []
        print(f"  {r['n_papers']:3d} papers · {r['canonical_name']:<40s} · venues={venues}")
    print()

    # ---- Section 3: name-only keys that shadow an OpenAlex-ID-keyed author
    # (same surname + compatible first name, both present → likely false split).
    print("=" * 72)
    print("Section 3: false-split candidates (name-only author overlaps an ID-keyed author)")
    print("=" * 72)
    # Group authors by surname.
    a["surname"] = a["canonical_name"].map(_surname)
    by_surname: dict[str, pd.DataFrame] = {}
    for sn, grp in a.groupby("surname"):
        if not sn:
            continue
        if args.surname and sn != args.surname.lower():
            continue
        kinds = set(grp["kind"].unique())
        if "name" in kinds and ("openalex_id" in kinds or "orcid" in kinds):
            by_surname[sn] = grp

    def fn_tokens(canonical: str) -> list[str]:
        if "," not in canonical:
            return []
        fn = canonical.split(",", 1)[1].strip()
        return [t for t in re.split(r"[\s.]+", fn) if t]

    MIN_PAPERS_NAME = 3      # name-only side must carry enough weight to be worth fixing
    MIN_PAPERS_ID = 3        # ID side too — single-paper IDs are noise
    n_flagged = 0
    n_common_flagged = 0
    printed_any = False
    for sn, grp in sorted(by_surname.items(), key=lambda kv: -len(kv[1])):
        name_rows = grp[(grp["kind"] == "name") & (grp["n_papers"] >= MIN_PAPERS_NAME)]
        id_rows = grp[(grp["kind"] != "name") & (grp["n_papers"] >= MIN_PAPERS_ID)]
        if name_rows.empty or id_rows.empty:
            continue
        for _, nr in name_rows.iterrows():
            nt = [t for t in fn_tokens(nr["canonical_name"]) if len(t) >= 2]
            if not nt:
                # Pure-initial name-only keys merge many different people — report but
                # don't try to match against specific ID records (would flood output).
                continue
            n_token_set = set(nt)
            for _, ir in id_rows.iterrows():
                it = [t for t in fn_tokens(ir["canonical_name"]) if len(t) >= 2]
                if not it:
                    continue
                # At least one full-length firstname token must appear on both sides.
                if not (n_token_set & set(it)):
                    continue
                n_flagged += 1
                if sn in COMMON_SURNAMES:
                    n_common_flagged += 1
                printed_any = True
                n_venues = list(nr["venues"]) if nr["venues"] is not None else []
                print(f"  [{sn}] name-only '{nr['canonical_name']}' (n={nr['n_papers']}, "
                      f"venues={n_venues}) "
                      f"~ id '{ir['canonical_name']}' (n={ir['n_papers']}, "
                      f"id={ir['author_key']})")
    if not printed_any:
        print("  (no high-confidence matches — raise MIN_PAPERS_* or inspect Section 2 manually)")
    print()
    print(f"flagged {n_flagged} candidates ({n_common_flagged} on common surnames)")
    print()
    print("To resolve a false split: add both openalex IDs under the same")
    print("`author_aliases` entry in config/pipeline.yaml, e.g.")
    print("  - canonical: \"Firstname Lastname\"")
    print("    openalex_ids: [A1234567890, A9876543210]")
    print("To resolve a false merge (name-only key covers multiple people): the")
    print("best fix is at ingest — upstream OpenAlex records missing authorship IDs")
    print("need to be re-fetched with `scripts/01_ingest.py --force` once OpenAlex")
    print("has disambiguated them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
