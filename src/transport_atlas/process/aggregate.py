"""Per-venue / per-year aggregates + compact papers JSON for the explorer."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from ..utils import config
from ..utils.logger import get_logger
from .authors import author_key

log = get_logger("aggregate")


def _authors_short(author_list) -> str:
    """Compact author display for the explorer table.

    - 4 or fewer authors: all shown
    - 5+ authors: first 3 + '…' + last author (usually the corresponding/senior
      author in Korean/Asian convention) + '(+N)' hidden-count badge.

    Preserving the last author matters for author-search discoverability — it's
    the person Korean readers are most likely to look for.
    """
    try:
        if author_list is None or len(author_list) == 0:
            return ""
    except TypeError:
        return ""
    names = [a.get("name") for a in author_list if isinstance(a, dict) and a.get("name")]
    if len(names) <= 4:
        return "; ".join(names)
    return "; ".join(names[:3]) + f"; … ; {names[-1]} (+{len(names) - 4})"


def _authors_full_lc(author_list) -> str:
    """Lowercase '; '-joined full author name list. Feeds the explorer author
    filter so users can find papers where the author is hidden behind the
    ``+N`` truncation in ``authors_short`` (37% of papers — up to 97 hidden).
    """
    try:
        if author_list is None or len(author_list) == 0:
            return ""
    except TypeError:
        return ""
    names = [a.get("name") for a in author_list if isinstance(a, dict) and a.get("name")]
    return "; ".join(n.lower() for n in names)


def _venue_stats(papers: pd.DataFrame, venues: list[dict]) -> list[dict]:
    """Per-venue author/collaboration stats (Sun & Rahwan 2017 Table 2) + top
    contributors / top papers per venue.

    Columns: papers, single_authored, authors, avg_authors, max_authors,
    collaborations, papers_per_author, avg_citations, plus
    top_authors_by_papers, top_authors_by_citations, top_papers.
    """
    from collections import Counter, defaultdict
    venue_short = {v["slug"]: v["short"] for v in venues}
    venue_name = {v["slug"]: v["name"] for v in venues}
    venue_pub = {v["slug"]: v.get("publisher", "") for v in venues}

    # Canonical display name for an author_key — take the longest name seen for
    # that key across all papers. Longest usually wins over initials ("J Kim" < "Jiwon Kim").
    key_to_name: dict[str, str] = {}

    # Per-venue aggregations
    per_venue_author_papers: dict[str, Counter] = defaultdict(Counter)
    per_venue_author_cites: dict[str, Counter] = defaultdict(Counter)
    per_venue_top_papers: dict[str, list] = defaultdict(list)

    # Pre-extract per-paper: venue, n_authors, author_keys, cites — single pass
    rows = []
    for _, r in papers.iterrows():
        al = r.get("authors")
        try:
            if al is None or len(al) == 0:
                al = []
        except TypeError:
            al = []
        keys = []
        for a in al:
            if isinstance(a, dict):
                k = author_key(a)
                if k:
                    keys.append(k)
                    nm = a.get("name") or ""
                    if len(nm) > len(key_to_name.get(k, "")):
                        key_to_name[k] = nm
        uniq_keys = set(keys)
        cites = (0 if (_c := r.get("cited_by_count")) is None
                 or (isinstance(_c, float) and _c != _c) else int(_c or 0))
        slug = r.get("venue_slug")
        rows.append({"venue_slug": slug, "n_authors": len(uniq_keys),
                     "author_keys": uniq_keys, "cites": cites})
        # Accumulate per-venue contributor counts
        for k in uniq_keys:
            per_venue_author_papers[slug][k] += 1
            per_venue_author_cites[slug][k] += cites
        # Keep top-cited papers per venue (sort later, cheaper than a heap here)
        per_venue_top_papers[slug].append({
            "title": (r.get("title") or "")[:160],
            "year": int(r["year"]) if pd.notna(r.get("year")) else None,
            "cites": cites,
            "doi": r.get("doi") or "",
            "authors_short": _authors_short(al),
        })
    df = pd.DataFrame(rows)

    out = []
    for slug in venue_short:
        sub = df[df["venue_slug"] == slug]
        if sub.empty:
            continue
        n_papers = len(sub)
        single = int((sub["n_authors"] == 1).sum())
        n_authors_arr = sub["n_authors"].to_numpy()
        collabs = int(((n_authors_arr * (n_authors_arr - 1)) // 2).sum())
        avg_authors = float(sub["n_authors"].mean())
        max_authors = int(sub["n_authors"].max())
        avg_cites = float(sub["cites"].mean())
        unique_authors = set().union(*sub["author_keys"].tolist()) if n_papers else set()
        n_unique = len(unique_authors)
        papers_per_author = (n_papers / n_unique) if n_unique else 0.0

        # Top contributors for this venue. Sort by (-count, name, key) so that
        # the published top-K is deterministic when multiple authors are tied
        # on the primary metric — without the secondary keys, dict-iteration
        # order at ties produced spurious table churn between regens.
        papers_counter = per_venue_author_papers.get(slug, Counter())
        cites_counter = per_venue_author_cites.get(slug, Counter())
        top_by_papers = [
            {"name": key_to_name.get(k, k), "key": k, "n_papers": c}
            for k, c in sorted(
                papers_counter.items(),
                key=lambda kv: (-kv[1], key_to_name.get(kv[0], kv[0]), kv[0]),
            )[:15]
        ]
        top_by_cites = [
            {"name": key_to_name.get(k, k), "key": k, "cites": c,
             "n_papers": papers_counter.get(k, 0)}
            for k, c in sorted(
                cites_counter.items(),
                key=lambda kv: (-kv[1], key_to_name.get(kv[0], kv[0]), kv[0]),
            )[:15]
        ]
        # Top 10 most-cited papers in this venue. DOI then title as
        # tie-breakers for determinism.
        top_papers = sorted(
            per_venue_top_papers.get(slug, []),
            key=lambda p: (-p["cites"], p.get("doi") or "", p.get("title") or ""),
        )[:10]

        out.append({
            "slug": slug,
            "short": venue_short.get(slug, slug),
            "name": venue_name.get(slug, slug),
            "publisher": venue_pub.get(slug, ""),
            "papers": n_papers,
            "single_authored": single,
            "single_pct": round(100 * single / n_papers, 1) if n_papers else 0.0,
            "authors": n_unique,
            "avg_authors": round(avg_authors, 2),
            "max_authors": max_authors,
            "collaborations": collabs,
            "papers_per_author": round(papers_per_author, 2),
            "avg_citations": round(avg_cites, 2),
            "top_authors_by_papers": top_by_papers,
            "top_authors_by_citations": top_by_cites,
            "top_papers": top_papers,
        })
    out.sort(key=lambda r: -r["papers"])
    return out


def run(*, write: bool = True) -> dict:
    interim = config.data_dir("interim")
    out = config.data_dir("processed")

    papers = pd.read_parquet(interim / "papers.parquet")
    venues = config.load_venues()
    venue_short = {v["slug"]: v["short"] for v in venues}
    venue_name = {v["slug"]: v["name"] for v in venues}

    # Explorer table — trimmed columns
    explorer = pd.DataFrame({
        "title": papers["title"].fillna(""),
        "authors_short": papers["authors"].map(_authors_short),
        "authors_full": papers["authors"].map(_authors_full_lc),
        "venue_short": papers["venue_slug"].map(venue_short).fillna(papers["venue_slug"]),
        "year": papers["year"].astype("Int64"),
        "cited_by_count": papers["cited_by_count"].fillna(0).astype(int),
        "doi": papers["doi"].fillna(""),
    })
    explorer = explorer.sort_values(["year", "cited_by_count"], ascending=[False, False])

    # By-year per venue
    by_year = duckdb.query("""
        SELECT year, venue_slug, COUNT(*) AS n
        FROM papers
        WHERE year IS NOT NULL AND year >= 1990 AND year <= 2026
        GROUP BY year, venue_slug
        ORDER BY year
    """).df()
    years = sorted(by_year["year"].unique().tolist())
    series = {}
    for slug in venue_short:
        row = by_year[by_year["venue_slug"] == slug].set_index("year")["n"]
        series[slug] = [int(row.get(y, 0)) for y in years]
    by_year_payload = {
        "years": years,
        "venues": [{"slug": s, "short": venue_short[s], "name": venue_name[s], "counts": series[s]}
                   for s in venue_short],
    }

    venue_stats = _venue_stats(papers, venues)

    summary = {
        "n_papers": int(len(papers)),
        "n_venues": int(papers["venue_slug"].nunique()),
        "year_min": int(papers["year"].min()) if papers["year"].notna().any() else None,
        "year_max": int(papers["year"].max()) if papers["year"].notna().any() else None,
    }
    log.info(f"aggregate summary: {summary}")

    if write:
        explorer.to_json(out / "papers.json", orient="records")
        (out / "by_year.json").write_text(json.dumps(by_year_payload))
        (out / "venue_stats.json").write_text(json.dumps(venue_stats))
        (out / "_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
