"""Dedup raw ingest → interim parquets.

Strategy:
  1. DOI collapse (primary). Lowercase, strip prefix.
  2. For papers without DOI, fuzzy-match on (normalized title + year + author surname overlap).
     title ratio > threshold AND same year AND ≥1 shared surname.

Output:
  data/interim/papers.parquet    — one row per unique paper (columns: paper_id, doi, title, year, venue_slug,
                                   cited_by_count, abstract, concepts, authors [list of dicts], type)
  data/interim/authors.parquet   — one row per canonical author (author_id, canonical_name, orcid, paper_ids,
                                   n_papers, last_year, venues)
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio

from ..utils import config
from ..utils.logger import get_logger
from .authors import (
    author_key,
    auto_alias_map_from_papers,
    canonical_last_first,
    coauthor_alias_map_from_papers,
    normalize_name,
    normalize_orcid,
    surname,
)
from .frontmatter import is_front_matter


def _build_alias_map() -> dict[str, str]:
    """Return {source_openalex_key: canonical_openalex_key} from config.

    Any author_key in the map is rewritten to its canonical target so downstream
    grouping treats the IDs as the same person.
    """
    pipe = config.load_pipeline()
    aliases = pipe.get("author_aliases") or []
    mp = {}
    for a in aliases:
        ids = a.get("openalex_ids") or []
        if len(ids) < 2:
            continue
        target = ids[0].lower()
        for other in ids[1:]:
            mp[other.lower()] = target
    return mp

log = get_logger("dedupe")


def _norm_title(t: str | None) -> str:
    if not t:
        return ""
    s = t.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _paper_id(doi: str | None, title: str | None, year: int | None) -> str:
    if doi:
        return f"doi:{doi}"
    key = f"{_norm_title(title)}|{year}"
    return "h:" + hashlib.sha1(key.encode()).hexdigest()[:16]


def _load_jsonl(path: Path, source: str) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            try:
                w = json.loads(line)
            except json.JSONDecodeError:
                continue
            w["_source"] = source
            out.append(w)
    return out


def _load_all(venues: list[dict]) -> pd.DataFrame:
    rows = []
    oa_dir = config.data_dir("raw/openalex")
    ieee_dir = config.data_dir("raw/ieee")
    for v in venues:
        slug = v["slug"]
        rows.extend(_load_jsonl(oa_dir / f"{slug}.jsonl", "openalex"))
        rows.extend(_load_jsonl(ieee_dir / f"{slug}.jsonl", "ieee"))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["doi"] = df["doi"].astype("object").where(df["doi"].notna(), None)
    df["title_norm"] = df["title"].map(_norm_title)
    df["year"] = df["year"].astype("Int64")
    # Filter front-matter (Editorial, TOC, etc.) — they inflate hubs + pollute topics
    n_before = len(df)
    fm_mask = df["title"].map(is_front_matter)
    df = df.loc[~fm_mask].reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped:
        log.info(f"filtered {dropped} front-matter entries ({dropped/n_before:.1%})")
    return df


def _fuzzy_groups(no_doi: pd.DataFrame, threshold: int) -> dict[int, int]:
    """Return index → cluster_id. O(n^2) per (year, first-surname) bucket."""
    cluster = {}
    next_id = 0
    no_doi = no_doi.copy()
    no_doi["first_surname"] = no_doi["authors"].map(
        lambda al: surname((al or [{}])[0].get("name", "")) if al else ""
    )
    for (year, fsn), grp in no_doi.groupby(["year", "first_surname"]):
        idxs = grp.index.tolist()
        local: dict[int, int] = {}
        for i_pos, i in enumerate(idxs):
            if i in local:
                continue
            local[i] = next_id
            ti = grp.loc[i, "title_norm"]
            surnames_i = {surname(a.get("name", "")) for a in (grp.loc[i, "authors"] or [])}
            for j in idxs[i_pos + 1:]:
                if j in local:
                    continue
                tj = grp.loc[j, "title_norm"]
                if not ti or not tj:
                    continue
                if ratio(ti, tj) < threshold:
                    continue
                surnames_j = {surname(a.get("name", "")) for a in (grp.loc[j, "authors"] or [])}
                if surnames_i & surnames_j:
                    local[j] = next_id
            next_id += 1
        cluster.update(local)
    return cluster


def _is_missing(v) -> bool:
    """True if v should be treated as missing (None / NaN / empty string / empty list)."""
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(v, (str, list, tuple, dict, set)) and len(v) == 0:
        return True
    return False


def _merge_records(group: pd.DataFrame) -> dict:
    """Merge rows referring to the same paper. Prefer OpenAlex metadata; fill missing from IEEE."""
    rows = group.sort_values("_source_priority").to_dict("records")
    base = rows[0]
    for r in rows[1:]:
        for k in ("abstract", "cited_by_count", "concepts"):
            if _is_missing(base.get(k)) and not _is_missing(r.get(k)):
                base[k] = r[k]
        base_authors = base.get("authors")
        base_authors = [] if _is_missing(base_authors) else list(base_authors)
        seen = {author_key(a) for a in base_authors if isinstance(a, dict)}
        new_authors = r.get("authors")
        if not _is_missing(new_authors):
            for a in new_authors:
                if isinstance(a, dict) and author_key(a) not in seen:
                    base_authors.append(a)
                    seen.add(author_key(a))
        base["authors"] = base_authors
    return base


def run(*, write: bool = True) -> dict:
    cfg = config.load_pipeline()["dedupe"]
    venues = config.load_venues()
    df = _load_all(venues)
    if df.empty:
        log.error("no ingest data found — run scripts/01_ingest.py first")
        return {"papers": 0, "authors": 0}
    n_raw = len(df)
    df["_source_priority"] = df["_source"].map({"openalex": 0, "ieee": 1}).fillna(9).astype(int)

    # Primary: DOI grouping
    df["cluster_id"] = None
    with_doi = df[df["doi"].notna()].copy()
    with_doi["cluster_id"] = "doi:" + with_doi["doi"].str.lower()

    without_doi = df[df["doi"].isna()].copy()
    fuzz_map = _fuzzy_groups(without_doi, cfg["fuzzy_title_threshold"])
    without_doi["cluster_id"] = without_doi.index.map(lambda i: f"fz:{fuzz_map.get(i, i)}")

    merged_rows = []
    for cid, grp in pd.concat([with_doi, without_doi]).groupby("cluster_id"):
        merged_rows.append(_merge_records(grp))
    papers = pd.DataFrame(merged_rows)
    papers["paper_id"] = papers.apply(
        lambda r: _paper_id(r.get("doi"), r.get("title"), r.get("year")), axis=1
    )
    papers = papers.drop_duplicates("paper_id")

    # Build authors table (applying manual aliases from config + auto-detected ORCID splits)
    alias_map = _build_alias_map()
    if alias_map:
        log.info(f"author aliases (manual): {len(alias_map)} mergers configured")
    auto_map = auto_alias_map_from_papers(papers)
    if auto_map:
        log.info(f"author aliases (auto-ORCID): {len(auto_map)} split-author mergers")
    # Coauthor-overlap pass — catches splits that ORCID logic misses (different/no ORCID
    # but clearly the same person based on collaborator fingerprint). Composed on top
    # of the ORCID merges so coauthor sets are already consolidated.
    coauthor_map = coauthor_alias_map_from_papers(papers, existing_aliases=auto_map)
    if coauthor_map:
        log.info(f"author aliases (auto-coauthor): {len(coauthor_map)} split-author mergers")
    # Manual entries take precedence over auto; ORCID over coauthor when both present
    merged_auto = {**coauthor_map, **auto_map}
    # Chain-resolve: if X→Y (ORCID) and Y→Z (coauthor), rewrite X→Z so the downstream
    # single-hop lookup in the records loop lands on the terminal canonical.
    def _resolve_chain(k: str, mp: dict[str, str], seen: set[str] | None = None) -> str:
        seen = seen or set()
        while k in mp and k not in seen:
            seen.add(k)
            k = mp[k]
        return k
    merged_auto = {k: _resolve_chain(v, merged_auto) for k, v in merged_auto.items()}
    for k, v in merged_auto.items():
        alias_map.setdefault(k, v)
    # Persist combined auto map for downstream phases (graph, similarity)
    (config.data_dir("interim") / "author_aliases_auto.json").write_text(
        json.dumps(merged_auto, indent=2)
    )
    records = []
    for _, r in papers.iterrows():
        authors_list = r.get("authors") or []
        for a in authors_list:
            if not isinstance(a, dict):
                continue
            k = author_key(a)
            if not k:
                continue  # drop unidentifiable authors (no id, no orcid, too-short name)
            k = alias_map.get(k, k)
            records.append({
                "author_key": k,
                "openalex_author_id": a.get("id") if a.get("id") and a.get("id").startswith("A") else None,
                "orcid": normalize_orcid(a.get("orcid")),
                "name": a.get("name"),
                "canonical": canonical_last_first(a.get("name")),
                "paper_id": r["paper_id"],
                "venue_slug": r["venue_slug"],
                "year": int(r["year"]) if pd.notna(r.get("year")) else None,
            })
    authors_long = pd.DataFrame(records)
    if authors_long.empty:
        log.warning("no authors extracted")
        authors = pd.DataFrame(columns=["author_key", "canonical_name", "n_papers", "last_year", "orcid", "paper_ids", "venues"])
    else:
        grp = authors_long.groupby("author_key")
        authors = grp.agg(
            canonical_name=("canonical", lambda s: s.dropna().mode().iloc[0] if s.dropna().size else s.iloc[0]),
            n_papers=("paper_id", "nunique"),
            last_year=("year", "max"),
            orcid=("orcid", lambda s: next((x for x in s if x), None)),
            paper_ids=("paper_id", lambda s: sorted(set(s))),
            venues=("venue_slug", lambda s: sorted(set(s))),
        ).reset_index()

    report = {
        "raw_rows": int(n_raw),
        "unique_papers": int(len(papers)),
        "dup_rate_pct": round(100 * (1 - len(papers) / max(n_raw, 1)), 2),
        "authors": int(len(authors)),
        "by_venue": papers.groupby("venue_slug").size().to_dict(),
    }
    log.info(f"dedupe: {report}")

    if write:
        out_dir = config.data_dir("interim")
        # Drop internal cols
        keep = ["paper_id", "doi", "title", "year", "date", "venue_slug", "type",
                "cited_by_count", "abstract", "concepts", "authors"]
        papers_out = papers[[c for c in keep if c in papers.columns]].reset_index(drop=True)
        papers_out.to_parquet(out_dir / "papers.parquet", index=False)
        authors.to_parquet(out_dir / "authors.parquet", index=False)
        (out_dir / "_dedupe_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
