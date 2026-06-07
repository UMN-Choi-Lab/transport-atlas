"""OpenAlex ingest — primary metadata source.

Each venue → `data/raw/openalex/<slug>.jsonl` + `<slug>_meta.json`.
Resume-safe: skips venues that already have a `_meta.json` marking completion.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

from tqdm import tqdm

from ..utils import config
from ..utils.logger import get_logger
from ._http import RateLimiter, get_json, make_session

log = get_logger("openalex")


def _resolve_source_id(session, base_url: str, issns: list[str], mailto: str, lim: RateLimiter) -> str | None:
    """Try each ISSN; return first OpenAlex source ID matching a journal or conference."""
    for issn in issns:
        lim.wait()
        url = f"{base_url}/sources"
        params = {"filter": f"issn:{issn}", "mailto": mailto}
        try:
            data = get_json(session, url, params=params)
        except Exception as e:
            log.warning(f"source lookup failed for ISSN {issn}: {e}")
            continue
        results = data.get("results", [])
        if results:
            sid = results[0]["id"].rsplit("/", 1)[-1]  # e.g. https://openalex.org/S123 -> S123
            log.info(f"resolved ISSN {issn} -> {sid} ({results[0].get('display_name')})")
            return sid
    return None


def _iter_works_for_source(
    session, base_url: str, source_id: str, mailto: str, lim: RateLimiter,
    per_page: int = 200, extra_filter: str | None = None,
):
    """Cursor pagination over works filtered by source id.

    `extra_filter` is appended to the OpenAlex filter clause (e.g.
    "from_publication_date:2026-03-01") for incremental top-ups.
    """
    flt = f"primary_location.source.id:{source_id}"
    if extra_filter:
        flt = f"{flt},{extra_filter}"
    cursor = "*"
    while cursor:
        lim.wait()
        params = {
            "filter": flt,
            "per-page": per_page,
            "cursor": cursor,
            "mailto": mailto,
            "select": (
                "id,doi,title,display_name,publication_year,publication_date,"
                "authorships,primary_location,cited_by_count,concepts,abstract_inverted_index,type"
            ),
        }
        data = get_json(session, f"{base_url}/works", params=params)
        for w in data.get("results", []):
            yield w
        cursor = data.get("meta", {}).get("next_cursor")


def _abstract_from_inverted(inv: dict | None) -> str | None:
    if not inv:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _compact_work(w: dict, venue_slug: str) -> dict:
    authorships = w.get("authorships") or []
    authors = []
    for a in authorships:
        au = a.get("author") or {}
        authors.append({
            "id": au.get("id", "").rsplit("/", 1)[-1] if au.get("id") else None,
            "name": au.get("display_name"),
            "orcid": au.get("orcid"),
            "position": a.get("author_position"),
            "institutions": [i.get("display_name") for i in (a.get("institutions") or []) if i.get("display_name")],
        })
    concepts = [{"name": c.get("display_name"), "level": c.get("level"), "score": c.get("score")}
                for c in (w.get("concepts") or [])[:8]]
    return {
        "openalex_id": w.get("id", "").rsplit("/", 1)[-1] if w.get("id") else None,
        "doi": (w.get("doi") or "").lower().replace("https://doi.org/", "") or None,
        "title": w.get("title") or w.get("display_name"),
        "year": w.get("publication_year"),
        "date": w.get("publication_date"),
        "venue_slug": venue_slug,
        "type": w.get("type"),
        "cited_by_count": w.get("cited_by_count", 0),
        "abstract": _abstract_from_inverted(w.get("abstract_inverted_index")),
        "authors": authors,
        "concepts": concepts,
    }


def ingest(venues: list[dict] | None = None, *, force: bool = False) -> dict[str, int]:
    cfg = config.load_pipeline()["openalex"]
    mailto = config.crossref_email() or "chois@umn.edu"
    out_dir = config.data_dir("raw/openalex")
    session = make_session(f"transport-atlas/0.1 (mailto:{mailto})")
    lim = RateLimiter(cfg["rate_limit_per_sec"])
    venues = venues or config.load_venues()

    resolution = {}
    counts: dict[str, int] = {}
    for v in venues:
        slug = v["slug"]
        out_jsonl = out_dir / f"{slug}.jsonl"
        meta = out_dir / f"{slug}_meta.json"
        if meta.exists() and not force:
            log.info(f"[{slug}] already ingested (meta present) — skipping")
            meta_data = json.loads(meta.read_text())
            counts[slug] = meta_data.get("count", 0)
            continue

        source_id = v.get("openalex_source_id") or _resolve_source_id(
            session, cfg["base_url"], v["issns"], mailto, lim
        )
        resolution[slug] = source_id
        if not source_id:
            log.warning(f"[{slug}] no OpenAlex source ID resolved; skipping")
            counts[slug] = 0
            continue

        count = 0
        t0 = time.time()
        with out_jsonl.open("w") as f:
            for w in tqdm(
                _iter_works_for_source(session, cfg["base_url"], source_id, mailto, lim, cfg["per_page"]),
                desc=f"[{slug}]",
                unit="work",
            ):
                cw = _compact_work(w, slug)
                f.write(json.dumps(cw) + "\n")
                count += 1
        meta.write_text(json.dumps({
            "slug": slug,
            "source_id": source_id,
            "count": count,
            "elapsed_sec": round(time.time() - t0, 1),
        }, indent=2))
        log.info(f"[{slug}] wrote {count} works in {time.time() - t0:.0f}s")
        counts[slug] = count

    (out_dir / "_resolution.json").write_text(json.dumps(resolution, indent=2))
    return counts


def ingest_recent(
    venues: list[dict] | None = None,
    *,
    lookback_days: int = 90,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Incremental top-up: append works *published* within the last
    ``lookback_days`` that aren't already on disk (dedup by openalex_id).

    Unlike :func:`ingest` (which skips any venue with a ``_meta.json``), this is
    meant to run on every scheduled update. The publication-date window is kept
    wide on purpose: OpenAlex frequently indexes a paper weeks after its nominal
    publication date, so a narrow window would miss late-indexed papers. The
    dedup-by-id makes re-scanning the overlap cheap and idempotent.

    Note: works that OpenAlex *backfills* with an old publication date (>lookback)
    are NOT caught here — run ``ingest(..., force=True)`` periodically (e.g.
    quarterly) to reconcile those.

    Returns ``{slug: {"added", "scanned", "total"}}``.
    """
    cfg = config.load_pipeline()["openalex"]
    mailto = config.crossref_email() or "chois@umn.edu"
    out_dir = config.data_dir("raw/openalex")
    session = make_session(f"transport-atlas/0.1 (mailto:{mailto})")
    lim = RateLimiter(cfg["rate_limit_per_sec"])
    venues = venues or config.load_venues()

    since = (date.today() - timedelta(days=lookback_days)).isoformat()
    extra = f"from_publication_date:{since}"
    log.info(f"incremental top-up: published since {since} "
             f"(lookback {lookback_days}d), dry_run={dry_run}")

    report: dict[str, dict] = {}
    for v in venues:
        slug = v["slug"]
        out_jsonl = out_dir / f"{slug}.jsonl"
        meta = out_dir / f"{slug}_meta.json"

        # Source id: reuse the one captured at first ingest; resolve only if absent.
        source_id = v.get("openalex_source_id")
        if not source_id and meta.exists():
            source_id = json.loads(meta.read_text()).get("source_id")
        if not source_id:
            source_id = _resolve_source_id(session, cfg["base_url"], v["issns"], mailto, lim)
        if not source_id:
            log.warning(f"[{slug}] no OpenAlex source id; skipping incremental")
            report[slug] = {"added": 0, "scanned": 0, "total": 0, "error": "no_source_id"}
            continue

        # Ids already on disk, to avoid re-appending the overlap window.
        seen: set[str] = set()
        if out_jsonl.exists():
            with out_jsonl.open() as f:
                for line in f:
                    try:
                        seen.add(json.loads(line).get("openalex_id"))
                    except json.JSONDecodeError:
                        continue

        scanned = added = 0
        new_lines: list[str] = []
        for w in _iter_works_for_source(
            session, cfg["base_url"], source_id, mailto, lim, cfg["per_page"],
            extra_filter=extra,
        ):
            scanned += 1
            cw = _compact_work(w, slug)
            if cw["openalex_id"] in seen:
                continue
            seen.add(cw["openalex_id"])
            new_lines.append(json.dumps(cw))
            added += 1

        total = len(seen)
        report[slug] = {"added": added, "scanned": scanned, "total": total}
        log.info(f"[{slug}] scanned {scanned} recent, +{added} new (total {total})")

        if dry_run or not new_lines:
            continue
        with out_jsonl.open("a") as f:
            for line in new_lines:
                f.write(line + "\n")
        meta_obj = json.loads(meta.read_text()) if meta.exists() else {"slug": slug}
        meta_obj.update({
            "slug": slug,
            "source_id": source_id,
            "count": total,
            "last_incremental": date.today().isoformat(),
            "last_incremental_added": added,
        })
        meta.write_text(json.dumps(meta_obj, indent=2))

    if not dry_run:
        (out_dir / "_incremental_report.json").write_text(json.dumps(report, indent=2))
    n_added = sum(r["added"] for r in report.values())
    log.info(f"incremental top-up complete: +{n_added} new works across {len(report)} venues")
    return report
