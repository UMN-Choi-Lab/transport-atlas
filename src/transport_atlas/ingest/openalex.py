"""OpenAlex ingest — primary metadata source.

Each venue → `data/raw/openalex/<slug>.jsonl` + `<slug>_meta.json`.
Resume-safe: skips venues that already have a `_meta.json` marking completion.
"""
from __future__ import annotations

import json
import time
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


def _iter_works_for_source(session, base_url: str, source_id: str, mailto: str, lim: RateLimiter, per_page: int = 200):
    """Cursor pagination over works filtered by source id."""
    cursor = "*"
    while cursor:
        lim.wait()
        params = {
            "filter": f"primary_location.source.id:{source_id}",
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
