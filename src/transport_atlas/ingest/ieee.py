"""IEEE Xplore ingest — abstract-level metadata for IEEE venues.

Docs: https://developer.ieee.org/docs/read/metadata_api_details
Rate limit varies by key tier; we stay conservative.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from tqdm import tqdm

from ..utils import config
from ..utils.logger import get_logger
from ._http import RateLimiter, get_json, make_session

log = get_logger("ieee")


def _search_venue(session, base_url: str, params: dict, lim: RateLimiter, max_records: int):
    start = 1
    while True:
        lim.wait()
        q = dict(params, start_record=start, max_records=min(params.get("max_records", 200), 200))
        data = get_json(session, base_url, params=q)
        arts = data.get("articles", []) or []
        if not arts:
            break
        for a in arts:
            yield a
        start += len(arts)
        if start > max_records:
            break
        total = data.get("total_records", 0)
        if start > total:
            break


def _compact(a: dict, venue_slug: str) -> dict:
    authors = []
    for au in (a.get("authors") or {}).get("authors", []) or []:
        authors.append({
            "id": au.get("id"),
            "name": au.get("full_name"),
            "orcid": au.get("orcid"),
            "position": None,
            "institutions": [au.get("affiliation")] if au.get("affiliation") else [],
        })
    return {
        "ieee_id": a.get("article_number"),
        "doi": (a.get("doi") or "").lower() or None,
        "title": a.get("title"),
        "year": int(a["publication_year"]) if a.get("publication_year") else None,
        "date": a.get("publication_date"),
        "venue_slug": venue_slug,
        "type": a.get("content_type"),
        "cited_by_count": a.get("citing_paper_count", 0),
        "abstract": a.get("abstract"),
        "authors": authors,
        "concepts": [{"name": k, "level": None, "score": None}
                     for k in (a.get("index_terms", {}).get("ieee_terms", {}).get("terms", []) or [])[:8]],
    }


def ingest(venues: list[dict] | None = None, *, force: bool = False) -> dict[str, int]:
    cfg = config.load_pipeline()["ieee"]
    api_key = config.ieee_key()
    if not api_key:
        log.error("IEEE_API_KEY not set; aborting IEEE ingest")
        return {}
    out_dir = config.data_dir("raw/ieee")
    session = make_session()
    lim = RateLimiter(cfg["rate_limit_per_sec"])
    venues = [v for v in (venues or config.load_venues()) if v["publisher"] == "ieee"]

    counts: dict[str, int] = {}
    for v in venues:
        slug = v["slug"]
        out_jsonl = out_dir / f"{slug}.jsonl"
        meta = out_dir / f"{slug}_meta.json"
        if meta.exists() and not force:
            log.info(f"[{slug}] meta present — skipping")
            counts[slug] = json.loads(meta.read_text()).get("count", 0)
            continue
        pub_num = v.get("ieee_publication_number")
        if not pub_num:
            log.warning(f"[{slug}] no ieee_publication_number; skipping")
            continue

        params = {
            "publication_number": pub_num,
            "apikey": api_key,
            "format": "json",
            "max_records": cfg["max_records_per_call"],
        }
        count = 0
        t0 = time.time()
        complete = False
        try:
            with out_jsonl.open("w") as f:
                for a in tqdm(
                    _search_venue(session, cfg["base_url"], params, lim, cfg["max_records_per_venue"]),
                    desc=f"[{slug}]",
                    unit="art",
                ):
                    f.write(json.dumps(_compact(a, slug)) + "\n")
                    count += 1
            complete = True
        except Exception as e:
            log.warning(f"[{slug}] IEEE query failed after {count} records: {e}")
        # Only persist meta on complete runs — partial writes should not lock out re-ingest.
        if complete:
            meta.write_text(json.dumps({"slug": slug, "publication_number": pub_num,
                                        "count": count, "elapsed_sec": round(time.time() - t0, 1),
                                        "complete": True}, indent=2))
            log.info(f"[{slug}] wrote {count} articles in {time.time() - t0:.0f}s")
        else:
            log.warning(f"[{slug}] incomplete — not writing _meta.json; rerun will resume from scratch")
        counts[slug] = count
    return counts
