"""Elsevier ScienceDirect full-text ingest for TR A-F.

Policy:
  - Store plaintext + section headings only. NEVER the raw XML.
  - Per-DOI file in data/raw/elsevier/<slug>/<doi_hash>.json; existence = done.
  - On 401/403: log and skip the DOI.
  - Rate limit: 2 req/s (institutional token default).
  - Requires ELSEVIER_KEY + ELSEVIER_INSTTOKEN.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from lxml import etree
from tqdm import tqdm

from ..utils import config
from ..utils.logger import get_logger
from ._http import RateLimiter, get_text, make_session

log = get_logger("elsevier")

NS = {
    "sd": "http://www.elsevier.com/xml/svapi/article/dtd",
    "ce": "http://www.elsevier.com/xml/common/dtd",
    "xocs": "http://www.elsevier.com/xml/xocs/dtd",
    "ja": "http://www.elsevier.com/xml/ja/dtd",
}


def _doi_hash(doi: str) -> str:
    return hashlib.sha1(doi.lower().encode()).hexdigest()[:16]


def _parse_full_text(xml: str) -> dict:
    """Extract plaintext + section headings from ScienceDirect XML article response."""
    root = etree.fromstring(xml.encode("utf-8"))
    sections: list[dict] = []
    for sec in root.iter("{http://www.elsevier.com/xml/common/dtd}section"):
        title_el = sec.find("ce:section-title", NS)
        title = "".join(title_el.itertext()).strip() if title_el is not None else None
        body_text = " ".join(
            " ".join(p.itertext())
            for p in sec.iter("{http://www.elsevier.com/xml/common/dtd}para")
        )
        if body_text:
            sections.append({"heading": title, "text": re.sub(r"\s+", " ", body_text).strip()})
    # Also collect body text if sections didn't match
    plaintext = " ".join(s["text"] for s in sections)
    if not plaintext:
        paras = [" ".join(p.itertext())
                 for p in root.iter("{http://www.elsevier.com/xml/common/dtd}para")]
        plaintext = re.sub(r"\s+", " ", " ".join(paras)).strip()
    # Keywords
    keywords = [
        (k.text or "").strip()
        for k in root.iter("{http://www.elsevier.com/xml/common/dtd}keyword")
        if k.text
    ]
    return {"sections": sections, "plaintext": plaintext, "keywords": keywords}


def _iter_dois_for_venue(slug: str, min_year: int) -> list[str]:
    src = config.data_dir("raw/openalex") / f"{slug}.jsonl"
    if not src.exists():
        return []
    out = []
    with src.open() as f:
        for line in f:
            w = json.loads(line)
            if w.get("doi") and (w.get("year") or 0) >= min_year:
                out.append(w["doi"])
    return out


def ingest(venues: list[dict] | None = None, *, force: bool = False) -> dict[str, int]:
    cfg = config.load_pipeline()["elsevier"]
    key = config.elsevier_key()
    itoken = config.elsevier_insttoken()
    if not (key and itoken):
        log.error("ELSEVIER_KEY or ELSEVIER_INSTTOKEN missing; aborting")
        return {}
    out_root = config.data_dir("raw/elsevier")
    session = make_session()
    lim = RateLimiter(cfg["rate_limit_per_sec"])
    venues = [v for v in (venues or config.load_venues()) if v.get("elsevier_full_text")]

    counts: dict[str, int] = {}
    for v in venues:
        slug = v["slug"]
        out_dir = out_root / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        dois = _iter_dois_for_venue(slug, cfg["min_year"])
        if not dois:
            log.info(f"[{slug}] no DOIs from OpenAlex yet; run openalex ingest first")
            continue
        done = 0
        skipped = 0
        failed = 0
        for doi in tqdm(dois, desc=f"[{slug}]", unit="doi"):
            target = out_dir / f"{_doi_hash(doi)}.json"
            if target.exists() and not force:
                done += 1
                continue
            lim.wait()
            url = f"{cfg['base_url']}/content/article/doi/{doi}"
            headers = {
                "X-ELS-APIKey": key,
                "X-ELS-Insttoken": itoken,
                "Accept": "application/xml",
            }
            try:
                status, text = get_text(session, url, headers=headers, timeout=cfg["timeout_sec"])
            except Exception as e:
                log.debug(f"{doi}: {e}")
                failed += 1
                continue
            if status == 404:
                target.write_text(json.dumps({"doi": doi, "status": 404}))
                skipped += 1
                continue
            if status in (401, 403):
                log.warning(f"{doi}: HTTP {status} (likely no entitlement)")
                target.write_text(json.dumps({"doi": doi, "status": status}))
                skipped += 1
                continue
            if status != 200:
                log.debug(f"{doi}: HTTP {status}")
                failed += 1
                continue
            try:
                parsed = _parse_full_text(text)
            except Exception as e:
                log.debug(f"{doi}: parse error {e}")
                failed += 1
                continue
            parsed["doi"] = doi
            parsed["status"] = 200
            target.write_text(json.dumps(parsed))
            done += 1
        counts[slug] = done
        log.info(f"[{slug}] full text: {done} done, {skipped} skipped, {failed} failed ({len(dois)} total)")
    return counts
