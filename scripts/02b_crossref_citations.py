#!/usr/bin/env python
"""Fetch Crossref `is-referenced-by-count` for every DOI in papers.parquet.

Writes checkpointed results to data/interim/crossref_citations.parquet and
updates data/interim/papers.parquet's `cited_by_count` when Crossref returns
a valid count. Adds `cited_by_source` column.

Resumable: re-running picks up where the last checkpoint left off.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transport_atlas.utils.config import crossref_email, data_dir
from transport_atlas.utils.logger import get_logger

LOG = get_logger("crossref_citations")

PAPERS_PATH = data_dir("interim") / "papers.parquet"
CHECKPOINT_PATH = data_dir("interim") / "crossref_citations.parquet"
FLUSH_EVERY = 1000  # flush checkpoint every N results (keeps resume cheap)
REQUEST_TIMEOUT = 30.0

# Crossref's polite pool advertises `x-rate-limit-limit: 10` req/s and
# `x-concurrency-limit: 3` in response headers. We stay under both by default.
DEFAULT_RATE_PER_SEC = 9.5   # stay just under Crossref's advertised 10 req/s limit
DEFAULT_CONCURRENCY_CAP = 3  # matches Crossref's x-concurrency-limit


class AsyncRateLimiter:
    """Token-bucket-ish limiter that spaces outgoing requests by 1/rate seconds."""

    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / max(rate_per_sec, 0.01)
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            if now < self._next_allowed:
                await asyncio.sleep(self._next_allowed - now)
                now = loop.time()
            self._next_allowed = max(now, self._next_allowed) + self.min_interval


def _normalize_doi(doi: str | None) -> str | None:
    if doi is None:
        return None
    s = str(doi).strip()
    if not s or s.lower() == "nan":
        return None
    low = s.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if low.startswith(prefix):
            low = low[len(prefix):]
            break
    return low.strip()


async def _fetch_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    limiter: AsyncRateLimiter,
    doi: str,
    email: str,
) -> tuple[str, int | None, str]:
    """Fetch single DOI. Returns (doi, crossref_cites_or_None, fetched_at_iso).

    Handles:
      - 200: parse and return is-referenced-by-count
      - 404: return None (record as miss, do not retry)
      - 429: exponential backoff (2s -> 30s cap), retry indefinitely
      - timeout/5xx: retry up to 3x with exponential backoff
      - other 4xx: log and return None
    """
    url = f"https://api.crossref.org/works/{doi}"
    params = {"mailto": email}
    backoff_429 = 2.0
    retry_5xx = 0
    max_5xx_retries = 3
    async with sem:
        while True:
            await limiter.wait()
            try:
                r = await client.get(url, params=params, timeout=REQUEST_TIMEOUT)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                retry_5xx += 1
                if retry_5xx > max_5xx_retries:
                    LOG.warning("timeout/transport exhausted for %s: %s", doi, e)
                    return doi, None, datetime.now(timezone.utc).isoformat()
                delay = min(2 ** retry_5xx + random.uniform(0, 1), 30.0)
                await asyncio.sleep(delay)
                continue

            status = r.status_code
            if status == 200:
                try:
                    j = r.json()
                    cites = j.get("message", {}).get("is-referenced-by-count")
                    if isinstance(cites, int) and cites >= 0:
                        return doi, int(cites), datetime.now(timezone.utc).isoformat()
                    # malformed response
                    LOG.warning("malformed response for %s: no int cites", doi)
                    return doi, None, datetime.now(timezone.utc).isoformat()
                except Exception as e:  # noqa: BLE001
                    LOG.warning("parse error for %s: %s", doi, e)
                    return doi, None, datetime.now(timezone.utc).isoformat()

            if status == 404:
                return doi, None, datetime.now(timezone.utc).isoformat()

            if status == 429:
                retry_after_hdr = r.headers.get("Retry-After")
                try:
                    retry_after = float(retry_after_hdr) if retry_after_hdr else backoff_429
                except (TypeError, ValueError):
                    retry_after = backoff_429
                delay = min(max(retry_after, backoff_429), 30.0)
                LOG.info("429 on %s, sleeping %.1fs", doi, delay)
                await asyncio.sleep(delay + random.uniform(0, 0.5))
                backoff_429 = min(backoff_429 * 2, 30.0)
                continue

            if 500 <= status < 600:
                retry_5xx += 1
                if retry_5xx > max_5xx_retries:
                    LOG.warning("5xx (%d) exhausted for %s", status, doi)
                    return doi, None, datetime.now(timezone.utc).isoformat()
                delay = min(2 ** retry_5xx + random.uniform(0, 1), 30.0)
                await asyncio.sleep(delay)
                continue

            # other status: log and skip
            LOG.warning("unexpected status %d for %s", status, doi)
            return doi, None, datetime.now(timezone.utc).isoformat()


def _load_checkpoint() -> pd.DataFrame:
    if CHECKPOINT_PATH.exists():
        df = pd.read_parquet(CHECKPOINT_PATH)
        LOG.info("loaded checkpoint: %d rows from %s", len(df), CHECKPOINT_PATH)
        return df
    return pd.DataFrame(columns=["doi", "crossref_cites", "fetched_at"])


def _flush(rows: list[dict], existing: pd.DataFrame) -> pd.DataFrame:
    if not rows:
        return existing
    new_df = pd.DataFrame(rows)
    merged = pd.concat([existing, new_df], ignore_index=True)
    # dedup — last write wins
    merged = merged.drop_duplicates(subset=["doi"], keep="last").reset_index(drop=True)
    merged.to_parquet(CHECKPOINT_PATH, index=False)
    return merged


async def _run_async(
    dois: list[str], workers: int, email: str, rate_per_sec: float,
) -> pd.DataFrame:
    existing = _load_checkpoint()
    done = set(existing["doi"].tolist())
    todo = [d for d in dois if d not in done]
    LOG.info("total DOIs: %d | already done: %d | to fetch: %d", len(dois), len(done), len(todo))

    if not todo:
        LOG.info("nothing to do — all DOIs already in checkpoint")
        return existing

    eff_workers = min(workers, DEFAULT_CONCURRENCY_CAP)
    if eff_workers != workers:
        LOG.info(
            "clamping workers %d -> %d (Crossref x-concurrency-limit is 3)",
            workers, eff_workers,
        )
    sem = asyncio.Semaphore(eff_workers)
    limiter = AsyncRateLimiter(rate_per_sec)
    LOG.info("rate limit: %.1f req/s, in-flight cap: %d", rate_per_sec, eff_workers)
    limits = httpx.Limits(max_keepalive_connections=eff_workers, max_connections=eff_workers * 2)
    headers = {"User-Agent": f"transport-atlas/0.1 (mailto:{email})"}

    buffer: list[dict] = []
    start = time.monotonic()
    completed = 0
    async with httpx.AsyncClient(
        limits=limits, headers=headers, http2=False, follow_redirects=True,
    ) as client:
        tasks = [asyncio.create_task(_fetch_one(client, sem, limiter, d, email)) for d in todo]
        for coro in asyncio.as_completed(tasks):
            doi, cites, fetched_at = await coro
            buffer.append({"doi": doi, "crossref_cites": cites, "fetched_at": fetched_at})
            completed += 1
            if completed % FLUSH_EVERY == 0:
                existing = _flush(buffer, existing)
                buffer = []
                elapsed = time.monotonic() - start
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = len(todo) - completed
                eta_s = remaining / rate if rate > 0 else 0.0
                LOG.info(
                    "progress: %d/%d (%.1f%%) | %.1f req/s | ETA %.1f min",
                    completed, len(todo), 100 * completed / len(todo), rate, eta_s / 60.0,
                )

    # final flush
    existing = _flush(buffer, existing)
    elapsed = time.monotonic() - start
    LOG.info("fetch loop done in %.1f s (%d new)", elapsed, completed)
    return existing


def _update_papers_parquet(checkpoint: pd.DataFrame) -> None:
    """Overwrite papers.parquet with Crossref-based cited_by_count + cited_by_source."""
    LOG.info("reading %s", PAPERS_PATH)
    papers = pd.read_parquet(PAPERS_PATH)
    LOG.info("papers shape: %s", papers.shape)

    # normalize DOIs on both sides for merge
    papers["_doi_norm"] = papers["doi"].map(_normalize_doi)
    ck = checkpoint.copy()
    ck["_doi_norm"] = ck["doi"].map(_normalize_doi)
    ck_by_doi = ck.drop_duplicates(subset=["_doi_norm"], keep="last").set_index("_doi_norm")

    # build override series
    def resolve(row):
        key = row["_doi_norm"]
        if key is None or key not in ck_by_doi.index:
            return pd.Series({"cited_by_count": row["cited_by_count"], "cited_by_source": "openalex"})
        cx = ck_by_doi.at[key, "crossref_cites"]
        if pd.isna(cx):
            return pd.Series({"cited_by_count": row["cited_by_count"], "cited_by_source": "openalex"})
        try:
            cx_int = int(cx)
        except (TypeError, ValueError):
            return pd.Series({"cited_by_count": row["cited_by_count"], "cited_by_source": "openalex"})
        if cx_int < 0:
            return pd.Series({"cited_by_count": row["cited_by_count"], "cited_by_source": "openalex"})
        return pd.Series({"cited_by_count": cx_int, "cited_by_source": "crossref"})

    # vectorized version for speed
    papers["_openalex_cites"] = papers["cited_by_count"]
    joined = papers.merge(
        ck_by_doi[["crossref_cites"]].rename(columns={"crossref_cites": "_cx_cites"}),
        left_on="_doi_norm",
        right_index=True,
        how="left",
    )
    has_cx = joined["_cx_cites"].notna() & (joined["_cx_cites"].fillna(-1) >= 0)
    joined["cited_by_count"] = joined["_openalex_cites"].where(~has_cx, joined["_cx_cites"].astype("Int64"))
    # cast back to a consistent dtype
    joined["cited_by_count"] = joined["cited_by_count"].astype("Int64").fillna(0).astype("int64")
    joined["cited_by_source"] = ["crossref" if b else "openalex" for b in has_cx]

    # drop helper columns
    drop_cols = ["_doi_norm", "_cx_cites", "_openalex_cites"]
    joined = joined.drop(columns=[c for c in drop_cols if c in joined.columns])

    # preserve original column order + cited_by_source at end
    original_cols = [
        "paper_id", "doi", "title", "year", "date", "venue_slug",
        "type", "cited_by_count", "abstract", "concepts", "authors",
    ]
    final_cols = [c for c in original_cols if c in joined.columns] + ["cited_by_source"]
    joined = joined[final_cols]

    LOG.info("writing %s (shape=%s)", PAPERS_PATH, joined.shape)
    joined.to_parquet(PAPERS_PATH, index=False)


def _compute_report_with_old(
    checkpoint: pd.DataFrame, papers_before: pd.DataFrame, elapsed_s: float,
) -> str:
    ck = checkpoint.copy()
    ck["_doi_norm"] = ck["doi"].map(_normalize_doi)
    ck_valid = ck[ck["crossref_cites"].notna()].copy()
    ck_valid["crossref_cites"] = ck_valid["crossref_cites"].astype(int)

    total_dois = int(papers_before["doi"].notna().sum())
    hits = len(ck_valid)

    papers_before = papers_before.copy()
    papers_before["_doi_norm"] = papers_before["doi"].map(_normalize_doi)
    merged = papers_before.merge(
        ck_valid[["_doi_norm", "crossref_cites"]],
        on="_doi_norm",
        how="inner",
    )
    merged["delta"] = merged["crossref_cites"] - merged["cited_by_count"]
    mean_delta = float(merged["delta"].mean()) if len(merged) else 0.0
    median_delta = float(merged["delta"].median()) if len(merged) else 0.0

    top = merged.sort_values("delta", ascending=False).head(10)

    lines: list[str] = []
    lines.append("# Crossref Citation Enrichment — Report")
    lines.append("")
    lines.append(f"- Wall-clock elapsed: **{elapsed_s:.1f} s** ({elapsed_s / 60:.1f} min)")
    lines.append(f"- Total DOIs: **{total_dois:,}**")
    lines.append(f"- Crossref hits: **{hits:,}** ({100 * hits / max(total_dois, 1):.1f}%)")
    lines.append(f"- Mean delta (Crossref − OpenAlex): **{mean_delta:.2f}**")
    lines.append(f"- Median delta (Crossref − OpenAlex): **{median_delta:.2f}**")
    lines.append("")
    lines.append("## Top 10 Biggest OpenAlex Undercounts")
    lines.append("")
    lines.append("| # | Title | Venue | OpenAlex | Crossref | Δ |")
    lines.append("|---|-------|-------|---------:|---------:|--:|")
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        title = (row.get("title") or "").replace("|", "\\|")
        if len(title) > 80:
            title = title[:77] + "..."
        venue = row.get("venue_slug") or ""
        lines.append(
            f"| {i} | {title} | {venue} | {int(row['cited_by_count'])} | "
            f"{int(row['crossref_cites'])} | {int(row['delta'])} |"
        )
    lines.append("")
    lines.append(f"- Checkpoint: `{CHECKPOINT_PATH}`")
    lines.append(f"- Updated papers parquet: `{PAPERS_PATH}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30, help="requested concurrency (capped to %d)" % DEFAULT_CONCURRENCY_CAP)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE_PER_SEC, help="max req/s (default %.1f)" % DEFAULT_RATE_PER_SEC)
    ap.add_argument("--force", action="store_true", help="ignore checkpoint and refetch all")
    args = ap.parse_args()

    email = crossref_email() or "chois@umn.edu"
    LOG.info("starting Crossref citation enrichment (workers=%d, force=%s)", args.workers, args.force)
    LOG.info("mailto=%s", email)

    t0_wall = datetime.now(timezone.utc)
    LOG.info("wall-clock start: %s", t0_wall.isoformat())

    LOG.info("reading %s", PAPERS_PATH)
    papers_before = pd.read_parquet(PAPERS_PATH)
    LOG.info("papers shape: %s", papers_before.shape)

    dois = (
        papers_before["doi"]
        .map(_normalize_doi)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    LOG.info("unique DOIs to process: %d", len(dois))

    if args.force and CHECKPOINT_PATH.exists():
        LOG.info("--force: removing existing checkpoint")
        CHECKPOINT_PATH.unlink()

    t0 = time.monotonic()
    checkpoint = asyncio.run(
        _run_async(dois, workers=args.workers, email=email, rate_per_sec=args.rate)
    )
    elapsed = time.monotonic() - t0
    t1_wall = datetime.now(timezone.utc)
    LOG.info("wall-clock end: %s (elapsed %.1fs)", t1_wall.isoformat(), elapsed)

    # update papers.parquet (uses the in-memory papers_before snapshot for report)
    _update_papers_parquet(checkpoint)

    report = _compute_report_with_old(checkpoint, papers_before, elapsed)
    print()
    print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
