#!/usr/bin/env python
"""Fetch as-of-train-cutoff citation snapshots from OpenAlex (Reviewer 4, pt 1).

The phantom eval weights each train paper's contribution to its authors'
centroids by w = 1 + log1p(citations).  Using the CURRENT cited_by_count for
pre-2020 papers is temporal leakage (a 2019 paper would be weighted by its
2026-era citation total).  This script reconstructs the citation count as of
the train cutoff from a SINGLE OpenAlex response per work:

    cited_asof_cutoff = max(0, cited_by_count - sum(counts_by_year[y]
                                                    for y > cutoff_year))

Both fields come from the same /works response, so the subtraction is
internally consistent; negative reconstructions (documented OpenAlex async
update artifacts) are clamped at 0 and logged.  counts_by_year == [] is normal
(zero-citation works, or old works whose citations all predate the ~10-year
window) and means cited_asof_cutoff == cited_by_count.

Scope: all train-period works (papers.parquet year <= cutoff).  W-ids are
recovered by joining interim paper_ids to data/raw/openalex/*.jsonl (which
retains openalex_id); interim rows that cannot be matched fall back to a
DOI-filter batch lookup, and the remainder is logged, never dropped silently.

Checkpointed + resumable:
    data/raw/openalex_citations/shard_XXXX.jsonl   (raw per-work records)
    data/raw/openalex_citations/_meta.json
Resume is the default (already-fetched requested ids are skipped);
--force wipes the store and refetches everything.

Compact mapping consumed by scripts/07_phantom_eval.py (--citations-asof):
    data/interim/citation_snapshots.parquet
    columns: paper_id, openalex_id, publication_year, cited_now,
             cited_since_cutoff, cited_asof_cutoff, clamped, cutoff_year

Polite pool: mailto comes from transport_atlas.utils.config (CROSSREF_EMAIL);
it is never printed or hardcoded.  Batches of 50 ids per request (the
documented per-filter OR cap), a conservative rate limit, and exponential
backoff on 429/5xx.

Usage (host python or the docker image; needs network — NOT run by tests):
    PYTHONPATH=src /usr/bin/python3 scripts/fetch_citation_snapshots.py
    ... --cutoff-year 2019 --rate 4 [--force] [--compact-only]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OPENALEX_BASE = "https://api.openalex.org"
BATCH_SIZE = 50          # documented per-filter OR cap
SHARD_SIZE = 5000        # records per JSONL shard
SELECT_FIELDS = "id,publication_year,cited_by_count,counts_by_year"
# OpenAlex documents counts_by_year coverage for roughly the last 10 years.
COUNTS_WINDOW_YEARS = 10


# ----------------------------------------------------------------------
# Pure arithmetic (unit-tested offline in tests/test_citation_snapshots.py)
# ----------------------------------------------------------------------
def sum_counts_since(counts_by_year: list[dict] | None, since_year: int) -> int:
    """Sum counts_by_year entries with year >= since_year (None/[] -> 0)."""
    if not counts_by_year:
        return 0
    total = 0
    for row in counts_by_year:
        try:
            yr = int(row.get("year"))
            ct = int(row.get("cited_by_count") or 0)
        except (TypeError, ValueError):
            continue
        if yr >= since_year:
            total += ct
    return total


def asof_citations(
    cited_by_count: int | None,
    counts_by_year: list[dict] | None,
    cutoff_year: int,
) -> tuple[int, bool]:
    """(citations as of end of cutoff_year, clamped?).

    asof = cited_by_count - sum(counts for years > cutoff_year), clamped at 0.
    Empty counts_by_year is NOT missing data: it means asof == cited_by_count
    (zero-citation work, or all citations predate the coverage window).
    """
    cited_now = int(cited_by_count or 0)
    since = sum_counts_since(counts_by_year, cutoff_year + 1)
    asof = cited_now - since
    if asof < 0:
        return 0, True
    return asof, False


def window_covers_cutoff(now_year: int, cutoff_year: int,
                         window_years: int = COUNTS_WINDOW_YEARS) -> bool:
    """True while counts_by_year's guaranteed window still covers cutoff+1.

    The subtraction needs every year > cutoff to be inside the coverage
    window [now_year - (window_years - 1), now_year].  Under the documented
    10-year guarantee this holds while now_year <= cutoff_year + window_years.
    """
    return (now_year - (window_years - 1)) <= (cutoff_year + 1)


def snapshot_record(requested_id: str, work: dict, cutoff_year: int) -> dict:
    """Build a raw-store record from one /works response object.

    Raw fields (cited_by_count, counts_by_year) are stored verbatim so other
    cutoffs can be derived later without refetching.
    """
    canonical = (work.get("id") or "").rsplit("/", 1)[-1] or None
    cby = work.get("counts_by_year") or []
    cited_now = int(work.get("cited_by_count") or 0)
    asof, clamped = asof_citations(cited_now, cby, cutoff_year)
    return {
        "requested_id": requested_id,
        "openalex_id": canonical,
        "publication_year": work.get("publication_year"),
        "cited_by_count": cited_now,
        "counts_by_year": cby,
        "cited_since_cutoff": sum_counts_since(cby, cutoff_year + 1),
        "cited_asof_cutoff": asof,
        "clamped": clamped,
        "cutoff_year": cutoff_year,
    }


def build_id_map(
    papers_rows: list[tuple],
    raw_records: list[dict],
    paper_id_fn,
) -> tuple[dict[str, str], list[tuple]]:
    """Map interim paper_id -> OpenAlex W-id via the raw ingest records.

    papers_rows: (paper_id, doi, title, year) tuples from papers.parquet.
    raw_records: dicts with openalex_id/doi/title/year from raw JSONL.
    paper_id_fn: the dedupe paper-id function (doi:… or h:…), injected so the
    join reproduces the exact dedupe key including the h: title-hash rows.

    Returns (mapping, unmatched_rows).
    """
    pid_map: dict[str, str] = {}
    doi_map: dict[str, str] = {}
    for rec in raw_records:
        wid = rec.get("openalex_id")
        if not wid:
            continue
        pid = paper_id_fn(rec.get("doi"), rec.get("title"), rec.get("year"))
        pid_map.setdefault(pid, wid)
        d = rec.get("doi")
        if isinstance(d, str) and d:
            doi_map.setdefault(d.strip().lower(), wid)

    mapping: dict[str, str] = {}
    unmatched: list[tuple] = []
    for row in papers_rows:
        paper_id, doi, _title, _year = row
        wid = pid_map.get(paper_id)
        if wid is None and isinstance(doi, str) and doi:
            wid = doi_map.get(doi.strip().lower())
        if wid is None:
            unmatched.append(row)
        else:
            mapping[paper_id] = wid
    return mapping, unmatched


# ----------------------------------------------------------------------
# HTTP (network path — exercised only in live runs, never by tests)
# ----------------------------------------------------------------------
def _request_json(session, url: str, params: dict, max_attempts: int = 6) -> dict:
    """GET with exponential backoff on 429/5xx and connection errors."""
    import requests

    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            r = session.get(url, params=params, timeout=60)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            retry_after = r.headers.get("Retry-After")
            wait = min(float(retry_after), 120.0) if retry_after else delay
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue
        r.raise_for_status()
        return r.json()
    if last_exc:
        raise last_exc
    raise RuntimeError(f"OpenAlex request failed after {max_attempts} attempts")


def _fetch_batch(session, wids: list[str], mailto: str) -> list[dict]:
    params = {
        "filter": "ids.openalex:" + "|".join(wids),
        "select": SELECT_FIELDS,
        "per-page": str(max(BATCH_SIZE, len(wids))),  # default 25 would truncate
        "mailto": mailto,
    }
    data = _request_json(session, f"{OPENALEX_BASE}/works", params)
    return data.get("results", [])


def _fetch_single(session, wid: str, mailto: str) -> dict | None:
    """Single-work fetch; follows merge redirects. None on 404."""
    import requests

    try:
        return _request_json(
            session, f"{OPENALEX_BASE}/works/{wid}",
            {"select": SELECT_FIELDS, "mailto": mailto})
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise


def _fetch_batch_by_doi(session, dois: list[str], mailto: str) -> list[dict]:
    params = {
        "filter": "doi:" + "|".join(dois),
        "select": SELECT_FIELDS + ",doi",
        "per-page": str(max(BATCH_SIZE, len(dois))),
        "mailto": mailto,
    }
    data = _request_json(session, f"{OPENALEX_BASE}/works", params)
    return data.get("results", [])


# ----------------------------------------------------------------------
# Checkpoint store
# ----------------------------------------------------------------------
class ShardStore:
    """Append-only JSONL shard store with resume support."""

    def __init__(self, store_dir: Path, shard_size: int = SHARD_SIZE):
        self.dir = store_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self._counts: dict[Path, int] = {}

    def shards(self) -> list[Path]:
        return sorted(self.dir.glob("shard_*.jsonl"))

    def load_fetched(self) -> dict[str, dict]:
        """requested_id -> record, across all shards (later wins)."""
        out: dict[str, dict] = {}
        for p in self.shards():
            n = 0
            with p.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rid = rec.get("requested_id")
                    if rid:
                        out[rid] = rec
                    n += 1
            self._counts[p] = n
        return out

    def _target_shard(self) -> Path:
        shards = self.shards()
        if shards:
            last = shards[-1]
            n = self._counts.get(last)
            if n is None:
                with last.open() as f:
                    n = sum(1 for _ in f)
                self._counts[last] = n
            if n < self.shard_size:
                return last
            nxt = int(last.stem.split("_")[1]) + 1
        else:
            nxt = 0
        p = self.dir / f"shard_{nxt:04d}.jsonl"
        self._counts.setdefault(p, 0)
        return p

    def append(self, records: list[dict]) -> None:
        i = 0
        while i < len(records):
            p = self._target_shard()
            room = self.shard_size - self._counts.get(p, 0)
            chunk = records[i:i + room]
            with p.open("a") as f:
                for rec in chunk:
                    f.write(json.dumps(rec) + "\n")
            self._counts[p] = self._counts.get(p, 0) + len(chunk)
            i += room

    def write_meta(self, meta: dict) -> None:
        (self.dir / "_meta.json").write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff-year", type=int, default=2019,
                    help="Train cutoff; citations counted as of END of this year.")
    ap.add_argument("--rate", type=float, default=4.0,
                    help="Max requests/sec (polite pool allows 10).")
    ap.add_argument("--force", action="store_true",
                    help="Wipe the checkpoint store and refetch everything. "
                         "Default is to RESUME (skip fetched ids).")
    ap.add_argument("--compact-only", action="store_true",
                    help="Skip fetching; rebuild the compact mapping from the "
                         "existing shard store.")
    ap.add_argument("--papers", type=Path,
                    default=ROOT / "data" / "interim" / "papers.parquet")
    ap.add_argument("--raw-dir", type=Path,
                    default=ROOT / "data" / "raw" / "openalex")
    ap.add_argument("--store-dir", type=Path,
                    default=ROOT / "data" / "raw" / "openalex_citations")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "interim" / "citation_snapshots.parquet")
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: cap the number of works to fetch (0 = all).")
    args = ap.parse_args()

    cutoff = args.cutoff_year
    now_year = date.today().year
    if not window_covers_cutoff(now_year, cutoff):
        print(f"[cites] FATAL: counts_by_year's ~{COUNTS_WINDOW_YEARS}-year "
              f"window no longer covers {cutoff + 1}..{now_year}; the "
              f"subtraction identity is unsound. Aborting.", file=sys.stderr)
        return 2

    import duckdb

    from transport_atlas.process.dedupe import _paper_id
    from transport_atlas.utils import config as ta_config

    # 1. Train-period papers (DuckDB per repo convention: >10k rows).
    con = duckdb.connect()
    papers_rows = con.execute(
        "SELECT paper_id, doi, title, year FROM read_parquet(?) "
        "WHERE year IS NOT NULL AND year <= ? ORDER BY paper_id",
        [str(args.papers), cutoff]).fetchall()
    print(f"[cites] train papers (year <= {cutoff}): {len(papers_rows):,}",
          flush=True)

    # 2. Recover W-ids from the raw ingest JSONL.
    raw_records: list[dict] = []
    for p in sorted(args.raw_dir.glob("*.jsonl")):
        with p.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_records.append({
                    "openalex_id": rec.get("openalex_id"),
                    "doi": rec.get("doi"),
                    "title": rec.get("title"),
                    "year": rec.get("year"),
                })
    mapping, unmatched = build_id_map(papers_rows, raw_records, _paper_id)
    print(f"[cites] paper_id -> W-id: {len(mapping):,} matched, "
          f"{len(unmatched):,} unmatched", flush=True)

    store = ShardStore(args.store_dir)
    if args.force and not args.compact_only:
        for p in store.shards():
            p.unlink()
        meta_p = args.store_dir / "_meta.json"
        if meta_p.exists():
            meta_p.unlink()
        print("[cites] --force: wiped checkpoint store", flush=True)
    fetched = store.load_fetched()

    wid_of: dict[str, str] = mapping  # paper_id -> requested W-id
    todo_wids = sorted({w for w in wid_of.values() if w not in fetched})
    if args.limit:
        todo_wids = todo_wids[:args.limit]

    n_unmatched_final = len(unmatched)
    if not args.compact_only:
        import requests

        mailto = ta_config.crossref_email()
        if not mailto:
            print("[cites] FATAL: CROSSREF_EMAIL not configured "
                  "(~/.claude/mcp-servers/refcheck/.env)", file=sys.stderr)
            return 2
        session = requests.Session()
        session.headers.update({"User-Agent": "transport-atlas/0.1"})
        min_interval = 1.0 / max(args.rate, 0.1)

        print(f"[cites] fetching {len(todo_wids):,} works "
              f"({len(fetched):,} already checkpointed) in batches of "
              f"{BATCH_SIZE} ...", flush=True)
        n_clamped = 0
        t0 = time.time()
        next_allowed = 0.0
        for b0 in range(0, len(todo_wids), BATCH_SIZE):
            batch = todo_wids[b0:b0 + BATCH_SIZE]
            now = time.monotonic()
            if now < next_allowed:
                time.sleep(next_allowed - now)
            next_allowed = time.monotonic() + min_interval
            results = _fetch_batch(session, batch, mailto)
            by_id = {(w.get("id") or "").rsplit("/", 1)[-1]: w for w in results}
            recs: list[dict] = []
            missing_in_batch = [w for w in batch if w not in by_id]
            for wid in batch:
                if wid in by_id:
                    recs.append(snapshot_record(wid, by_id[wid], cutoff))
            # Reconcile: merged-away ids fall out of the batch filter; refetch
            # singly via /works/<id>, which follows merge redirects.
            for wid in missing_in_batch:
                now = time.monotonic()
                if now < next_allowed:
                    time.sleep(next_allowed - now)
                next_allowed = time.monotonic() + min_interval
                w = _fetch_single(session, wid, mailto)
                if w is None:
                    recs.append({"requested_id": wid, "missing": True,
                                 "cutoff_year": cutoff})
                else:
                    recs.append(snapshot_record(wid, w, cutoff))
            for rec in recs:
                if rec.get("clamped"):
                    n_clamped += 1
                    print(f"[cites] clamped negative asof for "
                          f"{rec['requested_id']} (now={rec['cited_by_count']}"
                          f", since={rec['cited_since_cutoff']})", flush=True)
            store.append(recs)
            for rec in recs:
                fetched[rec["requested_id"]] = rec
            done = b0 + len(batch)
            if done % 5000 < BATCH_SIZE:
                print(f"[cites] {done:,}/{len(todo_wids):,} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # DOI-filter fallback for interim rows with no raw-JSONL W-id match.
        doi_fallback = [(pid, doi) for (pid, doi, _t, _y) in unmatched
                        if isinstance(doi, str) and doi]
        if doi_fallback:
            print(f"[cites] DOI-fallback lookup for {len(doi_fallback):,} "
                  f"unmatched rows ...", flush=True)
            resolved: dict[str, dict] = {}
            dois = [d for _pid, d in doi_fallback]
            for b0 in range(0, len(dois), BATCH_SIZE):
                batch = [f"https://doi.org/{d}" for d in dois[b0:b0 + BATCH_SIZE]]
                now = time.monotonic()
                if now < next_allowed:
                    time.sleep(next_allowed - now)
                next_allowed = time.monotonic() + min_interval
                for w in _fetch_batch_by_doi(session, batch, mailto):
                    d = (w.get("doi") or "").lower().replace(
                        "https://doi.org/", "")
                    if d:
                        resolved[d] = w
            recs = []
            for pid, doi in doi_fallback:
                w = resolved.get(doi.strip().lower())
                if w is not None:
                    wid = (w.get("id") or "").rsplit("/", 1)[-1]
                    rec = snapshot_record(wid, w, cutoff)
                    recs.append(rec)
                    fetched[wid] = rec
                    wid_of[pid] = wid
            if recs:
                store.append(recs)
            n_unmatched_final = len(unmatched) - sum(
                1 for pid, _d in doi_fallback if wid_of.get(pid))
            print(f"[cites] DOI fallback resolved {len(recs):,}; "
                  f"{n_unmatched_final:,} rows remain unmatched (logged, "
                  f"weighted c=0 downstream)", flush=True)

        store.write_meta({
            "cutoff_year": cutoff,
            "n_train_papers": len(papers_rows),
            "n_wids_targeted": len(set(wid_of.values())),
            "n_fetched": len(fetched),
            "n_clamped": n_clamped,
            "n_unmatched_paper_ids": n_unmatched_final,
            "select": SELECT_FIELDS,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    # 3. Compact mapping keyed by paper_id for scripts/07_phantom_eval.py.
    import pandas as pd

    rows_out = []
    n_missing_fetch = 0
    for (paper_id, _doi, _title, _year) in papers_rows:
        wid = wid_of.get(paper_id)
        rec = fetched.get(wid) if wid else None
        if rec is None or rec.get("missing"):
            n_missing_fetch += 1
            continue
        rows_out.append({
            "paper_id": paper_id,
            "openalex_id": rec.get("openalex_id") or wid,
            "publication_year": rec.get("publication_year"),
            "cited_now": int(rec.get("cited_by_count") or 0),
            "cited_since_cutoff": int(rec.get("cited_since_cutoff") or 0),
            "cited_asof_cutoff": int(rec.get("cited_asof_cutoff") or 0),
            "clamped": bool(rec.get("clamped", False)),
            "cutoff_year": cutoff,
        })
    if not rows_out:
        print("[cites] no fetched records — nothing to write "
              "(run without --compact-only first)", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[cites] wrote {args.out} ({len(df):,} rows; "
          f"{n_missing_fetch:,} train papers without a snapshot; "
          f"clamped={int(df['clamped'].sum()):,})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
