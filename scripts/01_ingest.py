#!/usr/bin/env python
"""Ingest metadata from OpenAlex / IEEE / Elsevier.

Usage:
  python scripts/01_ingest.py --source openalex
  python scripts/01_ingest.py --source ieee --venue t-its
  python scripts/01_ingest.py --source elsevier --venue tr-c
  python scripts/01_ingest.py --source all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transport_atlas.utils import config

# NB: ingest backends are imported lazily inside main() so an OpenAlex-only run
# (e.g. the scheduled update) doesn't pull elsevier.py's lxml/XML stack.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["openalex", "ieee", "elsevier", "all"], default="openalex")
    ap.add_argument("--venue", default=None, help="slug; omit for all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--incremental", action="store_true",
                    help="OpenAlex top-up: append only recently-published works "
                         "(dedup by id). Use for scheduled updates instead of --force.")
    ap.add_argument("--lookback-days", type=int, default=None,
                    help="publication-date window for --incremental "
                         "(default: config update.lookback_days, else 90)")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --incremental: report new-work counts without writing")
    args = ap.parse_args()

    venues = config.load_venues()
    if args.venue:
        venues = [v for v in venues if v["slug"] == args.venue]
        if not venues:
            print(f"unknown venue slug: {args.venue}")
            return 2

    if args.incremental:
        if args.source not in ("openalex", "all"):
            print("--incremental is only supported for --source openalex")
            return 2
        from transport_atlas.ingest import openalex
        lookback = args.lookback_days
        if lookback is None:
            lookback = (config.load_pipeline().get("update") or {}).get("lookback_days", 90)
        report = openalex.ingest_recent(venues, lookback_days=lookback, dry_run=args.dry_run)
        import json as _json
        print(_json.dumps(report, indent=2))
        return 0

    if args.source in ("openalex", "all"):
        from transport_atlas.ingest import openalex
        openalex.ingest(venues, force=args.force)
    if args.source in ("ieee", "all"):
        from transport_atlas.ingest import ieee
        ieee.ingest(venues, force=args.force)
    if args.source in ("elsevier", "all"):
        from transport_atlas.ingest import elsevier
        elsevier.ingest(venues, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
