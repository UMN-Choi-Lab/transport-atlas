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

from transport_atlas.ingest import elsevier, ieee, openalex
from transport_atlas.utils import config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["openalex", "ieee", "elsevier", "all"], default="openalex")
    ap.add_argument("--venue", default=None, help="slug; omit for all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    venues = config.load_venues()
    if args.venue:
        venues = [v for v in venues if v["slug"] == args.venue]
        if not venues:
            print(f"unknown venue slug: {args.venue}")
            return 2

    if args.source in ("openalex", "all"):
        openalex.ingest(venues, force=args.force)
    if args.source in ("ieee", "all"):
        ieee.ingest(venues, force=args.force)
    if args.source in ("elsevier", "all"):
        elsevier.ingest(venues, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
