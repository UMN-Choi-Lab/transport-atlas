#!/usr/bin/env python
"""End-to-end orchestrator: ingest → dedupe → graph → render.

Checkpointed: reruns skip completed work unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transport_atlas.ingest import elsevier, ieee, openalex
from transport_atlas.process import aggregate, coauthor_graph, dedupe
from transport_atlas.site import render
from transport_atlas.utils import config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-elsevier", action="store_true", help="skip full-text fetch")
    ap.add_argument("--skip-ieee", action="store_true")
    ap.add_argument("--venues", default=None, help="comma-sep slug list")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    venues = config.load_venues()
    if args.venues:
        keep = set(args.venues.split(","))
        venues = [v for v in venues if v["slug"] in keep]

    timings = {}
    t0 = time.time()

    t = time.time(); openalex.ingest(venues, force=args.force); timings["openalex"] = round(time.time() - t, 1)
    if not args.skip_ieee:
        t = time.time(); ieee.ingest(venues, force=args.force); timings["ieee"] = round(time.time() - t, 1)
    if not args.skip_elsevier:
        t = time.time(); elsevier.ingest(venues, force=args.force); timings["elsevier"] = round(time.time() - t, 1)
    t = time.time(); dedupe.run(); timings["dedupe"] = round(time.time() - t, 1)
    t = time.time(); aggregate.run(); timings["aggregate"] = round(time.time() - t, 1)
    t = time.time(); coauthor_graph.run(); timings["graph"] = round(time.time() - t, 1)
    t = time.time(); render.run(); timings["render"] = round(time.time() - t, 1)

    timings["total"] = round(time.time() - t0, 1)
    log_path = config.data_dir("processed") / f"_refresh_{time.strftime('%Y%m%d-%H%M%S')}.json"
    log_path.write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
