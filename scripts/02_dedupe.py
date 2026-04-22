#!/usr/bin/env python
"""Dedup ingest → data/interim/papers.parquet + authors.parquet."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transport_atlas.process import dedupe


def main() -> int:
    dedupe.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
