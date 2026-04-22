#!/usr/bin/env python
"""Build coauthor graph + site-ready aggregates."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transport_atlas.process import aggregate, coauthor_graph


def main() -> int:
    aggregate.run()
    coauthor_graph.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
