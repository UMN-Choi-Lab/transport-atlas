#!/usr/bin/env python
"""Re-flag the ``phantom`` field in ``author_similar.json`` in place.

The full similarity pipeline (``scripts/06_author_similarity.py``) computes
both the kNN list and the per-entry ``phantom`` flag. The flag is purely
derived from the coauthor-graph distance ``d`` already stored on each
entry, so when the threshold changes there's no need to redo the costly
embedding + kNN step — this script just reads the JSON, recomputes
``phantom = (d is None) or (d >= MIN_HOPS)`` for every entry, and writes
the file back.

Idempotent. Updates both ``data/processed/`` and ``site/data/`` copies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def reflag(path: Path, min_hops: int) -> tuple[int, int]:
    data = json.loads(path.read_text())
    total = 0
    flipped = 0
    for entries in data.values():
        for s in entries:
            old = bool(s.get("phantom"))
            d = s.get("d")
            new = (d is None) or (d >= min_hops)
            if old != new:
                flipped += 1
            s["phantom"] = new
            total += 1
    # allow_nan=False per project rule
    path.write_text(json.dumps(data, allow_nan=False))
    return total, flipped


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    # Mirror the constant in scripts/06_author_similarity.py
    sys.path.insert(0, str(repo / "scripts"))
    try:
        # Lazy-load via importlib because the script name starts with a digit
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_six_author_similarity", repo / "scripts" / "06_author_similarity.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        min_hops = int(mod.PHANTOM_MIN_HOPS)
    except Exception as exc:  # pragma: no cover
        print(f"[reflag] could not import PHANTOM_MIN_HOPS ({exc!r}); falling back to 2", file=sys.stderr)
        min_hops = 2

    targets = [
        repo / "data" / "processed" / "author_similar.json",
        repo / "site" / "data" / "author_similar.json",
    ]
    for p in targets:
        if not p.exists():
            print(f"[reflag] skip (not found): {p}")
            continue
        total, flipped = reflag(p, min_hops)
        print(f"[reflag] {p}: {flipped:,}/{total:,} entries re-flagged "
              f"(threshold d >= {min_hops})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
