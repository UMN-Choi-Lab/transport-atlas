#!/usr/bin/env python
"""Render static site under site/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transport_atlas.site import render


def main() -> int:
    out = render.run()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
