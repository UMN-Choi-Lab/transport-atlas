#!/usr/bin/env python
"""Annotate each node in coauthor_network.json with its full co-author list.

The graph ships only edges with >=2 collaborations (BASE_THRESHOLD), so
1-collab pairs are invisible in the view — users report "why isn't X
connected to Y?" when they share exactly one paper.

Shipping all 1-collab edges would balloon the JSON ~16x (71k -> 1.1M edges).
Instead, we add a per-node ``ac`` array: ``[[coauthor_node_id, n_collabs], ...]``
covering every coauthor who also has a node in the graph. The browser uses it
to populate the selected-author panel with the complete collaboration list,
regardless of edge threshold.

Runs in place. Safe to invoke multiple times (idempotent).
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transport_atlas.process.authors import author_key as _raw_author_key
from transport_atlas.process.coauthor_graph import _alias_map


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cn_path = repo / "data" / "processed" / "coauthor_network.json"
    if not cn_path.exists():
        print(f"[ac] missing {cn_path} — run scripts/03_graph.py first", file=sys.stderr)
        return 1

    cn = json.loads(cn_path.read_text())
    nodes = cn["nodes"]
    key_to_nid = {n["key"]: n["id"] for n in nodes if "key" in n}
    print(f"[ac] {len(nodes):,} nodes in graph", flush=True)

    papers = pd.read_parquet(repo / "data" / "interim" / "papers.parquet")
    alias_map = _alias_map()
    print(f"[ac] {len(papers):,} papers, {len(alias_map):,} aliases", flush=True)

    # For each paper, resolve author keys (with alias), keep only those in the graph
    pair_counts: Counter[tuple[int, int]] = Counter()
    t0 = time.time()
    kept_papers = 0
    for a_list in papers["authors"]:
        try:
            if a_list is None or len(a_list) == 0:
                continue
        except TypeError:
            continue
        nids = []
        for a in a_list:
            if not isinstance(a, dict):
                continue
            k = _raw_author_key(a)
            if not k:
                continue
            k = alias_map.get(k, k)
            nid = key_to_nid.get(k)
            if nid is not None:
                nids.append(nid)
        uniq = sorted(set(nids))
        if len(uniq) < 2:
            continue
        kept_papers += 1
        for a, b in combinations(uniq, 2):
            pair_counts[(a, b)] += 1

    print(f"[ac] {kept_papers:,} multi-author papers with ≥2 in-graph authors  "
          f"({time.time() - t0:.1f}s)", flush=True)
    print(f"[ac] distinct author pairs: {len(pair_counts):,}", flush=True)

    # Invert into per-node adjacency dict
    per_node: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (a, b), c in pair_counts.items():
        per_node[a].append((b, c))
        per_node[b].append((a, c))

    # Sort each list by collab count desc and attach. Cap at 200 per node to
    # keep file size sane (mean ~15, p99 ~150).
    MAX_LIST = 200
    total_pairs = 0
    for n in nodes:
        lst = per_node.get(n["id"], [])
        lst.sort(key=lambda p: -p[1])
        if len(lst) > MAX_LIST:
            lst = lst[:MAX_LIST]
        # Compact encoding: list of [id, collabs]
        n["ac"] = [[cid, cnt] for cid, cnt in lst]
        total_pairs += len(lst)
    print(f"[ac] mean ac size: {total_pairs / max(len(nodes), 1):.1f} entries/node", flush=True)

    # Write in place. Use allow_nan=False per project rule (browser-safe JSON).
    cn_path.write_text(json.dumps(cn, allow_nan=False))
    site_path = repo / "site" / "data" / "coauthor_network.json"
    if site_path.exists():
        site_path.write_text(json.dumps(cn, allow_nan=False))
        print(f"[ac] updated {site_path}", flush=True)
    print(f"[ac] wrote {cn_path} ({cn_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
