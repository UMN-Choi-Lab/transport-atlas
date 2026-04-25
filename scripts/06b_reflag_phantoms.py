#!/usr/bin/env python
"""Recompute the ``d`` (coauthor distance) and ``phantom`` fields of
``author_similar.json`` in place, using the *unthresholded* coauthor graph.

Why this exists:

- The full similarity pipeline (``scripts/06_author_similarity.py``) builds
  the BFS graph from ``coauthor_network.json`` edges. Those edges are
  thresholded at ≥2 shared papers (``BASE_THRESHOLD`` in coauthor_graph.py),
  so 1-collab pairs are invisible to the BFS and get labeled d≥2 — wrongly
  flagging real (one-shared-paper) coauthors as phantom.

- The ``ac`` per-node annotation written by 03b_annotate_all_coauthors.py
  *does* include 1-collab pairs. So we can rebuild the BFS graph from
  edges ∪ ac and get the correct distance label without re-running SPECTER2
  embeddings, kNN, or Leiden.

The script reads ``coauthor_network.json``, builds the unthresholded
adjacency, BFS each node up to ``BFS_CUTOFF`` hops (mirrors the constant
in 06_author_similarity.py), and rewrites every entry's ``d`` and
``phantom`` fields in ``author_similar.json``. Idempotent. Updates both
``data/processed/`` and ``site/data/`` copies.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path


def _load_constants(repo: Path) -> tuple[int, int]:
    """Mirror PHANTOM_MIN_HOPS and BFS_CUTOFF from 06_author_similarity.py."""
    spec = importlib.util.spec_from_file_location(
        "_six_author_similarity", repo / "scripts" / "06_author_similarity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.PHANTOM_MIN_HOPS), int(mod.BFS_CUTOFF)


def _bfs_distances(repo: Path, cutoff: int) -> dict[int, dict[int, int]]:
    """BFS from every node on the unthresholded coauthor graph (edges ∪ ac)."""
    import networkx as nx
    cn_path = repo / "data" / "processed" / "coauthor_network.json"
    cn = json.loads(cn_path.read_text())
    nodes = cn["nodes"]
    edges = cn["edges"]
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"])
    for e in edges:
        G.add_edge(e["source"], e["target"])
    n_edges_thresh = G.number_of_edges()
    n_with_ac = 0
    for n in nodes:
        ac = n.get("ac") or []
        if ac:
            n_with_ac += 1
        nid = n["id"]
        for cid, _cnt in ac:
            G.add_edge(nid, cid)
    print(f"[reflag] graph: {G.number_of_nodes():,} nodes, "
          f"thresholded edges={n_edges_thresh:,}, "
          f"unthresholded edges={G.number_of_edges():,}, "
          f"nodes with ac={n_with_ac:,}", flush=True)
    if n_with_ac == 0:
        print("[reflag] WARNING: no ac fields present — run "
              "scripts/03b_annotate_all_coauthors.py first; falling back "
              "to thresholded BFS (will not fix d=1 vs d=2 for 1-collab "
              "pairs)", file=sys.stderr)
    t0 = time.time()
    dist_of: dict[int, dict[int, int]] = {}
    for nid in G.nodes():
        dist_of[nid] = nx.single_source_shortest_path_length(G, nid, cutoff=cutoff)
    print(f"[reflag] BFS done in {time.time() - t0:.1f}s", flush=True)
    return dist_of


def _rewrite(path: Path, dist_of: dict[int, dict[int, int]],
             min_hops: int) -> tuple[int, int, int]:
    data = json.loads(path.read_text())
    total = 0
    flipped_phantom = 0
    changed_d = 0
    for src_str, entries in data.items():
        try:
            src_nid = int(src_str)
        except ValueError:
            continue
        src_dist = dist_of.get(src_nid, {})
        for s in entries:
            old_d = s.get("d")
            old_p = bool(s.get("phantom"))
            new_d = src_dist.get(s["id"])  # int or None
            new_p = (new_d is None) or (new_d >= min_hops)
            if old_d != new_d:
                changed_d += 1
            if old_p != new_p:
                flipped_phantom += 1
            s["d"] = new_d
            s["phantom"] = new_p
            total += 1
    path.write_text(json.dumps(data, allow_nan=False))
    return total, changed_d, flipped_phantom


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    try:
        min_hops, cutoff = _load_constants(repo)
    except Exception as exc:  # pragma: no cover
        print(f"[reflag] could not load constants ({exc!r}); "
              f"falling back to PHANTOM_MIN_HOPS=2, BFS_CUTOFF=3",
              file=sys.stderr)
        min_hops, cutoff = 2, 3
    print(f"[reflag] PHANTOM_MIN_HOPS={min_hops}, BFS_CUTOFF={cutoff}", flush=True)

    dist_of = _bfs_distances(repo, cutoff)

    targets = [
        repo / "data" / "processed" / "author_similar.json",
        repo / "site" / "data" / "author_similar.json",
    ]
    for p in targets:
        if not p.exists():
            print(f"[reflag] skip (not found): {p}")
            continue
        total, changed_d, flipped_p = _rewrite(p, dist_of, min_hops)
        print(f"[reflag] {p}: d changed={changed_d:,}/{total:,}, "
              f"phantom flipped={flipped_p:,}/{total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
