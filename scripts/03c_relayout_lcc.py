#!/usr/bin/env python
"""Rerun ForceAtlas2 on the LCC only and overwrite its node positions.

The default 03_graph.py pipeline runs FA2 on the whole base graph
(all authors with >=2 papers, all edges with >=2 collaborations). When a
small fraction of the graph is disconnected from the LCC, FA2 places the
small components on outer orbits and stretches the LCC into a spiked
shape. This post-step restricts the layout to the LCC only so the
figure matches the clean disc that Sun & Rahwan (2017) Fig. 3 shows.

Edits data/processed/coauthor_network.json in place (x/y for LCC nodes).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
NET_PATH = ROOT / "data" / "processed" / "coauthor_network.json"
PIPELINE_CFG = ROOT / "config" / "pipeline.yaml"


def _load_cfg() -> dict:
    import yaml
    return yaml.safe_load(PIPELINE_CFG.read_text())


def main() -> int:
    print(f"[lcc-layout] loading {NET_PATH.name}")
    net = json.loads(NET_PATH.read_text())
    nodes = net["nodes"]
    edges = net["edges"]
    print(f"[lcc-layout]   {len(nodes):,} nodes, {len(edges):,} edges")

    G = nx.Graph()
    id_to_key = {}
    for i, n in enumerate(nodes):
        nid = n.get("id", i)
        key = n.get("key") or n.get("label") or str(nid)
        G.add_node(nid)
        id_to_key[nid] = key
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if s in id_to_key and t in id_to_key:
            G.add_edge(s, t, weight=float(e.get("weight", 1.0)))

    lcc_nodes = max(nx.connected_components(G), key=len)
    Gl = G.subgraph(lcc_nodes).copy()
    print(f"[lcc-layout]   LCC: {Gl.number_of_nodes():,} nodes, "
          f"{Gl.number_of_edges():,} edges")

    cfg = _load_cfg()
    pipeline = cfg.get("pipeline", {})
    iters = int(pipeline.get("fa2_iterations", 500))
    scaling = float(pipeline.get("fa2_scaling_ratio", 2.0))
    print(f"[lcc-layout]   FA2: {iters} iters, scaling={scaling}")

    from fa2_modified import ForceAtlas2
    # LinLog mode + gentle gravity gives the round S&R-style hairball:
    # LinLog damps the quadratic repulsion so dense central cores don't
    # explode outward, while non-strong gravity retracts isolated leaves.
    # Stay with default gravity: strong gravity (even at 3) collapses
    # cluster structure into a uniform mush on a 28k-node LCC. The
    # original parameters give a shape closer to Sun & Rahwan Fig 3 even
    # if it isn't a perfectly circular disc at this scale.
    fa = ForceAtlas2(
        outboundAttractionDistribution=False,
        linLogMode=False,
        adjustSizes=False,
        edgeWeightInfluence=1.0,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        scalingRatio=scaling,
        strongGravityMode=False,
        gravity=1.0,
        verbose=False,
    )
    t0 = time.perf_counter()
    pos = fa.forceatlas2_networkx_layout(Gl, pos=None, iterations=iters)
    print(f"[lcc-layout]   FA2 wall-clock: {time.perf_counter() - t0:.1f}s")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    # Rescale to the same [-900, 900] box the pipeline uses.
    scaled = {n: (1800 * (p[0] - x0) / dx - 900,
                  1800 * (p[1] - y0) / dy - 900)
              for n, p in pos.items()}

    # Write back into the JSON for LCC nodes only; leave non-LCC x/y alone
    # so downstream figures that still want to see islands can opt in.
    touched = 0
    for n in nodes:
        nid = n.get("id")
        if nid in scaled:
            n["x"] = round(scaled[nid][0], 2)
            n["y"] = round(scaled[nid][1], 2)
            touched += 1
    print(f"[lcc-layout]   updated x/y for {touched:,} LCC nodes")

    NET_PATH.write_text(json.dumps(net, allow_nan=False))
    print(f"[lcc-layout] done -> {NET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
