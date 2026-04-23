#!/usr/bin/env python
"""Render static snapshots of the coauthor network and topic space.

Produces paper-ready figures from the already-computed layouts on disk.
The live atlas views (coauthor_network.html, topic_space.html) use JS
renderers; for the PDF paper we redraw the same underlying coordinates
with matplotlib.

Inputs:
    data/processed/coauthor_network.json      (nodes with x, y, community, d)
    data/processed/topic_coords.json          (author nid -> x, y, sc)
    data/processed/semantic_communities.json  (sc id -> label words)

Outputs:
    figures/04_coauthor_overview.pdf  (+ .png)
    figures/06_topic_space.pdf        (+ .png)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.prop_cycle": plt.cycler("color", OKABE_ITO),
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper" / "manuscript" / "figures"


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    plt.close(fig)


def _golden_hsl(cid: int) -> tuple[float, float, float]:
    """Deterministic HSL → RGB for community ids, matching the web atlas."""
    import colorsys
    hue = ((cid * 137.508) % 360) / 360.0
    sat = 0.70 if cid % 2 == 0 else 0.55
    light = 0.58 if cid % 3 == 0 else 0.50
    return colorsys.hls_to_rgb(hue, light, sat)


def render_coauthor_overview() -> None:
    net = json.loads((ROOT / "data" / "processed" / "coauthor_network.json")
                     .read_text())
    nodes = net["nodes"]
    edges = net["edges"]
    print(f"[atlas-fig] coauthor network: {len(nodes):,} nodes, "
          f"{len(edges):,} edges")

    xs = np.array([n["x"] for n in nodes], dtype=np.float32)
    ys = np.array([n["y"] for n in nodes], dtype=np.float32)
    ids = {n["id"]: i for i, n in enumerate(nodes)}
    comms = np.array([int(n.get("community") or 0) for n in nodes])
    papers = np.array([int(n.get("papers", 0) or 0) for n in nodes])
    degrees = np.array([int(n.get("d", 0) or 0) for n in nodes])
    labels = [n.get("label", "") for n in nodes]

    # Community colors: only color the top-12 largest communities; collapse
    # the long tail to a neutral grey so the plot stays readable.
    from collections import Counter
    comm_size = Counter(comms.tolist())
    top_comms = [c for c, _ in comm_size.most_common(12)]
    top_set = set(top_comms)
    node_colors = np.array([
        _golden_hsl(c) if c in top_set else (0.70, 0.70, 0.70)
        for c in comms
    ])

    # Edge coordinates — rasterize to keep the PDF small.
    edge_xy = np.array([
        [[xs[ids[e["source"]]], ys[ids[e["source"]]]],
         [xs[ids[e["target"]]], ys[ids[e["target"]]]]]
        for e in edges
        if e["source"] in ids and e["target"] in ids
    ], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    lc = LineCollection(edge_xy, colors=(0.4, 0.4, 0.4, 0.05),
                        linewidths=0.2, rasterized=True)
    ax.add_collection(lc)
    # Node size ∝ sqrt(papers) so prolific hubs are visible, but the cloud
    # of low-paper-count nodes stays small.
    sizes = 0.6 + 1.6 * np.sqrt(papers.clip(min=1))
    ax.scatter(xs, ys, s=sizes, c=node_colors, linewidths=0,
               alpha=0.85, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"Coauthor network — {len(nodes):,} authors, "
        f"{len(edges):,} edges; top-12 Leiden communities colored",
        fontsize=10,
    )

    # Legend for top-12 communities
    labels_by_id = {c["id"]: (c.get("label_words") or ["?"])[:2]
                    for c in net["meta"]["communities"]}
    legend_handles = []
    for cid in top_comms:
        label_bits = labels_by_id.get(cid, ["?"])
        label = f"C{cid}: " + " / ".join(label_bits[:2])
        if len(label) > 34:
            label = label[:32] + "…"
        legend_handles.append(
            plt.scatter([], [], s=30, c=[_golden_hsl(cid)], label=label)
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=6.2,
              frameon=True, ncol=2, borderpad=0.4, labelspacing=0.3)
    _save(fig, "05_coauthor_overview")
    print("  ✓ figures/05_coauthor_overview.pdf")

    # ------------------------------------------------------------------
    # Fig 5b — top-degree-labeled LCC (analog of Sun & Rahwan Fig 3).
    # Show only giant connected component and annotate authors with
    # degree >= LABEL_DEGREE (matches S&R's d >= 40 threshold at scale).
    # ------------------------------------------------------------------
    import networkx as nx
    G = nx.Graph()
    for i, _ in enumerate(nodes): G.add_node(i)
    for e in edges:
        if e["source"] in ids and e["target"] in ids:
            G.add_edge(ids[e["source"]], ids[e["target"]])
    lcc_idx = max(nx.connected_components(G), key=len)
    lcc_mask = np.array([i in lcc_idx for i in range(len(nodes))])
    print(f"[atlas-fig] LCC has {int(lcc_mask.sum()):,} nodes "
          f"(of {len(nodes):,})")

    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    lc_edges = [
        [[xs[a], ys[a]], [xs[b], ys[b]]]
        for a, b in G.subgraph(lcc_idx).edges()
    ]
    lc = LineCollection(lc_edges, colors=(0.4, 0.4, 0.4, 0.05),
                        linewidths=0.2, rasterized=True)
    ax.add_collection(lc)
    ax.scatter(xs[lcc_mask], ys[lcc_mask],
               s=(0.6 + 1.6 * np.sqrt(papers[lcc_mask].clip(min=1))),
               c=node_colors[lcc_mask], linewidths=0, alpha=0.85,
               rasterized=True)

    # Label authors with degree >= LABEL_DEGREE
    LABEL_DEGREE = 40
    to_label = [i for i in range(len(nodes))
                if lcc_mask[i] and degrees[i] >= LABEL_DEGREE]
    print(f"[atlas-fig] labeling {len(to_label)} authors with d >= "
          f"{LABEL_DEGREE}")
    for i in to_label:
        name = labels[i] or ""
        if "," in name:
            last, rest = name.split(",", 1)
            pretty = (last.strip().title() + ", " +
                      " ".join(p.strip().title() for p in rest.split()))
        else:
            pretty = name.title()
        ax.annotate(pretty, (xs[i], ys[i]), fontsize=5.5, alpha=0.85,
                    color="black",
                    textcoords="offset points", xytext=(2, 2))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"Largest connected component of the coauthor network "
        f"({int(lcc_mask.sum()):,} authors); "
        f"names shown for $d \\geq {LABEL_DEGREE}$",
        fontsize=9.5,
    )
    _save(fig, "05_lcc_labeled")
    print("  ✓ figures/05_lcc_labeled.pdf")


def render_topic_space() -> None:
    tc = json.loads((ROOT / "data" / "processed" / "topic_coords.json")
                    .read_text())
    sc_meta = json.loads((ROOT / "data" / "processed"
                          / "semantic_communities.json").read_text())
    print(f"[atlas-fig] topic space: {len(tc):,} author points")

    xs, ys, scs, papers = [], [], [], []
    for v in tc.values():
        xs.append(v["x"])
        ys.append(v["y"])
        scs.append(v.get("sc"))
        papers.append(int(v.get("p", 1) or 1))
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    scs = np.array([s if s is not None else -1 for s in scs])
    papers = np.array(papers)

    # Top-15 semantic communities get distinct colors, rest go grey.
    sorted_scs = sorted(sc_meta, key=lambda m: -m["size"])[:15]
    top_ids = [m["id"] for m in sorted_scs]
    top_set = set(top_ids)
    labels_by_sc = {m["id"]: (m.get("label_words") or ["?"])[:2]
                    for m in sc_meta}
    colors = np.array([
        _golden_hsl(int(s) + 7) if int(s) in top_set else (0.75, 0.75, 0.75)
        for s in scs
    ])

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    sizes = 2.0 + 1.5 * np.sqrt(papers.clip(min=1))
    ax.scatter(xs, ys, s=sizes, c=colors, linewidths=0, alpha=0.55,
               rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"Semantic topic space — {len(tc):,} authors in UMAP-2D; "
        f"top-15 semantic Leiden communities colored",
        fontsize=10,
    )

    legend_handles = []
    for sc_id in top_ids:
        bits = labels_by_sc.get(sc_id, ["?"])
        label = f"S{sc_id}: " + " / ".join(bits[:2])
        if len(label) > 34:
            label = label[:32] + "…"
        legend_handles.append(
            plt.scatter([], [], s=30, c=[_golden_hsl(int(sc_id) + 7)],
                        label=label)
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=6.2,
              frameon=True, ncol=2, borderpad=0.4, labelspacing=0.3)
    _save(fig, "06_topic_space")
    print("  ✓ figures/06_topic_space.pdf")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    render_coauthor_overview()
    render_topic_space()
    print("[atlas-fig] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
