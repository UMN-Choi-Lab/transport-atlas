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


# Okabe-Ito + Tol muted — 12 distinguishable hues for the top-12 communities.
_COMM_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#332288",
    "#117733", "#88CCEE", "#882255", "#44AA99",
]


def _comm_color(rank: int) -> tuple[float, float, float]:
    """Ordinal colour for a community's size rank (0 = largest)."""
    from matplotlib.colors import to_rgb
    return to_rgb(_COMM_PALETTE[rank % len(_COMM_PALETTE)])


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

    # Community colors: only color the top-12 largest NON-MISC communities
    # with a Okabe-Ito + Tol palette. The misc bucket (id 200 at this scale)
    # is the island-orbit sink that ForceAtlas2 places on a far outer ring,
    # and colouring it would put a giant ring of orange dots around the
    # actual mainland. Colouring only mainland communities leaves the
    # visible structure intact.
    misc_ids = {c["id"] for c in net["meta"]["communities"] if c.get("misc")}
    from collections import Counter
    comm_size = Counter(comms.tolist())
    ordered = [c for c, _ in comm_size.most_common() if c not in misc_ids]
    top_comms = ordered[:12]
    comm_rank = {c: i for i, c in enumerate(top_comms)}
    node_colors = np.array([
        _comm_color(comm_rank[c]) if c in comm_rank else (0.78, 0.78, 0.78)
        for c in comms
    ])

    # Restrict both sub-figures to the LCC so (a) and (b) share an
    # identical background (same node positions, same edges, same crop,
    # same colouring) and the only visible difference between the two
    # panels is the label layer (community ids vs. top-degree authors).
    import networkx as nx
    G = nx.Graph()
    for i, _ in enumerate(nodes):
        G.add_node(i)
    for e in edges:
        if e["source"] in ids and e["target"] in ids:
            G.add_edge(ids[e["source"]], ids[e["target"]])
    lcc_idx = max(nx.connected_components(G), key=len)
    lcc_mask = np.array([i in lcc_idx for i in range(len(nodes))])
    print(f"[atlas-fig] LCC has {int(lcc_mask.sum()):,} nodes "
          f"(of {len(nodes):,})")

    # Edge coordinates — rasterize to keep the PDF small.
    lcc_edge_xy = np.array([
        [[xs[a], ys[a]], [xs[b], ys[b]]]
        for a, b in G.subgraph(lcc_idx).edges()
    ], dtype=np.float32)

    # Crop rule shared by both panels: 2--98 percentile of the coloured
    # top-12 LCC communities, so the layout orbits don't squeeze the
    # visible structure. Computed once and reused on both axes.
    core_lcc_mask = lcc_mask & np.array([c in comm_rank for c in comms])
    if core_lcc_mask.any():
        pad = 0.06
        x_lo_d, x_hi_d = np.percentile(xs[core_lcc_mask], [2, 98])
        y_lo_d, y_hi_d = np.percentile(ys[core_lcc_mask], [2, 98])
        x_range = x_hi_d - x_lo_d
        y_range = y_hi_d - y_lo_d
        xlim = (x_lo_d - pad * x_range, x_hi_d + pad * x_range)
        ylim = (y_lo_d - pad * y_range, y_hi_d + pad * y_range)
    else:
        xlim = ylim = None

    def _draw_background(ax):
        lc = LineCollection(lcc_edge_xy, colors=(0.35, 0.35, 0.35, 0.06),
                            linewidths=0.22, rasterized=True)
        ax.add_collection(lc)
        # Node size: wider dynamic range so prolific hubs visually
        # dominate, following the style of Sun & Rahwan Fig 3.
        sizes_lcc = 2.0 + 4.5 * np.sqrt(papers[lcc_mask].clip(min=1))
        ax.scatter(xs[lcc_mask], ys[lcc_mask],
                   s=sizes_lcc, c=node_colors[lcc_mask],
                   linewidths=0, alpha=0.88, rasterized=True)
        if xlim is not None:
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig, ax = plt.subplots(figsize=(9.0, 8.2))
    _draw_background(ax)

    # In-plot community labels: start at each community's degree-weighted
    # centroid, then push outward along the radial direction so labels fan
    # away from the densest spot rather than stacking on the same pixels.
    # Then run a simple iterative repulsion to resolve any remaining
    # overlaps between label bounding boxes.
    labels_by_id = {c["id"]: (c.get("label_words") or ["?"])[:2]
                    for c in net["meta"]["communities"]}
    weights_core = degrees[core_lcc_mask].clip(min=1).astype(float) if core_lcc_mask.any() else None
    cx_core = float(np.average(xs[core_lcc_mask], weights=weights_core)) if core_lcc_mask.any() else 0.0
    cy_core = float(np.average(ys[core_lcc_mask], weights=weights_core)) if core_lcc_mask.any() else 0.0

    anchor_xy: list[tuple[float, float]] = []
    label_xy: list[list[float]] = []
    label_texts: list[str] = []
    label_ranks: list[int] = []
    for rank, cid in enumerate(top_comms):
        mask = (comms == cid) & lcc_mask
        if not mask.any():
            continue
        w = degrees[mask].clip(min=1).astype(float)
        cx = float(np.average(xs[mask], weights=w))
        cy = float(np.average(ys[mask], weights=w))
        bits = labels_by_id.get(cid, ["?"])
        txt = " / ".join(bits[:2])
        anchor_xy.append((cx, cy))
        label_xy.append([cx + 0.35 * (cx - cx_core),
                         cy + 0.35 * (cy - cy_core)])
        label_texts.append(f"C{cid}\n{txt}")
        label_ranks.append(rank)

    # Half-width / half-height of each label box in data units. Approximate
    # using the longest text line and the plot's current extent.
    x_lo_px, x_hi_px = ax.get_xlim()
    y_lo_px, y_hi_px = ax.get_ylim()
    x_span = x_hi_px - x_lo_px
    y_span = y_hi_px - y_lo_px
    box_half_w = []
    box_half_h = []
    for t in label_texts:
        longest_line = max(len(ln) for ln in t.split("\n"))
        n_lines = len(t.split("\n"))
        # 8pt bold text in a 9x8.2 inch figure: ~0.067" per char wide,
        # ~0.13" per line tall. Add padding for the rounded bbox.
        char_w_in = 0.072
        line_h_in = 0.14
        width_in = longest_line * char_w_in + 0.14   # padding
        height_in = n_lines * line_h_in + 0.12
        box_half_w.append(0.5 * width_in / 9.0 * x_span)
        box_half_h.append(0.5 * height_in / 8.2 * y_span)

    # Iterative repulsion: push overlapping label boxes apart.
    for _ in range(60):
        moved = False
        for i in range(len(label_xy)):
            for j in range(i + 1, len(label_xy)):
                dx = label_xy[j][0] - label_xy[i][0]
                dy = label_xy[j][1] - label_xy[i][1]
                min_dx = (box_half_w[i] + box_half_w[j]) * 1.08
                min_dy = (box_half_h[i] + box_half_h[j]) * 1.25
                overlap_x = min_dx - abs(dx)
                overlap_y = min_dy - abs(dy)
                if overlap_x > 0 and overlap_y > 0:
                    # Push apart along the cheaper axis
                    if overlap_x < overlap_y:
                        push = overlap_x * 0.55
                        sign = 1.0 if dx >= 0 else -1.0
                        label_xy[i][0] -= sign * push
                        label_xy[j][0] += sign * push
                    else:
                        push = overlap_y * 0.55
                        sign = 1.0 if dy >= 0 else -1.0
                        label_xy[i][1] -= sign * push
                        label_xy[j][1] += sign * push
                    moved = True
        if not moved:
            break

    for (anchor, (lx, ly), text, rank) in zip(
            anchor_xy, label_xy, label_texts, label_ranks):
        ax.annotate(
            text,
            xy=anchor,
            xytext=(lx, ly),
            ha="center", va="center", fontsize=8.0, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.9,
                      edgecolor=_comm_color(rank), linewidth=1.1),
            arrowprops=dict(arrowstyle="-", color=_comm_color(rank),
                            lw=0.8, alpha=0.85),
            zorder=5,
        )
    _save(fig, "05_coauthor_overview")
    print("  ✓ figures/05_coauthor_overview.pdf")

    # ------------------------------------------------------------------
    # Fig 5b — same LCC background as Fig 5a; only the label layer
    # changes (community C0--C11 -> top-degree author names). Using the
    # identical _draw_background helper guarantees pixel-for-pixel
    # matching node positions, edges, colours, and crop between the two
    # sub-figures, which the side-by-side caption in §5 depends on.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 8.2))
    _draw_background(ax)

    # Label ~40 authors, but select the top-K per top-community so the
    # labels are spread over the plot rather than piled up in the single
    # densest community. This matches Sun & Rahwan Fig 3 where each
    # colored region carries a few named anchors.
    LABELS_PER_COMM = 2       # top 2 by degree per colored community
    EXTRA_TOP = 4             # plus the 4 highest-degree authors overall
    per_comm: dict[int, list[tuple[int, int]]] = {c: [] for c in top_comms}
    for i in range(len(nodes)):
        if not lcc_mask[i]:
            continue
        c = int(comms[i])
        if c in per_comm:
            per_comm[c].append((i, int(degrees[i])))
    to_label_set: set[int] = set()
    for c, rows in per_comm.items():
        rows.sort(key=lambda p: -p[1])
        for idx, _ in rows[:LABELS_PER_COMM]:
            to_label_set.add(idx)
    lcc_author_degrees = sorted(
        ((i, int(degrees[i])) for i in range(len(nodes)) if lcc_mask[i]),
        key=lambda p: -p[1],
    )
    for idx, _ in lcc_author_degrees[:EXTRA_TOP]:
        to_label_set.add(idx)
    to_label = sorted(to_label_set, key=lambda i: -int(degrees[i]))
    min_d = int(degrees[to_label[-1]]) if to_label else 0
    print(f"[atlas-fig] labeling {len(to_label)} LCC authors across "
          f"{len(top_comms)} communities (min d = {min_d})")

    # Highlight labeled nodes with a larger outlined marker so they stand
    # out from the cloud.
    hl_x = xs[to_label]; hl_y = ys[to_label]
    hl_sizes = 18 + 14 * np.sqrt(papers[to_label].clip(min=1))
    ax.scatter(hl_x, hl_y, s=hl_sizes, facecolor="none",
               edgecolors="black", linewidths=0.8, alpha=0.95,
               rasterized=True, zorder=4)

    # Radial offset: place each label outward from the plot centroid so
    # labels fan away from the dense mass instead of piling on top of one
    # another. Magnitude scales with the local density.
    cx_all = float(np.mean(xs[lcc_mask]))
    cy_all = float(np.mean(ys[lcc_mask]))
    for i in to_label:
        name = labels[i] or ""
        if "," in name:
            last, rest = name.split(",", 1)
            pretty = (last.strip().title() + ", " +
                      " ".join(p.strip().title() for p in rest.split()))
        else:
            pretty = name.title()
        dx = float(xs[i]) - cx_all
        dy = float(ys[i]) - cy_all
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        # Larger radial offset so labels fan out of the dense centre
        off_x = 34.0 * dx / norm
        off_y = 34.0 * dy / norm
        ha = "left" if off_x >= 0 else "right"
        va = "bottom" if off_y >= 0 else "top"
        ax.annotate(
            pretty, (xs[i], ys[i]),
            fontsize=7.2, fontweight="bold", color="black",
            ha=ha, va=va,
            textcoords="offset points", xytext=(off_x, off_y),
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      alpha=0.92, edgecolor="#333", linewidth=0.4),
            arrowprops=dict(arrowstyle="-", color="#555", lw=0.5,
                            shrinkA=0, shrinkB=2),
            zorder=6,
        )
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
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

    # Top-12 non-misc semantic communities get distinct Okabe-Ito+Tol
    # colors; the long tail goes pale grey so the coloured subfields
    # stand out. Mirrors the coauthor-overview palette for consistency.
    sc_misc_ids = {m["id"] for m in sc_meta if m.get("misc")}
    sorted_scs = [m for m in sorted(sc_meta, key=lambda m: -m["size"])
                  if m["id"] not in sc_misc_ids][:12]
    top_ids = [m["id"] for m in sorted_scs]
    sc_rank = {sid: i for i, sid in enumerate(top_ids)}
    labels_by_sc = {m["id"]: (m.get("label_words") or ["?"])[:2]
                    for m in sc_meta}
    colors = np.array([
        _comm_color(sc_rank[int(s)]) if int(s) in sc_rank
        else (0.78, 0.78, 0.78)
        for s in scs
    ])

    fig, ax = plt.subplots(figsize=(9.0, 7.8))
    sizes = 2.5 + 2.0 * np.sqrt(papers.clip(min=1))
    ax.scatter(xs, ys, s=sizes, c=colors, linewidths=0, alpha=0.65,
               rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Crop to the 2-98 percentile of coloured-community coordinates so
    # sparse UMAP outliers don't squeeze the main mass into a small blob.
    core_pts = np.array([int(s) in sc_rank for s in scs])
    if core_pts.any():
        pad = 0.06
        x_lo, x_hi = np.percentile(xs[core_pts], [2, 98])
        y_lo, y_hi = np.percentile(ys[core_pts], [2, 98])
        x_range = x_hi - x_lo
        y_range = y_hi - y_lo
        ax.set_xlim(x_lo - pad * x_range, x_hi + pad * x_range)
        ax.set_ylim(y_lo - pad * y_range, y_hi + pad * y_range)

    # In-plot community labels at the size-weighted centroid of each
    # coloured community, with iterative repulsion so boxes don't
    # overlap. Mirrors the coauthor-overview treatment.
    x_lo_px, x_hi_px = ax.get_xlim()
    y_lo_px, y_hi_px = ax.get_ylim()
    x_span = x_hi_px - x_lo_px
    y_span = y_hi_px - y_lo_px
    cx_core = float(np.mean(xs[core_pts])) if core_pts.any() else 0.0
    cy_core = float(np.mean(ys[core_pts])) if core_pts.any() else 0.0

    label_texts: list[str] = []
    anchor_xy: list[tuple[float, float]] = []
    label_xy: list[list[float]] = []
    label_ranks: list[int] = []
    # Minimum radial offset in data units: ensures that labels for
    # communities whose centroid is close to the plot centre still start
    # with a visible outward push rather than piling up on the middle.
    min_radial = 0.08 * min(x_span, y_span)
    for rank, sid in enumerate(top_ids):
        mask = scs == sid
        if not mask.any():
            continue
        w = papers[mask].astype(float).clip(min=1)
        cx = float(np.average(xs[mask], weights=w))
        cy = float(np.average(ys[mask], weights=w))
        bits = labels_by_sc.get(sid, ["?"])
        txt = " / ".join(bits[:2])
        anchor_xy.append((cx, cy))
        # Initial radial offset: 40 % of the community-to-centre vector,
        # clamped to at least min_radial so near-centre labels still push
        # outward. The Fig 2 coauthor overview uses 35 %; topic space
        # has ~22 communities (vs 12), so a larger offset + stronger
        # repulsion is needed to avoid overlap.
        dx_c = cx - cx_core
        dy_c = cy - cy_core
        r = (dx_c * dx_c + dy_c * dy_c) ** 0.5 or 1.0
        radial = max(0.40 * r, min_radial)
        label_xy.append([cx + radial * dx_c / r,
                         cy + radial * dy_c / r])
        label_texts.append(f"S{sid}\n{txt}")
        label_ranks.append(rank)

    box_half_w = []
    box_half_h = []
    for t in label_texts:
        longest_line = max(len(ln) for ln in t.split("\n"))
        n_lines = len(t.split("\n"))
        char_w_in = 0.068
        line_h_in = 0.13
        width_in = longest_line * char_w_in + 0.14
        height_in = n_lines * line_h_in + 0.12
        box_half_w.append(0.5 * width_in / 9.0 * x_span)
        box_half_h.append(0.5 * height_in / 7.8 * y_span)

    # More aggressive padding (1.20 horizontal, 1.45 vertical) and
    # twice as many iterations than the coauthor overview, because the
    # topic-space cluster is denser (22 labels in the same plot area).
    for _ in range(150):
        moved = False
        for i in range(len(label_xy)):
            for j in range(i + 1, len(label_xy)):
                dx = label_xy[j][0] - label_xy[i][0]
                dy = label_xy[j][1] - label_xy[i][1]
                min_dx = (box_half_w[i] + box_half_w[j]) * 1.20
                min_dy = (box_half_h[i] + box_half_h[j]) * 1.45
                overlap_x = min_dx - abs(dx)
                overlap_y = min_dy - abs(dy)
                if overlap_x > 0 and overlap_y > 0:
                    if overlap_x < overlap_y:
                        push = overlap_x * 0.6
                        sign = 1.0 if dx >= 0 else -1.0
                        label_xy[i][0] -= sign * push
                        label_xy[j][0] += sign * push
                    else:
                        push = overlap_y * 0.6
                        sign = 1.0 if dy >= 0 else -1.0
                        label_xy[i][1] -= sign * push
                        label_xy[j][1] += sign * push
                    moved = True
        if not moved:
            break

    for (anchor, (lx, ly), text, rank) in zip(
            anchor_xy, label_xy, label_texts, label_ranks):
        ax.annotate(
            text,
            xy=anchor,
            xytext=(lx, ly),
            ha="center", va="center", fontsize=7.4, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      alpha=0.92,
                      edgecolor=_comm_color(rank), linewidth=1.0),
            arrowprops=dict(arrowstyle="-", color=_comm_color(rank),
                            lw=0.7, alpha=0.85),
            zorder=5,
        )
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
