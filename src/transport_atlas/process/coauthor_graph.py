"""Build coauthor graph + Leiden communities + TF-IDF topic labels.

Modeled after robopaper-atlas/_make_coauthor_network.py + _enrich_communities.py:
  - Giant-threshold graph (strong ties, weight>=5) for Leiden partition
  - Base-threshold graph (weight>=2) for propagation + island detection
  - Labels spread outward by weighted neighbor vote
  - Components >= ISLAND_MIN_SIZE get their own community; smaller → misc
  - HSL golden-angle palette, communities renumbered by size desc
  - TF-IDF keyword extraction over per-community paper title bags

Output: data/processed/coauthor_network.json (nodes, edges, meta.communities).
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ..utils import config
from ..utils.logger import get_logger
from .authors import author_key as _raw_author_key


def _alias_map() -> dict[str, str]:
    """Build {source_openalex_key: canonical_openalex_key}.

    Combines manual aliases from pipeline.yaml with auto-detected ORCID splits
    that dedupe.py persists to data/interim/author_aliases_auto.json.
    """
    import json as _json
    pipe = config.load_pipeline()
    aliases = pipe.get("author_aliases") or []
    mp = {}
    for a in aliases:
        ids = a.get("openalex_ids") or []
        if len(ids) < 2:
            continue
        target = ids[0].lower()
        for other in ids[1:]:
            mp[other.lower()] = target
    auto_path = config.data_dir("interim") / "author_aliases_auto.json"
    if auto_path.exists():
        auto_map = _json.loads(auto_path.read_text())
        for k, v in auto_map.items():
            mp.setdefault(k, v)  # manual entries take precedence
    return mp


_ALIAS_MAP = None

def author_key(a: dict) -> str:
    global _ALIAS_MAP
    if _ALIAS_MAP is None:
        _ALIAS_MAP = _alias_map()
    k = _raw_author_key(a)
    return _ALIAS_MAP.get(k, k) if k else k

log = get_logger("graph")

GIANT_THRESHOLD = 5         # edge-weight cutoff for Leiden (strong ties)
BASE_THRESHOLD = 2          # edge-weight cutoff for the exported graph
ISLAND_MIN_SIZE = 10        # smaller disconnected islands roll into "misc"
TOPIC_WORDS_PER_COMMUNITY = 8
TOP_AUTHORS_PER_COMMUNITY = 5
LEIDEN_SEED = 42
MISC_COLOR = "hsl(0, 0%, 45%)"

_STOP = {
    "a", "an", "and", "or", "the", "of", "on", "in", "for", "to", "with",
    "via", "using", "based", "new", "novel", "improved", "approach", "method",
    "methods", "toward", "towards", "from", "by", "at", "is", "are", "be",
    "this", "that", "these", "those", "we", "its", "their", "our", "as", "not",
    "also", "which", "than", "between", "through", "under", "over", "two",
    "three", "four", "one", "paper", "note", "brief", "case", "study",
    "application", "applications", "algorithm", "algorithms", "research",
    "results", "into", "such", "can", "has", "have", "had", "was", "were",
    "been", "being", "do", "does", "did", "done", "real", "online", "offline",
    "performance", "evaluation", "framework", "survey", "review", "experimental",
    "simulation", "experiments", "analysis", "problem", "problems", "work",
    "works", "technique", "techniques", "efficient", "effective", "fast",
    "adaptive", "dynamic", "static", "general", "generalized", "optimal",
    "optimization", "planning", "estimation", "tracking", "sensor", "sensors",
    "data", "time", "high", "low", "non", "multi", "single", "full", "field",
    "task", "tasks", "part", "parts",
    # Transportation-domain stop words (too generic to label a community)
    "transportation", "transport", "road", "roads", "traffic", "travel",
    "vehicle", "vehicles", "driver", "drivers", "driving", "vehicular",
    "system", "systems", "model", "models", "modeling", "modelling",
    "design", "control", "controller", "learning", "prediction", "predict",
    "network", "networks", "networking",
    # IEEE/venue boilerplate that leaks through paper titles
    "ieee", "society", "retracted", "retraction", "conference", "proceedings",
    "proceeding", "inc", "editorial", "editorial:", "international", "intl",
    "workshop", "symposium", "congress", "annual", "chapter",
    # Function words missed the first pass
    "where", "when", "what", "how", "why", "who", "whose", "whom",
    "any", "all", "each", "every", "some", "most", "many", "few", "several",
}


def golden_hsl(cid: int) -> str:
    """HSL golden-angle hue spacing — stable, perceptually varied colors."""
    hue = (cid * 137.508) % 360
    sat = 70 if cid % 2 == 0 else 55
    light = 58 if cid % 3 == 0 else 50
    return f"hsl({hue:.1f}, {sat}%, {light}%)"


def _build_coauthor(papers: pd.DataFrame, authors: pd.DataFrame, min_papers: int):
    """Return (nodes_set, edges_raw[(a,b,w)], per_author_papers, author_max_year,
    top_papers, paper_records, pair_years, pair_newman, cites_multi, cites_all,
    n_single_paper).

    pair_newman uses the Newman (2001b) weight 1/(n_p-1) per shared paper, which
    cancels the bias toward authors on large-team papers (IEEE T-ITS etc.).
    cites_multi / cites_all split citations by multi- vs single-author papers,
    mirroring Sun & Rahwan 2017 Table 3 (last two columns).
    """
    active = set(authors.loc[authors["n_papers"] >= min_papers, "author_key"])
    log.info(f"authors with >= {min_papers} papers: {len(active)}")

    per_author = Counter()
    author_max_year: dict[str, int] = {}
    paper_records: list[tuple[list[str], int, str, int]] = []  # (keys, year, title, cites)
    cites_multi: Counter = Counter()    # multi-author papers only (matches bv/pr scope)
    cites_all: Counter = Counter()      # incl. single-author (Daganzo-style solo work)
    n_single_paper: Counter = Counter() # # of single-authored papers per author

    for _, r in papers.iterrows():
        authors_list = r.get("authors")
        if authors_list is None or len(authors_list) == 0:
            continue
        keys = []
        for a in authors_list:
            if not isinstance(a, dict):
                continue
            k = author_key(a)
            if k and k in active:
                keys.append(k)
        if not keys:
            continue
        year = int(r["year"]) if pd.notna(r.get("year")) else 0
        title = (r.get("title") or "").strip().rstrip(".")
        _c = r.get("cited_by_count")
        cites = 0 if _c is None or (isinstance(_c, float) and _c != _c) else int(_c or 0)
        paper_records.append((keys, year, title, cites))
        per_author.update(keys)
        for k in keys:
            if year > author_max_year.get(k, 0):
                author_max_year[k] = year
            cites_all[k] += cites
        if len(set(keys)) >= 2:
            for k in set(keys):
                cites_multi[k] += cites
        else:
            for k in set(keys):
                n_single_paper[k] += 1

    # Edge weights by pair + list of collaboration years (for time-window filtering).
    # pair_newman accumulates 1/(n-1) per shared paper — Newman (2001b) weighting.
    pair_counts: Counter = Counter()
    pair_newman: dict[tuple[str, str], float] = defaultdict(float)
    pair_years: dict[tuple[str, str], list[int]] = defaultdict(list)
    for keys, year, _, _ in paper_records:
        uniq = sorted(set(keys))
        n = len(uniq)
        if n < 2:
            continue
        inv = 1.0 / (n - 1)
        for a, b in combinations(uniq, 2):
            pair_counts[(a, b)] += 1
            pair_newman[(a, b)] += inv
            if year:
                pair_years[(a, b)].append(int(year))

    edges_raw = [(a, b, c) for (a, b), c in pair_counts.items() if c >= BASE_THRESHOLD]
    log.info(f"edges (>= {BASE_THRESHOLD} collabs): {len(edges_raw):,}")

    nodes_set = {a for e in edges_raw for a in e[:2]}
    log.info(f"connected authors: {len(nodes_set):,}")

    # Top-3 most-cited papers per kept author
    top_papers_map: dict[str, list[dict]] = defaultdict(list)
    for keys, year, title, cites in paper_records:
        if not title:
            continue
        for k in keys:
            if k in nodes_set:
                top_papers_map[k].append({"t": title[:160], "y": year, "c": cites})
    for k, lst in top_papers_map.items():
        lst.sort(key=lambda x: -x["c"])
        top_papers_map[k] = lst[:3]

    return (nodes_set, edges_raw, per_author, author_max_year, top_papers_map,
            paper_records, pair_years, pair_newman, cites_multi, cites_all,
            n_single_paper)


def _leiden_on_mainland(edges_raw, seed: int) -> dict[str, int]:
    """Run Leiden on strong-tie mainland (weight>=GIANT_THRESHOLD)."""
    G_strong = nx.Graph()
    for a, b, w in edges_raw:
        if w >= GIANT_THRESHOLD:
            G_strong.add_edge(a, b, weight=w)
    if G_strong.number_of_nodes() == 0:
        return {}
    mainland = max(nx.connected_components(G_strong), key=len)
    Gi = G_strong.subgraph(mainland).copy()
    log.info(f"mainland (>= {GIANT_THRESHOLD}): {Gi.number_of_nodes():,} nodes, "
             f"{Gi.number_of_edges():,} edges")

    import igraph as ig
    import leidenalg
    ig_nodes = list(Gi.nodes())
    ig_idx = {v: i for i, v in enumerate(ig_nodes)}
    g = ig.Graph(
        n=len(ig_nodes),
        edges=[(ig_idx[u], ig_idx[v]) for u, v in Gi.edges()],
        directed=False,
    )
    g.es["weight"] = [Gi[u][v]["weight"] for u, v in Gi.edges()]
    part = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, weights="weight", seed=seed,
    )
    log.info(f"Leiden: {len(part)} mainland communities (Q={part.modularity:.4f})")
    out = {}
    for cid, members in enumerate(part):
        for i in members:
            out[ig_nodes[i]] = cid
    return out


def _propagate_and_islands(node_to_cid: dict[str, int], edges_raw) -> tuple[dict[str, int], int, int]:
    """Propagate labels along base-threshold edges; detect islands; return (cids, n_island, next_cid)."""
    adj: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for a, b, w in edges_raw:
        adj[a].append((b, w))
        adj[b].append((a, w))

    # Label propagation
    n_before = len(node_to_cid)
    for _ in range(20):
        changed = False
        for nid in list(adj.keys()):
            if nid in node_to_cid:
                continue
            votes: dict[int, int] = defaultdict(int)
            for nb, w in adj[nid]:
                if nb in node_to_cid:
                    votes[node_to_cid[nb]] += w
            if votes:
                node_to_cid[nid] = max(votes.items(), key=lambda kv: kv[1])[0]
                changed = True
        if not changed:
            break
    log.info(f"propagation attached {len(node_to_cid) - n_before:,} fringe nodes")

    # Islands (disconnected from mainland)
    unassigned = {nid for nid in adj if nid not in node_to_cid}
    G_un = nx.Graph()
    G_un.add_nodes_from(unassigned)
    for a, b, _ in edges_raw:
        if a in unassigned and b in unassigned:
            G_un.add_edge(a, b)
    island_comps = sorted(nx.connected_components(G_un), key=len, reverse=True)
    next_cid = (max(node_to_cid.values()) + 1) if node_to_cid else 0
    n_island = 0
    misc: list[str] = []
    for comp in island_comps:
        if len(comp) >= ISLAND_MIN_SIZE:
            for m in comp:
                node_to_cid[m] = next_cid
            next_cid += 1
            n_island += 1
        else:
            misc.extend(comp)
    misc_cid = next_cid if misc else None
    if misc_cid is not None:
        for m in misc:
            node_to_cid[m] = misc_cid
        next_cid += 1
    log.info(f"islands: {n_island} communities + {len(misc)} in misc")
    return node_to_cid, n_island, misc_cid


def _renumber_by_size(node_to_cid: dict[str, int], misc_cid: int | None):
    """Return (new_map, new_misc_cid) with communities sorted by size desc, misc pinned last."""
    sizes = Counter(node_to_cid.values())
    ordered = sorted((c for c in sizes if c != misc_cid), key=lambda c: -sizes[c])
    old_to_new = {old: new for new, old in enumerate(ordered)}
    new_misc = len(ordered) if misc_cid is not None else None
    if misc_cid is not None:
        old_to_new[misc_cid] = new_misc
    return {nid: old_to_new[c] for nid, c in node_to_cid.items()}, new_misc


def _tfidf_labels(comm_members: dict[int, list[str]], paper_records, label_of: dict[str, str],
                  n_comm: int, misc_cid: int | None) -> list[list[str]]:
    """Per-community top-k TF-IDF keywords from paper titles owned by each community."""
    # Build author → list of paper titles they were on
    author_titles: dict[str, list[str]] = defaultdict(list)
    for keys, _y, title, _c in paper_records:
        if not title:
            continue
        t = title.lower()
        for k in keys:
            author_titles[k].append(t)

    docs = [""] * n_comm
    for cid, members in comm_members.items():
        docs[cid] = " ".join(t for m in members for t in author_titles.get(m, []))

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
        max_df=0.5,
        min_df=3,
        stop_words=sorted(_STOP),
        token_pattern=r"[A-Za-z][A-Za-z\-]{2,}",
        ngram_range=(1, 2),
    )
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()

    labels = [[] for _ in range(n_comm)]
    for cid in range(n_comm):
        if cid == misc_cid:
            continue
        row = X[cid].toarray().flatten()
        top_idx = np.argsort(-row)[: TOPIC_WORDS_PER_COMMUNITY * 3]  # over-pull then dedup
        picked = []
        seen_roots = set()
        for i in top_idx:
            if row[i] <= 0:
                break
            w = vocab[i]
            # Skip near-duplicate bigrams ("travel behavior" dominates "travel")
            root = w.split()[0]
            if root in seen_roots and " " not in w:
                continue
            picked.append(w)
            seen_roots.add(root)
            if len(picked) >= TOPIC_WORDS_PER_COMMUNITY:
                break
        labels[cid] = picked
    return labels


def _compute_centralities(G: nx.Graph, pair_newman: dict[tuple[str, str], float],
                          bc_k: int | None, seed: int):
    """Compute per-node centralities + per-edge betweenness.

    Returns (node_metrics, edge_betweenness) where:
      node_metrics[key] = {d, s, s_newman, bc, bc_w, pr, pr_w}
      edge_betweenness[(u,v)] = float (edge betweenness on binary graph)

    `bc_k` — if set, use Brandes sampling with k pivots; else exact.
    """
    log.info(f"computing centralities on {G.number_of_nodes():,}n × {G.number_of_edges():,}e graph"
             + (f" (BC sampled, k={bc_k})" if bc_k else " (BC exact)"))

    # Attach inverse-weight for weighted-betweenness cost. Distance is short
    # when collaboration is frequent (many shared papers).
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        d["inv_weight"] = 1.0 / max(w, 1e-9)

    # Degree (# unique coauthors) and strength (# papers with ≥2 authors on shared edges)
    d = dict(G.degree())
    s = {n: sum(G[n][nb].get("weight", 1) for nb in G[n]) for n in G.nodes()}

    # Newman-weighted strength: sum of 1/(n-1) over incident edges — matches
    # Sun & Rahwan's Eq.(4). For an author i, this equals # of multi-author papers
    # i is on, regardless of paper size.
    s_newman: dict[str, float] = defaultdict(float)
    for (a, b), nw in pair_newman.items():
        if a in G and b in G and G.has_edge(a, b):
            s_newman[a] += nw
            s_newman[b] += nw

    bc_kwargs = {"normalized": True, "weight": None}
    if bc_k:
        bc_kwargs.update({"k": min(bc_k, G.number_of_nodes()), "seed": seed})
    bc = nx.betweenness_centrality(G, **bc_kwargs)
    bc_w = nx.betweenness_centrality(G, **{**bc_kwargs, "weight": "inv_weight"})

    pr = nx.pagerank(G, alpha=0.85, weight=None)
    pr_w = nx.pagerank(G, alpha=0.85, weight="weight")

    eb_kwargs = {"normalized": True, "weight": None}
    if bc_k:
        eb_kwargs.update({"k": min(bc_k, G.number_of_nodes()), "seed": seed})
    eb = nx.edge_betweenness_centrality(G, **eb_kwargs)

    node_metrics = {
        n: {
            "d": int(d.get(n, 0)),
            "s": int(s.get(n, 0)),
            "s_newman": float(s_newman.get(n, 0.0)),
            "bc": float(bc.get(n, 0.0)),
            "bc_w": float(bc_w.get(n, 0.0)),
            "pr": float(pr.get(n, 0.0)),
            "pr_w": float(pr_w.get(n, 0.0)),
        }
        for n in G.nodes()
    }
    return node_metrics, eb


def _rankings_table(node_metrics: dict, author_info: dict, cites_multi: Counter,
                    cites_all: Counter, n_single: Counter, top_k: int = 30) -> dict:
    """Per-metric top-K author tables (mirrors Sun & Rahwan 2017 Tables 3 & 4)."""
    def name(k: str) -> str:
        return (author_info.get(k) or {}).get("canonical_name") or k

    def rank_by(score_fn, label_fmt):
        ranked = sorted(node_metrics.keys(), key=lambda k: -score_fn(k))[:top_k]
        return [{"key": k, "name": name(k), "score": label_fmt(score_fn(k))} for k in ranked]

    return {
        "degree":       rank_by(lambda k: node_metrics[k]["d"],        lambda v: int(v)),
        "strength":     rank_by(lambda k: node_metrics[k]["s"],        lambda v: int(v)),
        "s_newman":     rank_by(lambda k: node_metrics[k]["s_newman"], lambda v: round(v, 3)),
        "citations":    rank_by(lambda k: cites_multi.get(k, 0),       lambda v: int(v)),
        "citations_all":rank_by(lambda k: cites_all.get(k, 0),         lambda v: int(v)),
        "betweenness":  rank_by(lambda k: node_metrics[k]["bc"],       lambda v: round(v * 100, 3)),
        "betweenness_w":rank_by(lambda k: node_metrics[k]["bc_w"],     lambda v: round(v * 100, 3)),
        "pagerank":     rank_by(lambda k: node_metrics[k]["pr"],       lambda v: round(v * 1000, 3)),
        "pagerank_w":   rank_by(lambda k: node_metrics[k]["pr_w"],     lambda v: round(v * 1000, 3)),
    }


def run(*, write: bool = True) -> dict:
    cfg = config.load_pipeline()["graph"]
    seed = config.load_pipeline()["seed"]
    interim = config.data_dir("interim")
    papers = pd.read_parquet(interim / "papers.parquet")
    authors = pd.read_parquet(interim / "authors.parquet")

    (nodes_set, edges_raw, per_author, author_max_year, top_papers, paper_records,
     pair_years, pair_newman, cites_multi, cites_all, n_single_paper) = \
        _build_coauthor(papers, authors, cfg["min_papers_per_author"])

    if not nodes_set:
        log.error("no connected authors — aborting")
        return {"nodes": 0, "edges": 0}

    # Build final graph for layout + metadata (set author names as node attrs)
    author_info = authors.set_index("author_key").to_dict("index")
    G = nx.Graph()
    for a, b, w in edges_raw:
        G.add_edge(a, b, weight=w)
    for n in G.nodes():
        info = author_info.get(n, {})
        G.nodes[n]["name"] = info.get("canonical_name") or n

    # Forceatlas2 layout on whole base graph (seeds simulation in browser)
    try:
        from fa2_modified import ForceAtlas2
        fa = ForceAtlas2(
            outboundAttractionDistribution=False, linLogMode=False,
            adjustSizes=False, edgeWeightInfluence=1.0, jitterTolerance=1.0,
            barnesHutOptimize=True, barnesHutTheta=1.2,
            scalingRatio=cfg["fa2_scaling_ratio"], strongGravityMode=False,
            gravity=1.0, verbose=False,
        )
        pos = fa.forceatlas2_networkx_layout(G, pos=None, iterations=cfg["fa2_iterations"])
    except Exception as e:
        log.warning(f"forceatlas2 failed ({e}); using spring_layout")
        pos = nx.spring_layout(G, seed=seed, iterations=100, weight="weight")

    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) or 1, (y1 - y0) or 1
    pos = {n: (1800 * (p[0] - x0) / dx - 900, 1800 * (p[1] - y0) / dy - 900)
           for n, p in pos.items()}

    # Communities: mainland Leiden → propagate → islands → misc
    mainland = _leiden_on_mainland(edges_raw, LEIDEN_SEED)
    node_to_cid, n_island, misc_cid = _propagate_and_islands(dict(mainland), edges_raw)
    node_to_cid, misc_cid = _renumber_by_size(node_to_cid, misc_cid)
    n_comm = max(node_to_cid.values()) + 1

    # TF-IDF labels + top authors (by weighted degree on base graph)
    comm_members: dict[int, list[str]] = defaultdict(list)
    for nid, cid in node_to_cid.items():
        comm_members[cid].append(nid)
    labels = _tfidf_labels(comm_members, paper_records,
                           {k: k for k in nodes_set}, n_comm, misc_cid)

    wdeg = {n: sum(G[n][nb].get("weight", 1) for nb in G[n]) for n in G.nodes()}

    # Centrality: degree, strength, betweenness (binary + weighted), pagerank, edge-BC.
    # bc_k = None → exact Brandes; int → pivot sampling (useful on very large graphs).
    bc_k = cfg.get("betweenness_k")  # keep None by default for reproducibility
    node_metrics, edge_bc = _compute_centralities(G, pair_newman, bc_k, seed)

    communities_meta = []
    for cid in range(n_comm):
        members = comm_members.get(cid, [])
        is_misc = (cid == misc_cid)
        if is_misc:
            color, top_auth, lbl = MISC_COLOR, [], []
        else:
            color = golden_hsl(cid)
            top_auth = sorted(members, key=lambda m: -wdeg.get(m, 0))[:TOP_AUTHORS_PER_COMMUNITY]
            lbl = labels[cid]
        communities_meta.append({
            "id": cid,
            "size": len(members),
            "color": color,
            "top_authors": [G.nodes[m].get("name", m) if m in G.nodes else m for m in top_auth],
            "label_words": lbl,
            "misc": is_misc,
        })

    # Build export payload
    key_to_id = {k: i for i, k in enumerate(sorted(G.nodes()))}
    d3_nodes = []
    for k in sorted(G.nodes()):
        info = author_info.get(k, {})
        name = info.get("canonical_name") or k
        cid = node_to_cid.get(k)
        x, y = pos.get(k, (0.0, 0.0))
        m = node_metrics.get(k, {})
        orcid = info.get("orcid") or None
        # If the key itself is an ORCID (author_key fallback), use it directly.
        if not orcid and isinstance(k, str) and len(k) == 19 and k[4] == "-":
            orcid = k
        oa_id = k if isinstance(k, str) and k.startswith("a") else None
        d3_nodes.append({
            "id": key_to_id[k],
            "key": k,  # stable author_key so downstream (sim) can match unambiguously
            "orcid": orcid,
            "oa": oa_id,  # OpenAlex author id for profile links
            "label": name,
            "papers": int(per_author.get(k, info.get("n_papers") or 0)),
            "last_year": int(author_max_year.get(k, info.get("last_year") or 0)),
            "community": cid,
            "x": round(x, 2),
            "y": round(y, 2),
            "top": top_papers.get(k, []),
            # Centralities (Sun & Rahwan 2017). Rounded to cap JSON size.
            "d": m.get("d", 0),
            "s": m.get("s", 0),
            "sn": round(m.get("s_newman", 0.0), 3),
            "bc": round(m.get("bc", 0.0) * 1e4, 2),      # scaled ×10^4
            "bcw": round(m.get("bc_w", 0.0) * 1e4, 2),
            "pr": round(m.get("pr", 0.0) * 1e4, 2),      # scaled ×10^4
            "prw": round(m.get("pr_w", 0.0) * 1e4, 2),
            "c": int(cites_multi.get(k, 0)),
            "ca": int(cites_all.get(k, 0)),
            "ns": int(n_single_paper.get(k, 0)),
        })
    # Edge-BC keys may come back as (u,v) or (v,u) from NetworkX — normalize.
    eb_lookup = {}
    for (u, v), val in edge_bc.items():
        eb_lookup[(u, v)] = val
        eb_lookup[(v, u)] = val
    d3_edges = []
    for a, b, w in edges_raw:
        years = sorted(pair_years.get((a, b), []))
        d3_edges.append({
            "source": key_to_id[a],
            "target": key_to_id[b],
            "weight": int(w),
            "wn": round(pair_newman.get((a, b), 0.0), 4),  # Newman 1/(n-1) weight
            "eb": round(eb_lookup.get((a, b), 0.0) * 1e5, 3),  # edge BC ×10^5
            "years": years,     # one int per co-authored paper
        })

    venues = sorted(set(papers["venue_slug"].dropna().tolist()))
    year_min = int(papers["year"].min()) if papers["year"].notna().any() else None
    year_max = int(papers["year"].max()) if papers["year"].notna().any() else None

    meta = {
        "nodes": len(d3_nodes),
        "edges": len(d3_edges),
        "min_author_papers": cfg["min_papers_per_author"],
        "min_edge_collabs": BASE_THRESHOLD,
        "giant_threshold": GIANT_THRESHOLD,
        "paper_year_min": year_min,
        "paper_year_max": year_max,
        "venues": [v.upper() for v in venues],
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "communities": communities_meta,
    }
    payload = {"meta": meta, "nodes": d3_nodes, "edges": d3_edges}

    # Top hubs by weighted degree (legacy single-metric list kept for the UI hub-panel).
    top_hubs = [
        {"key": key_to_id[k], "name": G.nodes[k].get("name", k),
         "degree": G.degree(k), "weighted": wdeg.get(k, 0),
         "n_papers": int(per_author.get(k, 0)),
         "last_year": int(author_max_year.get(k, 0))}
        for k, _ in sorted(wdeg.items(), key=lambda kv: -kv[1])[: cfg.get("top_hubs_count", 100)]
    ]

    # Multi-metric rankings (Sun & Rahwan 2017 Tables 3 & 4). Top-K per measure;
    # each row { key (int id), name, score }.
    top_k_rankings = cfg.get("ranking_top_k", 30)
    rankings = _rankings_table(node_metrics, author_info, cites_multi, cites_all,
                               n_single_paper, top_k=top_k_rankings)
    # Remap string author_key → int node id so the frontend can hydrate from DATA.nodes.
    for metric, rows in rankings.items():
        for r in rows:
            r["key"] = key_to_id.get(r["key"], -1)

    # Log top communities
    named = [c for c in communities_meta if not c["misc"]]
    log.info(f"top 5 communities:")
    for c in named[:5]:
        hubs = ", ".join(c["top_authors"][:3])
        kws = " · ".join(c["label_words"][:5])
        log.info(f"  #{c['id']} n={c['size']:4d} | hubs: {hubs} | kw: {kws}")

    report = {
        "nodes": len(d3_nodes),
        "edges": len(d3_edges),
        "communities": len(named),
        "communities_total": len(communities_meta),
        "mainland_size": len(mainland),
        "island_communities": n_island,
    }

    if write:
        out = config.data_dir("processed")
        (out / "coauthor_network.json").write_text(json.dumps(payload))
        (out / "top_hubs.json").write_text(json.dumps(top_hubs))
        (out / "author_rankings.json").write_text(json.dumps(rankings))
        (out / "_graph_report.json").write_text(json.dumps(report, indent=2))
        old = out / "coauthor_graph.json"
        if old.exists():
            old.unlink()
    return report
