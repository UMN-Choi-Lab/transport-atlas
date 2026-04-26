"""Render static site from processed JSON assets."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..utils import config
from ..utils.logger import get_logger

log = get_logger("render")

TEMPLATE_DIR = Path(__file__).parent / "templates"
# Same default that scripts/11_export_specter2_onnx.py writes about.
DEFAULT_SPECTER2_REPO = "Xenova/all-MiniLM-L6-v2"   # safe fallback; reviewers page
                                                     # gates itself if SPECTER2 isn't set
EMBED_OUT = Path(os.environ.get("EMBED_OUT", "/data2/chois/transport-atlas"))
PAGES = [
    ("index.html", "index", "index.html"),
    ("explorer.html", "explorer", "explorer.html"),
    ("by_year.html", "by_year", "by_year.html"),
    ("venues.html", "venues", "venues.html"),
    ("coauthor_network.html", "coauthor", "coauthor_network.html"),
    ("topic_space.html", "topic", "topic_space.html"),
    ("trajectories.html", "trajectories", "trajectories.html"),
    ("collab_style.html", "collab_style", "collab_style.html"),
    ("combined.html", "combined", "combined.html"),
    ("reviewers.html", "reviewers", "reviewers.html"),
]


def _resolve_specter2_repo() -> str:
    """Resolve which HF repo the reviewers page should load SPECTER2 from.

    Priority:
        1. SPECTER2_REPO env var (override)
        2. specter2_repo_id.txt written by scripts/11_export_specter2_onnx.py
        3. DEFAULT_SPECTER2_REPO (a placeholder; page warns when used)
    """
    if os.environ.get("SPECTER2_REPO"):
        return os.environ["SPECTER2_REPO"].strip()
    marker = EMBED_OUT / "specter2_repo_id.txt"
    if marker.exists():
        v = marker.read_text().strip()
        if v:
            return v
    return DEFAULT_SPECTER2_REPO


def _load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def run() -> dict:
    processed = config.data_dir("processed")
    out = config.site_dir()
    data_dst = out / "data"
    data_dst.mkdir(parents=True, exist_ok=True)

    # Copy data JSONs to site/data/ (topic-space files are optional — fail soft)
    for name in ["papers.json", "by_year.json", "coauthor_network.json", "top_hubs.json",
                 "author_rankings.json", "venue_stats.json",
                 "topic_coords.json", "author_similar.json", "author_trajectories.json",
                 "trajectory_taxonomy.json", "collab_style.json",
                 "semantic_communities.json", "combined_communities.json",
                 "reviewer_index.json", "reviewer_authors.bin"]:
        src = processed / name
        if src.exists():
            shutil.copy2(src, data_dst / name)
        else:
            log.warning(f"missing {src} — page may be empty")
    # Remove the old graphology file if it was copied in a prior run
    stale = data_dst / "coauthor_graph.json"
    if stale.exists():
        stale.unlink()

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    build_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    cache_bust = datetime.now().strftime("%Y%m%d-%H%M")

    summary = _load_json(processed / "_summary.json", {
        "n_papers": 0, "n_venues": 0, "year_min": None, "year_max": None,
    })
    graph_report = _load_json(processed / "_graph_report.json", {"nodes": 0, "edges": 0})
    venues = config.load_venues()

    specter2_repo = _resolve_specter2_repo()
    log.info(f"specter2 repo for reviewers page: {specter2_repo}")
    ctx_common = {
        "build_time": build_time,
        "cache_bust": cache_bust,
        "venues": venues,
        "summary": summary,
        "graph_report": graph_report,
        "specter2_repo": specter2_repo,
        "specter2_is_default": specter2_repo == DEFAULT_SPECTER2_REPO,
    }

    for tmpl, page, outfile in PAGES:
        html = env.get_template(tmpl).render(**ctx_common, page=page)
        (out / outfile).write_text(html)
        log.info(f"rendered {outfile}")

    return {"pages": [p[2] for p in PAGES], "build_time": build_time, "cache_bust": cache_bust}
