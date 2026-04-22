"""Render static site from processed JSON assets."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..utils import config
from ..utils.logger import get_logger

log = get_logger("render")

TEMPLATE_DIR = Path(__file__).parent / "templates"
PAGES = [
    ("index.html", "index", "index.html"),
    ("explorer.html", "explorer", "explorer.html"),
    ("by_year.html", "by_year", "by_year.html"),
    ("venues.html", "venues", "venues.html"),
    ("coauthor_network.html", "coauthor", "coauthor_network.html"),
    ("topic_space.html", "topic", "topic_space.html"),
    ("combined.html", "combined", "combined.html"),
]


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
                 "semantic_communities.json", "combined_communities.json"]:
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

    ctx_common = {
        "build_time": build_time,
        "cache_bust": cache_bust,
        "venues": venues,
        "summary": summary,
        "graph_report": graph_report,
    }

    for tmpl, page, outfile in PAGES:
        html = env.get_template(tmpl).render(**ctx_common, page=page)
        (out / outfile).write_text(html)
        log.info(f"rendered {outfile}")

    return {"pages": [p[2] for p in PAGES], "build_time": build_time, "cache_bust": cache_bust}
