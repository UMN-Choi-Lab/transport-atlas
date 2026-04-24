#!/usr/bin/env python
"""§3 + §4 descriptive tables and figures.

Emits to `paper/tables/` and `paper/figures/` (PDF + PNG, 300 DPI,
colorblind-friendly palette). Offline-reproducible from
`data/interim/papers.parquet`, `data/interim/authors.parquet`, and
`data/processed/venue_stats.json`.

Outputs:
    tables/03_corpus_summary.tex
    tables/04_venue_stats.tex
    tables/04_top_contributors.tex
    tables/04_top_papers.tex
    tables/app_a_venues.tex
    tables/app_b_coverage.tex
    figures/04_papers_by_year_stacked.pdf  (+ .png)
    figures/04_team_size_over_time.pdf     (+ .png)
    figures/04_lotka_productivity.pdf      (+ .png)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# ——— Paths ———
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from transport_atlas.utils import config as _cfg  # noqa: E402

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PAPER_DIR = ROOT / "paper"
# Tables and figures live under the Overleaf-pushable manuscript subtree so
# LaTeX can reference them as simply tables/... and figures/...
TABLES = PAPER_DIR / "manuscript" / "tables"
FIGURES = PAPER_DIR / "manuscript" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ——— Global plotting settings ———
# Okabe-Ito colorblind-friendly palette
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,      # TrueType for vector fonts
    "ps.fonttype": 42,
    "axes.prop_cycle": plt.cycler("color", OKABE_ITO),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SEED = 42
np.random.seed(SEED)


def _save(fig, stem: str) -> None:
    """Save a figure as both PDF (vector) and PNG (300 DPI)."""
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    plt.close(fig)


def _tex_escape(s: str) -> str:
    """Escape LaTeX-special characters in a free-text cell."""
    if not isinstance(s, str):
        s = str(s)
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
YEAR_MAX = 2025  # exclude partial 2026 snapshot from all analyses


def load_all():
    papers = pd.read_parquet(INTERIM / "papers.parquet")
    papers = papers[papers["year"] <= YEAR_MAX].copy()
    authors = pd.read_parquet(INTERIM / "authors.parquet")
    with open(PROCESSED / "venue_stats.json") as f:
        venue_stats = json.load(f)
    venues_cfg = _cfg.load_venues()
    return papers, authors, venue_stats, venues_cfg


# ---------------------------------------------------------------------------
# T1 — Corpus summary (single-row + per-decade)
# ---------------------------------------------------------------------------
def table_corpus_summary(papers: pd.DataFrame, authors: pd.DataFrame,
                         venue_stats: list[dict]) -> None:
    total_papers = len(papers)
    n_venues = len(venue_stats)
    y_min, y_max = int(papers["year"].min()), int(papers["year"].max())
    total_authors = len(authors)
    orcid_pct = 100 * authors["orcid"].notna().mean()
    total_cites = int(papers["cited_by_count"].sum())
    if "cited_by_source" in papers.columns:
        crossref_pct = 100 * (papers["cited_by_source"] == "crossref").mean()
    else:
        crossref_pct = None

    rows = [
        ("Papers", f"{total_papers:,}"),
        ("Venues", str(n_venues)),
        ("Year range", f"{y_min}--{y_max}"),
        ("Unique authors (deduped)", f"{total_authors:,}"),
        ("Authors with ORCID", f"{orcid_pct:.1f}\\%"),
        ("Total citations", f"{total_cites:,}"),
    ]
    if crossref_pct is not None:
        rows.append(("Citations sourced from Crossref", f"{crossref_pct:.1f}\\%"))
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Quantity} & \textbf{Value} \\",
        r"\midrule",
    ]
    for label, val in rows:
        lines.append(f"{label} & {val} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "03_corpus_summary.tex").write_text("\n".join(lines) + "\n")
    print(f"  ✓ tables/03_corpus_summary.tex")


# ---------------------------------------------------------------------------
# T2 — Per-venue stats (extended Sun & Rahwan Table 2)
# ---------------------------------------------------------------------------
def table_venue_stats(venue_stats: list[dict], venues_cfg: list[dict]) -> None:
    order = {v["slug"]: i for i, v in enumerate(venues_cfg)}
    vs = sorted(venue_stats, key=lambda v: order.get(v["slug"], 999))
    lines = [
        r"\begin{tabular}{p{5.2cm}rrrrrrrr}",
        r"\toprule",
        (r"\textbf{Venue} & \textbf{Papers} & \textbf{Authors} & "
         r"\textbf{Single} & \textbf{Avg} & \textbf{Max} & "
         r"\textbf{Collabs} & \textbf{P/A} & \textbf{Cites} \\"),
        (r"                & \textbf{} & \textbf{} & "
         r"\textbf{\%} & \textbf{auth.} & \textbf{auth.} & "
         r"\textbf{} & \textbf{} & \textbf{/paper} \\"),
        r"\midrule",
    ]
    for v in vs:
        lines.append(
            f"{_tex_escape(v['name'])} & "
            f"{v['papers']:,} & {v['authors']:,} & "
            f"{v['single_pct']:.1f} & {v['avg_authors']:.2f} & "
            f"{v['max_authors']} & {v['collaborations']:,} & "
            f"{v['papers_per_author']:.2f} & {v['avg_citations']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "04_venue_stats.tex").write_text("\n".join(lines) + "\n")
    print(f"  ✓ tables/04_venue_stats.tex")


# ---------------------------------------------------------------------------
# T3 — Top-10 contributors per venue
# ---------------------------------------------------------------------------
def table_top_contributors(venue_stats: list[dict], venues_cfg: list[dict]) -> None:
    order = {v["slug"]: i for i, v in enumerate(venues_cfg)}
    vs = sorted(venue_stats, key=lambda v: order.get(v["slug"], 999))
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"\textbf{Venue} & \textbf{Top-5 contributors (papers)} & \textbf{Top-5 contributors (citations)} \\",
        r"\midrule",
    ]
    for v in vs:
        top_p = v.get("top_authors_by_papers", [])[:5]
        top_c = v.get("top_authors_by_citations", [])[:5]
        by_p = "; ".join(
            f"{_tex_escape(a['name'])} ({a['n_papers']})"
            for a in top_p
        ) or "--"
        by_c = "; ".join(
            f"{_tex_escape(a['name'])} ({a['cites']:,})"
            for a in top_c
        ) or "--"
        venue_cell = _tex_escape(v["short"])
        lines.append(
            f"{venue_cell} & {by_p} & {by_c} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "04_top_contributors.tex").write_text("\n".join(lines) + "\n")
    print(f"  ✓ tables/04_top_contributors.tex")


# ---------------------------------------------------------------------------
# T4 — Top-3 most-cited papers per venue (full 10 is too big for the body)
# ---------------------------------------------------------------------------
def table_top_papers(venue_stats: list[dict], venues_cfg: list[dict]) -> None:
    order = {v["slug"]: i for i, v in enumerate(venues_cfg)}
    vs = sorted(venue_stats, key=lambda v: order.get(v["slug"], 999))
    lines = [
        r"\begin{tabular}{lp{7.0cm}rrl}",
        r"\toprule",
        (r"\textbf{Venue} & \textbf{Title} & "
         r"\textbf{Year} & \textbf{Cites} & \textbf{First author} \\"),
        r"\midrule",
    ]
    for v in vs:
        top_papers = v.get("top_papers", [])[:3]
        for i, p in enumerate(top_papers):
            first_author = (p.get("authors_short", "") or "").split(";", 1)[0].strip()
            title = p["title"]
            short_cell = _tex_escape(v["short"]) if i == 0 else ""
            lines.append(
                f"{short_cell} & "
                f"{_tex_escape(title)} & "
                f"{p.get('year') or '--'} & "
                f"{p['cites']:,} & "
                f"{_tex_escape(first_author)} \\\\"
            )
        if top_papers:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "04_top_papers.tex").write_text("\n".join(lines) + "\n")
    print(f"  ✓ tables/04_top_papers.tex")


# ---------------------------------------------------------------------------
# TA1 — Venue ISSN table (Appendix A)
# ---------------------------------------------------------------------------
def table_venues_isbn(papers: pd.DataFrame, venues_cfg: list[dict]) -> None:
    # Per-venue first/last year from observed papers
    first_year: dict[str, int] = {}
    last_year: dict[str, int] = {}
    n_papers: dict[str, int] = {}
    for slug, grp in papers.groupby("venue_slug"):
        first_year[slug] = int(grp["year"].min())
        last_year[slug] = int(grp["year"].max())
        n_papers[slug] = len(grp)

    lines = [
        r"\begin{tabular}{p{5.2cm}llrl}",
        r"\toprule",
        (r"\textbf{Venue} & \textbf{Abbr.} & \textbf{ISSN (print / online)} & "
         r"\textbf{Papers} & \textbf{Coverage} \\"),
        r"\midrule",
    ]
    for v in venues_cfg:
        slug = v["slug"]
        if slug not in n_papers:
            continue
        short = v.get("short", slug.upper())
        name = v.get("name", slug.upper())
        issns = v.get("issns") or []
        issn_str = " / ".join(issns[:2]) if issns else "--"
        cov = f"{first_year[slug]}--{last_year[slug]}"
        lines.append(
            f"{_tex_escape(name)} & "
            f"{_tex_escape(short)} & "
            f"{_tex_escape(issn_str)} & "
            f"{n_papers[slug]:,} & "
            f"{cov} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "app_a_venues.tex").write_text("\n".join(lines) + "\n")
    print(f"  ✓ tables/app_a_venues.tex")


# ---------------------------------------------------------------------------
# Appendix B — OpenAlex coverage by decade × venue
# ---------------------------------------------------------------------------
def table_coverage(papers: pd.DataFrame, venues_cfg: list[dict]) -> None:
    papers = papers.copy()
    papers["decade"] = (papers["year"] // 10) * 10
    decades = sorted(papers["decade"].unique())

    pivot = papers.pivot_table(
        index="venue_slug", columns="decade", values="paper_id",
        aggfunc="count", fill_value=0,
    )
    pivot = pivot[decades]
    # Row order: venues.yaml declaration order (matches Appendix A Table)
    order = {v["slug"]: i for i, v in enumerate(venues_cfg)}
    pivot = pivot.loc[sorted(pivot.index, key=lambda s: order.get(s, 999))]

    slug_to_name = {v["slug"]: v.get("name", v["slug"]) for v in venues_cfg}
    slug_to_short = {v["slug"]: v.get("short", v["slug"].upper()) for v in venues_cfg}

    lines = [
        r"\begin{tabular}{l" + "r" * len(decades) + "}",
        r"\toprule",
        r"\textbf{Venue} & "
        + " & ".join(f"\\textbf{{{d}s}}" for d in decades) + r" \\",
        r"\midrule",
    ]
    for slug, row in pivot.iterrows():
        short = slug_to_short.get(slug, slug.upper())
        nums = " & ".join(f"{int(n):,}" if n > 0 else "--" for n in row)
        lines.append(f"{_tex_escape(short)} & {nums} \\\\")
    total_row = pivot.sum(axis=0)
    lines.append(r"\midrule")
    lines.append(
        r"\textbf{Total} & " + " & ".join(
            f"\\textbf{{{int(n):,}}}" for n in total_row) + r" \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "app_b_coverage.tex").write_text("\n".join(lines) + "\n")
    print(f"  ✓ tables/app_b_coverage.tex")


# ---------------------------------------------------------------------------
# F1 — Papers-per-year, stacked by venue
# ---------------------------------------------------------------------------
def fig_papers_by_year_stacked(papers: pd.DataFrame,
                               venues_cfg: list[dict]) -> None:
    # Top 9 venues by total papers (for legibility); rest as "Other"
    top_n = 9
    papers = papers.copy()
    venue_totals = papers.groupby("venue_slug").size().sort_values(ascending=False)
    top_venues = venue_totals.head(top_n).index.tolist()

    slug_to_short = {v["slug"]: v.get("short", v["slug"]) for v in venues_cfg}

    years = np.arange(1967, 2026)
    # Build a matrix: rows = venue groups, cols = years
    data = []
    labels = []
    for slug in top_venues:
        counts = papers[papers["venue_slug"] == slug].groupby("year").size()
        data.append(counts.reindex(years, fill_value=0).values)
        labels.append(slug_to_short.get(slug, slug))
    # "Other" aggregate
    other = papers[~papers["venue_slug"].isin(top_venues)].groupby("year").size()
    data.append(other.reindex(years, fill_value=0).values)
    labels.append(f"Other ({len(venue_totals) - top_n} venues)")

    # Extended 9-color CB-safe palette (Okabe-Ito w/o black + Tol indigo)
    # so all top venues + "Other" (grey) are visually distinct in the stack.
    stack_palette = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
        "#D55E00", "#CC79A7", "#332288", "#117733",
    ][:len(top_venues)] + ["#999999"]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.stackplot(years, np.vstack(data), labels=labels, linewidth=0,
                 colors=stack_palette)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Papers (per year)", fontsize=9)
    ax.set_xlim(1967, 2025)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", ncols=2, fontsize=5.6, frameon=False,
              handlelength=1.2, columnspacing=0.8, labelspacing=0.3)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    _save(fig, "04_papers_by_year_stacked")
    print(f"  ✓ figures/04_papers_by_year_stacked.pdf")


# ---------------------------------------------------------------------------
# F2 — Avg authors per paper over time, by venue
# ---------------------------------------------------------------------------
def fig_team_size_over_time(papers: pd.DataFrame,
                            venues_cfg: list[dict]) -> None:
    papers = papers.copy()
    # Count authors per paper — fall back to 0 when authors list is None/empty
    def _n_authors(a_list):
        try:
            if a_list is None:
                return 0
            return sum(1 for a in a_list if isinstance(a, dict) and a.get("name"))
        except TypeError:
            return 0
    papers["n_authors"] = papers["authors"].map(_n_authors)

    # Top 6 venues for legibility
    venue_totals = papers.groupby("venue_slug").size().sort_values(ascending=False)
    top_venues = venue_totals.head(6).index.tolist()
    slug_to_short = {v["slug"]: v.get("short", v["slug"]) for v in venues_cfg}

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for i, slug in enumerate(top_venues):
        sub = papers[papers["venue_slug"] == slug]
        yearly = sub.groupby("year")["n_authors"].mean()
        # Smooth with 3-year rolling mean for readability
        smoothed = yearly.rolling(3, center=True, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed.values,
                label=slug_to_short.get(slug, slug),
                color=OKABE_ITO[i], linewidth=1.2)

    # Corpus-wide average as dashed reference line
    overall = papers.groupby("year")["n_authors"].mean().rolling(
        3, center=True, min_periods=1).mean()
    ax.plot(overall.index, overall.values, color="#666666",
            linewidth=1.2, linestyle="--", label="All 36 venues")

    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Avg authors per paper", fontsize=9)
    ax.set_xlim(1967, 2025)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=6, frameon=False,
              handlelength=1.2, labelspacing=0.3)
    ax.grid(alpha=0.2)
    # Sun & Rahwan window annotation at the top of the plot
    ax.axvspan(1990, 2015, alpha=0.08, color="#56B4E9")
    ax.text(2002.5, 0.98, "S\\&R 2017 window",
            ha="center", va="top", color="#56B4E9", fontsize=6, alpha=0.9,
            transform=ax.get_xaxis_transform())
    fig.tight_layout()
    _save(fig, "04_team_size_over_time")
    print(f"  ✓ figures/04_team_size_over_time.pdf")


# ---------------------------------------------------------------------------
# F3 — Lotka's law (log-log productivity histogram)
# ---------------------------------------------------------------------------
def fig_lotka(authors: pd.DataFrame) -> None:
    counts = Counter(authors["n_papers"].astype(int).tolist())
    ks = np.array(sorted(counts.keys()))
    ys = np.array([counts[k] for k in ks])
    # Fit α: log(f(k)) = -α log(k) + C  via OLS on log-log (k>=1, f>0)
    mask = (ks >= 1) & (ys > 0)
    lx = np.log(ks[mask])
    ly = np.log(ys[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    alpha = -slope
    # Sun & Rahwan reported α ≈ 2.6; Lotka's original prediction α = 2
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(ks, ys, "o", color=OKABE_ITO[4], markersize=4,
              alpha=0.7, label="Authors")
    # Fit line
    fit_y = np.exp(intercept) * ks[mask] ** slope
    ax.loglog(ks[mask], fit_y, color=OKABE_ITO[5], linewidth=1.5,
              label=fr"OLS fit: $\alpha = {alpha:.2f}$")
    # Lotka's reference line (α=2), rescaled to match observed high end
    ref_k = ks[mask].astype(float)
    # Align ref curve through the observed (k=1, y=f(1)) point if available
    c_lotka = ys[mask][0] / (ref_k[0] ** -2.0)
    ax.loglog(ref_k, c_lotka * ref_k ** -2.0,
              color="#888", linewidth=1, linestyle=":",
              label=r"Lotka prediction ($\alpha=2$)")
    ax.set_xlabel("Papers per author, $k$")
    ax.set_ylabel("Number of authors, $f(k)$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(which="both", alpha=0.2)
    _save(fig, "04_lotka_productivity")
    print(f"  ✓ figures/04_lotka_productivity.pdf  (α = {alpha:.3f})")

    # Persist the fitted α to a tiny JSON for citation in the prose
    (PAPER_DIR / "analysis" / "_lotka_fit.json").write_text(
        json.dumps({"alpha": float(alpha), "intercept": float(intercept),
                    "n_authors": int(len(authors))}, indent=2)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("[descriptive] loading…")
    papers, authors, venue_stats, venues_cfg = load_all()
    print(f"[descriptive]   papers={len(papers):,}  authors={len(authors):,}  "
          f"venues={len(venue_stats)}  years={int(papers['year'].min())}-"
          f"{int(papers['year'].max())}")

    print("[descriptive] tables…")
    table_corpus_summary(papers, authors, venue_stats)
    table_venue_stats(venue_stats, venues_cfg)
    table_top_contributors(venue_stats, venues_cfg)
    table_top_papers(venue_stats, venues_cfg)
    table_venues_isbn(papers, venues_cfg)
    table_coverage(papers, venues_cfg)

    print("[descriptive] figures…")
    fig_papers_by_year_stacked(papers, venues_cfg)
    fig_team_size_over_time(papers, venues_cfg)
    fig_lotka(authors)

    print("[descriptive] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
