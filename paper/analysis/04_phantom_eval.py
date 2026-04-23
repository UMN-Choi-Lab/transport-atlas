#!/usr/bin/env python
"""§8 — Render phantom-collaborator eval figures and tables.

Reads data/processed/phantom_eval.json (produced by scripts/07_phantom_eval.py)
and emits:
    tables/08_phantom_lift.tex
    tables/08_phantom_cases.tex
    figures/08_phantom_precision_at_k.pdf  (+.png)
    figures/08_phantom_calibration.pdf     (+.png)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.prop_cycle": plt.cycler("color", OKABE_ITO),
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper" / "manuscript" / "tables"
FIGURES = ROOT / "paper" / "manuscript" / "figures"
PHANTOM_JSON = ROOT / "data" / "processed" / "phantom_eval.json"


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    plt.close(fig)


def _tex_escape(s) -> str:
    if not isinstance(s, str): s = str(s)
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    for k, v in repl.items(): s = s.replace(k, v)
    return s


def _titlecase(s: str) -> str:
    if "," in s:
        last, rest = s.split(",", 1)
        parts = [p.strip() for p in rest.split()]
        return (last.strip().title() + ", " +
                " ".join(p.title() for p in parts))
    return s.title()


def main() -> int:
    if not PHANTOM_JSON.exists():
        print(f"[phantom-fig] missing {PHANTOM_JSON}", file=sys.stderr)
        return 1
    d = json.loads(PHANTOM_JSON.read_text())
    cfg = d["config"]
    metrics = d["metrics"]
    calib = d["calibration"]
    cases = d["cases"]

    ks = sorted(int(k.split("=")[1]) for k in metrics)
    methods = ("phantom", "random", "pref_attach", "same_venue")
    method_label = {
        "phantom":    r"Phantom (semantic)",
        "random":     r"Random",
        "pref_attach": r"Popularity-weighted",
        "same_venue": r"Same-venue",
    }

    # ------------------------------------------------------------------
    # Figure — precision@K curves (one line per method).
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for i, m in enumerate(methods):
        ys = [metrics[f"K={k}"][m]["micro_precision"] * 100 for k in ks]
        ax.plot(ks, ys, marker="o", color=OKABE_ITO[i % len(OKABE_ITO)],
                linewidth=2.0 if m == "phantom" else 1.2,
                label=method_label[m])
    ax.set_xlabel("$K$ (number of phantom partners proposed per author)")
    ax.set_ylabel(r"Precision @ $K$ (\%)")
    ax.set_title(
        "Phantom collaborators realise at $\\sim$" +
        f"{metrics['K=20']['phantom']['micro_precision']*100:.1f}" +
        r"\% vs $<$1\% for random / popularity / same-venue baselines"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_xticks(ks)
    _save(fig, "08_phantom_precision_at_k")
    print("  ✓ figures/08_phantom_precision_at_k.pdf")

    # ------------------------------------------------------------------
    # Table — lift over baselines (+micro/macro precision)
    # ------------------------------------------------------------------
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (r"\textbf{$K$} & \textbf{Method} & \textbf{Hits} & "
         r"\textbf{Predictions} & \textbf{micro-P (\%)} & "
         r"\textbf{macro-P (\%)} & \textbf{Lift vs phantom} \\"),
        r"\midrule",
    ]
    for k in ks:
        block = metrics[f"K={k}"]
        phantom_mp = block["phantom"]["micro_precision"]
        for m in methods:
            r = block[m]
            lift = ""
            if m == "phantom":
                lift = "--"
            else:
                base = r["micro_precision"]
                if base > 0:
                    lift = fr"$\times$ {phantom_mp / base:.2f}"
                else:
                    lift = r"$\infty$"
            bold_start, bold_end = ("", "")
            if m == "phantom":
                bold_start, bold_end = (r"\textbf{", r"}")
            lines.append(
                f"{k if m == 'phantom' else ''} & "
                f"{bold_start}{method_label[m]}{bold_end} & "
                f"{r['hits']:,} & {r['predictions']:,} & "
                f"{bold_start}{r['micro_precision']*100:.2f}{bold_end} & "
                f"{r['macro_precision']*100:.2f} & {lift} \\\\"
            )
        if k != ks[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "08_phantom_lift.tex").write_text("\n".join(lines) + "\n")
    print("  ✓ tables/08_phantom_lift.tex")

    # ------------------------------------------------------------------
    # Figure — calibration: similarity bucket → realized rate
    # ------------------------------------------------------------------
    if calib:
        xs = [c["sim_median"] for c in calib]
        ys = [c["realize_rate"] * 100 for c in calib]
        ns = [c["n_pairs"] for c in calib]
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(xs, ys, marker="o", color=OKABE_ITO[4], linewidth=1.5)
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n:,}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=6.5, alpha=0.7)
        base_rate = np.mean([c["realize_rate"] for c in calib]) * 100
        ax.axhline(base_rate, color=OKABE_ITO[5], linestyle=":", linewidth=1,
                   label=f"Mean rate ({base_rate:.3f}\\%)")
        ax.set_xlabel("Median pairwise cosine similarity (whitened, bucket)")
        ax.set_ylabel(r"Realised in 2020--2025 (\%)")
        ax.set_title(
            "Calibration: higher semantic similarity → higher realisation rate"
        )
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        _save(fig, "08_phantom_calibration")
        print("  ✓ figures/08_phantom_calibration.pdf")

    # ------------------------------------------------------------------
    # Table — top-30 realized-phantom cases
    # ------------------------------------------------------------------
    lines = [
        r"\begin{tabular}{lll rr}",
        r"\toprule",
        (r"\textbf{\#} & \textbf{Author A} & \textbf{Author B} & "
         r"\textbf{Cos sim} & \textbf{Train dist} \\"),
        r"\midrule",
    ]
    for i, c in enumerate(cases[:20], 1):
        d_txt = (r"$\geq 4$" if c["train_dist"] is None
                 else str(int(c["train_dist"])))
        lines.append(
            f"{i} & {_tex_escape(_titlecase(c['a_name']))} & "
            f"{_tex_escape(_titlecase(c['b_name']))} & "
            f"{c['sim']:.3f} & {d_txt} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "08_phantom_cases.tex").write_text("\n".join(lines) + "\n")
    print("  ✓ tables/08_phantom_cases.tex")

    # Prose summary (for quoting in §8 text)
    summary = {
        "train_cutoff":   cfg["train_cutoff_year"],
        "test_window":    cfg["test_years"],
        "n_eval_authors": cfg["n_eval_authors"],
        "n_authors_with_realized": cfg["n_authors_with_realized"],
        "best_K":         max(
            ks, key=lambda k: metrics[f"K={k}"]["phantom"]["micro_precision"]
        ),
        "phantom_precision_at_20": metrics["K=20"]["phantom"]["micro_precision"],
        "lift_vs_random_at_20":    metrics["K=20"]["random"]["lift_phantom_vs"],
        "lift_vs_pref_at_20":      metrics["K=20"]["pref_attach"]["lift_phantom_vs"],
        "lift_vs_venue_at_20":     metrics["K=20"]["same_venue"]["lift_phantom_vs"],
        "calibration_topbin_rate": (calib[-1]["realize_rate"]
                                    if calib else None),
        "calibration_botbin_rate": (calib[0]["realize_rate"]
                                    if calib else None),
    }
    (ROOT / "paper" / "analysis" / "_phantom_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("  ✓ paper/analysis/_phantom_summary.json")
    print("[phantom-fig] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
