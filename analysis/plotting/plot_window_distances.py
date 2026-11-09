#!/usr/bin/env python
# coding: utf-8
"""
Plot Window-Level Distance Distributions
==========================================
Frequency-distribution plot(s) for numeric columns of a NARDINI
sliding-window output file (out_windows.csv from nardini_MultiBlock's
--stage windows / --stage all), by default min_dist and inter_cluster_dist.

For each column: a histogram, a dashed line + label for the median, a
dotted line + label for the mean, and a text box giving Q10/Q25/Q75/Q90
for reference. Saves both a PNG and a PDF.

v 1.0 by Alexandre P. Magalhaes
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent

CONFIG = dict(
    input_csv       = str(SCRIPT_DIR / "Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.out_windows.csv"),
    columns         = ["min_dist", "inter_cluster_dist"],
    bins            = 25,
    arial_font_path = str(SCRIPT_DIR.parent / "Arial.ttf"),
    out_png         = str(SCRIPT_DIR / "figures" / "window_dist_distributions.png"),
    out_pdf         = str(SCRIPT_DIR / "figures" / "window_dist_distributions.pdf"),
)

# Column -> a nicer axis label. Anything not listed here just uses its own
# column name verbatim, so this script works on any numeric column.
LABELS = {
    "min_dist":           "min_dist  (squared distance to assigned centroid)",
    "inter_cluster_dist": "inter_cluster_dist  (gap to 2nd-nearest centroid)",
}

# A small fixed color rotation for however many columns are plotted.
COLORS = ["#3B6FA0", "#C1666B", "#4C9F70", "#8B6DB5", "#C98A3B"]

INK   = "#1a1a1a"
MUTED = "#6b7280"
GRID  = "#e5e7eb"


def setup_plot_style(arial_font_path: str) -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    family = "DejaVu Sans"
    if Path(arial_font_path).exists():
        font_manager.fontManager.addfont(arial_font_path)
        family = font_manager.FontProperties(fname=arial_font_path).get_name()
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [family, "Arial", "DejaVu Sans"]


def plot_column(ax, vals: np.ndarray, xlabel: str, color: str, bins: int) -> None:
    mean   = np.mean(vals)
    median = np.median(vals)
    q10, q25, q75, q90 = np.percentile(vals, [10, 25, 75, 90])

    counts, _, _ = ax.hist(vals, bins=bins, color=color, edgecolor="white",
                            linewidth=0.5, zorder=2)
    xrange = vals.max() - vals.min()
    dx = xrange * 0.02 if xrange > 0 else 0.02

    ax.axvline(mean, color=INK, linestyle=":", linewidth=1.5, zorder=3)
    ax.axvline(median, color=INK, linestyle="--", linewidth=1.5, zorder=3)

    ymax = counts.max()
    ax.set_ylim(top=ymax * 1.28)

    # Whichever of mean/median sits further left gets its label anchored to
    # the left (text growing leftward) so the two labels never collide.
    if median <= mean:
        med_ha, med_dx = "right", -dx
        mean_ha, mean_dx = "left", dx
    else:
        med_ha, med_dx = "left", dx
        mean_ha, mean_dx = "right", -dx

    ax.text(median + med_dx, ymax * 1.22, f"median = {median:.2f}", color=INK,
             fontsize=9.5, ha=med_ha, va="top", fontweight="bold")
    ax.text(mean + mean_dx, ymax * 1.10, f"mean = {mean:.2f}", color=INK,
             fontsize=9.5, ha=mean_ha, va="top", style="italic")

    stats_text = f"Q10 = {q10:.2f}\nQ25 = {q25:.2f}\nQ75 = {q75:.2f}\nQ90 = {q90:.2f}"
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top",
             fontsize=9.5, color=MUTED,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRID))

    ax.set_xlabel(xlabel, fontsize=10.5, color=INK)
    ax.set_ylabel("Number of windows", fontsize=10.5, color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main(cfg: dict) -> None:
    setup_plot_style(cfg["arial_font_path"])

    df = pd.read_csv(cfg["input_csv"])
    missing = [c for c in cfg["columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) {missing} not found in {cfg['input_csv']}. "
                          f"Available columns: {list(df.columns)}")

    n_windows = len(df)
    n_idrs = df["IDR_ID"].nunique() if "IDR_ID" in df.columns else None

    n_cols = len(cfg["columns"])
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 5.2), dpi=150)
    if n_cols == 1:
        axes = [axes]

    for ax, col, color in zip(axes, cfg["columns"], COLORS):
        vals = df[col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            raise ValueError(f"Column '{col}' has no non-null values to plot.")
        plot_column(ax, vals, LABELS.get(col, col), color, cfg["bins"])

    title = f"Window-level distance distributions  (n = {n_windows} windows"
    title += f" across {n_idrs} IDRs)" if n_idrs is not None else ")"
    fig.suptitle(title, fontsize=13.5, color=INK, y=1.02)
    fig.tight_layout()

    Path(cfg["out_png"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["out_pdf"]).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg["out_png"], dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(cfg["out_pdf"], facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {cfg['out_png']}")
    print(f"Saved -> {cfg['out_pdf']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot frequency distributions of numeric columns from a "
                     "NARDINI out_windows.csv file, with mean/median lines "
                     "and Q10/Q25/Q75/Q90 reference text."
    )
    parser.add_argument(
        "--input", default=CONFIG["input_csv"],
        help="Path to the out_windows.csv file to read."
    )
    parser.add_argument(
        "--columns", default=",".join(CONFIG["columns"]),
        help="Comma-separated list of numeric columns to plot "
             "(default: min_dist,inter_cluster_dist)."
    )
    parser.add_argument(
        "--bins", type=int, default=CONFIG["bins"],
        help="Number of histogram bins per column (default: 25)."
    )
    parser.add_argument(
        "--arial_font_path", default=CONFIG["arial_font_path"],
        help="Path to Arial.ttf for editable-text PDF output. Falls back to "
             "a generic sans-serif font if not found."
    )
    parser.add_argument(
        "--out_png", default=CONFIG["out_png"]
    )
    parser.add_argument(
        "--out_pdf", default=CONFIG["out_pdf"]
    )
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg.update(dict(
        input_csv       = args.input,
        columns         = [c.strip() for c in args.columns.split(",") if c.strip()],
        bins            = args.bins,
        arial_font_path = args.arial_font_path,
        out_png         = args.out_png,
        out_pdf         = args.out_pdf,
    ))

    main(cfg)
