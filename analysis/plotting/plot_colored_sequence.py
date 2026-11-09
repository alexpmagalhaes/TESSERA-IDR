#!/usr/bin/env python
# coding: utf-8
"""
Plot / Print IDR Sequences Colored by Residue (Biochemical) Type
==================================================================
Standalone port of the RUFF/King lab's "Print sequences colored by residue
type" Colab cell (Colorizer class + COLOR_MAP palettes) so it can run
outside a notebook, reading directly from an IDR metadata CSV such as
Hsapien_IDRs_ftl100aa.csv instead of an in-notebook `subseqs`/`subnames`
list.

Faithfully reproduces:
  - Both named color palettes ("Kiersten" and "Matt"), keyed by amino-acid
    chemical group exactly as in the original COLOR_MAP dicts.
  - The Colorizer class's per-residue coloring + bold/larger-font highlight
    of specific positions (the original's `mutpos` argument).

What's NOT ported (out of scope for a single metadata CSV): the
`highlight_nonrandom_feature == True` branch, which auto-picks which
positions to highlight from a full z-scored NARDINI+ composition/patterning
feature table (`pattcompfinaldf`) that isn't available here. Positions to
highlight can instead be supplied directly via --highlight (e.g. to bold a
GIN block/domain range found by nardini_MultiBlock_v099.py).

Two rendering paths, selectable with --format:
  - html  : a standalone HTML file with the exact colored <span> markup the
            original Colorizer.format_sequence() produced (monospace,
            17px/22px normal/highlighted) -- open it in any browser.
  - png/pdf/both (default) : a matplotlib figure wrapped to --wrap_width
            residues per line, for use in a paper/slide. Editable PDF text
            (pdf.fonttype/ps.fonttype = 42) and Arial labels, matching the
            rest of this project's figures. The sequence itself is rendered
            in a monospace font (Arial has no fixed-width variant, and the
            wrapped grid layout needs equal-width glyphs to line up).

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

# ══════════════════════════════════════════════════════════════════════════
#  COLOR_MAP palettes -- verbatim from the "Print sequences colored by
#  residue type" Colab cell.
# ══════════════════════════════════════════════════════════════════════════
COLOR_SETS = {
    "Kiersten": {
        "A":      "#8c8c8c",
        "CHNQST": "#1f8241",
        "DE":     "#e23a36",
        "FWY":    "#f58220",
        "G":      "#d865a6",
        "ILMV":   "#231f20",
        "KR":     "#189bd7",
        "P":      "#7c2880",
        "-":      "#000000",
    },
    "Matt": {
        "A":      "#8c8c8c",
        "CHNQST": "#077a15",
        "DE":     "#e23a36",
        "FWY":    "#f58220",
        "G":      "#86bd8d",
        "ILMV":   "#231f20",
        "KR":     "#1725c2",
        "P":      "#c25fc9",
    },
}

# Human-readable group labels for the legend (biochemical type each group
# represents), in a fixed, deliberate display order.
GROUP_LABELS = {
    "A":      "Alanine",
    "CHNQST": "Polar",
    "DE":     "Negative",
    "FWY":    "Aromatic",
    "G":      "Glycine",
    "ILMV":   "Aliphatic / hydrophobic",
    "KR":     "Positive",
    "P":      "Proline",
    "-":      "Gap",
}
GROUP_ORDER = ["ILMV", "FWY", "KR", "DE", "CHNQST", "A", "G", "P", "-"]

INK   = "#1a1a1a"
MUTED = "#6b7280"


def build_aacolor_map(color_set: str) -> dict:
    """Port of Colorizer._build_colormap: expand the amino-acid-group ->
    color map into a per-residue-letter -> color map.
    """
    if color_set not in COLOR_SETS:
        raise ValueError(f"Unknown --color_set '{color_set}'. "
                          f"Choices: {list(COLOR_SETS)}")
    color_map = COLOR_SETS[color_set]
    aacolor_map = {}
    for amino_acids, color in color_map.items():
        for amino_acid in amino_acids:
            aacolor_map[amino_acid] = color
    return aacolor_map


def parse_highlight_ranges(spec: str) -> set:
    """Parse a '--highlight' spec like '10-25,40-45' (1-based, inclusive,
    matching the IDR_START/IDR_END convention used elsewhere in this
    project) into a 0-based set of residue positions -- the same shape as
    the original notebook's `featpos`/`mutpos` list, just supplied
    explicitly instead of auto-detected from a feature table.
    """
    positions = set()
    if not spec:
        return positions
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
        else:
            start = end = int(chunk)
        if start < 1 or end < start:
            raise ValueError(f"Invalid --highlight range '{chunk}' "
                              f"(expected 1-based, start <= end).")
        positions.update(range(start - 1, end))  # 1-based inclusive -> 0-based
    return positions


# ══════════════════════════════════════════════════════════════════════════
#  Row lookup
# ══════════════════════════════════════════════════════════════════════════

def load_row(input_csv: str, idr_id: "str | None", index: "int | None") -> dict:
    df = pd.read_csv(input_csv)

    seq_col = next((c for c in ("IDR_Sequence", "Sequence") if c in df.columns), None)
    if seq_col is None:
        raise ValueError(f"'{input_csv}' has no 'IDR_Sequence' or 'Sequence' column.")
    if "IDR_ID" not in df.columns:
        raise ValueError(f"'{input_csv}' has no 'IDR_ID' column.")

    if idr_id is not None:
        matches = df.index[df["IDR_ID"] == idr_id].tolist()
        if not matches:
            raise ValueError(f"IDR_ID '{idr_id}' not found in '{input_csv}'.")
        row = df.loc[matches[0]]
    elif index is not None:
        if index < 0 or index >= len(df):
            raise ValueError(f"--index {index} out of range (0..{len(df) - 1}).")
        row = df.iloc[index]
    else:
        raise ValueError("Pass exactly one of --idr_id or --index.")

    seq = str(row[seq_col]).strip().upper()
    meta = {}
    for col in ("Gene name", "UniprotID", "Uniprot name", "IDR_START",
                "IDR_END", "IDR_length", "Description"):
        if col in df.columns:
            meta[col] = row[col]

    return dict(IDR_ID=row["IDR_ID"], Sequence=seq, meta=meta)


def build_title(record: dict) -> str:
    meta = record["meta"]
    parts = [str(record["IDR_ID"])]
    if meta.get("Gene name"):
        parts.append(str(meta["Gene name"]))
    if meta.get("UniprotID"):
        parts.append(str(meta["UniprotID"]))
    header = "  |  ".join(parts)
    span = ""
    if meta.get("IDR_START") is not None and meta.get("IDR_END") is not None:
        span = f"  (residues {int(meta['IDR_START'])}-{int(meta['IDR_END'])})"
    return header + span


# ══════════════════════════════════════════════════════════════════════════
#  HTML rendering -- direct port of Colorizer.format_sequence
# ══════════════════════════════════════════════════════════════════════════

def render_html(seq: str, aacolor_map: dict, highlight_positions: set,
                 title: str, normal_px: int = 17, highlight_px: int = 22
                 ) -> str:
    default_style = "font-family: Courier, monospace; background-color: transparent;"
    mapped = []
    for count, aa in enumerate(seq):
        if aa not in aacolor_map:
            raise RuntimeError(f'Unknown amino acid: "{aa}".')
        color = aacolor_map[aa]
        if count in highlight_positions:
            mapped.append(
                f'<span style="color: {color}; {default_style} '
                f'font-weight: bold; font-size: {highlight_px}px;">{aa}</span>'
            )
        else:
            mapped.append(
                f'<span style="color: {color}; {default_style} '
                f'font-size: {normal_px}px;">{aa}</span>'
            )
    seq_html = "".join(mapped)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
</head>
<body style="background: white; padding: 24px;">
<div style="font-family: Arial, sans-serif; font-size: 14px; color: {INK};
            margin-bottom: 10px; font-weight: bold;">{title}</div>
<div style="line-height: 1.5; word-break: break-all;">{seq_html}</div>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════
#  Matplotlib rendering (PNG/PDF)
# ══════════════════════════════════════════════════════════════════════════

def setup_plot_style(arial_font_path: str) -> str:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    label_family = "DejaVu Sans"
    if arial_font_path and Path(arial_font_path).exists():
        font_manager.fontManager.addfont(arial_font_path)
        label_family = font_manager.FontProperties(fname=arial_font_path).get_name()
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [label_family, "Arial", "DejaVu Sans"]
    return label_family


def _pick_monospace_family(mono_font_path: "str | None") -> str:
    if mono_font_path and Path(mono_font_path).exists():
        font_manager.fontManager.addfont(mono_font_path)
        return font_manager.FontProperties(fname=mono_font_path).get_name()
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("DejaVu Sans Mono", "Courier New", "Liberation Mono"):
        if candidate in available:
            return candidate
    return "monospace"


def _groups_present(seq: str, color_map: dict) -> list:
    """Which COLOR_MAP group keys actually occur in this sequence, in
    GROUP_ORDER -- keeps the legend from listing e.g. 'Gap' (both it and
    'Aliphatic/hydrophobic' happen to share the same black swatch) when no
    residue of that group is present.
    """
    present_letters = set(seq)
    return [g for g in GROUP_ORDER
            if g in color_map and present_letters.intersection(g)]


def render_matplotlib(seq: str, aacolor_map: dict, highlight_positions: set,
                       title: str, wrap_width: int, label_family: str,
                       mono_font_path: "str | None", color_set: str,
                       show_legend: bool, normal_fs: float = 11.5,
                       highlight_fs: float = 15.0):
    mono_family = _pick_monospace_family(mono_font_path)

    n = len(seq)
    n_lines = max(1, -(-n // wrap_width))  # ceil

    char_w_in   = 0.135
    row_h_in    = 0.38  # extra room vs. 0.30 for the every-10aa tick + label
    left_pad_in = 0.55   # room for the start-of-line position number
    top_pad_in  = 0.55
    bottom_pad_in = 0.55 + (0.35 if show_legend else 0.0)

    fig_w = left_pad_in + wrap_width * char_w_in + 0.3
    fig_h = top_pad_in + n_lines * row_h_in + bottom_pad_in
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    for line_idx in range(n_lines):
        line_start = line_idx * wrap_width
        line_seq   = seq[line_start:line_start + wrap_width]
        y = -line_idx

        # 1-based residue number at the start of this line.
        ax.text(-0.6, y, str(line_start + 1), fontsize=8.5, color=MUTED,
                 fontfamily=label_family, ha="right", va="center")

        for col, aa in enumerate(line_seq):
            pos = line_start + col
            color = aacolor_map.get(aa, "#000000")
            is_hi = pos in highlight_positions
            ax.text(col + 0.5, y, aa,
                     fontsize=highlight_fs if is_hi else normal_fs,
                     fontweight="bold" if is_hi else "normal",
                     color=color, fontfamily=mono_family,
                     ha="center", va="center")

            # Tick + position label every 10 residues, for counting along a
            # line without having to count individual letters.
            if (pos + 1) % 10 == 0:
                ax.plot([col + 0.5, col + 0.5], [y - 0.32, y - 0.20],
                         color=MUTED, lw=0.8, solid_capstyle="butt")
                ax.text(col + 0.5, y - 0.42, str(pos + 1), fontsize=6.5,
                         color=MUTED, fontfamily=label_family,
                         ha="center", va="top")

    ax.set_xlim(-2.5, wrap_width + 0.5)
    ax.set_ylim(-(n_lines - 1) - 0.7, 0.7)
    ax.axis("off")
    # A single two-line title (rather than a title + a separately-placed
    # fig.text note) so tight_layout reserves space for both lines
    # consistently regardless of how tall the overall figure ends up being
    # for a long sequence -- two independently-positioned text artists can
    # otherwise end up overlapping once tight_layout repacks a very tall
    # figure.
    subtitle = ("position numbers are relative to this IDR fragment "
                "(1 = its first residue)")
    ax.set_title(f"{title}\n{subtitle}", fontsize=12.5, color=INK,
                 fontfamily=label_family, loc="left", pad=14,
                 linespacing=1.8)

    if show_legend:
        color_map = COLOR_SETS[color_set]
        groups = _groups_present(seq, color_map)
        handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
                               markerfacecolor=color_map[g], markeredgecolor="none")
                   for g in groups]
        labels = [GROUP_LABELS[g] for g in groups]
        ax.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.08 / max(n_lines, 1) - 0.02),
                   ncol=min(len(labels), 5), frameon=False, fontsize=8.5,
                   prop=font_manager.FontProperties(family=label_family, size=8.5))

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Print/plot one IDR sequence colored by residue "
                     "(biochemical) type, ported from the RUFF/King lab's "
                     "'Print sequences colored by residue type' Colab cell."
    )
    parser.add_argument("--input", default="Hsapien_IDRs_ftl100aa.csv",
                         help="Path to the IDR metadata CSV (needs IDR_ID "
                              "and IDR_Sequence/Sequence columns).")
    sel = parser.add_mutually_exclusive_group(required=True)
    sel.add_argument("--idr_id", default=None,
                      help="IDR_ID to look up and plot, e.g. P00001_IDR_1.")
    sel.add_argument("--index", type=int, default=None,
                      help="Row index (0-based) to plot instead of --idr_id.")
    parser.add_argument("--color_set", choices=list(COLOR_SETS), default="Kiersten",
                         help="Which named palette to use (default: Kiersten, "
                              "matching the Colab cell's default).")
    parser.add_argument("--highlight", default=None,
                         help="1-based, inclusive residue range(s) to bold/"
                              "enlarge, e.g. '10-25' or '10-25,40-45' -- the "
                              "manual equivalent of the notebook's "
                              "auto-detected 'mutpos'/featpos highlighting.")
    parser.add_argument("--wrap_width", type=int, default=60,
                         help="Residues per line in the PNG/PDF render "
                              "(default: 60). Ignored for --format html, "
                              "which wraps to the browser width instead.")
    parser.add_argument("--format", default="png,pdf",
                         help="Comma-separated output format(s): png, pdf, "
                              "html (default: png,pdf).")
    parser.add_argument("--legend", dest="legend", action="store_true", default=True,
                         help="Show a biochemical-group color legend under "
                              "the PNG/PDF figure (default: on).")
    parser.add_argument("--no_legend", dest="legend", action="store_false")
    parser.add_argument("--arial_font_path",
                         default=str(SCRIPT_DIR.parent / "Arial.ttf"),
                         help="Path to Arial.ttf for titles/labels (editable-"
                              "text PDF output). Falls back to DejaVu Sans "
                              "if not found.")
    parser.add_argument("--mono_font_path", default=None,
                         help="Path to a monospace .ttf for the sequence "
                              "letters themselves (needed so residues line "
                              "up in the wrapped grid). Falls back to "
                              "DejaVu Sans Mono / Courier New / monospace.")
    parser.add_argument("--out_prefix", default=None,
                         help="Output path prefix (no extension). Default: "
                              "derived from the IDR_ID.")
    args = parser.parse_args()

    record = load_row(args.input, args.idr_id, args.index)
    seq = record["Sequence"]
    highlight_positions = parse_highlight_ranges(args.highlight)
    out_of_range = {p for p in highlight_positions if p >= len(seq)}
    if out_of_range:
        raise ValueError(f"--highlight includes position(s) beyond the "
                          f"sequence length ({len(seq)} aa): "
                          f"{sorted(p + 1 for p in out_of_range)[:5]}...")

    aacolor_map = build_aacolor_map(args.color_set)
    title = build_title(record)
    formats = {f.strip().lower() for f in args.format.split(",") if f.strip()}
    unknown = formats - {"png", "pdf", "html"}
    if unknown:
        raise ValueError(f"Unknown --format value(s): {unknown}")

    out_prefix = args.out_prefix or str(record["IDR_ID"])
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    if "html" in formats:
        html = render_html(seq, aacolor_map, highlight_positions, title)
        html_path = f"{out_prefix}.colored_sequence.html"
        Path(html_path).write_text(html)
        print(f"Saved -> {html_path}")

    if {"png", "pdf"} & formats:
        label_family = setup_plot_style(args.arial_font_path)
        fig = render_matplotlib(seq, aacolor_map, highlight_positions, title,
                                 args.wrap_width, label_family,
                                 args.mono_font_path, args.color_set, args.legend)
        if "png" in formats:
            png_path = f"{out_prefix}.colored_sequence.png"
            fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
            print(f"Saved -> {png_path}")
        if "pdf" in formats:
            pdf_path = f"{out_prefix}.colored_sequence.pdf"
            fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
            print(f"Saved -> {pdf_path}")
        plt.close(fig)

    print(f"\n{title}")
    print(f"{len(seq)} aa, color_set='{args.color_set}', "
          f"{len(highlight_positions)} residue(s) highlighted")


if __name__ == "__main__":
    main()
