#!/usr/bin/env python
# coding: utf-8
"""
NARDINI GIN Summary: Annotate, Parse to Long Format, Post-Process
====================================================================
Consolidates four former scripts into one, adapted for the new
nardini_MultiBlock_v099.py out_summary.csv, which already carries its
metadata columns (UniprotID, Uniprot name, Description, Gene name,
IDR_START, IDR_END, IDR_length) via metadata-passthrough -- so the old
00_IDR_annotations.py left-join step no longer has anything to do and is
dropped entirely.

Subcommands (chain them with `all`, or run any stage standalone):
  blocks   Explode out_summary's domain_details (one row per IDR) into a
           block-level long-format table (one row per IDR block). Replaces
           01_parse_clusters.py -- fully vectorized (pandas explode +
           json_normalize) instead of a per-row .apply + a redundant
           self-merge, and exposes every field the new block dicts carry
           (block_length, valley_score, valley_start/end, num_windows),
           not just the three the old script hardcoded.
  domains  Optional: build a UniProt domain-architecture long table (Total
           protein / Domain / IDR / IDR block rows) from an idmapping TSV
           + the block-level table. Replaces 02_parse_domains.py, minus
           its separate --idrs argument -- IDR_START/END/metadata are
           already on the summary, so that lookup table is redundant now.
  plots    Generalized analysis/plot suite replacing the notebook-style
           03_postProcessing.py: every cluster order, cluster pair, and
           file path that used to be hardcoded is now a CLI argument.
  all      Runs blocks -> [domains] -> plots in one process, keeping
           DataFrames in memory between stages instead of round-tripping
           through disk.

v 1.0 by Alexandre P. Magalhaes
"""

import argparse
import ast
import os
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
mpl.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = SCRIPT_DIR / "data"


def _default_arial_path() -> "str | None":
    """Arial.ttf to use for plots: $TESSERAIDR_ARIAL_TTF, else ./Arial.ttf,
    else the repository root (when run from a source checkout). If none
    exists, matplotlib falls back to a system Arial, then DejaVu Sans."""
    for cand in (os.environ.get("TESSERAIDR_ARIAL_TTF"), "Arial.ttf",
                 str(SCRIPT_DIR.parent.parent / "Arial.ttf")):
        if cand and Path(cand).exists():
            return cand
    return None
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"

# ══════════════════════════════════════════════════════════════════════════
#  Shared plot styling (consistent with the rest of this project's figures:
#  editable-text PDFs, Arial labels, light theme). The three near-duplicate
#  heatmap skins and the dark "space" theme in the original notebook are
#  consolidated to this one look -- see README for that decision.
# ══════════════════════════════════════════════════════════════════════════

def setup_plot_style(arial_font_path: "str | None") -> str:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    family = "DejaVu Sans"
    if arial_font_path and Path(arial_font_path).exists():
        font_manager.fontManager.addfont(arial_font_path)
        family = font_manager.FontProperties(fname=arial_font_path).get_name()
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [family, "Arial", "DejaVu Sans"]
    return family


def savefig(fig, out_prefix: str):
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_prefix}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_prefix}.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════
#  IO helpers
# ══════════════════════════════════════════════════════════════════════════

def load_table(filepath: str) -> pd.DataFrame:
    """Auto-detect separator: UniProt exports are TSV, everything else CSV."""
    root, ext = os.path.splitext(str(filepath).lower())
    if ext in (".gz", ".bz2", ".xz", ".zst", ".zip"):
        # Look past the compression suffix: "x.tsv.gz" is still a TSV.
        ext = os.path.splitext(root)[1]
    sep = "\t" if ext in (".tsv", ".txt") else ","
    return pd.read_csv(filepath, sep=sep)


def table_stem(filepath: str) -> str:
    """File name without its table and compression suffixes: x.csv.gz -> x."""
    name = Path(filepath).name
    for suffix in (".gz", ".bz2", ".xz", ".zst", ".zip"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return Path(name).stem


def parse_if_string(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    return val


# ══════════════════════════════════════════════════════════════════════════
#  STAGE 1: out_summary.csv -> block-level long format
#  (replaces 01_parse_clusters.py)
# ══════════════════════════════════════════════════════════════════════════

BLOCK_DICT_FIELDS = ["cluster", "block_start", "block_end", "block_length",
                     "valley_dist", "valley_score", "valley_start",
                     "valley_end", "num_windows"]


def explode_summary_to_blocks(summary_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (IDR_ID, block) instead of one row per IDR.

    domain_details already carries every per-block field the block-building
    step computed (block dicts from nardini_MultiBlock_v099.py's
    build_summary / build_summary_seed_expand); clusters_ordered is exactly
    [d['cluster'] for d in domain_details] and is dropped here as
    redundant. Uses pandas' vectorized explode + json_normalize rather than
    a per-row .apply -- the old script's self-merge back onto the
    un-exploded frame is also gone, since explode already keeps every other
    column attached to each new row.
    """
    df = summary_df.copy()

    if "IDR_ID" in df.columns:
        dupe_mask = df.duplicated(subset="IDR_ID", keep=False)
        if dupe_mask.any():
            n_dupe_ids = df.loc[dupe_mask, "IDR_ID"].nunique()
            n_dupe_rows = int(df.duplicated(subset="IDR_ID", keep="first").sum())
            print(f"Note: {n_dupe_ids} IDR_ID(s) had duplicate rows in the input "
                  f"summary ({n_dupe_rows} extra row(s) dropped, keeping the first "
                  f"occurrence) -- otherwise their blocks would be double-counted "
                  f"in every downstream co-occurrence/statistics step.")
            df = df.drop_duplicates(subset="IDR_ID", keep="first")

    df["domain_details"] = df["domain_details"].apply(parse_if_string)

    no_blocks = ~df["domain_details"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    if no_blocks.any():
        print(f"Note: {int(no_blocks.sum())} row(s) have no parsed blocks "
              f"and are excluded from the long-format table.")

    exploded = df.loc[~no_blocks].explode("domain_details", ignore_index=False)

    # IDR_block: 0-based position of this block within its IDR, matching
    # domain_details' left-to-right (start-sorted) order.
    exploded["IDR_block"] = exploded.groupby(level=0).cumcount()

    block_fields = pd.json_normalize(exploded["domain_details"].tolist())
    missing = [c for c in BLOCK_DICT_FIELDS if c not in block_fields.columns]
    if missing:
        print(f"Note: block dicts are missing field(s) {missing} "
              f"(older summary format?) -- continuing without them.")
    block_fields.index = exploded.index

    result = pd.concat(
        [exploded.drop(columns=["domain_details", "clusters_ordered"], errors="ignore"),
         block_fields],
        axis=1,
    )

    if "IDR_START" in result.columns and "block_start" in result.columns:
        result["IDR_block_start"] = result["IDR_START"] + result["block_start"]
        result["IDR_block_end"] = result["IDR_START"] + result["block_end"]

    return result.reset_index(drop=True)


def cmd_blocks(args):
    print(f"Reading {args.summary} ...")
    summary_df = pd.read_csv(args.summary)
    blocks_df = explode_summary_to_blocks(summary_df)
    out = args.out or f"{table_stem(args.summary)}_blocks.csv"
    blocks_df.to_csv(out, index=False)
    print(f"{len(summary_df)} IDRs -> {len(blocks_df)} blocks. Saved -> {out}")
    return blocks_df


# ══════════════════════════════════════════════════════════════════════════
#  STAGE 2 (optional): UniProt domain architecture
#  (replaces 02_parse_domains.py; no separate --idrs table needed anymore --
#  IDR_START/END and metadata are already columns on the summary/blocks df)
# ══════════════════════════════════════════════════════════════════════════

import re


def _parse_uniprot_domains(idmapping: pd.DataFrame) -> pd.DataFrame:
    """Pfam-style Domain [FT] free-text field -> one row per DOMAIN record.
    This step is left as a per-row parse (not vectorized): the field packs
    a variable number of 'DOMAIN start..end /note="name"' records per
    protein as free text, which pandas' vectorized string ops can't cleanly
    split without re-implementing a small parser either way.

    Expects idmapping to already carry a "UniprotID" column (i.e. called
    after build_domain_architecture's From -> UniprotID rename).
    """
    rows = []
    dom_col = idmapping["Domain [FT]"].astype(str)
    has_domain = dom_col.str.contains("DOMAIN", na=False)
    for uid, dom_str in zip(idmapping.loc[has_domain, "UniprotID"],
                             dom_col[has_domain]):
        for dom in dom_str.split("DOMAIN ")[1:]:
            match_se = re.match(r"^[<>]?(\d+)\.\.[<>]?(\d+)", dom)
            if not match_se:
                continue
            start, end = int(match_se.group(1)), int(match_se.group(2))
            match_name = re.search(r'/note="([^"]+)"', dom)
            dname = match_name.group(1) if match_name else "Unknown Domain"
            rows.append({"UniprotID": str(uid).strip(), "Domain name": dname,
                         "Start": start, "End": end})
    return pd.DataFrame(rows, columns=["UniprotID", "Domain name", "Start", "End"])


def build_domain_architecture(blocks_df: pd.DataFrame, idmapping: pd.DataFrame) -> pd.DataFrame:
    idmapping = idmapping.rename(columns={
        "From": "UniprotID", "Entry Name": "Uniprot name",
        "Protein names": "Description", "Gene Names": "Gene name",
    })
    idmapping["UniprotID"] = idmapping["UniprotID"].astype(str).str.strip()
    base_cols = ["UniprotID", "Uniprot name", "Description", "Gene name", "Organism"]
    base_info = idmapping[base_cols].drop_duplicates("UniprotID").set_index("UniprotID")

    # -- Total protein rows (one per UniProt entry) --
    total_df = idmapping[base_cols + ["Length"]].drop_duplicates("UniprotID").copy()
    total_df["Domain type"] = "Total protein"
    total_df["Domain name"] = total_df["Gene name"]
    total_df["Start"] = 0
    total_df["End"] = pd.to_numeric(total_df["Length"], errors="coerce").fillna(0).astype(int)

    # -- Domain rows, parsed from Domain [FT] --
    domain_df = _parse_uniprot_domains(idmapping)
    domain_df = domain_df.join(base_info, on="UniprotID")
    domain_df["Domain type"] = "Domain"
    domain_df["Length"] = domain_df["End"] - domain_df["Start"]

    # -- IDR rows: one per unique IDR_ID, already fully annotated on blocks_df --
    idr_cols = ["UniprotID", "Uniprot name", "Description", "Gene name",
                "IDR_ID", "IDR_START", "IDR_END"]
    idr_df = blocks_df[[c for c in idr_cols if c in blocks_df.columns]].drop_duplicates("IDR_ID").copy()
    idr_df = idr_df.join(base_info["Organism"], on="UniprotID")
    idr_df["Domain type"] = "IDR"
    idr_df = idr_df.rename(columns={"IDR_ID": "Domain name",
                                     "IDR_START": "Start", "IDR_END": "End"})
    idr_df["Length"] = idr_df["End"] - idr_df["Start"]

    # -- IDR block rows: one per GIN block, from the exploded blocks table --
    block_df = blocks_df[[c for c in ["UniprotID", "Uniprot name", "Description",
                                       "Gene name", "cluster", "IDR_block_start",
                                       "IDR_block_end"] if c in blocks_df.columns]].copy()
    block_df = block_df.join(base_info["Organism"], on="UniprotID")
    block_df["Domain type"] = "IDR block"
    # Bare cluster number, matching the original 02_parse_domains.py
    # convention -- not "GIN <n>".
    block_df["Domain name"] = block_df["cluster"].astype("Int64").astype(str)
    block_df = block_df.rename(columns={"IDR_block_start": "Start", "IDR_block_end": "End"})
    block_df["Length"] = block_df["End"] - block_df["Start"]

    columns_order = ["UniprotID", "Uniprot name", "Description", "Gene name",
                      "Domain type", "Domain name", "Start", "End", "Organism", "Length"]
    out = pd.concat([total_df, domain_df, idr_df, block_df], ignore_index=True)
    return out[columns_order]


def cmd_domains(args, blocks_df=None):
    if blocks_df is None:
        blocks_df = explode_summary_to_blocks(pd.read_csv(args.summary))
    print(f"Reading {args.idmapping} ...")
    idmapping = load_table(args.idmapping)
    arch_df = build_domain_architecture(blocks_df, idmapping)
    out = args.out or "domain_architecture.csv"
    arch_df.to_csv(out, index=False)
    print(f"{len(arch_df)} domain-architecture rows. Saved -> {out}")
    return arch_df


# ══════════════════════════════════════════════════════════════════════════
#  STAGE 3: generalized plot / analysis suite
#  (replaces 03_postProcessing.py)
# ══════════════════════════════════════════════════════════════════════════

def default_cluster_order(blocks_df: pd.DataFrame, top_n: "int | None" = None) -> list:
    """All clusters seen in the data, ranked by frequency (most common
    first) -- used whenever --cluster_order isn't given explicitly."""
    counts = blocks_df["cluster"].value_counts()
    order = counts.index.astype(int).tolist()
    return order[:top_n] if top_n else order


def parse_cluster_list(clusters_ordered_series: pd.Series) -> pd.Series:
    """clusters_ordered is either already-parsed (from an in-memory blocks
    run) or a stringified list (loaded back from CSV)."""
    return clusters_ordered_series.apply(parse_if_string)


def plot_n_domains_distribution(summary_df: pd.DataFrame, out_prefix: str, label_family: str):
    binned = summary_df["n_distinct_clusters"].apply(lambda x: "5+" if x >= 5 else str(int(x)))
    under5 = sorted(summary_df.loc[summary_df["n_distinct_clusters"] < 5,
                                    "n_distinct_clusters"].dropna().unique())
    order = [str(int(x)) for x in under5] + ["5+"]

    fig, ax = plt.subplots(figsize=(4, 4.5))
    sns.countplot(x=binned, order=order, color="#3B6FA0", ax=ax)
    ax.set_title("Frequency distribution of distinct GIN clusters per IDR",
                 fontsize=11.5, color=INK, fontfamily=label_family)
    ax.set_xlabel("Number of distinct clusters", fontsize=10.5, color=INK, fontfamily=label_family)
    ax.set_ylabel("Number of IDRs", fontsize=10.5, color=INK, fontfamily=label_family)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, out_prefix)


def plot_valley_dist_distribution(blocks_df: pd.DataFrame, out_prefix: str, label_family: str,
                                   compare_df: "pd.DataFrame | None" = None,
                                   compare_label: str = "Comparison",
                                   this_label: str = "This dataset"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(data=blocks_df, x="valley_dist", kde=True, color="#3B6FA0",
                 bins=40, alpha=0.45, label=this_label, ax=ax)
    mean1 = blocks_df["valley_dist"].mean()
    ax.axvline(mean1, color="#3B6FA0", linestyle="--", alpha=0.8)

    if compare_df is not None:
        sns.histplot(data=compare_df, x="valley_dist", kde=True, color="#C1666B",
                     bins=40, alpha=0.45, label=compare_label, ax=ax)
        mean2 = compare_df["valley_dist"].mean()
        ax.axvline(mean2, color="#C1666B", linestyle="--", alpha=0.8)
        trans = ax.get_xaxis_transform()
        ax.text(mean1, 0.94, f" mean = {mean1:.2f}", color="#3B6FA0", transform=trans, fontweight="bold")
        ax.text(mean2, 0.88, f" mean = {mean2:.2f}", color="#C1666B", transform=trans, fontweight="bold")
        ax.legend()
        title = "Valley distance: distribution comparison"
    else:
        trans = ax.get_xaxis_transform()
        ax.text(mean1, 0.94, f" mean = {mean1:.2f}", color="#3B6FA0", transform=trans, fontweight="bold")
        title = "Valley distance distribution"

    ax.set_title(title, fontsize=12, color=INK, fontfamily=label_family)
    ax.set_xlabel("valley_dist  (squared distance to assigned centroid)",
                  fontsize=10.5, color=INK, fontfamily=label_family)
    ax.set_ylabel("Number of blocks", fontsize=10.5, color=INK, fontfamily=label_family)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, out_prefix)


def build_cooccurrence_matrix(blocks_df: pd.DataFrame, cluster_order: list,
                               max_n_domains: "int | None" = None,
                               group_col: str = "IDR_ID") -> np.ndarray:
    """Symmetric co-occurrence count matrix: entry (i, j) = number of groups
    (IDRs by default) whose block clusters include both cluster_order[i]
    and cluster_order[j] (diagonal left at zero, matching the original's
    "no self pairs").

    Pass group_col="UniprotID" for the protein-level view: two clusters
    then "co-occur" for a protein if it has one IDR carrying the first
    cluster and any IDR (the same one or a different one) carrying the
    second -- i.e. all of that protein's IDRs pooled into one set, the same
    pooling export_anchor_combo_proteins already used at the protein level.
    """
    n = len(cluster_order)
    idx = {c: i for i, c in enumerate(cluster_order)}
    allowed = set(cluster_order)

    per_group_clusters = blocks_df.groupby(group_col)["cluster"].apply(
        lambda s: sorted(set(int(c) for c in s if int(c) in allowed))
    )
    if max_n_domains is not None:
        n_domains = blocks_df.groupby(group_col)["cluster"].size()
        per_group_clusters = per_group_clusters[per_group_clusters.index.map(
            lambda i: n_domains.get(i, 0) <= max_n_domains)]

    cooc = defaultdict(int)
    for clusters in per_group_clusters:
        for a, b in combinations(clusters, 2):
            cooc[(a, b)] += 1

    mat = np.zeros((n, n), dtype=int)
    for (a, b), count in cooc.items():
        i, j = idx[a], idx[b]
        mat[i, j] = mat[j, i] = count
    return mat, cooc


def plot_cooccurrence_heatmap(mat: np.ndarray, cluster_order: list, out_prefix: str,
                               label_family: str, title_suffix: str = ""):
    n = len(cluster_order)
    labels = [str(c) for c in cluster_order]
    df_heat = pd.DataFrame(mat, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.55), max(5.5, n * 0.5)))
    masked = np.ma.masked_where(mat == 0, mat)
    vmax = mat.max() if mat.max() > 0 else 1
    im = ax.imshow(masked, cmap="BuPu", norm=LogNorm(vmin=1, vmax=vmax), aspect="equal")
    im.cmap.set_bad("#f2f2f2")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Co-occurrence count (log scale)", color=INK, fontsize=10)

    for i in range(n):
        for j in range(n):
            if mat[i, j] > 0:
                brightness = np.log10(mat[i, j]) / np.log10(vmax) if vmax > 1 else 0
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=7.5,
                        color="white" if brightness > 0.55 else INK, fontweight="bold")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9, fontfamily=label_family)
    ax.set_yticklabels(labels, fontsize=9, fontfamily=label_family)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.invert_yaxis()
    ax.set_title(f"Cluster co-occurrence{title_suffix}", fontsize=13, color=INK,
                 fontfamily=label_family, pad=14)
    ax.set_xlabel("Cluster ID", fontsize=10.5, color=INK, fontfamily=label_family)
    ax.set_ylabel("Cluster ID", fontsize=10.5, color=INK, fontfamily=label_family)
    fig.tight_layout()
    savefig(fig, out_prefix)
    df_heat.to_csv(f"{out_prefix}.matrix.csv")
    print(f"Saved -> {out_prefix}.matrix.csv")


def compute_pair_statistics(blocks_df: pd.DataFrame, cluster_order: list,
                             max_n_domains: "int | None" = None,
                             group_col: str = "IDR_ID") -> pd.DataFrame:
    """Per-pair Frequency, Jaccard index, and Haldane-Anscombe-corrected
    odds ratio, over groups (IDRs by default) restricted to cluster_order
    (and optionally --max_n_domains). Replaces the hardcoded cell 24 of the
    old notebook. Pass group_col="UniprotID" for the protein-level view --
    see build_cooccurrence_matrix."""
    allowed = set(cluster_order)
    per_group = blocks_df.groupby(group_col)["cluster"].apply(
        lambda s: sorted(set(int(c) for c in s if int(c) in allowed)))
    if max_n_domains is not None:
        n_domains = blocks_df.groupby(group_col)["cluster"].size()
        per_group = per_group[per_group.index.map(lambda i: n_domains.get(i, 0) <= max_n_domains)]

    n_total = len(per_group)
    cluster_counts = {c: 0 for c in cluster_order}
    pair_counts = defaultdict(int)
    for clusters in per_group:
        for c in set(clusters):
            cluster_counts[c] += 1
        for a, b in combinations(sorted(set(clusters)), 2):
            pair_counts[(a, b)] += 1

    rows = []
    for (c1, c2), n_ab in pair_counts.items():
        n_a, n_b = cluster_counts[c1], cluster_counts[c2]
        jaccard = n_ab / (n_a + n_b - n_ab) if (n_a + n_b - n_ab) > 0 else 0.0
        b_, c_, d_ = n_a - n_ab, n_b - n_ab, n_total - (n_ab + (n_a - n_ab) + (n_b - n_ab))
        odds_ratio = ((n_ab + 0.5) * (d_ + 0.5)) / ((b_ + 0.5) * (c_ + 0.5))
        rows.append(dict(Cluster_1=c1, Cluster_2=c2, Frequency=n_ab,
                          Jaccard_Index=jaccard, Odds_Ratio=odds_ratio))
    return pd.DataFrame(rows)


def plot_cooccurrence_dotplot(pair_stats: pd.DataFrame, cluster_order: list,
                               out_prefix: str, label_family: str, title_suffix: str = ""):
    """size = Jaccard index, color = frequency (log scale). Purely from the
    summary data -- no external localization data needed."""
    idx = {c: i for i, c in enumerate(cluster_order)}
    n = len(cluster_order)
    x, y, freq, jacc = [], [], [], []
    for _, r in pair_stats.iterrows():
        for xi, yi in [(idx[r.Cluster_1], idx[r.Cluster_2]), (idx[r.Cluster_2], idx[r.Cluster_1])]:
            x.append(xi); y.append(yi); freq.append(r.Frequency); jacc.append(r.Jaccard_Index)
    x, y, freq, jacc = map(np.array, (x, y, freq, jacc))

    fig, ax = plt.subplots(figsize=(max(7, n * 0.6), max(6, n * 0.5)))
    sizes = 40 + jacc * 1200
    sc = ax.scatter(x, y, s=sizes, c=freq, cmap="viridis", norm=LogNorm(),
                     alpha=0.85, edgecolors="black", linewidth=0.6)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label("Co-occurrence frequency (log scale)", fontsize=10, color=INK)

    for jv in (0.1, 0.3, 0.6, 1.0):
        ax.scatter([], [], s=40 + jv * 1200, c="gray", alpha=0.5,
                   edgecolors="black", label=f"{jv:.1f}")
    ax.legend(title="Jaccard index", loc="upper left", bbox_to_anchor=(1.2, 1),
              labelspacing=1.3, borderpad=1, fontsize=9)

    labels = [str(c) for c in cluster_order]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontfamily=label_family)
    ax.set_yticklabels(labels, fontfamily=label_family)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("Cluster ID", fontsize=10.5, color=INK, fontfamily=label_family)
    ax.set_ylabel("Cluster ID", fontsize=10.5, color=INK, fontfamily=label_family)
    ax.set_title(f"Cluster co-occurrence  ·  size = Jaccard  ·  color = frequency{title_suffix}",
                 fontsize=11.5, color=INK, fontfamily=label_family)
    fig.tight_layout()
    savefig(fig, out_prefix)


# -- Optional: localization co-eligibility log2(OR) dot plot -----------------

def _parse_semicolon_locs(val) -> set:
    if pd.isna(val):
        return set()
    return {l.strip() for l in str(val).split(";") if l.strip()}


def _parse_list_locs(val) -> set:
    if pd.isna(val):
        return set()
    try:
        return {l.strip() for l in ast.literal_eval(val)}
    except (ValueError, SyntaxError):
        return set()


def plot_localization_cooccurrence_dotplot(blocks_df: pd.DataFrame, cluster_order: list,
                                            hpa_csv: str, gin_definitions_csv: str,
                                            out_prefix: str, label_family: str,
                                            max_n_domains: "int | None" = None,
                                            cap: float = 5.0):
    """size = raw co-occurrence count, color = log2(odds ratio) of the two
    clusters' HPA-localization-eligible protein sets overlapping. Needs
    --hpa_csv (columns: Uniprot, Main location, Additional location) and
    --gin_definitions (columns: GIN Type, Localization)."""
    hpa = pd.read_csv(hpa_csv)
    hpa = hpa[hpa["Uniprot"].isin(blocks_df["UniprotID"])].copy()
    main_locs = hpa["Main location"].apply(_parse_semicolon_locs)
    add_locs = hpa["Additional location"].apply(_parse_semicolon_locs)
    # NOTE: pandas' `|` on object-dtype Series of sets does NOT do an
    # element-wise set union -- it treats each set's truthiness and
    # returns a bool Series. Must combine element-wise instead.
    hpa["all_locs"] = [m | a for m, a in zip(main_locs, add_locs)]
    n_hpa = len(hpa)
    loc_to_proteins = defaultdict(set)
    for _, row in hpa.iterrows():
        for loc in row["all_locs"]:
            loc_to_proteins[loc].add(row["Uniprot"])

    gin = pd.read_csv(gin_definitions_csv)
    gin_locs = {row["GIN Type"]: _parse_list_locs(row["Localization"]) for _, row in gin.iterrows()}

    def eligible(clust):
        locs = gin_locs.get(clust, set())
        return set.union(*(loc_to_proteins.get(l, set()) for l in locs)) if locs else set()

    mat, cooc = build_cooccurrence_matrix(blocks_df, cluster_order, max_n_domains)
    pair_data = {}
    for (a, b), count in cooc.items():
        ea, eb = eligible(a), eligible(b)
        if ea and eb:
            both, only_a, only_b = len(ea & eb), len(ea - eb), len(eb - ea)
            neither = n_hpa - len(ea | eb)
            log2or = np.log2((both + 0.5) * (neither + 0.5) / ((only_a + 0.5) * (only_b + 0.5)))
        else:
            log2or = np.nan
        pair_data[(a, b)] = {"count": count, "log2or": log2or}

    idx = {c: i for i, c in enumerate(cluster_order)}
    n = len(cluster_order)
    xs, ys, sizes, colors_val, has_or = [], [], [], [], []
    for (a, b), d in pair_data.items():
        for xi, yi in [(idx[b], idx[a]), (idx[a], idx[b])]:
            xs.append(xi); ys.append(yi); sizes.append(d["count"])
            colors_val.append(d["log2or"]); has_or.append(not np.isnan(d["log2or"]))
    xs, ys, sizes = np.array(xs), np.array(ys), np.array(sizes, dtype=float)
    colors_val, has_or = np.array(colors_val, dtype=float), np.array(has_or)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.65), max(7, n * 0.55)))
    dot_sizes = 40 + sizes * 260 / max(sizes.max(), 1)
    colors_capped = np.clip(colors_val, -cap, cap)
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#2166ac", "#4393c3", "#92c5de", "#f7f7f7", "#f4a582", "#d6604d", "#b2182b"], N=512)
    norm = TwoSlopeNorm(vmin=-cap, vcenter=0, vmax=cap)

    for xi in range(n):
        ax.axvline(xi, color=GRID, lw=0.5, zorder=0)
        ax.axhline(xi, color=GRID, lw=0.5, zorder=0)
    if (~has_or).any():
        ax.scatter(xs[~has_or], ys[~has_or], s=dot_sizes[~has_or], color="#c7c9d9",
                   edgecolors="#9698ad", linewidths=0.5, zorder=3, alpha=0.85)
    if has_or.any():
        sc = ax.scatter(xs[has_or], ys[has_or], s=dot_sizes[has_or], c=colors_capped[has_or],
                        cmap=cmap, norm=norm, edgecolors="#444", linewidths=0.4, zorder=4)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.03)
        cbar.set_label("log2(odds ratio)\n[localization co-eligibility]", fontsize=9, color=INK)

    labels = [str(c) for c in cluster_order]
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9, fontfamily=label_family)
    ax.set_yticklabels(labels, fontsize=9, fontfamily=label_family)
    ax.invert_yaxis()

    max_count = max(int(sizes.max()), 1)
    # A handful of representative sizes (min/25th/median/75th/max of the
    # actual counts), not one legend entry per unique value -- with real
    # data there can be dozens of distinct counts, which made the legend
    # unreadable and pushed it off the figure.
    uniq_sizes = np.array(sorted(set(int(s) for s in sizes)))
    if len(uniq_sizes) <= 5:
        size_vals = uniq_sizes.tolist()
    else:
        qs = np.quantile(uniq_sizes, [0, 0.25, 0.5, 0.75, 1.0])
        size_vals = sorted(set(int(round(q)) for q in qs))
    size_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#888aaa",
                            markeredgecolor="#555", markersize=np.sqrt(40 + v * 260 / max_count),
                            label=f"n = {v}", linestyle="None") for v in size_vals]
    leg1 = ax.legend(handles=size_handles, title="Co-occurrence\ncount", title_fontsize=8,
                     fontsize=8, loc="lower right")
    grey_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="#c7c9d9",
                         markeredgecolor="#9698ad", markersize=7,
                         label="No localization\nassignment", linestyle="None")
    ax.legend(handles=[grey_handle], loc="upper right", fontsize=8)
    ax.add_artist(leg1)

    ax.set_title("Cluster co-occurrence  ·  size = count  ·  color = log2(OR) (HPA localization)",
                 fontsize=11, color=INK, fontfamily=label_family, pad=12)
    ax.set_xlabel("Cluster ID", fontsize=10.5, color=INK, fontfamily=label_family)
    ax.set_ylabel("Cluster ID", fontsize=10.5, color=INK, fontfamily=label_family)
    fig.tight_layout()
    savefig(fig, out_prefix)


# -- Dataset exports (generalized from cells 19-23) --------------------------

def export_diff_signature_proteins(summary_df: pd.DataFrame, out_csv: str):
    """Proteins with >1 IDR whose GIN block signatures (clusters_ordered)
    differ from each other. Fully generic already in the original -- no
    hardcoded values to remove."""
    sigs = summary_df.groupby("UniprotID")["clusters_ordered"].nunique()
    multi_idr_diff = sigs[(summary_df.groupby("UniprotID").size() > 1) & (sigs > 1)].index
    out = summary_df[summary_df["UniprotID"].isin(multi_idr_diff)]
    out.to_csv(out_csv, index=False)
    print(f"{out['UniprotID'].nunique()} protein(s), {len(out)} IDR row(s) "
          f"with differing IDR signatures. Saved -> {out_csv}")
    return out


METADATA_COLS = ["Uniprot name", "Description", "Gene name"]


def export_pair_uniprot_ids(blocks_df: pd.DataFrame, c1: int, c2: int, out_csv: str):
    """UniProt IDs (with metadata) of every IDR containing both cluster c1
    and c2 -- the generalized, CLI-driven form of the old hardcoded
    '23 and 26' / '18 and 26' / '7 and 26' cells."""
    meta_cols = [c for c in METADATA_COLS if c in blocks_df.columns]
    agg = {"UniprotID": ("UniprotID", "first"),
           "clusters": ("cluster", lambda s: set(int(c) for c in s))}
    agg.update({c: (c, "first") for c in meta_cols})
    per_idr = blocks_df.groupby("IDR_ID").agg(**agg)
    matches = per_idr[per_idr["clusters"].apply(lambda s: c1 in s and c2 in s)].copy()
    matches["Clusters_in_IDR"] = matches["clusters"].apply(
        lambda s: ",".join(str(x) for x in sorted(s)))
    out = matches.reset_index()[["IDR_ID", "UniprotID"] + meta_cols + ["Clusters_in_IDR"]]
    out.to_csv(out_csv, index=False)
    print(f"{len(out)} IDR(s) with both cluster {c1} and {c2}. Saved -> {out_csv}")
    return out


def export_pair_uniprot_ids_by_protein(blocks_df: pd.DataFrame, c1: int, c2: int, out_csv: str):
    """Protein-level counterpart of export_pair_uniprot_ids: matches a
    protein if cluster c1 and cluster c2 each appear somewhere across ANY
    of that protein's IDRs (pooled together), not necessarily the same
    IDR -- e.g. c1 in IDR_1 and c2 in IDR_2 of the same protein still
    counts. Same pooling as export_anchor_combo_proteins/the protein-level
    heatmap (group_col="UniprotID")."""
    meta_cols = [c for c in METADATA_COLS if c in blocks_df.columns]
    agg = {"clusters": ("cluster", lambda s: set(int(c) for c in s)),
           "IDR_IDs": ("IDR_ID", lambda s: sorted(set(s)))}
    agg.update({c: (c, "first") for c in meta_cols})
    per_protein = blocks_df.groupby("UniprotID").agg(**agg)
    matches = per_protein[per_protein["clusters"].apply(lambda s: c1 in s and c2 in s)].copy()
    matches["Clusters_present"] = matches["clusters"].apply(
        lambda s: ",".join(str(x) for x in sorted(s)))
    matches["IDR_IDs"] = matches["IDR_IDs"].apply(",".join)
    out = matches.reset_index()[["UniprotID"] + meta_cols + ["Clusters_present", "IDR_IDs"]]
    out.to_csv(out_csv, index=False)
    print(f"{len(out)} protein(s) with both cluster {c1} and {c2} across all their IDRs. "
          f"Saved -> {out_csv}")
    return out


def export_anchor_combo_proteins(blocks_df: pd.DataFrame, anchor: int, others: list, out_csv: str):
    """Proteins (with metadata) where some IDR of theirs contains `anchor`
    together with at least one cluster from `others` (generalizes the old
    hardcoded '26 with 23, or 26 with 18, or 26 with 7' rule -- pass any
    anchor and any list of partner clusters)."""
    meta_cols = [c for c in METADATA_COLS if c in blocks_df.columns]
    agg = {"clusters": ("cluster", lambda s: set(int(c) for c in s)),
           "IDR_IDs": ("IDR_ID", lambda s: sorted(set(s)))}
    agg.update({c: (c, "first") for c in meta_cols})
    per_protein = blocks_df.groupby("UniprotID").agg(**agg)
    others_set = set(others)
    matches = per_protein[per_protein["clusters"].apply(
        lambda s: anchor in s and bool(s & others_set))].copy()
    matches["Clusters_present"] = matches["clusters"].apply(
        lambda s: ",".join(str(x) for x in sorted(s)))
    matches["IDR_IDs"] = matches["IDR_IDs"].apply(",".join)
    out = matches.reset_index()[["UniprotID"] + meta_cols + ["Clusters_present", "IDR_IDs"]]
    out.to_csv(out_csv, index=False)
    print(f"{len(out)} protein(s) with cluster {anchor} + any of {sorted(others_set)}. "
          f"Saved -> {out_csv}")
    return out


# ══════════════════════════════════════════════════════════════════════════
#  plots subcommand driver
# ══════════════════════════════════════════════════════════════════════════

def cmd_plots(args, blocks_df=None, summary_df=None):
    if blocks_df is None:
        blocks_df = pd.read_csv(args.blocks)
    if summary_df is None and args.summary:
        summary_df = pd.read_csv(args.summary)

    label_family = setup_plot_style(args.arial_font_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if summary_df is not None:
        plot_n_domains_distribution(summary_df, str(outdir / "n_domains_distribution"), label_family)
    else:
        print("Skipping n_domains distribution (pass --summary for this plot).")

    compare_df = pd.read_csv(args.compare_blocks) if args.compare_blocks else None
    plot_valley_dist_distribution(blocks_df, str(outdir / "valley_dist_distribution"),
                                   label_family, compare_df=compare_df,
                                   compare_label=args.compare_label)

    cluster_order = ([int(c) for c in args.cluster_order.split(",")] if args.cluster_order
                      else default_cluster_order(blocks_df, top_n=args.top_n_clusters))
    print(f"Using cluster order ({len(cluster_order)} clusters): {cluster_order}")

    # IDR-level: two clusters "co-occur" only if the SAME IDR carries both.
    mat, _ = build_cooccurrence_matrix(blocks_df, cluster_order, args.max_n_domains)
    plot_cooccurrence_heatmap(mat, cluster_order, str(outdir / "cluster_cooccurrence_heatmap"),
                               label_family)

    pair_stats = compute_pair_statistics(blocks_df, cluster_order, args.max_n_domains)
    pair_stats.to_csv(outdir / "cluster_pair_statistics.csv", index=False)
    print(f"Saved -> {outdir / 'cluster_pair_statistics.csv'}")
    plot_cooccurrence_dotplot(pair_stats, cluster_order,
                               str(outdir / "cluster_cooccurrence_dotplot"), label_family)

    # Protein-level: two clusters "co-occur" if the protein has them
    # ANYWHERE across all of its IDRs pooled together (same IDR or not) --
    # a second view alongside the IDR-level one above, not a replacement.
    mat_prot, _ = build_cooccurrence_matrix(blocks_df, cluster_order, args.max_n_domains,
                                             group_col="UniprotID")
    plot_cooccurrence_heatmap(mat_prot, cluster_order,
                               str(outdir / "cluster_cooccurrence_heatmap_by_protein"),
                               label_family, title_suffix=" (protein level, all IDRs pooled)")

    pair_stats_prot = compute_pair_statistics(blocks_df, cluster_order, args.max_n_domains,
                                               group_col="UniprotID")
    pair_stats_prot.to_csv(outdir / "cluster_pair_statistics_by_protein.csv", index=False)
    print(f"Saved -> {outdir / 'cluster_pair_statistics_by_protein.csv'}")
    plot_cooccurrence_dotplot(pair_stats_prot, cluster_order,
                               str(outdir / "cluster_cooccurrence_dotplot_by_protein"),
                               label_family, title_suffix=" (protein level, all IDRs pooled)")

    if args.hpa_csv and args.gin_definitions:
        plot_localization_cooccurrence_dotplot(
            blocks_df, cluster_order, args.hpa_csv, args.gin_definitions,
            str(outdir / "cluster_cooccurrence_localization_dotplot"), label_family,
            max_n_domains=args.max_n_domains)
    elif args.hpa_csv or args.gin_definitions:
        print("Skipping localization dot plot: need BOTH --hpa_csv and --gin_definitions.")

    if summary_df is not None:
        export_diff_signature_proteins(summary_df, str(outdir / "diff_signature_proteins.csv"))

    for spec in args.pair_export or []:
        c1, c2 = (int(x) for x in spec.split(","))
        export_pair_uniprot_ids(blocks_df, c1, c2, str(outdir / f"pair_{c1}_{c2}_uniprot_ids.csv"))
        export_pair_uniprot_ids_by_protein(
            blocks_df, c1, c2, str(outdir / f"pair_{c1}_{c2}_uniprot_ids_by_protein.csv"))

    for spec in args.combo_export or []:
        anchor_s, others_s = spec.split(":")
        anchor = int(anchor_s)
        others = [int(x) for x in others_s.split(",")]
        export_anchor_combo_proteins(blocks_df, anchor, others,
                                      str(outdir / f"combo_{anchor}_with_{'_'.join(map(str, others))}.csv"))


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog="tesseraidr summary",
        description="Annotate/parse/plot a nardini_MultiBlock_v099.py out_summary.csv.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_blocks = sub.add_parser("blocks", help="Explode out_summary into a block-level long table.")
    p_blocks.add_argument("--summary", required=True, help="Path to out_summary.csv.")
    p_blocks.add_argument("--out", default=None, help="Output path (default: <summary>_blocks.csv).")

    p_domains = sub.add_parser("domains", help="Build the UniProt domain-architecture table.")
    p_domains.add_argument("--summary", required=True, help="Path to out_summary.csv.")
    p_domains.add_argument("--idmapping", required=True, help="UniProt idmapping TSV/CSV.")
    p_domains.add_argument("--out", default="domain_architecture.csv")

    p_plots = sub.add_parser("plots", help="Generalized analysis/plot suite.")
    p_plots.add_argument("--blocks", required=True, help="Block-level long CSV (from `blocks`).")
    p_plots.add_argument("--summary", default=None,
                          help="Optional out_summary.csv, for the n_domains plot and "
                               "the diff-signature export (per-IDR level info the "
                               "blocks table alone doesn't carry).")
    p_plots.add_argument("--outdir", default="figures")
    p_plots.add_argument("--cluster_order", default=None,
                          help="Comma-separated explicit cluster order, e.g. '7,3,19,24,...'. "
                               "Default: every cluster seen, ranked by frequency.")
    p_plots.add_argument("--top_n_clusters", type=int, default=None,
                          help="Cap the default (frequency-ranked) cluster order to the "
                               "top N most common clusters. Ignored if --cluster_order is given.")
    p_plots.add_argument("--max_n_domains", type=int, default=None,
                          help="Only include IDRs with at most this many blocks in the "
                               "co-occurrence heatmap/dot-plot/statistics (default: no cap).")
    p_plots.add_argument("--compare_blocks", default=None,
                          help="A second block-level CSV to overlay on the valley_dist plot.")
    p_plots.add_argument("--compare_label", default="Comparison",
                          help="Legend label for --compare_blocks.")
    p_plots.add_argument("--hpa_csv", default=None,
                          help="HPA subcellular-location CSV (columns: Uniprot, Main "
                               "location, Additional location). Enables the localization "
                               "log2(OR) dot plot together with --gin_definitions.")
    p_plots.add_argument("--gin_definitions", default=str(_DATA_DIR / "gin_definitions.csv.gz"),
                          help="gin_definitions.csv (columns: GIN Type, Localization). "
                               "Default: the copy bundled with tesseraidr.")
    p_plots.add_argument("--pair_export", action="append", default=None,
                          help="'C1,C2': export UniProt IDs of IDRs containing both "
                               "clusters. Repeatable.")
    p_plots.add_argument("--combo_export", action="append", default=None,
                          help="'ANCHOR:C1,C2,...': export proteins with an IDR "
                               "containing ANCHOR plus at least one of C1,C2,... "
                               "Repeatable.")
    p_plots.add_argument("--arial_font_path", default=_default_arial_path())

    p_all = sub.add_parser("all", help="Run blocks -> [domains] -> plots in one process.")
    p_all.add_argument("--summary", required=True)
    p_all.add_argument("--idmapping", default=None)
    p_all.add_argument("--outdir", default="figures")
    p_all.add_argument("--blocks_out", default=None,
                        help="Where to also save the block-level table (default: "
                             "<summary>_blocks.csv).")
    for a, kw in [
        ("--cluster_order", dict(default=None)),
        ("--top_n_clusters", dict(type=int, default=None)),
        ("--max_n_domains", dict(type=int, default=None)),
        ("--compare_blocks", dict(default=None)),
        ("--compare_label", dict(default="Comparison")),
        ("--hpa_csv", dict(default=None)),
        ("--gin_definitions", dict(default=str(_DATA_DIR / "gin_definitions.csv.gz"))),
        ("--arial_font_path", dict(default=_default_arial_path())),
    ]:
        p_all.add_argument(a, **kw)
    p_all.add_argument("--pair_export", action="append", default=None)
    p_all.add_argument("--combo_export", action="append", default=None)

    return parser


def main(argv=None, prog=None):
    parser = build_parser()
    if prog:
        parser.prog = prog
    args = parser.parse_args(argv)

    if args.command == "blocks":
        cmd_blocks(args)
    elif args.command == "domains":
        cmd_domains(args)
    elif args.command == "plots":
        cmd_plots(args)
    elif args.command == "all":
        summary_df = pd.read_csv(args.summary)
        blocks_df = explode_summary_to_blocks(summary_df)
        blocks_out = args.blocks_out or f"{table_stem(args.summary)}_blocks.csv"
        blocks_df.to_csv(blocks_out, index=False)
        print(f"{len(summary_df)} IDRs -> {len(blocks_df)} blocks. Saved -> {blocks_out}")

        if args.idmapping:
            cmd_domains(argparse.Namespace(idmapping=args.idmapping, out="domain_architecture.csv"),
                        blocks_df=blocks_df)

        cmd_plots(args, blocks_df=blocks_df, summary_df=summary_df)


if __name__ == "__main__":
    main()
