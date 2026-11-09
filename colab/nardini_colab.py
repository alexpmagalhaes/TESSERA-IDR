"""
TESSERA-IDR: glue between the Colab notebook and the repository code.

The notebook calls these functions so its cells stay short. Feature
extraction, cluster assignment and block building all come from the
pipeline modules in src/; the sequence and protein figures come from
analysis/plotting/. Written here, because they exist only in the notebook:
each block's best window, the IDRome rankings and Kolmogorov-Smirnov
comparison, and the residues behind a feature for the sequence highlight.
Everything is calibrated on the human IDRome.
"""

import io
import os
import re
import sys
import zipfile
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parent.parent
for sub in ("src", "analysis/plotting"):
    path = str(REPO / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

# nardini_summary_pipeline switches matplotlib to Agg on import, which would
# stop figures from showing inline; put the notebook's backend back.
_backend = mpl.get_backend()
import nardini_MultiBlock_v099 as multiblock          # noqa: E402
import nardini_fullSeq_numba_v096 as fullseq          # noqa: E402
import nardini_summary_pipeline as summary_pipeline   # noqa: E402
import plot_colored_sequence as colored_sequence      # noqa: E402
import plot_proteins                                  # noqa: E402
plt.switch_backend(_backend)

REFERENCE = REPO / "data" / "reference"
COMP_REF = REFERENCE / "All_sapiens_IDRs_comp_feat.csv.gz"
CENTROIDS = REFERENCE / "All_sapiens_IDRs_GIN_centroids.csv.gz"
SEED_THRESHOLDS = REFERENCE / "seed_thresholds_human.csv"
GIN_DEFINITIONS = REFERENCE / "gin_definitions.csv.gz"
HUMAN_IDRS = REPO / "data" / "input" / "Hsapien_IDRs_ftl100aa.csv.gz"
EXAMPLE_IDRS = REPO / "data" / "input" / "Hsapien_IDRs_ftl100aa_sample.csv.gz"
IDMAPPING = REPO / "data" / "annotation" / "idmapping_2026_05_29.tsv.gz"
PUBLISHED_BLOCKS = REPO / "results" / "multiblock" / "block_summary_seedExpantion.20260909.csv.gz"

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
GIN_COLORS = plot_proteins.myclustercolors

UNIPROT_FIELDS = "accession,id,protein_name,gene_names,organism_name,length,ft_domain"
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

logging.getLogger().setLevel(logging.WARNING)


# Fonts

ARIAL_CANDIDATES = (
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


def find_arial(extra: "str | None" = None) -> "str | None":
    for path in ((extra,) if extra else ()) + ARIAL_CANDIDATES:
        if path and Path(path).exists():
            return path
    return None


def setup_fonts(arial_path: "str | None" = None) -> str:
    """Register Arial when it is installed and make every figure use it,
    with editable (Type 42) text in PDFs. Returns the family in use."""
    family = colored_sequence.setup_plot_style(find_arial(arial_path))
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    return family


# GIN names

def gin_names() -> dict:
    gins = pd.read_csv(GIN_DEFINITIONS)
    return dict(zip(gins["GIN Type"].astype(int), gins["IDR Grammar"]))


def gin_label(cluster, names: "dict | None" = None) -> str:
    names = names if names is not None else gin_names()
    if pd.isna(cluster):
        return "unassigned"
    cluster = int(cluster)
    return f"GIN {cluster} · {names.get(cluster, '')}".rstrip(" ·")


# Input

def _clean_sequence(seq: str) -> str:
    return re.sub(r"[\s\d*]", "", str(seq)).upper()


def _parse_header(header: str) -> dict:
    """'>Q9Y388_IDR_1 uniprot=Q9Y388 start=112 gene=RBMX2' or a UniProt
    header '>sp|Q9Y388|RBMX2_HUMAN ...'. Only the first token is required."""
    tokens = header.split()
    record = {"IDR_ID": tokens[0] if tokens else ""}
    uniprot_style = re.match(r"^(?:sp|tr)\|([A-Z0-9-]+)\|(\S+)", record["IDR_ID"])
    if uniprot_style:
        record["UniprotID"] = uniprot_style.group(1)
        record["IDR_ID"] = uniprot_style.group(1)
    for token in tokens[1:]:
        key, sep, value = token.partition("=")
        if not sep:
            continue
        key = key.lower()
        if key in ("uniprot", "uniprotid", "acc"):
            record["UniprotID"] = value
        elif key == "start":
            record["IDR_START"] = int(value)
        elif key == "gene":
            record["Gene name"] = value
    return record


def parse_fasta(text: str) -> pd.DataFrame:
    records, header, chunks = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append({**_parse_header(header), "Sequence": "".join(chunks)})
            header, chunks = line[1:], []
        else:
            chunks.append(line)
    if header is not None:
        records.append({**_parse_header(header), "Sequence": "".join(chunks)})
    if not records and text.strip():
        # A bare sequence without a header.
        records.append({"IDR_ID": "sequence_1", "Sequence": text})
    return pd.DataFrame(records)


def standardize_input(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table the pipelines accept: IDR_ID + Sequence, plus any
    metadata. Fills IDR_END / IDR_length when IDR_START is known, and drops
    sequences the feature extraction cannot score."""
    df = df.copy()
    if "IDR_Sequence" in df.columns and "Sequence" not in df.columns:
        df = df.rename(columns={"IDR_Sequence": "Sequence"})
    if "Sequence" not in df.columns:
        raise ValueError("No 'Sequence' or 'IDR_Sequence' column found.")
    if "IDR_ID" not in df.columns:
        df.insert(0, "IDR_ID", [f"sequence_{i + 1}" for i in range(len(df))])

    df["IDR_ID"] = df["IDR_ID"].astype(str).str.strip()
    df["Sequence"] = df["Sequence"].map(_clean_sequence)
    # The bundled human table repeats some IDRs verbatim; keep one copy, and
    # number only the IDs that really point at different sequences.
    df = df.drop_duplicates(subset=["IDR_ID", "Sequence"])
    dupes = df["IDR_ID"].duplicated(keep=False)
    if dupes.any():
        df.loc[dupes, "IDR_ID"] += "_" + (df[dupes].groupby("IDR_ID").cumcount() + 1).astype(str)

    bad = df["Sequence"].map(lambda s: not s or not set(s) <= STANDARD_AA)
    if bad.any():
        print(f"Skipped {int(bad.sum())} sequence(s) with non-standard residues "
              f"(only the 20 standard amino acids are scored): "
              f"{', '.join(df.loc[bad, 'IDR_ID'].head(5))}")
        df = df.loc[~bad]

    df["IDR_length"] = df["Sequence"].str.len()
    if "IDR_START" in df.columns:
        start = pd.to_numeric(df["IDR_START"], errors="coerce")
        df["IDR_START"] = start.astype("Int64")
        if "IDR_END" not in df.columns:
            df["IDR_END"] = (start + df["IDR_length"] - 1).astype("Int64")

    if df.empty:
        raise ValueError("No usable sequences in the input.")
    front = ["IDR_ID", "UniprotID", "Gene name", "IDR_START", "IDR_END", "IDR_length"]
    order = [c for c in front if c in df.columns]
    order += [c for c in df.columns if c not in order and c != "Sequence"] + ["Sequence"]
    return df[order].reset_index(drop=True)


def load_example() -> pd.DataFrame:
    return standardize_input(pd.read_csv(EXAMPLE_IDRS))


def load_human_idrs(identifiers: "list[str]") -> pd.DataFrame:
    """IDRs of human proteins given by UniProt accession or gene name
    (gene names are matched without regard to case)."""
    human = pd.read_csv(HUMAN_IDRS)
    wanted = [u.strip() for u in identifiers if u.strip()]
    genes = human["Gene name"].astype(str).str.upper()
    hits = human[human["UniprotID"].isin(wanted) | genes.isin([w.upper() for w in wanted])]
    found = set(hits["UniprotID"]) | set(hits["Gene name"].astype(str).str.upper())
    missing = [w for w in wanted if w not in found and w.upper() not in found]
    if missing:
        print(f"No IDR >= 100 aa in the human set for: {', '.join(missing)}")
    if hits.empty:
        raise ValueError("None of these accessions or gene names has an IDR in the human set.")
    return standardize_input(hits)


def load_uploaded(name: str, content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8-sig")
    if name.lower().endswith((".fa", ".fasta", ".faa", ".txt")) or text.lstrip().startswith(">"):
        return standardize_input(parse_fasta(text))
    sep = "\t" if name.lower().endswith(".tsv") else ","
    return standardize_input(pd.read_csv(io.StringIO(text), sep=sep))


def write_pipeline_input(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# Pipelines

def _n_workers() -> int:
    return max(1, min(os.cpu_count() or 1, 32))


def run_full_sequence(input_csv: Path, outdir: Path, num_scrambles: int = 100) -> dict:
    """Score each IDR as a whole and assign it to the nearest GIN centroid
    (nardini_fullSeq_numba_v096.run_full_sequence)."""
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_csv).stem
    cfg = dict(
        fullseq.CONFIG,
        comp_reference_csv=str(COMP_REF),
        centroids_csv=str(CENTROIDS),
        num_scrambles=num_scrambles,
        n_workers=_n_workers(),
        output_sequences_csv=str(outdir / f"{stem}.fullseq.out_sequences.csv"),
        output_summary_csv=str(outdir / f"{stem}.fullseq.out_summary.csv"),
        output_dist_matrix_csv=str(outdir / f"{stem}.fullseq.out_dist_matrix.csv"),
    )
    df, metadata = fullseq.load_input_csv(str(input_csv))
    seq_df, summary_df, dist_df = fullseq.run_full_sequence(df, cfg, metadata_df=metadata)
    files = [cfg["output_sequences_csv"], cfg["output_summary_csv"], cfg["output_dist_matrix_csv"]]
    sequences, summary, distances = (pd.read_csv(f) for f in files)
    return dict(sequences=sequences, summary=summary, distances=distances, files=files)


def seed_reference_thresholds(min_dist_quantile: float, inter_dist_quantile: float) -> tuple:
    """Absolute seed cut-offs from the human proteome window distribution,
    for the requested quantiles (interpolated between the stored ones)."""
    ref = pd.read_csv(SEED_THRESHOLDS).sort_values("quantile")
    return (float(np.interp(min_dist_quantile, ref["quantile"], ref["min_dist"])),
            float(np.interp(inter_dist_quantile, ref["quantile"], ref["inter_cluster_dist"])))


def run_multiblock(input_csv: Path, outdir: Path, window_size: int = 70, step_size: int = 10,
                   num_scrambles: int = 100, block_method: str = "seed_expand",
                   seed_min_dist_quantile: float = 0.60, seed_inter_dist_quantile: float = 0.75,
                   seed_absorption_factor: float = 1.5, use_human_seed_thresholds: bool = True,
                   min_windows: int = 3, min_block_length: int = 70) -> dict:
    """Sliding-window assignment and block building
    (nardini_MultiBlock_v099.compute_windows + compute_blocks). The defaults
    are the settings of the published human run; min_windows only applies
    to meanboundary."""
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_csv).stem
    cfg = dict(
        multiblock.CONFIG,
        comp_reference_csv=str(COMP_REF),
        centroids_csv=str(CENTROIDS),
        window_size=window_size,
        step_size=step_size,
        num_scrambles=num_scrambles,
        n_workers=_n_workers(),
        block_method=block_method,
        seed_min_dist_quantile=seed_min_dist_quantile,
        seed_inter_dist_quantile=seed_inter_dist_quantile,
        seed_absorption_factor=seed_absorption_factor,
        min_windows=min_windows,
        min_block_length=min_block_length,
    )
    if block_method == "seed_expand" and use_human_seed_thresholds:
        if window_size != 70 or step_size != 10:
            print("Human seed thresholds come from 70 aa windows with step 10; "
                  "with other window settings they are only approximate.")
        thr_min, thr_inter = seed_reference_thresholds(seed_min_dist_quantile,
                                                       seed_inter_dist_quantile)
        cfg.update(seed_min_dist_threshold=thr_min, seed_inter_dist_threshold=thr_inter)
        print(f"Seed windows: min_dist <= {thr_min:.2f} and inter_cluster_dist >= "
              f"{thr_inter:.2f} (human proteome quantiles "
              f"{seed_min_dist_quantile:.2f} / {seed_inter_dist_quantile:.2f})")

    df, metadata = multiblock.load_input_csv(str(input_csv))
    short = [i for i, s in zip(df["IDR_ID"], df["Sequence"]) if len(s) < cfg["min_window_size"]]
    if short:
        print(f"{len(short)} sequence(s) shorter than {cfg['min_window_size']} aa get no "
              f"windows: {', '.join(short[:5])}")

    win_df = multiblock.compute_windows(df, cfg)
    summary_df = multiblock.compute_blocks(win_df, cfg)
    if metadata is not None:
        win_df = metadata.join(win_df, on="IDR_ID", how="inner")
        if len(summary_df) > 0:
            summary_df = metadata.join(summary_df, on="IDR_ID", how="inner")

    windows_csv = outdir / f"{stem}.MultiBlock.{block_method}.out_windows.csv"
    summary_csv = outdir / f"{stem}.MultiBlock.{block_method}.out_summary.csv"
    win_df.write_csv(windows_csv)
    summary_df.write_csv(summary_csv)
    return dict(windows=pd.read_csv(windows_csv), summary=pd.read_csv(summary_csv),
                files=[str(windows_csv), str(summary_csv)], cfg=cfg)


# Result tables

def _label_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    names = gin_names()
    blocks["GIN"] = blocks["cluster"].astype(int)
    blocks["IDR grammar"] = blocks["GIN"].map(names)
    blocks["block"] = blocks["IDR_block"] + 1
    return blocks


def published_blocks(idr_ids) -> pd.DataFrame:
    """Blocks of these IDRs in the published human table
    (block_summary_seedExpantion.20260909.csv), in the blocks_table format."""
    published = pd.read_csv(PUBLISHED_BLOCKS)
    hits = published[published["IDR_ID"].isin(list(idr_ids))].copy()
    missing = sorted(set(idr_ids) - set(hits["IDR_ID"]))
    if missing:
        print(f"Not in the published table ({len(missing)}; no block passed the filters, "
              f"or not a human IDR from the bundled set): {', '.join(missing[:8])}"
              f"{' ...' if len(missing) > 8 else ''}")
    return _label_blocks(hits.reset_index(drop=True)) if not hits.empty else pd.DataFrame()


def blocks_table(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per GIN block (nardini_summary_pipeline.explode_summary_to_blocks)."""
    if summary.empty:
        return pd.DataFrame()
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        blocks = summary_pipeline.explode_summary_to_blocks(summary)
    return _label_blocks(blocks)


def nearest_gins(distances: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Rank the GIN centroids by squared distance for each sequence."""
    cols = [c for c in distances.columns if c.startswith("cluster_")]
    mat = distances[cols].to_numpy(float)
    order = np.argsort(mat, axis=1)[:, :n]
    names = gin_names()
    out = pd.DataFrame({"IDR_ID": distances["IDR_ID"]})
    for rank, title in zip(range(n), ("Nearest GIN", "2nd", "3rd", "4th", "5th")):
        idx = order[:, rank]
        dist = mat[np.arange(len(mat)), idx]
        out[title] = [f"{i} · {names.get(i, '')}  ({d:.1f})" for i, d in zip(idx, dist)]
    return out


def display_nearest_gins(distances: pd.DataFrame, n: int = 3):
    """nearest_gins with the nearest GIN in its protein-diagram color;
    the number in brackets is the squared distance to that centroid."""
    table = nearest_gins(distances, n)

    def color(cell):
        bg = GIN_COLORS[int(cell.split(" ")[0]) % len(GIN_COLORS)]
        return f"background-color: {bg}; color: white; font-weight: bold"

    return (table.style.map(color, subset=["Nearest GIN"]).hide(axis="index")
            .set_properties(**{"text-align": "left"}))


def display_blocks(blocks: pd.DataFrame, best: "pd.DataFrame | None" = None):
    """The block table with each GIN cell in its protein-diagram color. With
    `best` (from best_windows), adds each block's best window and its
    min inter-cluster distance."""
    if blocks.empty:
        return pd.DataFrame({"result": ["No blocks passed the filters."]})
    if best is not None and not best.empty:
        extra = best[["IDR_ID", "IDR_block", "window_start", "window_end"]].copy()
        extra["best window"] = extra["window_start"].astype(str) + "-" + extra["window_end"].astype(str)
        extra["min inter-cluster dist"] = confidence(best).round(2).to_numpy()
        blocks = blocks.merge(extra[["IDR_ID", "IDR_block", "best window", "min inter-cluster dist"]],
                              on=["IDR_ID", "IDR_block"], how="left")
    cols = [c for c in ["IDR_ID", "Gene name", "block", "GIN", "IDR grammar",
                        "block_start", "block_end", "block_length",
                        "IDR_block_start", "IDR_block_end", "valley_dist",
                        "best window", "min inter-cluster dist"] if c in blocks.columns]
    view = blocks[cols].rename(columns={
        "block_start": "start (IDR)", "block_end": "end (IDR)", "block_length": "length",
        "IDR_block_start": "start (protein)", "IDR_block_end": "end (protein)",
        "valley_dist": "best distance"})

    def color(gin):
        bg = GIN_COLORS[int(gin) % len(GIN_COLORS)]
        return f"background-color: {bg}; color: white; font-weight: bold; text-align: center"

    text_cols = [c for c in ("IDR_ID", "Gene name", "IDR grammar") if c in view.columns]
    styled = (view.style.map(color, subset=["GIN"]).hide(axis="index")
              .set_properties(subset=text_cols, **{"text-align": "left"}))
    if "min inter-cluster dist" in view.columns:
        styled = styled.background_gradient(cmap="Blues", subset=["min inter-cluster dist"])
    return styled.format({"best distance": "{:.2f}", "min inter-cluster dist": "{:.2f}"}, na_rep="")


# Colored sequence (analysis/plotting/plot_colored_sequence.py)

def block_choices(blocks: pd.DataFrame, idr_id: str) -> list:
    """(label, highlight spec) pairs for the colored-sequence viewer.
    Specs are 1-based inclusive ranges, as plot_colored_sequence expects."""
    choices = [("No highlight", "")]
    if blocks.empty:
        return choices
    for _, b in blocks[blocks["IDR_ID"] == idr_id].iterrows():
        start, end = int(b["block_start"]) + 1, int(b["block_end"])
        label = f"Block {b['block']} · GIN {b['GIN']} {b['IDR grammar']} ({start}-{end})"
        choices.append((label, f"{start}-{end}"))
    return choices


def colored_sequence_figure(record: pd.Series, highlight: str = "", color_set: str = "Kiersten",
                            wrap_width: int = 60, legend: bool = True):
    seq = record["Sequence"]
    meta = {k: record[k] for k in ("Gene name", "UniprotID", "IDR_START", "IDR_END")
            if k in record and pd.notna(record[k])}
    title = colored_sequence.build_title(dict(IDR_ID=record["IDR_ID"], Sequence=seq, meta=meta))
    positions = {p for p in colored_sequence.parse_highlight_ranges(highlight) if p < len(seq)}
    colors = colored_sequence.build_aacolor_map(color_set)
    family = mpl.rcParams["font.sans-serif"][0]
    fig = colored_sequence.render_matplotlib(seq, colors, positions, title, wrap_width,
                                             family, None, color_set, legend)
    html = colored_sequence.render_html(seq, colors, positions, title)
    return fig, html


# Protein diagram (analysis/plotting/plot_proteins.py)

def _uniprot_annotation(accessions: list) -> pd.DataFrame:
    """Protein length, names and Domain [FT] from the UniProt REST API, in
    the same columns as the bundled idmapping export."""
    if not accessions:
        return pd.DataFrame()
    query = " OR ".join(f"accession:{a}" for a in accessions)
    url = (f"{UNIPROT_STREAM}?format=tsv&fields={UNIPROT_FIELDS}"
           f"&query={urllib.parse.quote(query)}")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            table = pd.read_csv(io.StringIO(response.read().decode()), sep="\t")
    except Exception as err:
        print(f"UniProt lookup failed ({err}); those proteins are drawn without domains.")
        return pd.DataFrame()
    table.insert(0, "From", table["Entry"])
    return table


def domain_table(blocks: pd.DataFrame, fetch_from_uniprot: bool = True) -> pd.DataFrame:
    """Long table for plot_proteins: Total protein / Domain / IDR / IDR block
    rows (nardini_summary_pipeline.build_domain_architecture). Sequences with
    no UniProt ID are drawn on their own, as an IDR-only track."""
    if blocks.empty:
        return pd.DataFrame()
    has_protein = (blocks.get("UniprotID", pd.Series(index=blocks.index, dtype=object)).notna()
                   & blocks.get("IDR_START", pd.Series(index=blocks.index, dtype=object)).notna())
    parts = []

    annotated = blocks[has_protein]
    if not annotated.empty:
        wanted = annotated["UniprotID"].astype(str).unique().tolist()
        idmap = summary_pipeline.load_table(str(IDMAPPING))
        idmap = idmap[idmap["From"].astype(str).isin(wanted)]
        missing = sorted(set(wanted) - set(idmap["From"].astype(str)))
        if missing and fetch_from_uniprot:
            idmap = pd.concat([idmap, _uniprot_annotation(missing)], ignore_index=True)
        if not idmap.empty:
            idmap["Gene Names"] = idmap["Gene Names"].astype(str).str.split().str[0]
            organisms = idmap.set_index("From")["Organism"].astype(str)
            other = organisms[~organisms.str.startswith("Homo sapiens")]
            if not other.empty:
                print(f"Not human according to UniProt: {', '.join(f'{a} ({o})' for a, o in other.items())}. "
                      f"GIN centroids and the feature normalisation are human-specific.")
            arch = summary_pipeline.build_domain_architecture(annotated, idmap)
            arch = _drop_domains_on_length_mismatch(arch, annotated)
            parts.append(arch[arch["UniprotID"].isin(idmap["From"].astype(str))])
            annotated = annotated[~annotated["UniprotID"].isin(idmap["From"].astype(str))]
        if not annotated.empty:
            print(f"No UniProt annotation for {', '.join(annotated['UniprotID'].unique())}; "
                  f"drawn as IDR-only tracks.")

    standalone = pd.concat([blocks[~has_protein], annotated])
    for idr_id, group in standalone.groupby("IDR_ID", sort=False):
        length = int(group["IDR_length"].iloc[0])
        base = {"UniprotID": idr_id, "Gene name": group.get("Gene name", pd.Series([idr_id])).iloc[0],
                "Organism": "IDR only"}
        if pd.isna(base["Gene name"]):
            base["Gene name"] = idr_id
        rows = [dict(base, **{"Domain type": "Total protein", "Domain name": idr_id,
                              "Start": 0, "End": length})]
        for _, b in group.iterrows():
            rows.append(dict(base, **{"Domain type": "IDR block", "Domain name": str(int(b["GIN"])),
                                      "Start": int(b["block_start"]), "End": int(b["block_end"])}))
        parts.append(pd.DataFrame(rows))

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _drop_domains_on_length_mismatch(arch: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """If an IDR ends beyond the UniProt length, the IDR was cut from another
    isoform, so the UniProt domain positions do not apply; drop them."""
    lengths = arch[arch["Domain type"] == "Total protein"].set_index("UniprotID")["End"]
    ends = blocks.groupby("UniprotID")["IDR_END"].max()
    bad = [u for u, e in ends.items() if u in lengths and pd.notna(e) and int(e) > int(lengths[u])]
    if bad:
        print(f"IDR coordinates run past the UniProt sequence length for {', '.join(bad)} "
              f"(probably another isoform); their domains are not drawn.")
        arch = arch[~((arch["UniprotID"].isin(bad)) & (arch["Domain type"] == "Domain"))]
    return arch


def protein_figure(domains: pd.DataFrame, proteins: "list | None" = None,
                   show_domains: bool = True, width: float = 12):
    if not show_domains:
        domains = domains[domains["Domain type"] != "Domain"]
    return plot_proteins.plot_protein_diagrams(domains, proteins, width=width)


def track_ids(blocks: pd.DataFrame, domains: pd.DataFrame) -> list:
    """Protein-track keys in input order: the UniProt ID when the protein
    is annotated, otherwise the IDR_ID."""
    keys = []
    present = set(domains["UniprotID"])
    for _, b in blocks.drop_duplicates("IDR_ID").iterrows():
        key = b.get("UniprotID") if b.get("UniprotID") in present else b["IDR_ID"]
        if key in present and key not in keys:
            keys.append(key)
    return keys


# Saving

def save_figure(fig, prefix: Path, formats=("png", "pdf")):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["svg.fonttype"] = "none"
    for ext in formats:
        fig.savefig(f"{prefix}.{ext}", dpi=300, facecolor="white", bbox_inches="tight")


def zip_results(folder: Path) -> Path:
    archive = folder.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(folder.parent))
    return archive


# Grammar features (z-scores) for whole IDRs and for blocks

FEATURES = pd.read_csv(CENTROIDS, nrows=0).columns[1:].tolist()
CENTROID_MATRIX = pd.read_csv(CENTROIDS)[FEATURES].to_numpy(float)
PATTERNING = [f for f in FEATURES if "-" in f]
COMPOSITION = [f for f in FEATURES if f not in PATTERNING]
HUMAN_FULLSEQ_Z = REFERENCE / "human_idrome_fullseq_zscores.csv.gz"
PUBLISHED_BEST_WINDOWS = REFERENCE / "published_blocks_best_window_zscores.csv.gz"

# Features whose low values mean "well mixed" or a shifted ratio rather than
# simply "less of something"; only these are ranked from the bottom.
LOW_IS_INFORMATIVE = PATTERNING + ["R/K Ratio", "E/D Ratio", "NCPR", "Hydrophobicity", "Iso point"]


def best_windows(blocks: pd.DataFrame, windows: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """One row of z-scores per block: the window centred inside the block
    that is closest to the block's own GIN centroid. Published blocks are
    looked up in the precomputed reference; recomputed blocks use the
    out_windows table of the same run."""
    if blocks.empty:
        return pd.DataFrame()
    keys = blocks[["IDR_ID", "IDR_block", "block", "GIN", "IDR grammar", "block_start", "block_end"]]
    if windows is None:
        ref = pd.read_csv(PUBLISHED_BEST_WINDOWS)
        out = keys.merge(ref.drop(columns="cluster"), on=["IDR_ID", "IDR_block"], how="inner")
    else:
        win = windows.dropna(subset=["cluster"]).drop_duplicates(["IDR_ID", "start", "end"])
        groups = dict(tuple(win.groupby("IDR_ID")))
        rows = []
        for b in keys.itertuples(index=False):
            g = groups.get(b.IDR_ID)
            if g is None:
                continue
            centre = (g["start"] + g["end"]) / 2
            sub = g[(centre >= b.block_start) & (centre < b.block_end)]
            if sub.empty:
                sub = g[(g["start"] < b.block_end) & (g["end"] > b.block_start)]
            dist = np.nansum((sub[FEATURES].to_numpy(float) - CENTROID_MATRIX[b.GIN]) ** 2, axis=1)
            w = sub.iloc[int(np.argmin(dist))]
            rows.append(dict(b._asdict(), window_start=int(w["start"]), window_end=int(w["end"]),
                             window_cluster=int(w["cluster"]), min_dist=float(w["min_dist"]),
                             inter_cluster_dist=float(w["inter_cluster_dist"]),
                             dist_to_block_gin=float(dist.min()), **w[FEATURES].astype(float).to_dict()))
        out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["label"] = (out["IDR_ID"] + " · block " + out["block"].astype(str)
                    + " · GIN " + out["GIN"].astype(str))
    return out


def idr_zscores(inputs: pd.DataFrame, fullseq_sequences: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """Whole-IDR z-scores: from this run's full-sequence output when there is
    one, otherwise looked up in the human IDRome reference."""
    if fullseq_sequences is not None and not fullseq_sequences.empty:
        z = fullseq_sequences.copy()
    else:
        ref = pd.read_csv(HUMAN_FULLSEQ_Z)
        z = ref[ref["IDR_ID"].isin(inputs["IDR_ID"])].copy()
        missing = sorted(set(inputs["IDR_ID"]) - set(z["IDR_ID"]))
        if missing:
            print(f"No whole-IDR z-scores for {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}; "
                  f"run the full-sequence mode in step 3 to compute them.")
    if z.empty:
        return z
    names = gin_names()
    z["GIN"] = z["cluster"].astype("Int64")
    z["label"] = z["IDR_ID"] + " · GIN " + z["GIN"].astype(str) + " " + z["GIN"].map(names).fillna("")
    order = {i: n for n, i in enumerate(inputs["IDR_ID"])}
    return z.sort_values("IDR_ID", key=lambda s: s.map(order)).reset_index(drop=True)


def centroid_distances(z: pd.DataFrame) -> np.ndarray:
    """Squared distances to the 30 centroids, computed as in the pipeline."""
    return np.array([np.nansum((CENTROID_MATRIX - row) ** 2, axis=1)
                     for row in z[FEATURES].to_numpy(float)])


def confidence(z: pd.DataFrame) -> pd.Series:
    """Min inter-cluster distance: second-nearest minus nearest centroid.
    Large = clearly one grammar; near 0 = a close call between two."""
    d = np.sort(centroid_distances(z), axis=1)
    return pd.Series(d[:, 1] - d[:, 0], index=z.index)


def zscore_heatmap(z: pd.DataFrame, cmap: str = "PRGn", vmax: float = 3.0,
                   min_abs: float = 1.0, clustering: str = "Both"):
    """Heatmap of the 90 z-scores, one row per IDR or block. Features are
    kept if at least one row reaches |z| >= min_abs."""
    import seaborn as sns
    data = z.set_index("label")[FEATURES]
    data = data.loc[:, data.abs().max(axis=0) >= min_abs]
    if data.empty:
        print(f"No feature reaches |z| >= {min_abs}; lower the cut-off.")
        return None
    rows = clustering in ("Both", "Rows only") and len(data) > 1
    cols = clustering in ("Both", "Features only") and data.shape[1] > 1
    size = (max(6, 0.28 * data.shape[1] + 3), max(2.5, 0.38 * len(data) + 1.8))
    grid = sns.clustermap(data.fillna(0), row_cluster=rows, col_cluster=cols, method="ward",
                          cmap=cmap, vmin=-vmax, vmax=vmax, center=0, linewidths=0.5,
                          linecolor="white", figsize=size, dendrogram_ratio=(0.08, 0.12),
                          cbar_pos=(0.01, 0.9, 0.012, 0.08),
                          cbar_kws={"label": "z-score"}, xticklabels=True, yticklabels=True)
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.ax_heatmap.tick_params(axis="x", labelsize=8)
    grid.ax_heatmap.tick_params(axis="y", labelsize=8)
    return grid.figure


def centroid_distance_figure(z: pd.DataFrame, ncols: int = 3):
    """Sorted distance to all 30 centroids, one panel per IDR or block; the
    nearest GIN is filled, the gap to the runner-up is the confidence."""
    dist = centroid_distances(z)
    n = len(z)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.6 * nrows), squeeze=False)
    for ax, (_, row), d in zip(axes.flat, z.iterrows(), dist):
        order = np.argsort(d)
        ax.plot(range(30), d[order], "-", color="#9ca3af", lw=1, zorder=1)
        ax.scatter(range(30), d[order], s=22, c=["white"] * 30, edgecolors="#4b5563", lw=0.8, zorder=2)
        # Fill the assigned GIN (the block's GIN, or the nearest for a whole IDR).
        assigned = int(row["GIN"]) if "GIN" in row and pd.notna(row["GIN"]) else int(order[0])
        pos = int(np.where(order == assigned)[0][0])
        ax.scatter([pos], [d[assigned]], s=44, c=[GIN_COLORS[assigned % len(GIN_COLORS)]],
                   edgecolors="black", lw=0.8, zorder=3)
        ax.set_xticks(range(30))
        ax.set_xticklabels(order, fontsize=6.5)
        note = "" if pos == 0 else f" · window nearest to GIN {order[0]}"
        ax.set_title(f"{row['label']}\nmin inter-cluster distance {d[order[1]] - d[order[0]]:.1f}{note}",
                     fontsize=8.5, loc="left")
        ax.set_ylabel("Squared distance", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.supxlabel("GIN, nearest to farthest", fontsize=9)
    fig.tight_layout()
    return fig


# Residues behind a feature, for the colored-sequence highlight

FEATURE_RESIDUES = {f"Frac {aa}": aa for aa in "ACDEFGHIKLMNPQRSTVWY"}
FEATURE_RESIDUES.update({"Frac K+R": "KR", "Frac D+E": "DE", "Frac Polar": "QNSTGCH",
                         "Frac Aliphatic": "ALMIV", "Frac Aromatic": "FWY", "FCR": "KRDE",
                         "NCPR": "KRDE", "Frac Chain Expanding": "KRDEP", "R/K Ratio": "RK",
                         "E/D Ratio": "ED"})
RESIDUE_TYPES = {"pol": "STNQCH", "hyd": "ILMV", "pos": "RK", "neg": "ED",
                 "aro": "FWY", "ala": "A", "pro": "P", "gly": "G"}
NO_RESIDUE_MAP = {"Hydrophobicity", "Disorder Promoting", "Iso point", "PPII"}


def _patch_spans(seq: str, residues: str, min_count: int, max_gap: int = 2, pair: "str | None" = None):
    """Spans where the residues recur with at most max_gap other residues
    between them, keeping spans with at least min_count of them (or of the
    dipeptide `pair`). Same patch definition as the pipeline's features."""
    hits = [i for i, aa in enumerate(seq) if aa in residues]
    spans, start = [], None
    for prev, cur in zip([None] + hits, hits + [None]):
        if start is None:
            start = cur
        if cur is None or (prev is not None and cur - prev > max_gap + 1):
            end = prev + 1
            piece = seq[start:end]
            count = piece.count(pair) if pair else sum(aa in residues for aa in piece)
            if count >= min_count:
                spans.append((start, end))
            start = cur
    return spans


def feature_positions(seq: str, feature: str) -> set:
    """0-based positions of the residues behind one feature."""
    if feature in FEATURE_RESIDUES:
        return {i for i, aa in enumerate(seq) if aa in FEATURE_RESIDUES[feature]}
    if "-" in feature:
        a, b = feature.split("-")
        residues = RESIDUE_TYPES[a] + RESIDUE_TYPES[b]
        return {i for i, aa in enumerate(seq) if aa in residues}
    if feature.endswith(" Patch"):
        spans = _patch_spans(seq, feature[0], min_count=4)
    elif feature == "RG Frac":
        spans = _patch_spans(seq, "RG", min_count=2, pair="RG")
    else:
        return set()
    return {i for s, e in spans for i in range(s, e)}


def top_feature(zrow: pd.Series) -> "tuple[str, float] | None":
    """The most non-random feature that maps onto residues: the highest
    positive z-score, or a strongly negative one for features where low
    values mean well mixed (patterning) or a shifted ratio."""
    best, score = None, 0.0
    for f in FEATURES:
        if f in NO_RESIDUE_MAP:
            continue
        z = float(zrow[f])
        s = z if z > 0 else (-z if f in LOW_IS_INFORMATIVE else 0)
        if s > score:
            best, score = f, s
    return (best, float(zrow[best])) if best else None


def top_feature_choices(record: pd.Series, idr_z: pd.DataFrame, block_z: pd.DataFrame) -> list:
    """Extra (label, 1-based range spec) choices for the colored-sequence
    viewer: the top feature of the whole IDR and of each block's best window."""
    seq, choices = record["Sequence"], []

    def spec(positions):
        return ",".join(str(p + 1) for p in sorted(positions))

    whole = idr_z[idr_z["IDR_ID"] == record["IDR_ID"]] if not idr_z.empty else idr_z
    if not whole.empty:
        top = top_feature(whole.iloc[0])
        if top:
            choices.append((f"Top feature, whole IDR: {top[0]} (z = {top[1]:.1f})",
                            spec(feature_positions(seq, top[0]))))
    blocks = block_z[block_z["IDR_ID"] == record["IDR_ID"]] if not block_z.empty else block_z
    for _, b in blocks.iterrows():
        top = top_feature(b)
        if top:
            inside = {p for p in feature_positions(seq, top[0]) if b["block_start"] <= p < b["block_end"]}
            choices.append((f"Top feature, block {b['block']} (GIN {b['GIN']}): {top[0]} (z = {top[1]:.1f})",
                            spec(inside)))
    return choices


# Comparisons with the human IDRome

def reference_table(level: str) -> pd.DataFrame:
    """The human IDRome reference: whole IDRs, or the best windows of the
    published blocks."""
    if level == "blocks":
        ref = pd.read_csv(PUBLISHED_BEST_WINDOWS)
        ref["key"] = ref["IDR_ID"] + "#" + ref["IDR_block"].astype(str)
    else:
        ref = pd.read_csv(HUMAN_FULLSEQ_Z)
        ref["key"] = ref["IDR_ID"]
    return ref


def exceptional_features(z: pd.DataFrame, level: str, rank_cutoff: int = 150) -> pd.DataFrame:
    """Features where an IDR (or block) ranks in the top `rank_cutoff` of the
    human IDRome. High: largest z first, all features. Low: smallest z first,
    patterning and ratio-type features only. The item itself is left out of
    the reference, and ties count against it (a value shared by 300 IDRs
    ranks 300), which matters at the +/-3 cap."""
    ref = reference_table(level)
    keys = (z["IDR_ID"] + "#" + z["IDR_block"].astype(str)) if level == "blocks" else z["IDR_ID"]
    rows = []
    for key, (_, item) in zip(keys, z.iterrows()):
        other = ref[ref["key"] != key]
        n = len(other) + 1
        for f in FEATURES:
            v = float(item[f])
            high = int((other[f] >= v).sum()) + 1
            if high <= rank_cutoff:
                rows.append(dict(item=item["label"], feature=f, z=v, direction="high", rank=high, of=n))
            if f in LOW_IS_INFORMATIVE:
                low = int((other[f] <= v).sum()) + 1
                if low <= rank_cutoff:
                    rows.append(dict(item=item["label"], feature=f, z=v, direction="low", rank=low, of=n))
    out = pd.DataFrame(rows, columns=["item", "feature", "z", "direction", "rank", "of"])
    out["_order"] = out["item"].map({lab: i for i, lab in enumerate(z["label"])})
    return out.sort_values(["_order", "rank"]).drop(columns="_order").reset_index(drop=True)


def compare_to_idrome(z: pd.DataFrame, level: str, min_abs_mean: float = 0.0) -> pd.DataFrame:
    """Two-sample Kolmogorov-Smirnov test per feature, your IDRs (or blocks)
    against the rest of the human IDRome. Signed -log10(p): positive when
    your mean is higher. Only p < 0.05 with |your mean| >= min_abs_mean."""
    from scipy.stats import ks_2samp
    ref = reference_table(level)
    keys = set((z["IDR_ID"] + "#" + z["IDR_block"].astype(str)) if level == "blocks" else z["IDR_ID"])
    rest = ref[~ref["key"].isin(keys)]
    rows = []
    for f in FEATURES:
        mine, other = z[f].dropna().to_numpy(float), rest[f].dropna().to_numpy(float)
        p = ks_2samp(mine, other).pvalue
        m = mine.mean()
        signed = 0.0
        if p < 0.05 and abs(m) >= min_abs_mean:
            signed = min(-np.log10(max(p, 1e-300)), 300) * (1 if m > other.mean() else -1)
        rows.append(dict(feature=f, your_mean=m, idrome_mean=other.mean(), p_value=p,
                         signed_log10_p=signed))
    return pd.DataFrame(rows)


def comparison_figure(table: pd.DataFrame, cmap: str = "PRGn", vmax: float = 10.0, title: str = ""):
    import seaborn as sns
    hits = table[table["signed_log10_p"] != 0].sort_values("signed_log10_p")
    if hits.empty:
        print("No feature differs at p < 0.05 with the current filter.")
        return None
    fig, ax = plt.subplots(figsize=(max(4, 0.32 * len(hits) + 2), 2.2))
    sns.heatmap(hits.set_index("feature")[["signed_log10_p"]].T, cmap=cmap, vmin=-vmax, vmax=vmax,
                center=0, linewidths=0.5, linecolor="black", square=True, ax=ax,
                cbar_kws={"label": "signed -log10(p)", "shrink": 0.8}, yticklabels=False)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    return fig


# Tables for display

def colored_sequence_html(seq: str, color_set: str = "Kiersten") -> str:
    colors = colored_sequence.build_aacolor_map(color_set)
    spans = "".join(f'<span style="color:{colors.get(aa, "#000")}">{aa}</span>' for aa in seq)
    return f'<div style="font-family:Courier,monospace;word-break:break-all;min-width:420px">{spans}</div>'


def _gin_cell(value):
    try:
        bg = GIN_COLORS[int(value) % len(GIN_COLORS)]
    except (TypeError, ValueError):
        return ""
    return f"background-color: {bg}; color: white; font-weight: bold; text-align: center"


def overview_table(inputs: pd.DataFrame, idr_z: pd.DataFrame, blocks: pd.DataFrame,
                   color_set: str = "Kiersten"):
    """One row per IDR: whole-IDR GIN with its confidence, the block order,
    and the sequence colored by residue type."""
    view = inputs[[c for c in ["IDR_ID", "Gene name", "UniprotID", "IDR_START", "IDR_END",
                               "IDR_length", "Sequence"] if c in inputs.columns]].copy()
    if not idr_z.empty:
        whole = idr_z[["IDR_ID", "GIN"]].copy()
        whole["IDR grammar"] = whole["GIN"].map(gin_names())
        whole["min inter-cluster dist"] = confidence(idr_z).round(2).to_numpy()
        view = view.merge(whole, on="IDR_ID", how="left")
    if not blocks.empty:
        order = blocks.groupby("IDR_ID", sort=False)["GIN"].apply(lambda g: " → ".join(map(str, g)))
        view["blocks (GIN order)"] = view["IDR_ID"].map(order).fillna("")
    view = view[[c for c in view.columns if c != "Sequence"] + ["Sequence"]]
    styled = view.style.hide(axis="index").format(
        {"Sequence": lambda s: colored_sequence_html(s, color_set), "min inter-cluster dist": "{:.2f}"},
        na_rep="", escape=None)
    if "GIN" in view.columns:
        styled = styled.map(_gin_cell, subset=["GIN"])
    if "min inter-cluster dist" in view.columns:
        styled = styled.background_gradient(cmap="Blues", subset=["min inter-cluster dist"])
    return styled.set_properties(**{"text-align": "left", "vertical-align": "top"})


def gin_definitions_table():
    gins = pd.read_csv(GIN_DEFINITIONS).sort_values("GIN Type")
    gins["Localization"] = gins["Localization"].map(
        lambda s: ", ".join(parse_list(s)) if isinstance(s, str) else "")
    return (gins.style.hide(axis="index").map(_gin_cell, subset=["GIN Type"])
            .format({"Median Min Inter Clust Dist": "{:.1f}"}, na_rep="")
            .set_properties(**{"text-align": "left"}))


def parse_list(text: str) -> list:
    import json
    try:
        return json.loads(text)
    except ValueError:
        return [text]
