<p align="center">
  <img src="logo/logo.png" alt="TESSERA-IDR logo" width="600">
</p>

# TESSERA-IDR

**T**iling-based **E**xploration of **S**equence **S**ignatures for **E**xtended
**R**esidue **A**nnotation in Intrinsically Disordered Regions.

Developed by Alexandre P. Magalhaes. Contact: [magalhae@molgen.mpg.de](mailto:magalhae@molgen.mpg.de)

<p align="center">
  <a href="https://colab.research.google.com/github/alexpmagalhaes/TESSERA-IDR/blob/main/colab/TESSERA-IDR_colab.ipynb">
    <img src="https://img.shields.io/badge/Open%20in-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open TESSERA-IDR in Google Colab" height="56">
  </a>
</p>

TESSERA-IDR assigns human IDRs to the 30 published GIN (grammars of IDRs) centroids using
the NARDINI+ feature pipeline, with the omega/kappa sliding-window off-by-one
bug fixed
([issue #2](https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-/issues/2)).
The published centroids are kept unchanged, so differences from the original
analysis come from the kernel fix alone.

Two assignment modes:

- **Full sequence** (`src/nardini_fullSeq_numba_v096.py`): each IDR is scored
  once as a whole and assigned to its nearest centroid.
- **Sliding window / multi-block** (`src/nardini_MultiBlock_v099.py`): each IDR
  is scanned with a 70 aa window (step 10), every window is assigned, and
  blocks are grown from high-confidence seed windows (`seed_expand`).

The published blocks are `results/multiblock/block_summary_seedExpantion.20260909.csv.gz`.
They come from `seed_expand` with seeds at `min_dist` <= the 0.60 quantile and
`inter_cluster_dist` >= the 0.75 quantile of all human windows, absorption
factor 1.5, and blocks of at least 70 aa (no minimum window count). These are
the script's defaults.

Input: 10,371 human IDRs of length >= 100 aa (`data/input/Hsapien_IDRs_ftl100aa.csv.gz`).

Also included:

- **Centroid analysis** (`analysis/centroid_analysis/`): distances of each
  IDR to the 30 GIN centroids, compared across RBPs and nucleolar / nuclear
  speckle protein lists, and cluster enrichment per localization.

`README_original_bundle.md` is the original working notes for this bundle,
including the full description of every command-line option. Its file paths
refer to the old flat layout.

## Run it in Google Colab

<p align="center">
  <a href="https://colab.research.google.com/github/alexpmagalhaes/TESSERA-IDR/blob/main/colab/TESSERA-IDR_colab.ipynb">
    <img src="https://img.shields.io/badge/Open%20in-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open TESSERA-IDR in Google Colab" height="56">
  </a>
</p>

No installation needed: the notebook runs TESSERA-IDR in your browser on Google's
servers. Give it human IDRs by UniProt accession or gene name, paste FASTA, or
upload a table, and it returns:

- **GIN assignment**, for the whole IDR and block by block along it (or the
  published blocks, for human IDRs), with a confidence score for each: the
  min inter-cluster distance.
- **An overview table** with every IDR's sequence colored by residue type, and
  the GIN definitions with their localizations.
- **The 90 NARDINI+ z-scores** as a clustered heatmap, and the distance of each
  IDR or block to all 30 GIN centroids.
- **The colored sequence** with a block, or the residues behind its most
  non-random feature, in bold.
- **The protein architecture**: UniProt domains, the IDR and its GIN blocks.
- **Exceptional grammars**: features where an IDR ranks among the most extreme
  in the human IDRome.
- **A comparison of your IDR set with the rest of the human IDRome**
  (Kolmogorov-Smirnov test per feature).

Every table and figure is saved and can be downloaded as one zip or copied to
Google Drive (PNG, PDF with editable text, or SVG).

> **Human IDRs only.** The GIN centroids, the compositional normalisation and the
> reference tables all come from the human IDRome. Support for other species
> will be added soon.

### How the notebook works

Open `colab/TESSERA-IDR_colab.ipynb` in Colab, run the first cell and give it the
repository (GitHub URL, a Drive folder, or an uploaded `.zip`), then follow the
numbered steps. `colab/nardini_colab.py` connects the notebook to `src/` and
`analysis/plotting/`: feature extraction, GIN assignment, block building, the
colored sequence (`plot_colored_sequence.py`) and the protein diagram
(`plot_proteins.py`) come from there unchanged. Only the notebook's own
additions (best windows, rankings, the IDRome comparison and the feature
highlight) are written in `nardini_colab.py`. Blocks are represented by their
best window: the 70 aa window centred inside the block that is closest to the
block's GIN centroid.

Reference tables used by the notebook (`data/reference/`):

- `human_idrome_fullseq_zscores.csv.gz`: whole-IDR z-scores and GIN for the
  10,371 human IDRs (full-sequence mode, fixed kernel, 100 scrambles; duplicate
  input rows removed).
- `published_blocks_best_window_zscores.csv.gz`: the best window of each of the
  10,592 published blocks, with its 90 z-scores, taken from the Sep 8 windows
  file behind the published blocks.

For human IDRs, the notebook shows the published blocks directly, and it can
also recompute them with the published settings. For `seed_expand` it uses
absolute seed cut-offs from `data/reference/seed_thresholds_human.csv`
(quantiles of `min_dist` and `inter_cluster_dist` over the 251,223 windows of
the human run), because quantiles of a handful of windows would define
"confident" differently from the proteome run. Applied to the human windows
of the 7-IDR sample, these cut-offs give exactly the published blocks. A fresh
recomputation can still differ slightly, because the patterning z-scores come
from random scrambles. On the command line, the same cut-offs are
`--seed_min_dist_threshold` / `--seed_inter_dist_threshold`.

## Layout

```
pyproject.toml           packaging for the `tesseraidr` package / command
src/
  tesseraidr/            the installable package (core pipeline)
    fullseq.py             full-sequence assignment (fixed kernel)
    multiblock.py          sliding-window assignment + block building
    summary.py             summary -> block table, domain table, plots
    cli.py                 the `tesseraidr` command
    data/                  bundled reference tables (centroids, comp. features,
                           GIN definitions, human seed thresholds)
  nardini_fullSeq_numba_v096.py    thin wrappers around tesseraidr.*, kept so
  nardini_MultiBlock_v099.py       `python src/<script>.py` and the Colab
  nardini_summary_pipeline.py      notebook keep working
scripts/hpc/             local and SLURM wrappers (run from any directory)
colab/                   Google Colab notebook + its helper module
analysis/
  plotting/              sequence, domain-diagram, window-distance and schematic plots
  expression/            proteomics ranking / GSEA notebooks
  centroid_analysis/     Centroid_analysis.ipynb + its inputs and figures
    Lists/               Enrichment_plot.ipynb, protein lists, enrichment plots
data/
  reference/             published comp. feature reference, 30 GIN centroids, GIN names
  input/                 IDR table (full + 7-IDR sample), protein list for plots
  annotation/            UniProt ID mapping, HPA subcellular location
results/
  multiblock/            main block table (block_summary_seedExpantion.20260909.csv.gz),
                         out_summary files, block_summary_.csv.gz, domain_architecture
  example_run/           outputs of the 7-IDR sample, for a quick check
  summary_plots/         co-occurrence, n_domains, valley_dist plots and exports
  figures/               IDR length and window distance distributions
  sequence_plots/        residue-colored IDR sequences
  protein_plots/         domain diagram for data/input/Proteins_to_plot_20261103.txt
```

## Compressed tables

Every table in `data/`, `results/` and `analysis/` is gzip-compressed
(`.csv.gz`, `.tsv.gz`). All scripts and notebooks read the `.gz` files
directly (pandas and polars decompress on the fly), so there is nothing to
unpack. To look at one in a terminal: `gzip -dc file.csv.gz | head`.
The pipeline writes its own outputs as plain `.csv`; compress them with
`gzip -9` before committing.

## Installation

TESSERA-IDR installs as a Python package (Python >= 3.9) that provides the
`tesseraidr` command. From the repository root:

```
pip install .                 # or: pip install -e .  (editable, for development)
pip install ".[analysis]"     # also installs the extra notebook dependencies
```

or straight from GitHub:

```
pip install "git+https://github.com/alexpmagalhaes/TESSERA-IDR.git"
```

The reference tables the pipeline needs (GIN centroids, compositional
feature normalisation, GIN definitions, human seed thresholds) are bundled
with the package, so `--comp_ref`, `--centroids` and `--gin_definitions`
can be omitted. `tesseraidr data` prints where they are.

```
tesseraidr --help
tesseraidr multiblock --input idrs.csv --n_workers 8 \
    --out_windows run.out_windows.csv --out_summary run.out_summary.csv
tesseraidr fullseq    --input idrs.csv --out_sequences run.out_sequences.csv \
    --out_summary run.fullseq_summary.csv
tesseraidr summary all --summary run.out_summary.csv --outdir figures
```

The options are the same as those of the original scripts (`tesseraidr
<command> --help`). In Python:

```python
from tesseraidr import multiblock, fullseq, summary
df, meta = multiblock.load_input_csv("idrs.csv")
windows, blocks = multiblock.run_sliding_window(df, multiblock.CONFIG.copy(), metadata_df=meta)
```

Without installing, `pip install -r requirements.txt` and
`python src/nardini_MultiBlock_v099.py ...` still work as before.

`localcider` and `numba` can conflict with numpy 2 on some HPC module stacks;
if either fails to import, pin `numpy<2`. Numba caches compiled kernels on
first run; on a read-only home directory set
`export NUMBA_CACHE_DIR=/scratch/$USER/numba_cache`.

Plots use Arial when `Arial.ttf` is found (`--arial_font_path`, else
`$TESSERAIDR_ARIAL_TTF`, else `./Arial.ttf`, else the repo root, else a
system Arial). The font is not distributed here; the
scripts fall back to a default font without it.

## Running

All wrappers `cd` to the repo root, so they can be launched from anywhere.

```
scripts/hpc/run_MB_local.sh 4       # sliding window, 4 workers
scripts/hpc/run_local.sh 4          # full sequence
sbatch scripts/hpc/run_MB_slurm.sh  # edit the #SBATCH lines first
```

Rebuild the published blocks from the windows file (block-building stage
only). On the Sep 8 windows file, this gives a summary byte-identical to
`Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.seedExpand.q60.out_summary.csv.gz`:

```
python src/nardini_MultiBlock_v099.py --stage blocks \
  --in_windows results/multiblock/Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.out_windows.csv.gz \
  --input data/input/Hsapien_IDRs_ftl100aa.csv.gz \
  --centroids data/reference/All_sapiens_IDRs_GIN_centroids.csv.gz \
  --block_method seed_expand \
  --seed_min_dist_quantile 0.60 --seed_inter_dist_quantile 0.75 \
  --seed_absorption_factor 1.5 --min_block_length 70 \
  --out_summary results/multiblock/Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.seedExpand.q60.out_summary.csv
```

`--min_windows` only affects `--block_method meanboundary`.

Post-processing (block table, domain architecture, plots):

```
python src/nardini_summary_pipeline.py all \
  --summary results/multiblock/Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.seedExpand.q60.out_summary.csv.gz \
  --idmapping data/annotation/idmapping_2026_05_29.tsv.gz \
  --hpa_csv data/annotation/HPA_subcellular_location.csv.gz \
  --gin_definitions data/reference/gin_definitions.csv.gz \
  --outdir results/summary_plots \
  --top_n_clusters 20 \
  --pair_export 23,26 --pair_export 18,26 --pair_export 7,26 \
  --combo_export 26:23,18,7
```

## Not included (download separately)

| File | Why it is not in the repo | Where it goes |
|---|---|---|
| `*.out_windows.csv.gz` (74 MB) | Regenerated by `run_MB_*.sh`; above GitHub's 50 MB warning size | `results/multiblock/` |
| `RBPbase_Hs_DescriptiveID.xlsx` | Third-party RBP annotation used by `analysis/expression/Dual_grammar.ipynb` | `analysis/expression/` |
| Proteomics matrices (`Imputed_matrix.csv`, `Ranked_matrix.csv`, ...) | Lab proteomics data; the notebooks read them from absolute local paths | see the paths inside each notebook |
| `RBP_HS.csv` (299 MB) | Full RBP2GO human table, above GitHub's 100 MB file limit. `High_confidence_human_RBPs.csv.gz`, the subset the notebook uses, is included | [RBP2GO](https://rbp2go.dkfz.de) |
| HPA `subcellular_location.tsv`, `ENSEMBL_to_Uniprot_location_proteins_202312_no_duplicates.tsv` | From the RUFF_KING repo's `general_data/HPA/`; `Lists/Enrichment_plot.ipynb` reads them, and the old-math IDR table, from absolute local paths | [RUFF_KING_Grammars_of_IDRs_using_NARDINI-](https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-) |

## Notes

- **Main output (published):** `results/multiblock/block_summary_seedExpantion.20260909.csv.gz`,
  one row per IDR block.
- `block_summary_.csv.gz` is an export from `nardini_summary_pipeline.py` that
  adds `valley_start` / `valley_end`; the files in `results/summary_plots/`
  and `domain_architecture.csv.gz` come from that run.

## Attribution and citation

The full-sequence pipeline (`tesseraidr fullseq`, formerly
`nardini_fullSeq_numba_v096.py`) is a **reimplementation (rewrite) of
NARDINI+** and the GIN cluster assignment from Ruff, King et al. It is not
the original code. The per-window features in `tesseraidr multiblock` use the
same reimplementation. The feature definitions, the human-IDRome
normalisation and the 30 GIN centroids bundled in `src/tesseraidr/data/`
come from that work. If you use TESSERA-IDR, please cite:

- Ruff KM, King MR, Ying AW, Liu V, Pant A, Lieberman WE, Shinn MK, Su X,
  Kadoch C, Pappu RV. Molecular grammars of predicted intrinsically
  disordered regions that span the human proteome. *Cell* 189(1):323-342.e17
  (2026). [doi:10.1016/j.cell.2025.10.019](https://doi.org/10.1016/j.cell.2025.10.019).
  Original code:
  [RUFF_KING_Grammars_of_IDRs_using_NARDINI-](https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-)
  (MIT License).
- Cohan MC, Shinn MK, Lalmansingh JM, Pappu RV. Uncovering non-random binary
  patterns within sequences of intrinsically disordered proteins. *J Mol Biol*
  434(2):167373 (2022).
  [doi:10.1016/j.jmb.2021.167373](https://doi.org/10.1016/j.jmb.2021.167373).

## License

TESSERA-IDR is released under the [MIT License](LICENSE). The original
NARDINI+ / GIN code is also MIT-licensed (Copyright (c) 2025 kierstenruff);
its copyright notice is kept in `LICENSE`.
