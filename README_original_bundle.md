# Nardini GIN Assignment — Fixed Kernel, Original Centroids

This folder is a self-contained bundle for running two related NARDINI+ GIN
cluster assignment pipelines with the omega/kappa off-by-one bug fixed
(https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-/issues/2),
while keeping the **original, published GIN centroids** unchanged:

- **Full-sequence assignment** (`nardini_fullSeq_numba_v096.py`) — scores
  each IDR once, as a single whole-sequence window, and assigns it to the
  nearest of the 30 centroids. This is the paper's "map arbitrary IDRs onto
  the 30 published GIN centroids" step (notebook 2 equivalent).
- **Sliding-window / multi-block assignment** (`nardini_MultiBlock_v099.py`)
  — scans each IDR with a sliding window, assigns every window to its
  nearest centroid, then denoises and merges same-cluster windows into
  contiguous "blocks" (domains) per IDR — the actual GIN block-detection
  workload this project's other scripts (`out_sequences`/`out_summary`
  files, `w70s10...` naming) were built around.

We already checked, on the 4,529-IDR clustering subset, that holding the
centroids fixed and only correcting the kernel changes very little: 96.9% of
IDRs are still nearest to their own originally-published centroid (every one
of the 30 clusters retained at least 94% of its members). This run applies
that same fixed-kernel + old-centroids assignment to the full IDR set in
`Hsapien_IDRs_ftl100aa.csv` (10,371 human IDRs, length >= 100 aa) rather
than only the clustering subset.

## Contents

- `nardini_fullSeq_numba_v096.py` — the full-sequence assignment script. For
  each input sequence it computes the 90-dim compositional + patterning
  z-score feature vector (with the corrected `_omega`/`_kappa` sliding-window
  kernel) and assigns it to the nearest of the 30 centroids by squared
  Euclidean distance.
- `nardini_MultiBlock_v099.py` — the sliding-window / multi-block assignment
  script. Same corrected `_omega`/`_kappa` kernel and centroid assignment,
  but applied per sliding window (`--window_size`/`--step_size`) rather than
  once per whole sequence; denoises the per-window cluster calls and merges
  contiguous same-cluster windows into blocks (domains) per IDR. Its two
  stages can be split with `--stage`:
  - `--stage windows` — build windows, compute features, assign clusters,
    write `--out_windows` only. This is the expensive, multiprocessing +
    numba part.
  - `--stage blocks` — read a previously-written windows CSV from
    `--in_windows` (always explicit — never inferred) and run only the
    denoise/merge/summarize step, writing `--out_summary`. Cheap and
    single-threaded; no sequences, `--comp_ref`, or `--centroids` needed.
    Pass `--input` too if you want the summary to carry the same metadata
    columns as the windows stage did (optional — omit for a bare summary).
  - `--stage all` (default) — both stages in one run, as before; this is
    what `run_MB_local.sh`/`run_MB_slurm.sh` use.
  Splitting is useful on HPC when you want the heavy windows stage on a
  many-core node and the cheap blocks stage (e.g. to re-tune
  `--min_windows`/`--min_block_length` without recomputing features) on a
  login node or a tiny job.
  The blocks stage also has two selectable block-building strategies via
  `--block_method`:
  - `meanboundary` (default, unchanged behavior) — denoise per-window
    calls, merge contiguous same-cluster windows into raw blocks, keep
    blocks spanning >= `--min_windows` windows, resolve any overlap
    between adjacent blocks at the arithmetic mean of their boundary
    coordinates, merge same-cluster neighbors, then drop anything shorter
    than `--min_block_length`.
  - `seed_expand` — instead of keeping every window and resolving overlap
    afterward, start only from statistically confident "seed" windows
    (`min_dist` at/below `--seed_min_dist_quantile`, `inter_cluster_dist`
    at/above `--seed_inter_dist_quantile`, both computed globally over the
    whole windows input) and grow each seed outward one window at a time.
    A boundary window is absorbed into a seed's block ("smart absorption")
    if its squared distance to *that seed's* centroid is within
    `--seed_absorption_factor` times its own `min_dist` — i.e. it fits the
    neighboring block almost as well as it fits its own best centroid.
    Seeds expand in priority order (most statistically confident first),
    and expansion is blocked the moment it would reach a window another,
    higher-priority seed already claimed, so window-index overlap can't
    happen; the same arithmetic-mean boundary step as `meanboundary` is
    still applied afterward, because neighboring sliding windows share
    residues by construction even when their window-index sets are
    disjoint. The point of this method is that a window's fate (kept vs.
    dropped, and which block it joins) is decided by its own distance
    statistics rather than only by its position in the sequence.
    Requires the windows input to carry the full per-window 90-dim
    feature columns (i.e. an `--stage windows` output, not a
    stripped-down file) and `--centroids` to be supplied at the blocks
    stage too.
- Both scripts read `--input` as either a bare `IDR_ID,Sequence` table or, as
  here, a metadata-rich CSV — any columns beyond `IDR_ID` and the sequence
  column (`Sequence` or `IDR_Sequence`) are detected automatically and
  carried through onto every output row, so no separate join is needed
  afterward.
- `All_sapiens_IDRs_comp_feat.csv` — the published compositional feature
  reference (per-feature Mean/Std used to z-score the 54 compositional
  features). Unchanged from the original analysis.
- `All_sapiens_IDRs_GIN_centroids.csv` — the **original, published** 30 GIN
  centroids (90 features each), computed with the buggy kernel. Kept as-is
  per your request, so this run isolates the effect of the kernel fix alone.
- `Hsapien_IDRs_ftl100aa.csv` — the input: the full metadata table (10,371
  rows) with `UniprotID`, `Uniprot name`, `Description`, `Gene name`,
  `IDR_START`, `IDR_END`, `IDR_length`, `IDR_ID`, `IDR_Sequence`. Passed
  directly to `--input` for both scripts; every column except
  `IDR_Sequence` itself is carried through into the outputs.
- `Hsapien_IDRs_ftl100aa.splf.csv` — the bare `IDR_ID,Sequence` version of
  the same input, kept for reference / other tools that want the minimal
  format. Not used by any of the `run_*` scripts now that both read the
  metadata file directly.
- `requirements.txt` — Python packages the scripts import.
- `run_local.sh` / `run_slurm.sh` — wrappers for the full-sequence script
  (`nardini_fullSeq_numba_v096.py`).
- `run_MB_local.sh` / `run_MB_slurm.sh` — wrappers for the sliding-window
  script (`nardini_MultiBlock_v099.py`).
- The `_slurm` variants are SLURM `sbatch` templates — edit the `#SBATCH`
  lines (partition, account, walltime, cpus, memory) for your cluster
  before submitting. The multi-block run scans far more windows than the
  full-sequence run scores whole sequences, so it needs a longer walltime
  budget (the template defaults to 48h vs. 24h).

## Environment

The script needs: `numpy`, `pandas`, `polars`, `scipy`, `tqdm`,
`multiprocess`, `localcider`, `numba`.

The most reliable option is to replicate whatever conda/venv environment
this pipeline already runs in on your Mac (e.g. `conda env export --from-history
> environment.yml` there, then `conda env create -f environment.yml` on the
HPC login node). `requirements.txt` is provided as a pip-based fallback if
you'd rather build a fresh environment:

```
pip install -r requirements.txt
```

`localcider` and `numba` can be picky about the numpy version on some HPC
module stacks — if either fails to import, try pinning `numpy<2` first.

Numba compiles and caches the `_omega`/`_kappa` kernels on first run. If your
HPC home directory is read-only or quota-limited, point the cache at scratch
space before running:

```
export NUMBA_CACHE_DIR=/scratch/$USER/numba_cache
```

## Usage

Full-sequence assignment:

```
python nardini_fullSeq_numba_v096.py \
  --input Hsapien_IDRs_ftl100aa.csv \
  --comp_ref All_sapiens_IDRs_comp_feat.csv \
  --centroids All_sapiens_IDRs_GIN_centroids.csv \
  --num_scrambles 100 \
  --n_workers 32 \
  --out_sequences Hsapien_IDRs_ftl100aa.fullseq.fixedKernel.out_sequences.csv \
  --out_summary Hsapien_IDRs_ftl100aa.fullseq.fixedKernel.out_summary.csv \
  --out_dist_matrix Hsapien_IDRs_ftl100aa.fullseq.fixedKernel.out_dist_matrix.csv
```

Sliding-window / multi-block assignment:

```
python nardini_MultiBlock_v099.py \
  --input Hsapien_IDRs_ftl100aa.csv \
  --comp_ref All_sapiens_IDRs_comp_feat.csv \
  --centroids All_sapiens_IDRs_GIN_centroids.csv \
  --window_size 70 \
  --step_size 10 \
  --num_scrambles 100 \
  --n_workers 32 \
  --min_windows 3 \
  --min_block_length 70 \
  --out_windows Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.out_windows.csv \
  --out_summary Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.out_summary.csv
```

`--n_workers` should match the number of CPUs you request. `run_local.sh` /
`run_slurm.sh` and `run_MB_local.sh` / `run_MB_slurm.sh` wrap these same
commands; edit the paths/flags directly if you want different output names,
window/step sizes, or scramble counts.

To re-run just the blocks stage with the `seed_expand` strategy against an
existing `out_windows.csv` (cheap — no sequences, no numba/multiprocessing):

```
python nardini_MultiBlock_v099.py \
  --stage blocks \
  --in_windows Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.out_windows.csv \
  --centroids All_sapiens_IDRs_GIN_centroids.csv \
  --block_method seed_expand \
  --seed_min_dist_quantile 0.25 \
  --seed_inter_dist_quantile 0.75 \
  --seed_absorption_factor 1.5 \
  --min_windows 3 \
  --min_block_length 70 \
  --out_summary Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.seedExpand.out_summary.csv
```

(Add `--input Hsapien_IDRs_ftl100aa.csv` too if you want the same metadata
columns joined onto this summary.)

## Output

Same files the original (buggy-kernel) pipeline produced, just with a
`fixedKernel` tag in the name so they don't collide with any prior run's
outputs in the same directory. All output rows carry `IDR_ID` and the 7
input metadata columns first, since the input metadata rides along
automatically — no separate join step is needed to compare against a prior
old-kernel run; just re-join the corresponding output file onto it by
`IDR_ID`.

Full-sequence assignment (`nardini_fullSeq_numba_v096.py`):

- `*.out_sequences.csv` — one row per IDR: metadata columns, then the full
  90-dim z-scored feature vector, the assigned cluster (0-29, indexing the
  *original* centroids), `min_dist` (squared distance to that centroid) and
  `inter_cluster_dist` (gap to the second-nearest centroid — small values
  flag borderline calls).
- `*.out_summary.csv` — one row per IDR with `n_distinct_clusters` (always 1
  here, since each IDR is scored as a single whole-sequence window) and
  `domain_details` (the same information in the paper's block-record
  format, for compatibility with downstream scripts that expect it).
- `*.out_dist_matrix.csv` — metadata columns followed by the full
  (IDRs x 30 centroids) squared-distance matrix, if you want to inspect
  close calls yourself rather than relying only on
  `min_dist`/`inter_cluster_dist`.

Sliding-window / multi-block assignment (`nardini_MultiBlock_v099.py`):

- `*.out_windows.csv` — one row per (IDR, window): metadata columns, then
  `start`/`end`/`window_seq`, the assigned cluster for that window, and its
  90-dim feature vector plus `min_dist`/`inter_cluster_dist`.
- `*.out_summary.csv` — one row per IDR that has at least
  `--min_distinct_clusters` distinct blocks after denoising and merging:
  metadata columns, then `n_distinct_clusters`, `n_domains`,
  `clusters_ordered` (the linear sequence of block cluster IDs), and
  `domain_details` (per-block start/end/length/valley info).

## Post-processing: annotate, parse to long format, plot

`nardini_summary_pipeline.py` takes an `out_summary.csv` from
`nardini_MultiBlock_v099.py` (metadata already joined in, so no separate
annotation step is needed) and explodes it to a block-level long table,
optionally builds a UniProt domain-architecture table, and generates the
full analysis/plot suite (n_domains distribution, valley_dist distribution,
cluster co-occurrence heatmap/dot plot + statistics, an optional HPA
localization co-eligibility dot plot, and CSV exports for specific
cluster-pair/combo protein sets). It has four subcommands -- `blocks`,
`domains`, `plots`, and `all` (chains all three in memory) -- run
`python nardini_summary_pipeline.py --help` or `... <subcommand> --help`
for the full option list.

Full run, from this directory, using the files already here:

```
python nardini_summary_pipeline.py all \
  --summary Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.seedExpand.q60.saf125.out_summary.csv \
  --idmapping idmapping_2026_05_29.tsv \
  --hpa_csv HPA_subcellular_location.csv \
  --gin_definitions gin_definitions.csv \
  --outdir figures \
  --top_n_clusters 20 \
  --pair_export 23,26 --pair_export 18,26 --pair_export 7,26 \
  --combo_export 26:23,18,7
```

- `--idmapping` is optional -- omit it to skip the domain-architecture
  table (`domain_architecture.csv`).
- `--hpa_csv`/`--gin_definitions` are optional and only take effect
  together -- they enable the localization log2(odds ratio) dot plot;
  omit both to skip it.
- `--top_n_clusters 20` caps the co-occurrence heatmap/dot plots to the 20
  most frequent clusters, for legibility -- drop it to include every
  cluster seen, or use `--cluster_order 7,3,19,...` for an explicit order.
- `--pair_export`/`--combo_export` are repeatable and generalize the old
  notebook's hardcoded cluster-pair exports (`23,26`, `18,26`, `7,26`; and
  "26 co-occurring with 23, 18, or 7") -- pass any cluster IDs you want.
- `--arial_font_path` defaults to `Arial.ttf` one directory up from the
  script (i.e. `../Arial.ttf` from here), so it doesn't need to be passed
  explicitly when run from this folder layout.

`blocks`/`domains`/`plots` can also be run as separate steps (e.g. to
re-plot with different `--cluster_order`/`--max_n_domains` without
re-exploding the summary) -- see `--help` on each subcommand.
