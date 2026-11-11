#!/usr/bin/env python
# coding: utf-8
"""
NARDINI Sliding-Window Cluster Assignment
==========================================
Applies the NARDINI+ feature pipeline on a sliding window over each IDR
sequence and assigns each window to the nearest pre-calculated GIN centroid.

Attribution
-----------
The per-window features and GIN assignment reimplement (rewrite) NARDINI+
of Ruff, King et al. (Cell 2026, doi:10.1016/j.cell.2025.10.019; original
code MIT License, Copyright (c) 2025 kierstenruff,
https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-),
with patterning z-scores from NARDINI (Cohan et al., J Mol Biol 2022,
doi:10.1016/j.jmb.2021.167373). See tesseraidr.fullseq for the full
references. The sliding-window block building is new to TESSERA-IDR.

v 0.98 by Alexandre P. Magalhaes

v 0.99 by Alexandre P. Magalhaes
  - Fixed an off-by-one in _omega/_kappa's sliding-window loop (see
    https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-/issues/2):
    a sequence of length L has L-b+1 complete size-b windows, not L-b+2; the
    extra iteration's slice ran past the array end, numpy silently truncated
    it, and the code still divided by the hardcoded b. Also switched both
    functions to an O(1)-per-step rolling sum now that the window count is
    correct (~1.7x / ~2.5x faster for _omega / _kappa respectively). Same fix
    already applied to nardini_fullSeq_numba_v096.py.
  - --input may now be a metadata-rich CSV (e.g. UniprotID, Gene name,
    Description, IDR_START/END, ...) instead of a bare IDR_ID/Sequence
    table. Any columns beyond IDR_ID and the sequence column are detected
    automatically and carried through onto every output row (repeated
    across all windows/blocks of the same IDR), so results don't need a
    separate manual join back onto the input metadata afterward. The
    sequence column may be named 'Sequence' or 'IDR_Sequence'.
  - Added a second, selectable block-building strategy for the blocks
    stage: --block_method {meanboundary, seed_expand}. The existing
    denoise/overlap-resolution pipeline is now "meanboundary" (unchanged
    behavior; see below for the current default). "seed_expand" instead starts from
    statistically confident windows (quantile thresholds on min_dist /
    inter_cluster_dist: --seed_min_dist_quantile, --seed_inter_dist_quantile)
    and grows them outward window-by-window with a smart-absorption test
    against each seed's centroid (--seed_absorption_factor), so windows are
    kept or dropped based on their own statistics rather than discarded
    purely by position, and overlap is prevented by construction instead of
    resolved afterward. Requires the windows input to carry the full
    per-window feature columns and --centroids to be supplied at the
    blocks stage.
  - Defaults now reproduce the published blocks
    (block_summary_seedExpantion.20260909.csv): --block_method seed_expand,
    --window_size 70, --seed_min_dist_quantile 0.60,
    --seed_inter_dist_quantile 0.75, --seed_absorption_factor 1.5,
    --min_block_length 70. seed_expand applies no --min_windows filter
    (that option only affects meanboundary), which is the rule the
    published run used.
  - seed_expand accepts absolute seed cut-offs (--seed_min_dist_threshold,
    --seed_inter_dist_threshold) that override the quantiles, so a small
    input (e.g. a Colab run) can use the seed definition of the full human
    run instead of quantiles of its own few windows.
"""

import os
import re
import math
import time
import warnings
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats
from tqdm import tqdm
from multiprocess import Pool
from localcider.sequenceParameters import SequenceParameters
from numba import njit  # <-- Added Numba import

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Thread-count pins ─────────────────────────────────────────────────────────
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _tv in _THREAD_VARS:
    os.environ.setdefault(_tv, "1")

# ── Worker-process cap ────────────────────────────────────────────────────────
_SAFE_WORKER_CAP = 32

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  –  edit this block to match your environment
# ══════════════════════════════════════════════════════════════════════════════
_DATA_DIR = Path(__file__).resolve().parent / "data"

CONFIG = dict(
    comp_reference_csv   = str(_DATA_DIR / "All_sapiens_IDRs_comp_feat.csv.gz"),
    centroids_csv        = str(_DATA_DIR / "All_sapiens_IDRs_GIN_centroids.csv.gz"),
    window_size          = 70,
    step_size            = 10,
    min_window_size      = 50,
    num_scrambles        = 100,
    n_workers            = None,
    output_windows_csv   = "nardini_window_results.csv",
    output_summary_csv   = "nardini_multi_cluster_summary.csv",
    min_distinct_clusters = 1,
    zscore_clip           = 3.0,
    # -- Reprocessing / summary-building parameters (used by build_summary) --
    min_windows           = 3,    # meanboundary only: minimum number of windows a block must span
    min_block_length      = 70,   # minimum residue length of a final domain
    # -- Block-building strategy selector (used by compute_blocks) --
    block_method              = "seed_expand",  # or "meanboundary"
    seed_min_dist_quantile    = 0.60,  # seed_expand: seeds need min_dist <= this quantile
    seed_inter_dist_quantile  = 0.75,  # seed_expand: seeds need inter_cluster_dist >= this quantile
    seed_absorption_factor    = 1.5,   # seed_expand: absorb if dist_to_centroid <= factor * own min_dist
    seed_min_dist_threshold   = None,  # seed_expand: absolute min_dist cut-off, overrides the quantile
    seed_inter_dist_threshold = None,  # seed_expand: absolute inter_cluster_dist cut-off, overrides the quantile
)
# ══════════════════════════════════════════════════════════════════════════════

# ── Amino-acid type definitions (patterning) ──────────────────────────────────
_POL  = list("STNQCH")
_HYD  = list("ILMV")
_POS  = list("RK")
_NEG  = list("ED")
_ARO  = list("FWY")
_ALA  = list("A")
_PRO  = list("P")
_GLY  = list("G")

TYPEALL    = [_POL, _HYD, _POS, _NEG, _ARO, _ALA, _PRO, _GLY]
PATTNAMES  = ["pol", "hyd", "pos", "neg", "aro", "ala", "pro", "gly"]
NUM_TYPES  = 8

_PATT_FEATS   = []
_TARGET_IDX   = []
for _i in range(NUM_TYPES):
    for _j in range(_i, NUM_TYPES):
        _PATT_FEATS.append(f"{PATTNAMES[_i]}-{PATTNAMES[_j]}")
        _TARGET_IDX.append(_i * 8 + _j)
TARGET_INDICES = np.array(_TARGET_IDX)

PATCH_AAS   = "ACDEFGHIKLMNPQRSTVY"
NUM_INT     = 2
MIN_BLOCK   = 4

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS (Optimized with Numba & Vectorization)
# ══════════════════════════════════════════════════════════════════════════════

@njit(fastmath=True, cache=True)
def _omega(mask: np.ndarray, b: int, sig_all: float) -> float:
    L = len(mask)
    n_windows = L - b + 1
    if n_windows < 1:
        return 0.0
    sig_x = np.zeros(n_windows)

    count = 0.0
    for i in range(b):
        count += mask[i]
    frac = count / b
    sig_x[0] = (frac - (1.0 - frac)) ** 2

    for x in range(1, n_windows):
        count += mask[x + b - 1] - mask[x - 1]
        frac = count / b
        sig_x[x] = (frac - (1.0 - frac)) ** 2

    return float(np.mean((sig_x - sig_all) ** 2))

@njit(fastmath=True, cache=True)
def _kappa(mask1: np.ndarray, mask2: np.ndarray, b: int, sig_all: float) -> float:
    L = len(mask1)
    n_windows = L - b + 1
    if n_windows < 1:
        return 0.0
    sig_x = np.zeros(n_windows)

    count1 = 0.0
    count2 = 0.0
    for i in range(b):
        count1 += mask1[i]
        count2 += mask2[i]
    if count1 + count2 == 0:
        sig_x[0] = 0.0
    else:
        f1, f2 = count1 / b, count2 / b
        sig_x[0] = (f1 - f2) ** 2 / (f1 + f2)

    for x in range(1, n_windows):
        count1 += mask1[x + b - 1] - mask1[x - 1]
        count2 += mask2[x + b - 1] - mask2[x - 1]
        if count1 + count2 == 0:
            sig_x[x] = 0.0
        else:
            f1, f2 = count1 / b, count2 / b
            sig_x[x] = (f1 - f2) ** 2 / (f1 + f2)

    return float(np.mean((sig_x - sig_all) ** 2))

def _patch_fraction(seq: str, aa: str, L: int) -> float:
    flags = np.zeros(L, dtype=int)
    pos = [i for i, r in enumerate(seq) if r == aa]

    if len(pos) < MIN_BLOCK:
        return 0.0

    for p in range(len(pos) - 1):
        if pos[p + 1] - pos[p] <= NUM_INT + 1:
            flags[pos[p]:pos[p + 1] + 1] = 1

    for p in pos:
        flags[p] = 1

    total = 0
    in_block = False
    start_idx = 0

    for i in range(L):
        if flags[i] == 1 and not in_block:
            in_block = True
            start_idx = i
        elif flags[i] == 0 and in_block:
            in_block = False
            sub = seq[start_idx:i]
            if sub.count(aa) >= MIN_BLOCK:
                total += len(sub)

    if in_block:
        sub = seq[start_idx:L]
        if sub.count(aa) >= MIN_BLOCK:
            total += len(sub)

    return total / L

def _rg_patch_fraction(seq: str, L: int) -> float:
    flags = np.zeros(L, dtype=int)
    pos = [i for i, r in enumerate(seq) if r in ("R", "G")]

    for p in range(len(pos) - 1):
        if pos[p + 1] - pos[p] <= NUM_INT + 1:
            flags[pos[p]:pos[p + 1] + 1] = 1

    for p in pos:
        flags[p] = 1

    total = 0
    in_block = False
    start_idx = 0

    for i in range(L):
        if flags[i] == 1 and not in_block:
            in_block = True
            start_idx = i
        elif flags[i] == 0 and in_block:
            in_block = False
            sub = seq[start_idx:i]
            if sub.count("RG") >= 2:
                total += len(sub)

    if in_block:
        sub = seq[start_idx:L]
        if sub.count("RG") >= 2:
            total += len(sub)

    return total / L

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_compositional_raw(seq: str) -> np.ndarray:
    L = len(seq)
    INVALID = set("XUZJBO")
    if L < 1 or INVALID.intersection(seq):
        return np.full(54, np.nan)

    ob  = SequenceParameters(seq)
    aaf = ob.get_amino_acid_fractions()

    feats = [
        aaf["A"], aaf["C"], aaf["D"], aaf["E"], aaf["F"],
        aaf["G"], aaf["H"], aaf["I"], aaf["K"], aaf["L"],
        aaf["M"], aaf["N"], aaf["P"], aaf["Q"], aaf["R"],
        aaf["S"], aaf["T"], aaf["V"], aaf["W"], aaf["Y"],
        aaf["K"] + aaf["R"],
        aaf["D"] + aaf["E"],
        aaf["Q"] + aaf["N"] + aaf["S"] + aaf["T"] +
        aaf["G"] + aaf["C"] + aaf["H"],
        aaf["A"] + aaf["L"] + aaf["M"] + aaf["I"] + aaf["V"],
        aaf["F"] + aaf["W"] + aaf["Y"],
        np.log10(((L * aaf["R"]) + 1) / ((L * aaf["K"]) + 1)),
        np.log10(((L * aaf["E"]) + 1) / ((L * aaf["D"]) + 1)),
        ob.get_fraction_expanding(),
        ob.get_FCR(),
        ob.get_NCPR(),
        ob.get_mean_hydropathy(),
        ob.get_fraction_disorder_promoting(),
        ob.get_isoelectric_point(),
        ob.get_PPII_propensity(mode="hilser"),
    ]

    for aa in PATCH_AAS:
        feats.append(_patch_fraction(seq, aa, L))

    feats.append(_rg_patch_fraction(seq, L))

    return np.array(feats, dtype=float)

def compute_patterning_zscores(seq: str, num_scrambles: int = 100) -> np.ndarray:
    L = len(seq)
    masks = np.zeros((NUM_TYPES, L), dtype=float)
    for i, residues in enumerate(TYPEALL):
        for r in residues:
            masks[i] += np.array([1.0 if x == r else 0.0 for x in seq])

    fracs     = masks.sum(axis=1) / L
    sig_omega = np.zeros(NUM_TYPES)
    sig_kappa = np.zeros((NUM_TYPES, NUM_TYPES))

    for i in range(NUM_TYPES):
        if fracs[i] > 0.10:
            sig_omega[i] = (fracs[i] - (1 - fracs[i])) ** 2
        for j in range(i + 1, NUM_TYPES):
            if fracs[i] > 0.10 and fracs[j] > 0.10:
                c1, c2 = fracs[i], fracs[j]
                sig_kappa[i, j] = (c1 - c2) ** 2 / (c1 + c2)

    org_arr  = np.zeros((NUM_TYPES, NUM_TYPES))
    good_idx = []
    for i in range(NUM_TYPES):
        if fracs[i] > 0.10:
            org_arr[i, i] = _omega(masks[i], 5, sig_omega[i])
            good_idx.append(i * 8 + i)
        for j in range(i + 1, NUM_TYPES):
            if fracs[i] > 0.10 and fracs[j] > 0.10:
                org_arr[i, j] = _kappa(masks[i], masks[j], 5, sig_kappa[i, j])
                good_idx.append(i * 8 + j)
    org_1d = org_arr.reshape(-1)

    scr_vals = np.zeros((num_scrambles, 64))
    for x in range(num_scrambles):
        perm      = np.random.permutation(L)
        scr_masks = masks[:, perm]
        scr_arr   = np.zeros((NUM_TYPES, NUM_TYPES))
        for i in range(NUM_TYPES):
            if fracs[i] > 0.10:
                scr_arr[i, i] = _omega(scr_masks[i], 5, sig_omega[i])
            for j in range(i + 1, NUM_TYPES):
                if fracs[i] > 0.10 and fracs[j] > 0.10:
                    scr_arr[i, j] = _kappa(scr_masks[i], scr_masks[j], 5, sig_kappa[i, j])
        scr_vals[x] = scr_arr.reshape(-1)

    z = np.zeros(64)
    for idx in good_idx:
        data = scr_vals[:, idx]
        try:
            a, loc, scale = stats.gamma.fit(data)
            cmean = stats.gamma.mean(a, loc, scale)
            cvar  = stats.gamma.var(a, loc, scale)
            z[idx] = 0.0 if org_1d[idx] == 0 else (org_1d[idx] - cmean) / math.sqrt(max(cvar, 1e-12))
        except Exception:
            z[idx] = 0.0

    return z[TARGET_INDICES]

# ══════════════════════════════════════════════════════════════════════════════
#  WORKER (called in multiprocessing pool)
# ══════════════════════════════════════════════════════════════════════════════

_MEAN_ARR   = None
_STD_ARR    = None
_CENTMAT    = None
_NUMSCRAMS  = None

def _init_worker(mean_arr, std_arr, centmat, numscrams):
    for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "BLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[_tv] = "1"
    global _MEAN_ARR, _STD_ARR, _CENTMAT, _NUMSCRAMS
    _MEAN_ARR  = mean_arr
    _STD_ARR   = std_arr
    _CENTMAT   = centmat
    _NUMSCRAMS = numscrams

def _process_window(args):
    idr_id, start, end, wseq, feat_names = args
    result = dict(IDR_ID=idr_id, start=start, end=end, window_seq=wseq)

    raw_comp = compute_compositional_raw(wseq)
    if np.isnan(raw_comp).all():
        result.update(dict(cluster=np.nan, min_dist=np.nan,
                           inter_cluster_dist=np.nan))
        for name in feat_names:
            result[name] = np.nan
        return result

    z_comp = (raw_comp - _MEAN_ARR) / _STD_ARR
    z_patt = compute_patterning_zscores(wseq, _NUMSCRAMS)

    z_all     = np.concatenate([z_comp, z_patt])
    z_clipped = np.clip(z_all, -CONFIG["zscore_clip"], CONFIG["zscore_clip"])

    dists  = np.nansum((z_clipped[None, :] - _CENTMAT) ** 2, axis=1)
    order  = np.argsort(dists)
    cluster      = int(order[0])
    min_dist     = float(dists[order[0]])
    inter_dist   = float(dists[order[1]] - dists[order[0]]) if len(order) > 1 else np.nan

    result.update(dict(cluster=cluster,
                       min_dist=min_dist,
                       inter_cluster_dist=inter_dist))
    for name, val in zip(feat_names, z_clipped.tolist()):
        result[name] = val

    return result

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_references(cfg: dict):
    comp_ref  = pd.read_csv(cfg["comp_reference_csv"])
    mean_arr  = comp_ref["Mean"].values.astype(float)
    std_arr   = comp_ref["Std"].values.astype(float)
    comp_feat_names = comp_ref["Feature"].tolist()

    cent_df   = pd.read_csv(cfg["centroids_csv"])
    cent_cols = cent_df.columns.tolist()[1:]
    centmat   = cent_df[cent_cols].to_numpy().astype(float)

    all_feat_names = comp_feat_names + _PATT_FEATS

    if centmat.shape[1] != 90:
        raise ValueError(
            f"Centroid matrix has {centmat.shape[1]} columns but expected 90 "
            f"(54 compositional + 36 patterning)."
        )

    log.info("Reference files loaded — %d clusters, %d comp features.",
             centmat.shape[0], len(comp_feat_names))
    return mean_arr, std_arr, centmat, all_feat_names

def build_windows(idr_id: str, seq: str, window_size: int,
                  step_size: int, min_window_size: int,
                  feat_names: list) -> list:
    L       = len(seq)
    windows = []
    for start in range(0, L, step_size):
        end  = min(start + window_size, L)
        wlen = end - start
        if wlen < min_window_size:
            break
        windows.append((idr_id, start, end, seq[start:end], feat_names))
    return windows

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY REPROCESSING  (mean-boundary variant — folded in from
#  nardini_reprocess_summary_meanboundary.py)
# ══════════════════════════════════════════════════════════════════════════════
#
# Pipeline
# --------
#   1. Denoise window assignments (multi-pass A-B-A / A-B-C rules).
#   2. Build raw blocks by merging contiguous same-cluster windows.
#   3. Eligibility filter: keep blocks that span >= min_windows windows
#      (default 3).  Blocks below this threshold are discarded entirely.
#   4. Overlap resolution: sort by block_start; for each adjacent pair that
#      overlaps, place the boundary at the arithmetic mean:
#
#          boundary = floor( (left_block_end + right_block_start) / 2 )
#
#      Left block is clipped to [..., boundary], right block to [boundary, ...].
#      Blocks whose length becomes <= 0 after clipping are dropped.
#   5. Merge consecutive same-cluster blocks and compile the summary.

_QUALITY_EPS = 1e-9

def _valley_quality(min_d, inter_d) -> float:
    """Combined-quality score at a window: min_dist / inter_cluster_dist.
    Low score = close to its centroid AND clearly separated from the
    runner-up = confidently and unambiguously assigned. Shared by both
    block-building strategies (build_summary and build_summary_seed_expand).
    """
    if inter_d is None or (isinstance(inter_d, float) and math.isnan(inter_d)):
        return 0.0
    return float(min_d) / max(float(inter_d), _QUALITY_EPS)


def _resolve_overlaps_and_merge(candidates: list, window_size: int) -> list:
    """Shared post-processing step for both block-building strategies.

    Sliding windows overlap in residue coordinates by construction (step
    size < window size), so two blocks can end up with overlapping
    [block_start, block_end) ranges even when the underlying strategy
    already guarantees their *window-index* sets are disjoint (as
    build_summary_seed_expand's claimed-array expansion does). This step
    is what actually removes any residue-coordinate overlap, and is also
    where meanboundary's own overlap resolution lives:

      1. Sort by block_start, then make a single left-to-right pass; for
         each adjacent pair that overlaps, clip the boundary to the
         arithmetic mean of the left block's end and the right block's
         start (floor-divided), clipping the left block to [..., boundary]
         and the right block to [boundary, ...].
      2. Drop any block that collapsed to <= 0 length after clipping, and
         clamp valley coordinates to stay within the (possibly clipped)
         block extent.
      3. Merge consecutive same-cluster blocks (two blocks of the same
         cluster with nothing but a gap between them are one domain);
         keep whichever valley is more confident (lower valley_dist).

    Returns the final list of domain dicts (pre min_block_length filter).
    """
    if not candidates:
        return []

    placed = sorted((dict(**c) for c in candidates), key=lambda b: b["block_start"])

    for i in range(len(placed) - 1):
        curr = placed[i]
        nxt  = placed[i + 1]

        if curr["block_end"] > nxt["block_start"]:           # overlap
            boundary = (curr["block_end"] + nxt["block_start"]) // 2

            curr["block_end"]    = boundary
            curr["block_length"] = boundary - curr["block_start"]

            nxt["block_start"]   = boundary
            nxt["block_length"]  = nxt["block_end"] - boundary

    final_placed = []
    for b in placed:
        if b["block_length"] <= 0:
            continue
        b["valley_start"] = max(b["valley_start"], b["block_start"])
        b["valley_end"]   = min(b["valley_end"],   b["block_end"])
        if b["valley_end"] <= b["valley_start"]:
            b["valley_start"] = b["block_start"]
            b["valley_end"]   = b["block_start"] + window_size
        final_placed.append(b)

    placed = final_placed
    placed.sort(key=lambda d: d["block_start"])

    domains = []
    for d in placed:
        if domains and domains[-1]["cluster"] == d["cluster"]:
            prev = domains[-1]
            prev["block_end"]    = d["block_end"]
            prev["block_length"] = d["block_end"] - prev["block_start"]
            if d["valley_dist"] < prev["valley_dist"]:
                prev["valley_dist"]  = d["valley_dist"]
                prev["valley_score"] = d["valley_score"]
                prev["valley_start"] = d["valley_start"]
                prev["valley_end"]   = d["valley_end"]
            prev["num_windows"] += d["num_windows"]
        else:
            domains.append(dict(**d))

    return domains


def build_summary(win_df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """
    Denoise → build raw blocks → filter by window count → resolve overlaps
    with arithmetic-mean boundaries → compile summary.

    Parameters
    ----------
    win_df : polars DataFrame
        At minimum: IDR_ID, start, end, cluster, min_dist, inter_cluster_dist.
    cfg : dict
        Must contain: window_size, min_windows, min_block_length, min_distinct_clusters.

    Returns
    -------
    polars DataFrame with one row per IDR that has >= min_distinct_clusters.
    """
    window_size           = cfg["window_size"]
    min_windows           = cfg["min_windows"]
    min_block_length      = cfg["min_block_length"]
    min_distinct_clusters = cfg["min_distinct_clusters"]

    ids = win_df["IDR_ID"].unique(maintain_order=True).to_list()
    log.info("Reprocessing %d unique IDRs …", len(ids))

    core_cols = ["IDR_ID", "start", "end", "cluster", "min_dist", "inter_cluster_dist"]
    win_core  = win_df.select(core_cols)

    summary_rows = []

    for idr_id in ids:
        sub = win_core.filter(pl.col("IDR_ID") == idr_id).sort("start")
        if sub.is_empty():
            continue

        windows = sub.to_dicts()

        # -----------------------------------------------------------------
        # Multi-Pass Denoising
        #
        # A "single-window run" is a window whose cluster differs from both
        # its left and right neighbours (checked by start coordinate only).
        #
        # Two rules, applied repeatedly until the sequence stops changing:
        #   A-B-A  (same cluster on both sides) -> fix: assign B to A
        #   A-B-C  (different clusters, or a sequence boundary)
        #                                        -> eliminate: remove B
        #
        # Running until convergence handles cascading cases, e.g. fixing
        # one A-B-A window can expose a new one adjacent to it.
        # -----------------------------------------------------------------
        changed = True
        while changed and len(windows) >= 2:
            changed = False
            to_fix    = {}
            to_remove = set()

            for i in range(len(windows)):
                left_c  = windows[i - 1]["cluster"] if i > 0              else None
                right_c = windows[i + 1]["cluster"] if i < len(windows)-1  else None
                curr_c  = windows[i]["cluster"]

                # Only act on interior windows — never touch sequence boundaries
                if left_c is None or right_c is None:
                    continue

                left_diff  = (left_c  != curr_c)
                right_diff = (right_c != curr_c)

                if left_diff and right_diff:          # single-window run
                    if left_c == right_c:
                        to_fix[i] = left_c            # A-B-A: fix to A
                    else:
                        to_remove.add(i)              # A-B-C: remove B

            for i, c in to_fix.items():
                windows[i]["cluster"] = c
                changed = True

            if to_remove:
                windows = [w for i, w in enumerate(windows)
                           if i not in to_remove]
                changed = True

        if not windows:
            continue

        # -----------------------------------------------------------------
        # Raw Block Building
        #
        # Merge contiguous same-cluster windows into blocks.
        # The block's representative window ("valley") is the one with the
        # lowest min_dist.  Priority score = combined quality at the valley:
        #
        #   valley_score = valley_dist / inter_cluster_dist_at_valley
        #
        # Low score = close to centroid AND clearly separated from the
        # next-nearest centroid = confidently and unambiguously assigned.
        # -----------------------------------------------------------------
        raw_blocks = []
        current_cluster = windows[0]["cluster"]
        run_starts      = [windows[0]["start"]]
        run_ends        = [windows[0]["end"]]
        run_dists       = [windows[0]["min_dist"]]
        run_inter_dists = [windows[0].get("inter_cluster_dist")]

        def commit_block():
            best_idx     = int(np.argmin(run_dists))
            valley_score = _valley_quality(run_dists[best_idx],
                                           run_inter_dists[best_idx])
            raw_blocks.append(dict(
                cluster      = current_cluster,
                block_start  = min(run_starts),
                block_end    = max(run_ends),
                block_length = max(run_ends) - min(run_starts),
                valley_dist  = run_dists[best_idx],
                valley_score = valley_score,
                valley_start = run_starts[best_idx],
                valley_end   = run_ends[best_idx],
                num_windows  = len(run_starts),
            ))

        for w in windows[1:]:
            if w["cluster"] == current_cluster:
                run_starts.append(w["start"])
                run_ends.append(w["end"])
                run_dists.append(w["min_dist"])
                run_inter_dists.append(w.get("inter_cluster_dist"))
            else:
                commit_block()
                current_cluster = w["cluster"]
                run_starts      = [w["start"]]
                run_ends        = [w["end"]]
                run_dists       = [w["min_dist"]]
                run_inter_dists = [w.get("inter_cluster_dist")]

        commit_block()

        # -----------------------------------------------------------------
        # Eligibility Filter (applied after denoising)
        #
        # A block is eligible if it spans at least min_windows windows.
        # This is a count-based criterion that is independent of the
        # physical window size, so it remains meaningful even when
        # window_size or step_size are varied.
        # -----------------------------------------------------------------
        candidates = [b for b in raw_blocks if b["num_windows"] >= min_windows]
        if not candidates:
            continue

        # -----------------------------------------------------------------
        # Overlap Resolution (arithmetic-mean boundary) + merge consecutive
        # same-cluster blocks. See _resolve_overlaps_and_merge for details.
        # -----------------------------------------------------------------
        domains = _resolve_overlaps_and_merge(candidates, window_size)

        if not domains:
            continue

        # -----------------------------------------------------------------
        # Minimum Block Length Filter (applied after final merge)
        #
        # Drop any domain whose residue length is below min_block_length.
        # This removes spuriously short blocks that survive denoising and
        # overlap resolution but are too small to be biologically meaningful.
        # -----------------------------------------------------------------
        domains = [d for d in domains if d["block_length"] >= min_block_length]

        if not domains:
            continue

        # -----------------------------------------------------------------
        # Summary Compilation
        # -----------------------------------------------------------------
        distinct_clusters = set(d["cluster"] for d in domains)
        if len(distinct_clusters) < min_distinct_clusters:
            continue

        summary_rows.append(dict(
            IDR_ID              = idr_id,
            n_distinct_clusters = len(distinct_clusters),
            n_domains           = len(domains),
            clusters_ordered    = str([d["cluster"] for d in domains]),
            domain_details      = str(domains),
        ))

    log.info("Multi-cluster IDRs found: %d / %d", len(summary_rows), len(ids))

    if summary_rows:
        return pl.from_dicts(summary_rows)

    return pl.DataFrame(schema={
        "IDR_ID":               pl.Utf8,
        "n_distinct_clusters":  pl.Int32,
        "n_domains":            pl.Int32,
        "clusters_ordered":     pl.Utf8,
        "domain_details":       pl.Utf8,
    })


_SEED_SUMMARY_SCHEMA = {
    "IDR_ID":               pl.Utf8,
    "n_distinct_clusters":  pl.Int32,
    "n_domains":            pl.Int32,
    "clusters_ordered":     pl.Utf8,
    "domain_details":       pl.Utf8,
}


def load_centroids_matrix(centroids_csv: str) -> "tuple[np.ndarray, list]":
    """Load just the centroid feature matrix (no compositional reference
    needed) for the seed_expand block method's smart-absorption test. The
    first column is assumed to be a cluster id/label; the rest are the
    feature columns, in the same order/names as the per-window feature
    columns already present in an out_windows.csv (as written by
    compute_windows / --stage windows) -- matching is done by name, not
    position, so column order doesn't need to match exactly.
    """
    cent_df   = pd.read_csv(centroids_csv)
    feat_cols = cent_df.columns.tolist()[1:]
    centmat   = cent_df[feat_cols].to_numpy().astype(float)
    return centmat, feat_cols


def build_summary_seed_expand(win_df: pl.DataFrame, cfg: dict,
                               centmat: np.ndarray, feat_cols: list
                               ) -> pl.DataFrame:
    """
    Alternative block-building strategy ("seed & expand"): rather than
    keeping every window and resolving overlaps after the fact, start from
    only the statistically *confident* windows (low min_dist, high
    inter_cluster_dist -- close to their centroid and not a close call
    against the runner-up) and grow those seeds outward window-by-window
    until growth is no longer justified. Overlap is prevented by
    construction -- expansion never crosses into territory a
    higher-priority seed has already claimed -- rather than resolved
    afterward with an arithmetic-mean boundary.

    Steps
    -----
      1. Seed definition (global, quantile-based): a window is a seed if
         min_dist <= Q(seed_min_dist_quantile) AND inter_cluster_dist >=
         Q(seed_inter_dist_quantile), with quantiles taken over every valid
         window in the whole input -- not per-IDR -- so "confident" is a
         property of the overall distance distribution, not of location.
      2. Per IDR: group consecutive (by window order) same-cluster seed
         windows into seed runs, and rank runs by mean valley-quality
         (min_dist / inter_cluster_dist, ascending = most confident first).
      3. Claim: each run's own windows are assigned to it. Distinct seed
         runs are disjoint by construction, so this step cannot conflict.
      4. Expand: process runs most-confident-first; grow each run's window
         range left/right one window at a time. A candidate window is
         absorbed ("smart absorption") if its squared distance to *this
         run's* assigned centroid is at most `seed_absorption_factor` times
         its own min_dist -- i.e. it fits the neighboring block almost as
         well as it fits its own best centroid. Expansion in a direction
         stops when that test fails, an already-claimed window is hit
         (guaranteeing no overlap), or a sequence boundary is reached.
      5. Compile blocks from the final window -> run assignment, resolve
         residue overlaps and merge same-cluster neighbours as
         build_summary does, then keep blocks of at least min_block_length
         residues. min_windows is not applied: short seeded blocks that
         merge with a same-cluster neighbour are kept whole.

    Parameters
    ----------
    win_df : polars DataFrame
        Must contain IDR_ID, start, end, cluster, min_dist,
        inter_cluster_dist, plus every column in `feat_cols` (the per-window
        90-dim feature vector) -- unlike build_summary, this strategy needs
        the full feature vector to run the smart-absorption test.
    cfg : dict
        Must contain: window_size, min_block_length, min_distinct_clusters,
        seed_min_dist_quantile, seed_inter_dist_quantile,
        seed_absorption_factor.
    centmat : np.ndarray, shape (n_clusters, n_features)
        The centroid matrix, row i = centroid for cluster i (same indexing
        as the `cluster` column).
    feat_cols : list of str
        Names of the feature columns in win_df, in the same row order as
        centmat's columns.
    """
    window_size           = cfg["window_size"]
    min_block_length      = cfg["min_block_length"]
    min_distinct_clusters = cfg["min_distinct_clusters"]
    q_min_dist            = cfg["seed_min_dist_quantile"]
    q_inter_dist          = cfg["seed_inter_dist_quantile"]
    absorption_factor     = cfg["seed_absorption_factor"]

    core_plus_feat = (["IDR_ID", "start", "end", "cluster",
                       "min_dist", "inter_cluster_dist"] + feat_cols)
    valid = win_df.select(core_plus_feat).filter(pl.col("cluster").is_not_null())
    if valid.is_empty():
        return pl.DataFrame(schema=_SEED_SUMMARY_SCHEMA)

    min_dist_all   = valid["min_dist"].to_numpy().astype(float)
    inter_dist_all = valid["inter_cluster_dist"].to_numpy().astype(float)
    thr_min_dist   = float(np.nanquantile(min_dist_all, q_min_dist))
    thr_inter_dist = float(np.nanquantile(inter_dist_all, q_inter_dist))
    # Absolute thresholds override the quantiles. Use them when scoring a
    # small input against a seed definition taken from a larger reference
    # run (e.g. the human proteome), where quantiles of the small input
    # would give different cut-offs.
    if cfg.get("seed_min_dist_threshold") is not None:
        thr_min_dist = float(cfg["seed_min_dist_threshold"])
    if cfg.get("seed_inter_dist_threshold") is not None:
        thr_inter_dist = float(cfg["seed_inter_dist_threshold"])
    log.info("Seed thresholds (seed_expand): min_dist <= %.4f (q=%.2f), "
             "inter_cluster_dist >= %.4f (q=%.2f), absorption_factor=%.2f",
             thr_min_dist, q_min_dist, thr_inter_dist, q_inter_dist,
             absorption_factor)

    ids = valid["IDR_ID"].unique(maintain_order=True).to_list()
    log.info("Reprocessing %d unique IDRs (seed_expand) …", len(ids))

    summary_rows = []

    for idr_id in ids:
        sub = valid.filter(pl.col("IDR_ID") == idr_id).sort("start")
        if sub.is_empty():
            continue

        n           = sub.height
        starts      = sub["start"].to_numpy().astype(int)
        ends        = sub["end"].to_numpy().astype(int)
        clusters    = sub["cluster"].to_numpy().astype(int)
        min_dists   = sub["min_dist"].to_numpy().astype(float)
        inter_dists = sub["inter_cluster_dist"].to_numpy().astype(float)
        feats       = sub.select(feat_cols).to_numpy().astype(float)

        is_seed = (min_dists <= thr_min_dist) & (inter_dists >= thr_inter_dist)

        # -----------------------------------------------------------------
        # Seed-run identification: maximal runs of consecutive, same-
        # cluster seed windows. A non-seed window (even same cluster) or a
        # cluster change breaks a run.
        # -----------------------------------------------------------------
        runs = []
        i = 0
        while i < n:
            if not is_seed[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and is_seed[j + 1] and clusters[j + 1] == clusters[i]:
                j += 1
            qualities = [_valley_quality(min_dists[k], inter_dists[k])
                         for k in range(i, j + 1)]
            runs.append(dict(cluster=int(clusters[i]), i0=i, i1=j,
                              quality=float(np.mean(qualities))))
            i = j + 1

        if not runs:
            continue

        # Most confident (lowest quality score) first.
        runs.sort(key=lambda r: r["quality"])

        claimed = np.full(n, -1, dtype=int)
        for rid, r in enumerate(runs):
            claimed[r["i0"]:r["i1"] + 1] = rid

        # -----------------------------------------------------------------
        # Priority-ordered greedy expansion. Each run expands to
        # completion (both directions) before the next, lower-priority
        # run gets a turn, so higher-confidence seeds always win contested
        # territory; expansion into an already-claimed window is simply
        # blocked, which is what prevents any overlap.
        # -----------------------------------------------------------------
        for rid, r in enumerate(runs):
            centroid = centmat[r["cluster"]]

            lo = r["i0"]
            while lo - 1 >= 0 and claimed[lo - 1] == -1:
                cand = lo - 1
                dist_to_centroid = float(np.sum((feats[cand] - centroid) ** 2))
                if dist_to_centroid <= min_dists[cand] * absorption_factor:
                    claimed[cand] = rid
                    lo = cand
                else:
                    break

            hi = r["i1"]
            while hi + 1 < n and claimed[hi + 1] == -1:
                cand = hi + 1
                dist_to_centroid = float(np.sum((feats[cand] - centroid) ** 2))
                if dist_to_centroid <= min_dists[cand] * absorption_factor:
                    claimed[cand] = rid
                    hi = cand
                else:
                    break

            r["i0"], r["i1"] = lo, hi

        # -----------------------------------------------------------------
        # Compile raw blocks straight from the final claim assignment --
        # membership per run is contiguous by construction (expansion only
        # ever extends the range by one window at a time), so no
        # window-INDEX overlap is possible here. Adjacent blocks' residue
        # RANGES can still overlap, though: sliding windows overlap by
        # construction (step_size < window_size), so the last window
        # claimed by one run and the first window claimed by its neighbor
        # already share residues even though no window index is claimed
        # twice. _resolve_overlaps_and_merge (below) is what removes that
        # residue-coordinate overlap.
        # -----------------------------------------------------------------
        raw_blocks = []
        for rid, r in enumerate(runs):
            members = np.where(claimed == rid)[0]
            if len(members) == 0:
                continue
            m0, m1 = int(members.min()), int(members.max())
            block_min_dists   = min_dists[m0:m1 + 1]
            block_inter_dists = inter_dists[m0:m1 + 1]
            best_local  = int(np.argmin(block_min_dists))
            valley_idx  = m0 + best_local
            raw_blocks.append(dict(
                cluster      = r["cluster"],
                block_start  = int(starts[m0:m1 + 1].min()),
                block_end    = int(ends[m0:m1 + 1].max()),
                block_length = int(ends[m0:m1 + 1].max() - starts[m0:m1 + 1].min()),
                valley_dist  = float(min_dists[valley_idx]),
                valley_score = _valley_quality(min_dists[valley_idx],
                                               inter_dists[valley_idx]),
                valley_start = int(starts[valley_idx]),
                valley_end   = int(ends[valley_idx]),
                num_windows  = int(m1 - m0 + 1),
            ))

        # No min_windows filter here: a block grown from a confident seed is
        # kept whatever its window count, and only the residue-length filter
        # below applies. This is the rule of the published run
        # (block_summary_seedExpantion.20260909.csv).
        candidates = raw_blocks
        if not candidates:
            continue

        # Overlap resolution (arithmetic-mean boundary) + merge consecutive
        # same-cluster blocks -- see _resolve_overlaps_and_merge. Needed
        # even though claimed[] already guarantees disjoint window-index
        # sets, because overlapping sliding windows still give adjacent
        # blocks overlapping residue ranges (see comment above).
        domains = _resolve_overlaps_and_merge(candidates, window_size)

        if not domains:
            continue

        domains = [d for d in domains if d["block_length"] >= min_block_length]
        if not domains:
            continue

        distinct_clusters = set(d["cluster"] for d in domains)
        if len(distinct_clusters) < min_distinct_clusters:
            continue

        summary_rows.append(dict(
            IDR_ID              = idr_id,
            n_distinct_clusters = len(distinct_clusters),
            n_domains           = len(domains),
            clusters_ordered    = str([d["cluster"] for d in domains]),
            domain_details      = str(domains),
        ))

    log.info("Multi-cluster IDRs found (seed_expand): %d / %d",
             len(summary_rows), len(ids))

    if summary_rows:
        return pl.from_dicts(summary_rows)

    return pl.DataFrame(schema=_SEED_SUMMARY_SCHEMA)


def compute_windows(df_pl: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Stage 1: build sliding windows, compute the 90-dim feature vector for
    each one (with the corrected _omega/_kappa kernel), and assign each
    window to its nearest centroid. This is the expensive, multiprocessing
    + numba step. Returns the raw per-window results (no metadata joined
    in, no summary/block-building — that's compute_blocks below).
    """
    for col in ("IDR_ID", "Sequence"):
        if col not in df_pl.columns:
            raise ValueError(f"Input DataFrame must contain column '{col}'.")

    ids  = df_pl["IDR_ID"].to_list()
    seqs = [s.upper() for s in df_pl["Sequence"].to_list()]

    mean_arr, std_arr, centmat, feat_names = load_references(cfg)

    log.info("Building windows (size=%d, step=%d) for %d sequences …",
             cfg["window_size"], cfg["step_size"], len(ids))

    all_windows = []
    for idr_id, seq in zip(ids, seqs):
        all_windows.extend(
            build_windows(idr_id, seq,
                          cfg["window_size"],
                          cfg["step_size"],
                          cfg["min_window_size"],
                          feat_names)
        )
    log.info("Total windows to process: %d", len(all_windows))

    _requested = cfg["n_workers"]
    if _requested is None:
        n_workers = min(os.cpu_count() or 1, _SAFE_WORKER_CAP)
    else:
        n_workers = min(int(_requested), _SAFE_WORKER_CAP)
    log.info("Worker processes: %d (cap=%d).", n_workers, _SAFE_WORKER_CAP)
    t0 = time.time()

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(mean_arr, std_arr, centmat, cfg["num_scrambles"])
    ) as pool:
        records = list(
            tqdm(pool.imap(_process_window, all_windows, chunksize=32),
                 total=len(all_windows),
                 desc="Windows")
        )

    elapsed = time.time() - t0
    log.info("Done in %.1f s (%.2f windows/s).", elapsed,
             len(all_windows) / max(elapsed, 1e-6))

    # infer_schema_length=None forces a full scan of `records` when inferring
    # each column's dtype. Without this, Polars only samples the first 100
    # rows; since `_process_window` emits `cluster` as a Python int for valid
    # windows but as `np.nan` (a float) for invalid ones, an early run of
    # all-valid windows locks in an integer builder that then errors the
    # first time it hits a NaN further down the list.
    win_df = pl.from_dicts(records, infer_schema_length=None)
    win_df = win_df.with_columns([
        # strict=False: invalid windows carry cluster=NaN (float) rather than
        # an int; a strict cast to Int32 raises on NaN, so cast leniently and
        # let it become null instead — the downstream filter on
        # `cluster.is_not_null()` already relies on exactly that.
        pl.col("cluster").cast(pl.Int32, strict=False),
        pl.col("start").cast(pl.Int32),
        pl.col("end").cast(pl.Int32),
    ])

    return win_df


def compute_blocks(win_df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Stage 2: reprocess per-window cluster assignments into the domain
    summary. Cheap — no sequence data, no numba/multiprocessing.

    Two selectable strategies (cfg["block_method"]):
      - "meanboundary" (default): the original denoise -> raw blocks ->
        eligibility filter -> overlap resolution (arithmetic-mean boundary)
        -> merge -> compile pipeline. Only needs core_cols below.
      - "seed_expand": define statistically confident seed windows and grow
        them outward with a smart-absorption test, so overlap is prevented
        by construction. Needs the full per-window feature columns (as
        written by --stage windows) plus a centroids CSV (cfg["centroids_csv"]).
    """
    core_cols = ["IDR_ID", "start", "end", "cluster", "min_dist", "inter_cluster_dist"]
    missing = [c for c in core_cols if c not in win_df.columns]
    if missing:
        raise ValueError(
            f"Windows input is missing required column(s) {missing}; "
            f"expected at least {core_cols}."
        )

    method = cfg.get("block_method", "meanboundary")

    if method == "seed_expand":
        centroids_csv = cfg.get("centroids_csv")
        if not centroids_csv:
            raise ValueError(
                "block_method='seed_expand' requires a centroids CSV "
                "(--centroids) to run the smart-absorption distance test."
            )
        centmat, feat_cols = load_centroids_matrix(centroids_csv)
        missing_feat = [c for c in feat_cols if c not in win_df.columns]
        if missing_feat:
            raise ValueError(
                f"block_method='seed_expand' needs the per-window 90-dim "
                f"feature columns in the windows input ({len(missing_feat)} "
                f"missing, e.g. {missing_feat[:5]}); make sure --in_windows "
                f"points at an out_windows.csv produced by --stage windows "
                f"(not a file stripped down to only the core columns)."
            )
        return build_summary_seed_expand(win_df, cfg, centmat, feat_cols)

    win_core = win_df.select(core_cols).filter(pl.col("cluster").is_not_null())
    return build_summary(win_core, cfg)


def run_sliding_window(df_pl: pl.DataFrame, cfg: dict,
                        metadata_df: "pl.DataFrame | None" = None
                        ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convenience wrapper that runs both stages back to back (the original,
    single-shot behavior) and writes both output files.
    """
    win_df = compute_windows(df_pl, cfg)
    summary_df = compute_blocks(win_df, cfg)

    if metadata_df is not None:
        win_df = metadata_df.join(win_df, on="IDR_ID", how="inner")
        if len(summary_df) > 0:
            summary_df = metadata_df.join(summary_df, on="IDR_ID", how="inner")

    win_df.write_csv(cfg["output_windows_csv"])
    log.info("Window results → %s", cfg["output_windows_csv"])

    summary_df.write_csv(cfg["output_summary_csv"])
    log.info("Summary        → %s", cfg["output_summary_csv"])

    return win_df, summary_df

# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: LOAD INPUT FROM CSV / DIRECT CALL
# ══════════════════════════════════════════════════════════════════════════════

def load_input_csv(path: str) -> "tuple[pl.DataFrame, pl.DataFrame | None]":
    """Load the input CSV and split it into the (IDR_ID, Sequence) pair the
    pipeline needs plus, if the file carries any other columns (UniprotID,
    Gene name, Description, IDR_START/END, ...), a metadata frame keyed on
    IDR_ID that gets joined back onto every output later. Accepts either
    'Sequence' or 'IDR_Sequence' as the sequence column name.
    """
    df = pl.read_csv(path)
    if "IDR_ID" not in df.columns:
        raise ValueError("Input CSV must contain an 'IDR_ID' column.")

    seq_col = next((c for c in ("Sequence", "IDR_Sequence") if c in df.columns), None)
    if seq_col is None:
        raise ValueError("Input CSV must contain a 'Sequence' or 'IDR_Sequence' column.")
    if seq_col != "Sequence":
        df = df.rename({seq_col: "Sequence"})

    extra_cols = [c for c in df.columns if c not in ("IDR_ID", "Sequence")]
    metadata_df = df.select(["IDR_ID"] + extra_cols) if extra_cols else None
    if metadata_df is not None:
        log.info("Carrying %d extra input column(s) through to the outputs: %s",
                  len(extra_cols), extra_cols)

    return df.select(["IDR_ID", "Sequence"]), metadata_df

# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="tesseraidr multiblock",
        description="NARDINI Sliding-Window Cluster Assignment"
    )
    parser.add_argument(
        "--stage", choices=["windows", "blocks", "all"], default="all",
        help="Which part to run. 'windows': build windows, compute features, "
             "assign clusters, and write --out_windows only (no block-building). "
             "'blocks': read an existing windows CSV from --in_windows and run "
             "only the block-building step, writing --out_summary. 'all' "
             "(default): both stages in one run, as before."
    )
    parser.add_argument(
        "--in_windows", default=None,
        help="Path to a previously-written out_windows.csv to reprocess into "
             "blocks. Required when --stage blocks; ignored otherwise. Never "
             "inferred automatically — always pass it explicitly."
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to a CSV with 'IDR_ID' and 'Sequence' columns (optionally "
             "with extra metadata columns). Required when --stage is 'windows' "
             "or 'all'. Optional when --stage is 'blocks': if given, its extra "
             "columns are still joined onto --out_summary by IDR_ID; if "
             "omitted, --out_summary carries no metadata columns."
    )
    parser.add_argument(
        "--comp_ref",
        default=CONFIG["comp_reference_csv"],
        help="Path to the compositional reference CSV (Mean/Std per feature)."
    )
    parser.add_argument(
        "--centroids",
        default=CONFIG["centroids_csv"],
        help="Path to the GIN centroids CSV."
    )
    parser.add_argument(
        "--window_size",  type=int, default=CONFIG["window_size"]
    )
    parser.add_argument(
        "--step_size",    type=int, default=CONFIG["step_size"]
    )
    parser.add_argument(
        "--num_scrambles", type=int, default=CONFIG["num_scrambles"]
    )
    parser.add_argument(
        "--n_workers", type=int, default=CONFIG["n_workers"]
    )
    parser.add_argument(
        "--out_windows",
        default=CONFIG["output_windows_csv"]
    )
    parser.add_argument(
        "--out_summary",
        default=CONFIG["output_summary_csv"]
    )
    parser.add_argument(
        "--min_windows", type=int, default=CONFIG["min_windows"],
        help="meanboundary only: minimum number of windows a block must "
             "span to be considered (default: 3). Applied immediately after "
             "denoising, before overlap resolution. seed_expand ignores it."
    )
    parser.add_argument(
        "--min_block_length", type=int, default=CONFIG["min_block_length"],
        help="Minimum residue length (block_end - block_start) a domain must have "
             "to appear in the final output (default: 70).  Applied after all "
             "merging is complete."
    )
    parser.add_argument(
        "--block_method", choices=["meanboundary", "seed_expand"],
        default=CONFIG["block_method"],
        help="Block-building strategy for the blocks stage (default: "
             "seed_expand, as in the published run). 'meanboundary': the original denoise -> raw "
             "blocks -> overlap resolution (arithmetic-mean boundary) -> "
             "merge pipeline. 'seed_expand': define statistically confident "
             "'seed' windows (quantile thresholds on min_dist / "
             "inter_cluster_dist) and grow them outward window-by-window "
             "using a smart-absorption test against each seed's centroid, "
             "most-confident-seed-first -- overlap is prevented by "
             "construction instead of resolved afterward. Requires the "
             "windows input to carry the full per-window feature columns "
             "and --centroids to be supplied."
    )
    parser.add_argument(
        "--seed_min_dist_quantile", type=float,
        default=CONFIG["seed_min_dist_quantile"],
        help="seed_expand only: a window qualifies as a seed only if its "
             "min_dist is at or below this quantile of min_dist across all "
             "windows in the input (default: 0.60)."
    )
    parser.add_argument(
        "--seed_inter_dist_quantile", type=float,
        default=CONFIG["seed_inter_dist_quantile"],
        help="seed_expand only: a window qualifies as a seed only if its "
             "inter_cluster_dist is at or above this quantile of "
             "inter_cluster_dist across all windows in the input "
             "(default: 0.75)."
    )
    parser.add_argument(
        "--seed_absorption_factor", type=float,
        default=CONFIG["seed_absorption_factor"],
        help="seed_expand only: a boundary window next to a seed block is "
             "absorbed into it if the window's squared distance to that "
             "block's centroid is at most this factor times the window's "
             "own min_dist (default: 1.5); otherwise expansion stops there."
    )
    parser.add_argument(
        "--seed_min_dist_threshold", type=float, default=None,
        help="seed_expand only: absolute min_dist cut-off for seeds. "
             "Overrides --seed_min_dist_quantile; use it to apply a "
             "reference run's cut-off (data/reference/seed_thresholds_human.csv) "
             "to a small input."
    )
    parser.add_argument(
        "--seed_inter_dist_threshold", type=float, default=None,
        help="seed_expand only: absolute inter_cluster_dist cut-off for "
             "seeds. Overrides --seed_inter_dist_quantile."
    )
    return parser


def main(argv=None, prog=None):
    parser = build_parser()
    if prog:
        parser.prog = prog
    args = parser.parse_args(argv)

    if args.stage in ("windows", "all") and not args.input:
        parser.error("--input is required when --stage is 'windows' or 'all'.")
    if args.stage == "blocks" and not args.in_windows:
        parser.error("--in_windows is required when --stage is 'blocks'.")

    cfg = CONFIG.copy()
    cfg.update(dict(
        comp_reference_csv       = args.comp_ref,
        centroids_csv            = args.centroids,
        window_size              = args.window_size,
        step_size                = args.step_size,
        num_scrambles            = args.num_scrambles,
        n_workers                = args.n_workers,
        output_windows_csv       = args.out_windows,
        output_summary_csv       = args.out_summary,
        min_windows              = args.min_windows,
        min_block_length         = args.min_block_length,
        block_method             = args.block_method,
        seed_min_dist_quantile   = args.seed_min_dist_quantile,
        seed_inter_dist_quantile = args.seed_inter_dist_quantile,
        seed_absorption_factor   = args.seed_absorption_factor,
        seed_min_dist_threshold  = args.seed_min_dist_threshold,
        seed_inter_dist_threshold = args.seed_inter_dist_threshold,
    ))

    if args.stage == "windows":
        df, metadata_df = load_input_csv(args.input)
        log.info("Loaded %d sequences from %s", len(df), args.input)

        win_df = compute_windows(df, cfg)
        if metadata_df is not None:
            win_df = metadata_df.join(win_df, on="IDR_ID", how="inner")

        win_df.write_csv(cfg["output_windows_csv"])
        log.info("Window results → %s", cfg["output_windows_csv"])

        print("\n── Window results (first 5 rows) ───────────────────────────────")
        print(win_df.select(["IDR_ID", "start", "end",
                              "cluster", "min_dist",
                              "inter_cluster_dist"]).head(5))

    elif args.stage == "blocks":
        win_df = pl.read_csv(args.in_windows)
        log.info("Loaded %d window rows from %s", len(win_df), args.in_windows)

        metadata_df = None
        if args.input:
            _, metadata_df = load_input_csv(args.input)

        summary_df = compute_blocks(win_df, cfg)
        if metadata_df is not None and len(summary_df) > 0:
            summary_df = metadata_df.join(summary_df, on="IDR_ID", how="inner")

        summary_df.write_csv(cfg["output_summary_csv"])
        log.info("Summary        → %s", cfg["output_summary_csv"])

        print("\n── Multi-cluster IDRs ───────────────────────────────────────────")
        print(summary_df.head(10))

    else:  # "all"
        df, metadata_df = load_input_csv(args.input)
        log.info("Loaded %d sequences from %s", len(df), args.input)

        window_df, summary_df = run_sliding_window(df, cfg, metadata_df=metadata_df)

        print("\n── Window results (first 5 rows) ───────────────────────────────")
        print(window_df.select(["IDR_ID", "start", "end",
                                 "cluster", "min_dist",
                                 "inter_cluster_dist"]).head(5))

        print("\n── Multi-cluster IDRs ───────────────────────────────────────────")
        print(summary_df.head(10))


if __name__ == "__main__":
    main()
