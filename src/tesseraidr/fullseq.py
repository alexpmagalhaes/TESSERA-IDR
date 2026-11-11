#!/usr/bin/env python
# coding: utf-8
"""
NARDINI Full-Sequence Cluster Assignment
=========================================
Applies the NARDINI+ feature pipeline on the full length of each IDR
sequence and assigns it to the nearest pre-calculated GIN centroid.

Attribution
-----------
This module is a reimplementation (rewrite) of the NARDINI+ feature
pipeline and GIN cluster assignment of Ruff, King et al.; it is not the
original code. The algorithm, the 54 compositional + 36 patterning features,
the human-IDRome feature normalisation and the 30 GIN centroids come from:

  Ruff KM, King MR, Ying AW, Liu V, Pant A, Lieberman WE, Shinn MK, Su X,
  Kadoch C, Pappu RV. Molecular grammars of predicted intrinsically
  disordered regions that span the human proteome. Cell 189(1):323-342.e17
  (2026). doi:10.1016/j.cell.2025.10.019
  Original code (MIT License, Copyright (c) 2025 kierstenruff):
  https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-

The patterning z-scores follow NARDINI:

  Cohan MC, Shinn MK, Lalmansingh JM, Pappu RV. Uncovering non-random binary
  patterns within sequences of intrinsically disordered proteins. J Mol Biol
  434(2):167373 (2022). doi:10.1016/j.jmb.2021.167373

Please cite these papers when using this module.

v 0.96 by Alexandre P. Magalhaes
  - Fixed an off-by-one in _omega/_kappa's sliding-window loop (see
    https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-/issues/2):
    a sequence of length L has L-b+1 complete size-b windows, not L-b+2; the
    extra iteration's slice ran past the array end, numpy silently truncated
    it, and the code still divided by the hardcoded b. Also switched both
    functions to an O(1)-per-step rolling sum now that the window count is
    correct (~1.7x / ~2.5x faster for _omega / _kappa respectively).
  - --input may now be a metadata-rich CSV (e.g. UniprotID, Gene name,
    Description, IDR_START/END, ...) instead of a bare IDR_ID/Sequence
    table. Any columns beyond IDR_ID and the sequence column are detected
    automatically and carried through onto every output row, so results
    don't need a separate manual join back onto the input metadata
    afterward. The sequence column may be named 'Sequence' or
    'IDR_Sequence'.
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
    num_scrambles        = 100,
    n_workers            = None,
    output_sequences_csv = "nardini_sequence_results.csv",
    output_summary_csv   = "nardini_multi_cluster_summary.csv",
    output_dist_matrix_csv = None,   # set a filename to enable, e.g. "nardini_dist_matrix.csv"
    min_distinct_clusters = 1,
    zscore_clip           = 3.0,
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
                           inter_cluster_dist=np.nan,
                           _all_dists=np.full(_CENTMAT.shape[0], np.nan).tolist()))
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
                       inter_cluster_dist=inter_dist,
                       _all_dists=dists.tolist()))
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

def run_full_sequence(df_pl: pl.DataFrame, cfg: dict,
                       metadata_df: "pl.DataFrame | None" = None
                       ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    for col in ("IDR_ID", "Sequence"):
        if col not in df_pl.columns:
            raise ValueError(f"Input DataFrame must contain column '{col}'.")

    ids  = df_pl["IDR_ID"].to_list()
    seqs = [s.upper() for s in df_pl["Sequence"].to_list()]

    mean_arr, std_arr, centmat, feat_names = load_references(cfg)

    log.info("Processing %d full-length sequences …", len(ids))

    # One item per sequence: start=0, end=len(seq) covers the full sequence.
    all_sequences = [
        (idr_id, 0, len(seq), seq, feat_names)
        for idr_id, seq in zip(ids, seqs)
    ]
    log.info("Total sequences to process: %d", len(all_sequences))

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
            tqdm(pool.imap(_process_window, all_sequences, chunksize=32),
                 total=len(all_sequences),
                 desc="Sequences")
        )

    elapsed = time.time() - t0
    log.info("Done in %.1f s (%.2f sequences/s).", elapsed,
             len(all_sequences) / max(elapsed, 1e-6))

    # ── Distance matrix ───────────────────────────────────────────────────────
    # Extract per-cluster distances collected by the worker and build a tidy
    # (n_sequences × n_clusters) matrix.  The _all_dists key is removed from
    # records so it does not pollute seq_df.
    n_clusters   = centmat.shape[0]
    cluster_cols = [f"cluster_{i}" for i in range(n_clusters)]
    dist_rows    = []
    for rec in records:
        all_dists = rec.pop("_all_dists")   # remove in-place; list of floats
        dist_rows.append(
            {"IDR_ID": rec["IDR_ID"], **dict(zip(cluster_cols, all_dists))}
        )
    dist_matrix_df = pl.from_dicts(dist_rows)

    if metadata_df is not None:
        dist_matrix_df = metadata_df.join(dist_matrix_df, on="IDR_ID", how="inner")

    if cfg.get("output_dist_matrix_csv"):
        dist_matrix_df.write_csv(cfg["output_dist_matrix_csv"])
        log.info("Distance matrix  → %s  (%d sequences × %d clusters)",
                 cfg["output_dist_matrix_csv"], len(dist_rows), n_clusters)
    # ─────────────────────────────────────────────────────────────────────────

    seq_df = pl.from_dicts(records)
    seq_df = seq_df.with_columns([
        pl.col("cluster").cast(pl.Int32),
        pl.col("start").cast(pl.Int32),
        pl.col("end").cast(pl.Int32),
    ])

    summary_rows = []
    core_cols  = ["IDR_ID", "start", "end", "cluster", "min_dist"]
    seq_core   = seq_df.select(core_cols).filter(pl.col("cluster").is_not_null())

    for idr_id in ids:
        sub = seq_core.filter(pl.col("IDR_ID") == idr_id).sort("start")
        if sub.is_empty():
            continue

        # Convert to a list of dicts for sequential processing
        windows = sub.to_dicts()

        domains = []
        current_cluster = windows[0]["cluster"]
        run_starts = [windows[0]["start"]]
        run_ends   = [windows[0]["end"]]
        run_dists  = [windows[0]["min_dist"]]

        # Helper function to close out a block when the cluster changes
        def commit_domain():
            # The "valley" is the local minimum distance within this contiguous block
            best_idx = int(np.nanargmin(run_dists))
            domains.append(dict(
                cluster      = current_cluster,
                block_start  = min(run_starts),
                block_end    = max(run_ends),
                block_length = max(run_ends) - min(run_starts),
                valley_dist  = run_dists[best_idx],
                valley_start = run_starts[best_idx],
                valley_end   = run_ends[best_idx],
                num_windows  = len(run_starts)
            ))

        # Iterate through windows to find contiguous runs
        for w in windows[1:]:
            if w["cluster"] == current_cluster:
                # Extend the current block
                run_starts.append(w["start"])
                run_ends.append(w["end"])
                run_dists.append(w["min_dist"])
            else:
                # Cluster changed -> commit the previous block and start a new one
                commit_domain()
                current_cluster = w["cluster"]
                run_starts = [w["start"]]
                run_ends   = [w["end"]]
                run_dists  = [w["min_dist"]]

        # Commit the final block at the end of the sequence
        commit_domain()

        # ---------------------------------------------------------------------
        # 1. Non-Maximum Suppression (NMS) to remove overlapping valleys
        # ---------------------------------------------------------------------
        # Sort domains by distance so we process the strongest (lowest dist) valleys first
        domains_sorted = sorted(domains, key=lambda d: d["valley_dist"])
        filtered_domains = []

        for d in domains_sorted:
            overlap = False
            for fd in filtered_domains:
                # Two valleys overlap if their start/end intervals intersect
                if max(d["valley_start"], fd["valley_start"]) < min(d["valley_end"], fd["valley_end"]):
                    overlap = True
                    break

            if not overlap:
                filtered_domains.append(d)

        # Restore the linear sequence order for the remaining distinct domains
        domains = sorted(filtered_domains, key=lambda d: d["block_start"])

        # ---------------------------------------------------------------------
        # 2. Adjust block boundaries so the full domains don't overlap
        # ---------------------------------------------------------------------
        for i in range(len(domains) - 1):
            curr_d = domains[i]
            next_d = domains[i + 1]

            if curr_d["block_end"] > next_d["block_start"]:
                overlap_mid = (curr_d["block_end"] + next_d["block_start"]) // 2
                boundary = max(curr_d["valley_end"], min(overlap_mid, next_d["valley_start"]))

                curr_d["block_end"] = boundary
                next_d["block_start"] = boundary

                curr_d["block_length"] = curr_d["block_end"] - curr_d["block_start"]
                next_d["block_length"] = next_d["block_end"] - next_d["block_start"]

        # ---------------------------------------------------------------------
        # 3. Merge adjacent domains of the SAME cluster (Post-filtering cleanup)
        # ---------------------------------------------------------------------
        merged_domains = []
        if domains:
            current = domains[0]
            for next_d in domains[1:]:
                if current["cluster"] == next_d["cluster"]:
                    # Expand the block boundaries to encompass both
                    current["block_start"] = min(current["block_start"], next_d["block_start"])
                    current["block_end"] = max(current["block_end"], next_d["block_end"])
                    current["block_length"] = current["block_end"] - current["block_start"]
                    current["num_windows"] += next_d.get("num_windows", 0)

                    # Retain the deepest/strongest valley of the two
                    if next_d["valley_dist"] < current["valley_dist"]:
                        current["valley_dist"] = next_d["valley_dist"]
                        current["valley_start"] = next_d["valley_start"]
                        current["valley_end"] = next_d["valley_end"]
                else:
                    merged_domains.append(current)
                    current = next_d
            merged_domains.append(current)

        domains = merged_domains

        # ---------------------------------------------------------------------
        # 4. Final Summary Compilation
        # ---------------------------------------------------------------------
        distinct_clusters = set(d["cluster"] for d in domains)
        if len(distinct_clusters) < cfg["min_distinct_clusters"]:
            continue

        # Extract the sequence of clusters exactly as they appear linearly
        clusters_ordered = [d["cluster"] for d in domains]

        # Append the finalized, non-overlapping domains to the summary
        summary_rows.append(dict(
            IDR_ID              = idr_id,
            n_distinct_clusters = len(distinct_clusters),
            n_domains           = len(domains),
            clusters_ordered    = str(clusters_ordered),
            domain_details      = str(domains),
        ))

    # ---------------------------------------------------------------------
    # OUTSIDE THE LOOP: Convert to DataFrame and save
    # (Make sure this aligns with `for idr_id in ids:`)
    # ---------------------------------------------------------------------
    if summary_rows:
        summary_df = pl.from_dicts(summary_rows)
    else:
        summary_df = pl.DataFrame(schema={
            "IDR_ID":               pl.Utf8,
            "n_distinct_clusters":  pl.Int32,
            "n_domains":            pl.Int32,
            "clusters_ordered":     pl.Utf8,
            "domain_details":       pl.Utf8,
        })

    log.info("Sequences processed: %d / %d", len(summary_rows), len(ids))

    if metadata_df is not None:
        seq_df = metadata_df.join(seq_df, on="IDR_ID", how="inner")
        if len(summary_df) > 0:
            summary_df = metadata_df.join(summary_df, on="IDR_ID", how="inner")

    seq_df.write_csv(cfg["output_sequences_csv"])
    log.info("Sequence results → %s", cfg["output_sequences_csv"])

    summary_df.write_csv(cfg["output_summary_csv"])
    log.info("Summary          → %s", cfg["output_summary_csv"])

    return seq_df, summary_df, dist_matrix_df

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
        prog="tesseraidr fullseq",
        description="NARDINI Full-Sequence Cluster Assignment. A reimplementation "
                    "of NARDINI+ / GIN from Ruff, King et al., Cell 2026 "
                    "(doi:10.1016/j.cell.2025.10.019); please cite it."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a CSV with 'IDR_ID' and 'Sequence' columns."
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
        "--num_scrambles", type=int, default=CONFIG["num_scrambles"]
    )
    parser.add_argument(
        "--n_workers", type=int, default=CONFIG["n_workers"]
    )
    parser.add_argument(
        "--out_sequences",
        default=CONFIG["output_sequences_csv"]
    )
    parser.add_argument(
        "--out_summary",
        default=CONFIG["output_summary_csv"]
    )
    parser.add_argument(
        "--out_dist_matrix",
        default=CONFIG["output_dist_matrix_csv"],
        help="Optional path for the (sequences × clusters) distance matrix CSV. "
             "Omit or pass 'none' to skip."
    )
    return parser


def main(argv=None, prog=None):
    parser = build_parser()
    if prog:
        parser.prog = prog
    args = parser.parse_args(argv)

    cfg = CONFIG.copy()
    cfg.update(dict(
        comp_reference_csv   = args.comp_ref,
        centroids_csv        = args.centroids,
        num_scrambles        = args.num_scrambles,
        n_workers            = args.n_workers,
        output_sequences_csv = args.out_sequences,
        output_summary_csv   = args.out_summary,
        output_dist_matrix_csv = (
            None if args.out_dist_matrix in (None, "none", "None")
            else args.out_dist_matrix
        ),
    ))

    df, metadata_df = load_input_csv(args.input)
    log.info("Loaded %d sequences from %s", len(df), args.input)

    seq_df, summary_df, dist_matrix_df = run_full_sequence(df, cfg, metadata_df=metadata_df)

    print("\n── Sequence results (first 5 rows) ─────────────────────────────")
    print(seq_df.select(["IDR_ID", "start", "end",
                          "cluster", "min_dist",
                          "inter_cluster_dist"]).head(5))

    print("\n── Summary ──────────────────────────────────────────────────────")
    print(summary_df.head(10))

    if cfg["output_dist_matrix_csv"]:
        print("\n── Distance matrix (first 5 rows, first 6 cluster cols) ─────────")
        cluster_cols = [c for c in dist_matrix_df.columns if c.startswith("cluster_")][:5]
        print(dist_matrix_df.select(["IDR_ID"] + cluster_cols).head(5))


if __name__ == "__main__":
    main()
