#!/usr/bin/env bash
# Plain interactive/login-node test run of the sliding-window (multi-block)
# GIN assignment (no scheduler). For the full 10,371-IDR input, prefer
# run_MB_slurm.sh on a compute node instead.
#
# Usage: ./run_MB_local.sh [N_WORKERS]

cd "$(dirname "$0")/../.."   # repo root

N_WORKERS="${1:-4}"

python src/nardini_MultiBlock_v099.py \
  --input data/input/Hsapien_IDRs_ftl100aa.csv.gz \
  --comp_ref data/reference/All_sapiens_IDRs_comp_feat.csv.gz \
  --centroids data/reference/All_sapiens_IDRs_GIN_centroids.csv.gz \
  --window_size 70 \
  --step_size 10 \
  --num_scrambles 100 \
  --n_workers "$N_WORKERS" \
  --block_method seed_expand \
  --seed_min_dist_quantile 0.60 \
  --seed_inter_dist_quantile 0.75 \
  --seed_absorption_factor 1.5 \
  --min_block_length 70 \
  --out_windows results/multiblock/Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.out_windows.csv \
  --out_summary results/multiblock/Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.seedExpand.q60.out_summary.csv
