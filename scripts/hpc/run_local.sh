#!/usr/bin/env bash
# Plain interactive/login-node test run (no scheduler). For the full
# 10,371-IDR input, prefer run_slurm.sh on a compute node instead.
#
# Usage: ./run_local.sh [N_WORKERS]

cd "$(dirname "$0")/../.."   # repo root
mkdir -p results/fullseq

N_WORKERS="${1:-4}"

python src/nardini_fullSeq_numba_v096.py \
  --input data/input/Hsapien_IDRs_ftl100aa.csv.gz \
  --comp_ref data/reference/All_sapiens_IDRs_comp_feat.csv.gz \
  --centroids data/reference/All_sapiens_IDRs_GIN_centroids.csv.gz \
  --num_scrambles 100 \
  --n_workers "$N_WORKERS" \
  --out_sequences results/fullseq/Hsapien_IDRs_ftl100aa.fullseq.fixedKernel.out_sequences.csv \
  --out_summary results/fullseq/Hsapien_IDRs_ftl100aa.fullseq.fixedKernel.out_summary.csv \
  --out_dist_matrix results/fullseq/Hsapien_IDRs_ftl100aa.fullseq.fixedKernel.out_dist_matrix.csv
