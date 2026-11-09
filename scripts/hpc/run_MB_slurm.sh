#!/bin/bash
#SBATCH --job-name=nardini_gin_multiblock
#SBATCH --output=nardini_gin_multiblock_%j.out
#SBATCH --error=nardini_gin_multiblock_%j.err
#SBATCH --time=48:00:00            # EDIT: walltime for your queue (sliding-window scans far more windows than the full-sequence run — budget more time)
#SBATCH --cpus-per-task=32         # EDIT: match --n_workers below
#SBATCH --mem=64G                  # EDIT: adjust as needed
#SBATCH --partition=YOUR_PARTITION # EDIT
# #SBATCH --account=YOUR_ACCOUNT   # EDIT: uncomment/set if your cluster requires it

# EDIT: load/activate whatever environment has numpy/pandas/polars/scipy/
# tqdm/multiprocess/localcider/numba installed, e.g.:
# module load anaconda3
# conda activate nardini_env

# Uncomment if your HPC home directory is read-only or quota-limited —
# numba needs somewhere writable to cache compiled kernels:
# export NUMBA_CACHE_DIR=/scratch/$USER/numba_cache

cd "$(dirname "$0")/../.."   # repo root

python src/nardini_MultiBlock_v099.py \
  --input data/input/Hsapien_IDRs_ftl100aa.csv.gz \
  --comp_ref data/reference/All_sapiens_IDRs_comp_feat.csv.gz \
  --centroids data/reference/All_sapiens_IDRs_GIN_centroids.csv.gz \
  --window_size 70 \
  --step_size 10 \
  --num_scrambles 100 \
  --n_workers "${SLURM_CPUS_PER_TASK:-32}" \
  --block_method seed_expand \
  --seed_min_dist_quantile 0.60 \
  --seed_inter_dist_quantile 0.75 \
  --seed_absorption_factor 1.5 \
  --min_block_length 70 \
  --out_windows results/multiblock/Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.out_windows.csv \
  --out_summary results/multiblock/Hsapien_IDRs_ftl100aa.MultiBlock.fixedKernel.seedExpand.q60.out_summary.csv
