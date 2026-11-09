#!/bin/bash
#SBATCH --job-name=nardini_gin_assign
#SBATCH --output=nardini_gin_assign_%j.out
#SBATCH --error=nardini_gin_assign_%j.err
#SBATCH --time=24:00:00            # EDIT: walltime for your queue
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
mkdir -p results/fullseq

python src/nardini_fullSeq_numba_v096.py \
  --input data/input/Hsapien_IDRs_ftl100aa.csv.gz \
  --comp_ref data/reference/All_sapiens_IDRs_comp_feat.csv.gz \
  --centroids data/reference/All_sapiens_IDRs_GIN_centroids.csv.gz \
  --num_scrambles 100 \
  --n_workers "${SLURM_CPUS_PER_TASK:-32}" \
  --out_sequences results/fullseq/Hsapien_IDRs_ftl100aa.fullseq.fixedKernel.out_sequences.csv \
  --out_summary results/fullseq/Hsapien_IDRs_ftl100aa.fullseq.fixedKernel.out_summary.csv \
  --out_dist_matrix results/fullseq/Hsapien_IDRs_ftl100aa.fullseq.fixedKernel.out_dist_matrix.csv
