#!/bin/bash
#SBATCH --job-name=symfc-band
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

symfc-vasp band --config run.yaml --fit-dir run/force_constants --analysis-output run/analysis

