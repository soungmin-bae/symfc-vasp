#!/bin/bash
#SBATCH -J svN3k
#SBATCH -p i8cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 00:30:00
#SBATCH --exclusive

set -euo pipefail
ulimit -s unlimited
source /etc/profile.d/modules.sh
module purge
module load oneapi_compiler/2023.0.0 oneapi_mkl/2023.0.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-128}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-128}"
export OPENBLAS_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-128}" \
  symfc-vasp run \
    --trajectory OUTCAR \
    --unitcell POSCAR-unitcell \
    --supercell POSCAR-supercell \
    --dim 2 2 2 \
    --skip 5000 \
    --samples 3000 \
    --order 2 3 \
    --rc2 7 --rc3 4 \
    --mesh 11 11 11 \
    --output run_N3000 \
  > run_N3000.stdout 2> run_N3000.stderr
