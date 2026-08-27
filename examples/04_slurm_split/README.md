# Two-stage Slurm workflow

Copy `run.yaml` and the VASP trajectory into the submission directory. Adapt
only the partition, walltime, and CPU count in the scripts, then submit:

```bash
fit_job=$(sbatch --parsable submit-01-fit.sh)
sbatch --dependency="afterok:${fit_job}" submit-02-analysis.sh
```
