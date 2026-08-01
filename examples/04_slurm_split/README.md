# Three-stage Slurm workflow

Copy `run.yaml` and the three VASP input files into the submission directory.
Adapt only the partition, walltime, and CPU count in the scripts, then submit:

```bash
fit_job=$(sbatch --parsable submit-01-fit.sh)
band_job=$(sbatch --parsable --dependency="afterok:${fit_job}" submit-02-band.sh)
sbatch --dependency="afterok:${band_job}" submit-03-mesh.sh
```

