# OUTCAR: FC2, FC3, band, and mesh

Place `OUTCAR` in this directory. Optional known reference POSCAR files can be
listed in `run.yaml`. Then run:

```bash
symfc-vasp full --config run.yaml OUTCAR --output run_N3000 --analysis-output run_N3000
```

The example skips 5,000 equilibration frames and selects 3,000 frames
uniformly from the remainder.
