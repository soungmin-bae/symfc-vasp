# OUTCAR: FC2, FC3, band, and mesh

Place `OUTCAR`, `POSCAR-supercell`, and `POSCAR-unitcell` in this directory,
then run:

```bash
symfc-vasp run --config run.yaml --output run_N3000
```

The example skips 5,000 equilibration frames and selects 3,000 frames
uniformly from the remainder.

