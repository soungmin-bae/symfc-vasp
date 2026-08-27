# vasprun.xml trajectory

`vasprun.xml` may be used instead of `OUTCAR` when every ionic calculation
contains the cell, positions, and forces.

```bash
symfc-vasp full --config run.yaml vasprun.xml \
  --output run_vasprun --analysis-output run_vasprun
```
