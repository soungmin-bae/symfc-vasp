# vasprun.xml trajectory

`vasprun.xml` may be used instead of `OUTCAR` when every ionic calculation
contains the cell, positions, and forces.

```bash
symfc-vasp inspect --config run.yaml
symfc-vasp run --config run.yaml --output run_vasprun
```

