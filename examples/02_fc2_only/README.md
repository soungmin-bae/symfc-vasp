# FC2-only fit

This case extracts a harmonic finite-temperature force constant without FC3
or mode-Gruneisen analysis.

```bash
symfc-vasp fit --config run.yaml OUTCAR --output run_fc2
symfc-vasp phonon run_fc2 --analysis-output run_fc2
```
