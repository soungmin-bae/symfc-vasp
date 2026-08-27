# Deuterium mass with H-labelled structures

VASP structures normally retain the chemical symbol `H` even when `POMASS`
uses the deuterium mass. This example applies 2.014 amu to every H site during
phonopy/phono3py postprocessing:

```bash
symfc-vasp full --config run.yaml OUTCAR \
  --output run_D_N3000 --analysis-output run_D_N3000
```

The mass override changes phonon frequencies and mode-Gruneisen parameters.
It does not change the fitted FC2/FC3 tensors or rename H atoms to D.
