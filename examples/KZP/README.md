# KZP OUTCAR-only example

This example contains 200 fixed-cell VASP MLFF force evaluations for a
288-atom KZP supercell. It demonstrates automatic reconstruction of the
reference structure directly from an `OUTCAR`, without supplying a unit-cell
file.

From this directory:

```bash
xz -dkf OUTCAR.xz
symfc-vasp full OUTCAR \
  --output run \
  --analysis-output run
```

The trajectory mean is refined to an R-3c 36-atom unit cell and a 2x2x2
supercell relation. The command fits FC2 and writes the phonon dispersion and
reusable phonopy files under `run/`.

To select only part of a longer trajectory, add `--skip`, `--samples`, and
`--selection uniform` as needed.
