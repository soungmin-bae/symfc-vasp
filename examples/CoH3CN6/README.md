# CoH3(CN)6 random-displacement example

This example shows the complete fixed-cell workflow:

```text
harmonic FORCE_CONSTANTS
  -> 200 phonopy random-displacement structures
  -> VASP MLFF force evaluation with IBRION=11
  -> symfc FC2/FC3
  -> phonon and mode-Gruneisen outputs
```

The included `OUTCAR.xz` is the result of the VASP step, so the
`symfc-vasp` part can be tested without running VASP.

## Test symfc-vasp

From this directory:

```bash
xz -dkf OUTCAR.xz
symfc-vasp full OUTCAR \
  --fc3 \
  --unitcell POSCAR-unitcell \
  --born BORN \
  --mesh 5 5 5 \
  --output run \
  --analysis-output run
```

The calculation uses the supplied 16-atom P-31m harmonic reference and infers
its 2x2x2, 128-atom supercell relation. Fitted force constants and all analysis
files are written under `run/`. Because the bundled MLFF `OUTCAR` contains one
aligned potential energy per force frame, the run also writes phonon DOS,
harmonic thermal properties, and `tdep_energy_offset.yaml`.

For a faster FC2-only check, omit `--fc3`:

```bash
symfc-vasp full OUTCAR \
  --unitcell POSCAR-unitcell \
  --born BORN \
  --output run \
  --analysis-output run
```

## Recreate the VASP trajectory

In addition to the files in this directory, prepare:

- a VASP executable with MLFF and interactive-position support;
- a trained `ML_FF` for Co, H, C, and N;
- a matching licensed `POTCAR`.

The supplied `FORCE_CONSTANTS.harmonic` and
`phonopy_disp_harmonic.yaml` form the harmonic parent calculation. Generate
200 random-displacement supercells at 0 K:

```bash
cp FORCE_CONSTANTS.harmonic FORCE_CONSTANTS
phonopy phonopy_disp_harmonic.yaml \
  --rd 200 \
  --rd-temperature 0 \
  --random-seed 42
```

Prepare the interactive VASP input. `POSCAR-001` is evaluated as the initial
`POSCAR`; structures 2 through 200 are read from standard input:

```bash
NIONS=128
cp POSCAR-001 POSCAR
for i in {2..200}; do
  tail -n "$NIONS" "$(printf 'POSCAR-%03d' "$i")"
done > positions.stdin
```

Place `ML_FF` and `POTCAR` in this directory, then run VASP with the supplied
`INCAR` and `KPOINTS`:

```bash
vasp_std < positions.stdin
```

The resulting fixed-cell `OUTCAR` or `vasprun.xml` can be passed directly to
the `symfc-vasp full` command shown above. NPT trajectories are not supported.
