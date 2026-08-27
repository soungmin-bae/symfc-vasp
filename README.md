# symfc-vasp

`symfc-vasp` fits symmetry-adapted force constants from a fixed-cell VASP
`OUTCAR` or `vasprun.xml`, then creates phonopy phonon and optional
FC2+FC3 mode-Gruneisen outputs.

It is an independent workflow package, not an official symfc, phonopy, or
VASP project.

## Scientific scope

The supported input is a fixed-cell VASP trajectory containing Cartesian
positions and Born-Oppenheimer forces. The supported result boundary is:

```text
fixed-cell OUTCAR/vasprun.xml
  -> symmetry-validated reference structure
  -> symfc FC2 or FC2+FC3
  -> phonopy phonon dispersion
  -> phono3py FC3-based tensor mode-Gruneisen band and mesh data
```

This package does not run VASP and does not implement variable-cell MD, QHA,
finite-strain `phonopy-gruneisen`, thermal-expansion integration, or free-energy
calculation. Those require separate physical inputs and convergence studies.

## Install

```bash
git clone https://github.com/soungmin-bae/symfc-vasp.git
cd symfc-vasp
python -m pip install -e '.[dev]'
symfc-vasp --version
```

## Commands

The public CLI is deliberately small:

```text
symfc-vasp fit [OUTCAR|vasprun.xml]
symfc-vasp phonon FIT_DIR
symfc-vasp gruneisen FIT_DIR
symfc-vasp full [OUTCAR|vasprun.xml]
```

All commands write into the current directory by default. Use
`--output DIRECTORY` for fitting or `--analysis-output DIRECTORY` for
postprocessing only when separate directories are wanted.

## Python API

The CLI is an adapter over the typed public API. A complete FC2 workflow can
be driven directly from Python:

```python
from pathlib import Path

from symfc_vasp import (
    AnalysisConfig,
    FitConfig,
    TrajectoryConfig,
    WorkflowConfig,
    run_workflow,
)

fit, phonon, gruneisen = run_workflow(
    WorkflowConfig(
        fit=FitConfig(
            trajectory=TrajectoryConfig(
                Path("OUTCAR"), skip=5000, samples=3000,
                selection="uniform",
            ),
            output_dir=Path("result"),
        ),
        analysis=AnalysisConfig(output_dir=Path("result")),
    )
)

print(fit.fc2.shape)
print(phonon.frequencies.min(), phonon.frequencies.max())
```

Set `orders=(2, 3)` in `FitConfig` to obtain `fit.fc3` and a non-`None`
`gruneisen` result. The lower-level public functions are `read_trajectory`,
`build_reference`, `fit_force_constants`, `calculate_phonons`, and
`calculate_gruneisen`.

## Minimal workflow

Fit FC2 from a trajectory. FC2 is the default and no cutoff is imposed unless
`--rc2` is supplied.

```bash
symfc-vasp fit OUTCAR
```

This reads every available frame (`--skip 0`, stride selection) and writes
flat output files in the current directory:

```text
FORCE_CONSTANTS       fc2.hdf5
POSCAR-mean           POSCAR-unitcell
POSCAR-supercell      SPOSCAR
supercell_matrix.dat  symfc_input.npz
selected_indices.txt  symmetry_report.yaml
fit_request.yaml      symfc_summary.yaml
symfc_solver.log
```

Then calculate a normal phonopy-compatible band structure in the same
directory. `.` is explicit so a bare command only shows help:

```bash
symfc-vasp phonon .
```

It writes `band.conf`, `phonopy_disp.yaml`, `band.yaml`, `band.pdf`,
`band.png`, `phonon_band.dat`, and `phonopy-band.dat`. `phonon_band.dat` is
grouped branch-by-branch, so it is directly usable with:

```gnuplot
plot "phonon_band.dat" using 1:2 with lines
```

`phonopy-band.dat` is the exact output of `phonopy-bandplot --gnuplot band.yaml`.
`band.conf` can be replayed directly:

```bash
phonopy --config band.conf -p -s
```

## FC3 and mode-Gruneisen analysis

FC3 is an explicit extra calculation:

```bash
symfc-vasp fit OUTCAR --fc3
```

`--rc3 auto` is the default when `--fc3` is present. It selects the largest
FC3 cutoff satisfying the configured equations-per-parameter threshold. A
fixed cutoff is explicit:

```bash
symfc-vasp fit OUTCAR --fc3 --rc3 4.0
```

After FC3 is available, generate band-path and q-mesh tensor Gruneisen data,
plots, YAML files, tables, and reproducibility inputs. Postprocessing uses the
fitted `supercell_matrix.dat`, including a general 3x3 matrix, so `--dim` does
not need to be supplied again:

```bash
symfc-vasp gruneisen . --mesh 11 11 11
```

The one-command equivalent is:

```bash
symfc-vasp full OUTCAR --fc3 --mesh 11 11 11
```

Without `--fc3`, `full` stops after FC2 fitting and phonon dispersion.

## Reference structure

With only a trajectory, the package constructs a periodic mean structure,
scans and standardizes symmetry with spglib, builds the matching supercell,
and validates a species-preserving atom map. The generated `POSCAR-unitcell`
is the reference used for FC and phonon calculation.

The terminal prints the selected space group, `symprec`, primitive/supercell
relation, integer matrix, and map residual. `symmetry_report.yaml` stores the
full tolerance scan and atom-mapping information.

The automatic symmetry scan reaches `0.3 A` by default so a finite random
displacement sample can recover its parent symmetry. Reduce or increase this
only with `--reference-symprec-max` after inspecting `symmetry_report.yaml`.

Every tolerance candidate is tested for an integer cell relation and a
species-preserving one-to-one atom map. `symmetry_report.yaml` records rejected
candidates, the stable tolerance plateau, the selected space group, and all
mapping residuals.

If trajectory-only standardization fails its atom-map validation and an
existing `POSCAR-unitcell` is present in the same directory, that known
displacement parent is automatically used as a verified fallback. The event
and original failure are recorded in `structure_report.yaml` as
`automatic-existing-unitcell-fallback`; mapping tolerances are not silently
loosened.

When a known reference unit cell must be used unchanged, supply it explicitly:

```bash
symfc-vasp full OUTCAR \
  --unitcell POSCAR-unitcell \
  --samples 3000 --selection uniform \
  --fc3
```

An optional `--supercell POSCAR` fixes the supplied supercell atom order. The
optional `--dim 2 2 2` is a diagonal consistency check; omit it for a general
integer supercell matrix.

## Common options

```bash
# Equilibrated subset
symfc-vasp fit OUTCAR --skip 5000 --samples 3000 --selection uniform

# Preserve the selected mean instead of fitting fluctuations around it
symfc-vasp fit OUTCAR --no-center-selected

# Non-analytical correction and isotope postprocessing
symfc-vasp full OUTCAR --fc3 --born BORN --mass H 2.014

# Keep fitting and analysis separate
symfc-vasp fit OUTCAR --output fit
symfc-vasp phonon fit --analysis-output analysis
symfc-vasp gruneisen fit --analysis-output analysis
```

`--born BORN` enables phonopy-format non-analytical correction. If `BORN`
exists beside the fitted files, `phonon` and `gruneisen` discover it
automatically. `--mass H 2.014` changes only the masses used in phonon and
Gruneisen postprocessing; it does not refit the force constants.

For site-selective isotope masses, one-based primitive-cell indices override
the element-level value:

```bash
symfc-vasp phonon . --mass H 2.014 --mass-index 2 3.016
```

Flat outputs are protected against accidental mixing. If an existing fit or
analysis manifest refers to a different input hash, rerun with `--force` only
after confirming that replacement is intended. Original trajectory, POSCAR,
and BORN files are never modified.

## Diagnostics and provenance

`symfc_summary.yaml` records the resolved selection, structure relation,
software versions, thread environment, basis parameter counts,
equations-per-parameter ratios, permutation/translational residuals, peak
memory, and in-sample force reconstruction. The latter is not cross-validation
and is named `in_sample_reconstruction` deliberately.

`phonon_summary.yaml` records the Gamma frequencies, three acoustic
frequencies, imaginary band-point count, NAC validation, masses, input hashes,
timing, and peak memory. `analysis_summary.yaml` provides the corresponding
FC3/Gruneisen provenance.

## Optional `run.yaml`

`run.yaml` is optional. It provides defaults, and explicit CLI options always
override it. Existing files using `force_constants.orders: [2, 3]` remain
compatible and are interpreted as `--fc3`.

```bash
symfc-vasp full --config run.yaml OUTCAR --samples 4000
```

Use `symfc-vasp <command> -h` for the exact options and concise examples.
