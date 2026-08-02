# symfc-vasp

`symfc-vasp` extracts finite-temperature force constants from fixed-cell VASP
molecular-dynamics trajectories. It connects four operations in one
reproducible workflow:

1. read positions and Born-Oppenheimer forces from `OUTCAR` or `vasprun.xml`;
2. select equilibrated trajectory frames;
3. fit symmetry-adapted FC2 and FC3 with symfc;
4. calculate phonons and tensor mode-Gruneisen parameters with phonopy and
   phono3py.

This is an independent workflow package and is not an official symfc or VASP
project.

## Installation

Install from a source checkout:

```bash
git clone https://github.com/soungmin-bae/symfc-vasp.git
cd symfc-vasp
python -m pip install .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

Confirm the installation:

```bash
symfc-vasp --version
symfc-vasp --help
```

## Required input

A calculation directory normally contains:

```text
case/
├── OUTCAR              # or vasprun.xml
├── POSCAR-supercell    # atom order used by VASP MD
├── POSCAR-unitcell     # unit-cell symmetry reference
└── run.yaml            # optional reusable workflow configuration
```

The trajectory must use a fixed simulation cell. `POSCAR-supercell` must have
the same atom order as the VASP trajectory. The cell of `POSCAR-supercell`
must be generated from `POSCAR-unitcell` by the supplied supercell matrix.

## Quick start

Inspect the trajectory before fitting:

```bash
symfc-vasp inspect \
  --trajectory OUTCAR \
  --unitcell POSCAR-unitcell \
  --supercell POSCAR-supercell \
  --dim 2 2 2 \
  --skip 5000 \
  --samples 3000 \
  --selection uniform
```

Run FC2, FC3, band-path, and q-mesh analysis:

```bash
symfc-vasp run \
  --trajectory OUTCAR \
  --unitcell POSCAR-unitcell \
  --supercell POSCAR-supercell \
  --dim 2 2 2 \
  --skip 5000 \
  --samples 3000 \
  --selection uniform \
  --order 2 3 \
  --rc2 7 --rc3 4 \
  --mass H 2.014 \
  --mesh 11 11 11 \
  --output run_N3000
```

`--mass` changes the isotope mass used by phonopy and phono3py without
changing the element labels or refitting FC2/FC3. For a deuterated trajectory
whose POSCAR still labels the isotope as H, use `--mass H 2.014`. The resolved
`run.yaml` stores this as:

```yaml
mass_overrides:
  H: 2.014
```

The applied and default masses are also recorded in
`analysis/analysis_summary.yaml`. This makes H/D mass postprocessing explicit
and reproducible.

Selected configurations are centered by default. Per-atom mean displacement
and mean force are removed so that the fitted force constants describe
fluctuations around the sampled finite-temperature mean. Use
`--no-center-selected` for an uncentered control.

## Reuse `run.yaml`

Each complete run writes its resolved settings to `run.yaml`. Reuse them with:

```bash
symfc-vasp run \
  --config previous_run/run.yaml \
  --output rerun_N3000
```

Explicit CLI options override YAML values:

```bash
symfc-vasp run \
  --config previous_run/run.yaml \
  --samples 5000 \
  --output run_N5000
```

The YAML files have distinct roles:

- `run.yaml`: resolved input settings and provenance;
- `analysis/analysis_summary.yaml`: dimensions of generated band and mesh data;
- `FINAL_VALIDATION.yaml`: final completeness and finite-value checks.

## Split a run into three stages

Short scheduler queues can use stable FC and analysis boundaries.

```bash
# 1. Trajectory extraction and FC2/FC3 fitting
symfc-vasp fit \
  --config run.yaml \
  --output run_N3000/force_constants

# 2. Phonon dispersion and band-path mode-Gruneisen plots
symfc-vasp band \
  --config run.yaml \
  --fit-dir run_N3000/force_constants \
  --analysis-output run_N3000/analysis

# 3. q-mesh mode-Gruneisen data and final validation
symfc-vasp mesh \
  --config run.yaml \
  --fit-dir run_N3000/force_constants \
  --analysis-output run_N3000/analysis
symfc-vasp validate run_N3000
```

`phonon` is an alias of `band`. `gruneisen` runs both band and mesh stages.

## Outputs

```text
run_N3000/
├── run.yaml
├── FINAL_VALIDATION.yaml
├── force_constants/
│   ├── symfc_input.npz
│   ├── selected_indices.txt
│   ├── symfc_summary.yaml
│   ├── FORCE_CONSTANTS
│   ├── fc2.hdf5
│   └── fc3.hdf5
└── analysis/
    ├── phonon_dispersion.tsv
    ├── phonon_dispersion.pdf
    ├── mode_gruneisen_qpath.tsv
    ├── mode_gruneisen_q_resolved.pdf
    ├── mode_gruneisen_on_phonon_dispersion.pdf
    ├── gruneisen_qmesh_11x11x11.hdf5
    ├── mode_gruneisen_qmesh_11x11x11.pdf
    ├── phonopy_disp.yaml
    ├── band.conf
    ├── phono3py-gruneisen-band.conf
    ├── phono3py-gruneisen-mesh.conf
    ├── phonon_band.dat
    ├── gruneisen_qmesh_11x11x11.dat
    ├── plot_phonon_dispersion.gp
    ├── plot_mode_gruneisen_q_resolved.gp
    ├── plot_mode_gruneisen_on_phonon_dispersion.gp
    ├── plot_mode_gruneisen_qmesh.gp
    └── README_REPRODUCE.md
```

The analysis directory is a reproducible postprocessing bundle. The `.dat`
files are human-readable and directly usable by gnuplot. Each `.gp` script
defaults to PDF and can be opened interactively with, for example:

```bash
gnuplot plot_phonon_dispersion.gp
gnuplot -e 'plot_terminal="qt"' plot_phonon_dispersion.gp
```

Relative links to `FORCE_CONSTANTS`, `fc2.hdf5`, and `fc3.hdf5` are created
automatically. Together with `POSCAR-unitcell` and the generated configuration
files, they permit independent phonopy and phono3py reruns from `analysis/`.
`phonopy_disp.yaml` embeds the unit cell, supercell matrix, effective masses,
and fitted FC2, so the standard phonopy command works directly:

```bash
phonopy -p band.conf -s
```

## Parallel execution

symfc is used as one Python process. Allocate one scheduler task and multiple
CPUs per task. NumPy, SciPy, and optional MKL-backed kernels use the available
threads. Do not launch one Python process per MPI rank.

```bash
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=1
symfc-vasp fit --config run.yaml --output run/force_constants
```

Portable examples are provided under [`examples/`](examples/).
