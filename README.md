# symfc-vasp

`symfc-vasp` fits symmetry-adapted force constants from fixed-cell VASP
trajectories and calculates phonon dispersions and mode-Gruneisen parameters.
It accepts `OUTCAR` and `vasprun.xml`. NPT trajectories are not supported.

## Installation

```bash
git clone https://github.com/soungmin-bae/symfc-vasp.git
cd symfc-vasp
python -m pip install .
```

Python 3.11-3.13, phonopy 4.x, phono3py 4.x, and symfc 1.7.x are supported.

## Basic usage

Fit FC2 and calculate the phonon dispersion:

```bash
symfc-vasp full OUTCAR
```

Add FC3 and mode-Gruneisen calculations:

```bash
symfc-vasp full OUTCAR --fc3
```

`vasprun.xml` can be used in the same way:

```bash
symfc-vasp full vasprun.xml --fc3
```

By default, results are written in the current directory. The input trajectory
is never modified.

### Select trajectory frames

This example discards 5,000 equilibration frames and uniformly selects 3,000
frames from the remainder:

```bash
symfc-vasp full OUTCAR \
  --skip 5000 \
  --samples 3000 \
  --selection uniform \
  --fc3
```

Without selection options, all available frames are used.

### Use a known reference structure

When only a trajectory is supplied, the reference structure and integer
supercell matrix are determined from its periodic mean structure. For a
random-displacement calculation, the original unit cell can be supplied
explicitly:

```bash
symfc-vasp full OUTCAR --unitcell POSCAR-unitcell
```

Use `--supercell POSCAR-supercell` as well when its atom order must be
preserved.

### NAC and isotope masses

Use a phonopy-format `BORN` file for non-analytical correction:

```bash
symfc-vasp full OUTCAR --fc3 --born BORN
```

Masses can be changed during phonon and mode-Gruneisen analysis without
refitting the force constants:

```bash
symfc-vasp phonon . --mass H 2.014
symfc-vasp gruneisen . --mass H 2.014
```

## Separate stages

The same workflow can be run one stage at a time:

```bash
symfc-vasp fit OUTCAR --fc3 --output fit
symfc-vasp phonon fit --analysis-output analysis
symfc-vasp gruneisen fit --analysis-output analysis
```

FC2 is fitted by default. `--fc3` enables FC3. No FC2 cutoff is applied unless
`--rc2` is given; FC3 selects a cutoff automatically unless `--rc3` is given.

Use `symfc-vasp <command> -h` for the complete option list.

## Outputs

The fitting stage writes:

```text
FORCE_CONSTANTS
fc2.hdf5
fc3.hdf5                 # with --fc3
POSCAR-unitcell
POSCAR-supercell
supercell_matrix.dat
symfc_input.npz
symfc_summary.yaml
symfc_solver.log
```

Phonon and mode-Gruneisen analysis writes standard phonopy YAML files,
human-readable tables, reusable configuration and plotting files, and PDF/PNG
figures. Important files include:

```text
band.conf
band.yaml
phonopy_disp.yaml
phonon_band.dat
gruneisen_band.yaml
gruneisen_mesh.yaml
phonon_dispersion.pdf
mode_gruneisen_q_resolved.pdf
mode_gruneisen_on_phonon_dispersion.pdf
```

## Complete example

[`examples/CoH3CN6`](examples/CoH3CN6) contains a compact 200-structure VASP
MLFF `OUTCAR` and harmonic inputs for generating the same type of trajectory.
The example can be run immediately or repeated with a user-provided `ML_FF`
and licensed `POTCAR`.

## Python API

```python
from pathlib import Path

from symfc_vasp import FitConfig, TrajectoryConfig, fit_force_constants

result = fit_force_constants(
    FitConfig(
        trajectory=TrajectoryConfig(Path("OUTCAR")),
        output_dir=Path("fit"),
    )
)

print(result.fc2.shape)
```

## Citation

When publishing results produced with this workflow, please cite the relevant
underlying projects:

- [symfc](https://symfc.github.io/symfc/)
- [phonopy](https://phonopy.github.io/phonopy/)
- [phono3py](https://phonopy.github.io/phono3py/)
- [spglib](https://spglib.readthedocs.io/)
- [SeeK-path](https://seekpath.readthedocs.io/)

## Author

**Author:** Soungmin Bae

**Affiliation:** Yokohama City University

**Contact:** [soungminbae@gmail.com](mailto:soungminbae@gmail.com)
