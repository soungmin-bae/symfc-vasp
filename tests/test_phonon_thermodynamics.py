import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from symfc_vasp.engine import phonon


def test_phonon_stage_writes_dos_and_thermal_properties(tmp_path: Path):
    example = Path(__file__).parents[1] / "examples" / "CoH3CN6"
    fit_dir = tmp_path / "fit"
    output = tmp_path / "analysis"
    fit_dir.mkdir()
    shutil.copy2(example / "POSCAR-unitcell", fit_dir / "POSCAR-unitcell")
    shutil.copy2(example / "FORCE_CONSTANTS.harmonic", fit_dir / "FORCE_CONSTANTS")
    np.savetxt(fit_dir / "supercell_matrix.dat", np.diag([2, 2, 2]), fmt="%d")
    args = SimpleNamespace(
        fit_dir=fit_dir,
        analysis_output=output,
        dim=None,
        symprec=1e-5,
        band_points=3,
        mesh=(3, 3, 3),
        frequency_cutoff=0.05,
        tmin=0.0,
        tmax=100.0,
        tstep=50.0,
        born=None,
        mass=None,
        mass_index=None,
        force=False,
    )
    phonon(args)
    expected = {
        "phonon_dos.dat",
        "phonon_dos.pdf",
        "phonon_dos.png",
        "thermal_properties.yaml",
        "thermal_properties.tsv",
        "thermal_properties.pdf",
        "thermal_properties.png",
    }
    assert expected <= {path.name for path in output.iterdir()}
    summary = yaml.safe_load((output / "phonon_summary.yaml").read_text())
    assert summary["dos"]["mesh"] == [3, 3, 3]
    assert summary["thermal_properties"]["points"] == 3
