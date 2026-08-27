from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from symfc_vasp import (
    AnalysisConfig,
    FitConfig,
    ReferenceConfig,
    TrajectoryConfig,
    calculate_phonons,
    fit_force_constants,
)
from symfc_vasp.models import FitResult, PhononResult


def test_fit_config_is_translated_without_argparse(monkeypatch, tmp_path: Path):
    output = tmp_path / "fit"
    output.mkdir()
    for name, shape in (("fc2.hdf5", (1, 1, 3, 3)),):
        with h5py.File(output / name, "w") as handle:
            handle["fc2"] = np.zeros(shape)
    np.savetxt(output / "supercell_matrix.dat", np.eye(3, dtype=int), fmt="%d")
    np.savez(output / "symfc_input.npz", generated_to_vasp=np.array([0]))
    (output / "POSCAR-unitcell").write_text("unit")
    (output / "POSCAR-supercell").write_text("super")
    (output / "symfc_summary.yaml").write_text("fit: {}\n")

    captured = {}

    def fake_fit(args):
        captured.update(vars(args))
        return output

    monkeypatch.setattr("symfc_vasp.api.engine.fit", fake_fit)
    result = fit_force_constants(
        FitConfig(
            trajectory=TrajectoryConfig(tmp_path / "OUTCAR", skip=3, samples=7),
            reference=ReferenceConfig(dim=(2, 2, 2)),
            output_dir=output,
            orders=(2,),
        )
    )
    assert isinstance(result, FitResult)
    assert captured["skip"] == 3
    assert captured["samples"] == 7
    assert captured["dim"] == (2, 2, 2)
    assert result.fc2.shape == (1, 1, 3, 3)


def test_phonon_result_exposes_band_arrays(monkeypatch, tmp_path: Path):
    (tmp_path / "band.yaml").write_text(
        "phonon:\n"
        "- q-position: [0, 0, 0]\n"
        "  distance: 0.0\n"
        "  label: GAMMA\n"
        "  band:\n"
        "  - frequency: 1.25\n"
    )
    (tmp_path / "phonon_summary.yaml").write_text("spacegroup: P1\n")
    monkeypatch.setattr("symfc_vasp.api.engine.phonon", lambda args: tmp_path)
    result = calculate_phonons(AnalysisConfig())
    assert isinstance(result, PhononResult)
    assert result.frequencies.shape == (1, 1)
    assert result.frequencies[0, 0] == 1.25
    assert result.labels == ("GAMMA",)
