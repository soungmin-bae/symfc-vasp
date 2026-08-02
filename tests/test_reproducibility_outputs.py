from pathlib import Path

import numpy as np

from symfc_vasp.reproducibility import (
    write_band_dat,
    write_band_gnuplot_scripts,
    write_mesh_dat,
    write_mesh_gnuplot_script,
    write_phonon_inputs,
    write_reproduction_readme,
)


def test_reproducibility_bundle_contains_plain_data_inputs_and_scripts(tmp_path: Path):
    segments = [np.linspace([0, 0, 0], [0.5, 0, 0], 2)]
    labels = [("Γ", "M")]
    rows = np.array(
        [
            [0, 0, 0.0, 1, 1.0, -2.0, -1.0, 0.5, -1.5, -0.8333],
            [0, 1, 1.0, 1, 2.0, -1.0, 0.0, 1.0, -0.5, 0.0],
        ]
    )
    mesh_rows = np.array(
        [[0, 0, 0, 0, 0, 1, 1, 1.0, -2.0, -1.0, 0.5, -0.8333]]
    )
    write_band_dat(tmp_path / "phonon_band.dat", rows)
    write_phonon_inputs(
        tmp_path, segments, labels, [2, 2, 2], 21, [11, 11, 11], [58.933, 2.014]
    )
    write_band_gnuplot_scripts(tmp_path, [0, 1], labels, -10, 20, -100, 2300, 0.05)
    write_mesh_dat(tmp_path / "gruneisen_qmesh_11x11x11.dat", mesh_rows)
    write_mesh_gnuplot_script(tmp_path, [11, 11, 11], -10, 20, 0.05)
    write_reproduction_readme(tmp_path, [11, 11, 11])

    expected = {
        "band.conf",
        "phono3py-gruneisen-band.conf",
        "phono3py-gruneisen-mesh.conf",
        "run_phonopy_phono3py.sh",
        "phonon_band.dat",
        "gruneisen_qmesh_11x11x11.dat",
        "plot_phonon_dispersion.gp",
        "plot_mode_gruneisen_q_resolved.gp",
        "plot_mode_gruneisen_on_phonon_dispersion.gp",
        "plot_mode_gruneisen_qmesh.gp",
        "README_REPRODUCE.md",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert "DIM = 2 2 2" in (tmp_path / "band.conf").read_text()
    assert "MASS = 58.933 2.014" in (tmp_path / "band.conf").read_text()
    assert "BAND =" in (tmp_path / "phono3py-gruneisen-band.conf").read_text()
    assert "MESH = 11 11 11" in (tmp_path / "phono3py-gruneisen-mesh.conf").read_text()
    assert 'plot_terminal="qt"' in (tmp_path / "README_REPRODUCE.md").read_text()
