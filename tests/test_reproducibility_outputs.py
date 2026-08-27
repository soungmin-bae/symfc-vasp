from pathlib import Path

import numpy as np

from symfc_vasp.reproducibility import (
    write_band_dat,
    write_plain_phonon_band_dat,
    write_band_gnuplot_scripts,
    write_mesh_dat,
    write_mesh_gnuplot_script,
    write_phonon_inputs,
    write_phonopy_yaml,
    write_reproduction_readme,
    write_gruneisen_mesh_yaml,
    write_tensor_plotter_bundle,
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
    assert "#BAND_CONNECTION = .TRUE." in (tmp_path / "band.conf").read_text()
    assert "#EIGENVECTORS = .TRUE." in (tmp_path / "band.conf").read_text()
    assert 'plot_terminal="qt"' in (tmp_path / "README_REPRODUCE.md").read_text()


def test_plain_phonon_band_data_is_grouped_by_mode_for_gnuplot(tmp_path: Path):
    path = tmp_path / "phonon_band.dat"
    write_plain_phonon_band_dat(
        path,
        [np.array([0.0, 0.5])],
        [np.array([[1.0, 3.0], [2.0, 4.0]])],
    )
    blocks = [
        [line for line in block.splitlines() if line]
        for block in path.read_text().split("\n\n")
        if block.strip()
    ]
    assert blocks[0][1:] == ["0 1 0 1", "0.5 2 0 1"]
    assert blocks[1] == ["0 3 0 2", "0.5 4 0 2"]


def test_tensor_mesh_yaml_and_plotter_bundle_are_self_contained(tmp_path: Path):
    class Unit:
        cell = np.eye(3)

    qpoints = np.array([[0.0, 0.0, 0.0]])
    weights = np.array([1])
    frequencies = np.array([[1.25]])
    tensors = np.array([[[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]])
    path = write_gruneisen_mesh_yaml(
        tmp_path, Unit(), [11, 11, 11], qpoints, weights, frequencies, tensors
    )
    text = path.read_text()
    assert "gruneisen_tensor:" in text
    assert "gruneisen:    2.0000000000" in text
    names = {item.name for item in write_tensor_plotter_bundle(tmp_path, [11, 11, 11])}
    assert {
        "export_gruneisen_mesh_yaml.py",
        "gruneisen_tensor_io.py",
        "plot_gruneisen_band_tensor.py",
        "plot_gruneisen_band_ab-axis.py",
        "plot_gruneisen_band_c-axis.py",
        "plot_gruneisen_mesh_3d_tensor.py",
        "plot_gruneisen_mesh_3d_ab-axis.py",
        "plot_gruneisen_mesh_3d_c-axis.py",
        "plot_mode_gruneisen_q_resolved.py",
    } <= names


def test_phonopy_yaml_contains_supercell_masses_and_force_constants(tmp_path: Path):
    from phonopy.interface.vasp import read_vasp_from_strings
    from phonopy.interface.phonopy_yaml import PhonopyYaml

    unit = read_vasp_from_strings(
        """H\n1.0\n2 0 0\n0 2 0\n0 0 2\nH\n1\nDirect\n0 0 0\n"""
    )
    unit.masses = [2.014]
    fc2 = np.zeros((8, 8, 3, 3))
    path = write_phonopy_yaml(tmp_path, unit, [2, 2, 2], fc2, symprec=1e-5)

    yaml_data = PhonopyYaml()
    yaml_data.read(path)
    assert path.name == "phonopy_disp.yaml"
    assert yaml_data.supercell_matrix.tolist() == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert np.asarray(yaml_data.force_constants).shape == (8, 8, 3, 3)
    assert yaml_data.unitcell.masses.tolist() == [2.014]


def test_reproducibility_writers_accept_general_supercell_matrix(tmp_path: Path):
    from phonopy.interface.vasp import read_vasp_from_strings
    from phonopy.interface.phonopy_yaml import PhonopyYaml

    unit = read_vasp_from_strings(
        """H\n1.0\n2 0 0\n0 2 0\n0 0 2\nH\n1\nDirect\n0 0 0\n"""
    )
    matrix = [[2, 1, 0], [0, 2, 0], [0, 0, 2]]
    fc2 = np.zeros((8, 8, 3, 3))
    path = write_phonopy_yaml(tmp_path, unit, matrix, fc2, symprec=1e-5)
    yaml_data = PhonopyYaml()
    yaml_data.read(path)
    assert yaml_data.supercell_matrix.tolist() == matrix
    segments = [np.linspace([0, 0, 0], [0.5, 0, 0], 2)]
    write_phonon_inputs(tmp_path, segments, [("Γ", "M")], matrix, 21, [11, 11, 11], [1.0])
    assert "DIM = 2 1 0 0 2 0 0 0 2" in (tmp_path / "band.conf").read_text()


def test_band_conf_joins_only_contiguous_path_segments(tmp_path: Path):
    segments = [
        np.asarray([[0, 0, 0], [0.5, 0, 0]]),
        np.asarray([[0.5, 0, 0], [1 / 3, 1 / 3, 0]]),
        np.asarray([[0, 0, 0.5], [0.5, 0, 0.5]]),
    ]
    labels = [("Γ", "M"), ("M", "K"), ("A", "L")]
    write_phonon_inputs(
        tmp_path, segments, labels, [2, 2, 2], 21, [11, 11, 11], [1.008]
    )
    config = (tmp_path / "band.conf").read_text()

    band_line = next(line for line in config.splitlines() if line.startswith("BAND ="))
    assert band_line.count(",") == 1
    assert "0 0 0 0.5 0 0 0.3333333333 0.3333333333 0" in band_line
    assert "BAND_LABELS = GAMMA M K A L" in config
