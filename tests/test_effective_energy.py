from pathlib import Path

import numpy as np
import yaml

from symfc_vasp.effective_energy import (
    calculate_effective_energy_offset,
    effective_energy_statistics,
    harmonic_energies,
)


def test_harmonic_offset_is_recovered_exactly(tmp_path: Path):
    rng = np.random.default_rng(4)
    displacements = rng.normal(scale=0.03, size=(200, 2, 3))
    matrix = rng.normal(size=(6, 6))
    matrix = matrix.T @ matrix
    fc2 = matrix.reshape(2, 3, 2, 3).transpose(0, 2, 1, 3)
    harmonic = harmonic_energies(displacements, fc2)
    energies = -17.25 + harmonic
    result = calculate_effective_energy_offset(
        output=tmp_path,
        displacements=displacements,
        energies=energies,
        fc2=fc2,
        source_indices=np.arange(1000, 1200),
        energy_field="energy_without_entropy",
        energy_metadata={"includes_ionic_kinetic_energy": False},
        primitive_cells_per_supercell=2,
        bootstrap_samples=20,
    )
    assert np.isclose(
        result["effective_energy_offset"]["value_eV_supercell"], -17.25, atol=1e-12
    )
    assert np.isclose(
        result["effective_energy_offset"]["value_eV_primitive_cell"], -8.625
    )
    assert (tmp_path / "tdep_energy_residuals.tsv").is_file()
    assert (tmp_path / "tdep_energy_diagnostics.pdf").is_file()
    saved = yaml.safe_load((tmp_path / "tdep_energy_offset.yaml").read_text())
    assert saved["statistics"]["num_snapshots"] == 200


def test_constant_potential_shift_changes_only_the_offset(tmp_path: Path):
    displacements = np.linspace(-0.1, 0.1, 60).reshape(20, 1, 3)
    fc2 = np.eye(3).reshape(1, 1, 3, 3)
    harmonic = harmonic_energies(displacements, fc2)
    first = calculate_effective_energy_offset(
        output=tmp_path,
        displacements=displacements,
        energies=harmonic + 2.0,
        fc2=fc2,
        source_indices=np.arange(20),
        energy_field="free_energy",
        energy_metadata={},
        primitive_cells_per_supercell=1,
        bootstrap_samples=0,
    )
    second_dir = tmp_path / "shifted"
    second_dir.mkdir()
    second = calculate_effective_energy_offset(
        output=second_dir,
        displacements=displacements,
        energies=harmonic + 7.0,
        fc2=fc2,
        source_indices=np.arange(20),
        energy_field="free_energy",
        energy_metadata={},
        primitive_cells_per_supercell=1,
        bootstrap_samples=0,
    )
    assert np.isclose(
        second["effective_energy_offset"]["value_eV_supercell"]
        - first["effective_energy_offset"]["value_eV_supercell"],
        5.0,
    )


def test_correlated_drifting_two_component_series_is_warned_not_split():
    rng = np.random.default_rng(7)
    first = np.repeat(-1.0, 1000) + rng.normal(scale=0.05, size=1000)
    second = np.repeat(1.0, 1000) + rng.normal(scale=0.05, size=1000)
    values = np.concatenate((first, second))
    statistics, warnings = effective_energy_statistics(
        values, np.arange(len(values)), bootstrap_samples=20
    )
    assert statistics["effective_sample_size"] < len(values)
    assert statistics["two_gaussian_diagnostic"]["descriptive_only"] is True
    assert any("non-stationary" in warning for warning in warnings)
    assert any("no automatic basin splitting" in warning for warning in warnings)
