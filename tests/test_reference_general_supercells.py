from pathlib import Path

import numpy as np
import pytest
import spglib
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from symfc_vasp.outcar_reference import (
    _minimum_image,
    _periodic_mean,
    _translation_invariant_projection_distances,
    build_outcar_reference,
)


@pytest.fixture
def silicon_primitive() -> PhonopyAtoms:
    """Rhombohedral primitive setting of diamond Si (MP mp-149)."""
    return PhonopyAtoms(
        cell=[
            [3.333573, 0.0, 1.924639],
            [1.111191, 3.142924, 1.924639],
            [0.0, 0.0, 3.849278],
        ],
        scaled_positions=[[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]],
        symbols=["Si", "Si"],
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([2, 3, 7]),
        np.array([[1, -1, 0], [0, 1, 0], [1, 1, 1]]),
    ],
)
def test_reference_recovers_general_supercells_from_noisy_mean(
    tmp_path: Path, silicon_primitive: PhonopyAtoms, matrix: np.ndarray
):
    supercell = Phonopy(
        silicon_primitive, supercell_matrix=matrix, primitive_matrix="P"
    ).supercell
    rng = np.random.default_rng(20260831)
    positions = np.asarray(supercell.positions)[None, :, :] + rng.uniform(
        -0.3, 0.3, size=(200, len(supercell), 3)
    )

    unit, generated, mapping, report = build_outcar_reference(
        outcar=tmp_path / "OUTCAR",
        positions=positions,
        output=tmp_path,
        symprec_max=0.3,
        map_tolerance=1.0,
        symbols=tuple(supercell.symbols),
        cell=np.asarray(supercell.cell),
    )

    dataset = spglib.get_symmetry_dataset(
        (unit.cell, unit.scaled_positions, unit.numbers), symprec=1e-5
    )
    assert dataset is not None
    assert dataset.number == 227
    assert len(unit) == 2
    assert len(generated) == len(supercell)
    assert abs(round(np.linalg.det(report["supercell_matrix"]))) == abs(
        round(np.linalg.det(matrix))
    )
    assert sorted(mapping.tolist()) == list(range(len(supercell)))
    assert report["mapping"]["max_distance_A"] < 0.1
    assert report["symmetry_projection"]["validation_symprec_A"] == 1e-5


def test_periodic_mean_uses_cell_metric_for_skew_basis():
    cell = np.array([[4.0, 0.0, 0.0], [-4.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    parent_frac = np.array([[0.01, 0.99, 0.5], [0.97, 0.02, 0.25]])
    rng = np.random.default_rng(19)
    noise = rng.uniform(-0.3, 0.3, size=(200, len(parent_frac), 3))
    cartesian = parent_frac[None, :, :] @ cell + noise
    frames = np.mod(cartesian @ np.linalg.inv(cell), 1.0)

    mean_frac = _periodic_mean(frames, cell)
    measured = _minimum_image(mean_frac - parent_frac, cell)

    assert np.allclose(measured, np.mean(noise, axis=0), atol=1e-12, rtol=0)


def test_projection_distortion_ignores_equivalent_origin_shift():
    cell = np.array([[5.0, 0.0, 0.0], [-1.5, 4.0, 0.0], [0.2, 0.3, 6.0]])
    reference = np.array([[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]])
    shift = np.array([0.17, -0.11, 0.08])

    distances = _translation_invariant_projection_distances(
        np.mod(reference + shift, 1.0), reference, cell
    )

    assert np.allclose(distances, 0.0, atol=1e-12, rtol=0)
