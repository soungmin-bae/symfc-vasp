from pathlib import Path

import numpy as np

from symfc_vasp.outcar_reference import (
    _affine_operation_subgroups,
    build_outcar_reference,
)


def test_reference_selection_requires_valid_stable_mapping(tmp_path: Path):
    positions = np.asarray([[[0.001, 0.0, 0.0]], [[-0.001, 0.0, 0.0]]])
    unit, generated, mapping, report = build_outcar_reference(
        outcar=tmp_path / "OUTCAR",
        positions=positions,
        output=tmp_path,
        symprec_max=0.01,
        map_tolerance=0.05,
        symbols=("Si",),
        cell=np.eye(3),
    )
    assert len(unit) == len(generated) == 1
    assert mapping.tolist() == [0]
    assert report["schema"] == "symfc-vasp-symmetry-report-v3"
    assert report["primitive_search_success"] is True
    assert report["selection_policy"]["stable_candidate_used"] is True
    assert all("valid" in candidate for candidate in report["symmetry_scan"])


def test_affine_subgroups_separate_centering_translations():
    point_rotations = [
        np.diag(signs)
        for signs in np.asarray(list(np.ndindex(2, 2, 2))) * 2 - 1
    ]
    rotations = np.asarray([
        rotation for rotation in point_rotations for _ in range(2)
    ])
    translations = np.asarray([
        translation
        for _ in point_rotations
        for translation in (np.zeros(3), np.array([0.5, 0.5, 0.0]))
    ])

    subgroups = _affine_operation_subgroups(rotations, translations)
    uncentred_mmm = tuple(range(0, len(rotations), 2))

    assert uncentred_mmm in subgroups
