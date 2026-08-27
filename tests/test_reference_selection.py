from pathlib import Path

import numpy as np

from symfc_vasp.outcar_reference import build_outcar_reference


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
    assert report["schema"] == "symfc-vasp-symmetry-report-v2"
    assert report["selection_policy"]["stable_candidate_used"] is True
    assert all("valid" in candidate for candidate in report["symmetry_scan"])
