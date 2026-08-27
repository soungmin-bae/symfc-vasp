import numpy as np
import pytest
import yaml

from symfc_vasp.engine import (
    _guard_analysis_output,
    _guard_fit_output,
    force_constant_symmetry_diagnostics,
    sha256,
    validate_fit_dataset,
)


def test_fit_dataset_rejects_one_frame_and_zero_variance():
    with pytest.raises(ValueError, match="at least two"):
        validate_fit_dataset(np.zeros((1, 2, 3)), np.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="zero numerical variance"):
        validate_fit_dataset(np.zeros((2, 2, 3)), np.ones((2, 2, 3)))


def test_force_constant_permutation_diagnostics_detect_error():
    fc2 = np.zeros((2, 2, 3, 3))
    fc2[0, 1, 0, 1] = 1.0
    diagnostics = force_constant_symmetry_diagnostics(fc2, None)
    assert diagnostics["fc2_permutation_max_abs"] == 1.0


def test_force_constant_permutation_diagnostics_accept_symmetric_tensors():
    fc2 = np.zeros((2, 2, 3, 3))
    fc3 = np.zeros((2, 2, 2, 3, 3, 3))
    diagnostics = force_constant_symmetry_diagnostics(fc2, fc3)
    assert diagnostics["fc2_permutation_max_abs"] == 0.0
    assert diagnostics["fc3_permutation_sample_max_abs"] == 0.0


def test_input_hash_guards_reject_mixed_flat_outputs(tmp_path):
    old = tmp_path / "old.OUTCAR"
    new = tmp_path / "new.OUTCAR"
    old.write_text("old")
    new.write_text("new")
    (tmp_path / "symfc_summary.yaml").write_text(
        yaml.safe_dump({"trajectory": {"sha256": sha256(old)}})
    )
    with pytest.raises(FileExistsError, match="different input trajectory"):
        _guard_fit_output(tmp_path, new, force=False)
    _guard_fit_output(tmp_path, new, force=True)

    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    (fit_dir / "FORCE_CONSTANTS").write_text("new-fc")
    (tmp_path / "phonon_summary.yaml").write_text(
        yaml.safe_dump({"inputs": {"FORCE_CONSTANTS_sha256": "old-hash"}})
    )
    with pytest.raises(FileExistsError, match="different FORCE_CONSTANTS"):
        _guard_analysis_output(
            tmp_path, fit_dir, summary_name="phonon_summary.yaml",
            key="FORCE_CONSTANTS_sha256", source_name="FORCE_CONSTANTS", force=False,
        )
