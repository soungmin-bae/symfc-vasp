import numpy as np

from symfc_vasp.engine import force_metrics


def test_force_metrics_reports_undefined_r2_for_zero_force_variance():
    displacements = np.zeros((1, 1, 3))
    forces = np.zeros((1, 1, 3))
    fc2 = np.zeros((1, 1, 3, 3))

    metrics = force_metrics(displacements, forces, fc2, None, max_samples=5)

    assert metrics["r2"] is None
    assert metrics["r2_status"] == "undefined: validation force variance is zero"
