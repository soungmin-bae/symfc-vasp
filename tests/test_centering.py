import numpy as np


def test_centered_arrays_have_zero_sample_mean():
    rng = np.random.default_rng(7)
    displacements = rng.normal(size=(20, 4, 3)) + 0.4
    forces = rng.normal(size=(20, 4, 3)) - 0.2
    displacements -= displacements.mean(axis=0, keepdims=True)
    forces -= forces.mean(axis=0, keepdims=True)
    assert np.allclose(displacements.mean(axis=0), 0, atol=1e-14)
    assert np.allclose(forces.mean(axis=0), 0, atol=1e-14)

