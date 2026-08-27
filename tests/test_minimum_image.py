import itertools

import numpy as np

from symfc_vasp.engine import center_periodic_trajectory, minimum_image_displacements


def brute_force(delta, cell):
    candidates = [
        (delta + np.asarray(shift, dtype=float)) @ cell
        for shift in itertools.product(range(-2, 3), repeat=3)
    ]
    return min(candidates, key=lambda vector: np.dot(vector, vector))


def test_hexagonal_cell_uses_cartesian_shortest_image():
    cell = np.array(
        [
            [12.8, 0.0, 0.0],
            [-6.4, 11.085125, 0.0],
            [0.0, 0.0, 11.3],
        ]
    )
    delta = np.array([0.49, -0.49, 0.0])
    old_component_wrapped = (delta - np.rint(delta)) @ cell
    result = minimum_image_displacements(delta, cell)
    expected = brute_force(delta, cell)

    assert np.linalg.norm(old_component_wrapped) > 10.0
    assert np.allclose(result, expected)
    assert np.linalg.norm(result) < 6.5


def test_minimum_image_supports_batched_vectors():
    cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.5, 0.0], [0.2, 0.1, 5.0]])
    deltas = np.array([[0.49, -0.49, 0.1], [-0.48, 0.47, -0.49]])
    result = minimum_image_displacements(deltas, cell)
    expected = np.array([brute_force(delta, cell) for delta in deltas])
    assert np.allclose(result, expected)


def test_periodic_center_handles_samples_across_boundary():
    cell = np.diag([10.0, 10.0, 10.0])
    frames = np.array([[[0.98, 0.2, 0.3]], [[0.99, 0.2, 0.3]], [[0.01, 0.2, 0.3]], [[0.02, 0.2, 0.3]]])
    reference = np.array([[0.5, 0.2, 0.3]])
    displacements, shift, iterations = center_periodic_trajectory(frames, reference, cell)

    assert iterations <= 3
    assert np.allclose(displacements.mean(axis=0), 0.0, atol=1e-12)
    assert np.max(np.abs(displacements)) <= 0.2 + 1e-12
    assert np.allclose(np.linalg.norm(shift, axis=1), 5.0)
