import numpy as np


def test_saved_dataset_contains_required_restart_arrays(tmp_path):
    path = tmp_path / "symfc_input.npz"
    np.savez_compressed(
        path,
        displacements=np.zeros((3, 2, 3)),
        forces=np.ones((3, 2, 3)),
        source_indices=np.array([10, 20, 30]),
        cell=np.eye(3),
    )
    saved = np.load(path)
    assert {"displacements", "forces", "source_indices", "cell"} <= set(saved.files)

