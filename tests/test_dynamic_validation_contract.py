import numpy as np


def test_uniform_selection_contract_is_monotonic_and_unique():
    indices = np.linspace(5000, 44999, 3000, dtype=int)
    assert len(indices) == 3000
    assert indices[0] == 5000
    assert indices[-1] == 44999
    assert len(np.unique(indices)) == len(indices)
    assert set(np.diff(indices)) == {13, 14}

