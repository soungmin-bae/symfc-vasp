import numpy as np
import pytest

from symfc_vasp.selection import select_indices


def test_validation_stride_contract():
    indices = select_indices(20000, skip=5000, samples=3000)
    assert len(indices) == 3000
    assert indices[0] == 5000
    assert indices[-1] == 19995
    assert np.all(np.diff(indices) == 5)


def test_inconsistent_stride_is_rejected():
    with pytest.raises(ValueError, match="not requested"):
        select_indices(20000, skip=5000, samples=3000, stride=4)

