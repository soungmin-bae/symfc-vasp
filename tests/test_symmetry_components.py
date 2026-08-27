import numpy as np

from symfc_vasp.symmetry_components import component_keys, crystal_system


def test_component_policy_covers_all_crystal_systems():
    assert crystal_system(1) == "triclinic"
    assert crystal_system(15) == "monoclinic"
    assert crystal_system(62) == "orthorhombic"
    assert crystal_system(141) == "tetragonal"
    assert crystal_system(167) == "trigonal"
    assert crystal_system(194) == "hexagonal"
    assert crystal_system(221) == "cubic"
    assert component_keys("cubic") == ["hydro"]
    assert component_keys("hexagonal") == ["ab", "c"]
    assert component_keys("orthorhombic") == ["a", "b", "c"]
    assert component_keys("monoclinic")[-3:] == ["shear_ab", "shear_bc", "shear_ca"]


def test_standard_frame_is_orthonormal_for_skew_lattice():
    from symfc_vasp.symmetry_components import _standard_frame

    frame = _standard_frame(np.array([[4.0, 0, 0], [-2.0, 3.0, 0], [0.4, 0.1, 5.0]]))
    assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-12)
