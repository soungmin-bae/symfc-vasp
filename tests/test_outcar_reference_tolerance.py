from symfc_vasp.outcar_reference import _symprec_grid


def test_default_random_displacement_symmetry_scan_reaches_03_angstrom():
    grid = _symprec_grid(0.3)
    assert 0.15 in grid
    assert 0.2 in grid
    assert 0.3 in grid
