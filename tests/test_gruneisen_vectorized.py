import numpy as np

from symfc_vasp.gruneisen import vectorized_expectation


def test_vectorized_gruneisen_expectation_matches_explicit_loops():
    rng = np.random.default_rng(7)
    natom = 3
    modes = 3 * natom
    d_d_du = rng.normal(size=(natom, natom, 3, 3, 3, 3)) + 1j * rng.normal(
        size=(natom, natom, 3, 3, 3, 3)
    )
    eigenvectors = rng.normal(size=(3 * natom, modes)) + 1j * rng.normal(
        size=(3 * natom, modes)
    )
    expected = np.zeros((modes, 3, 3))
    for mode in range(modes):
        for strain_i in range(3):
            for strain_j in range(3):
                for atom_i in range(natom):
                    for atom_j in range(natom):
                        for cart_i in range(3):
                            for cart_j in range(3):
                                expected[mode, strain_i, strain_j] += (
                                    eigenvectors[atom_i * 3 + cart_i, mode].conjugate()
                                    * d_d_du[atom_i, atom_j, cart_i, cart_j, strain_i, strain_j]
                                    * eigenvectors[atom_j * 3 + cart_j, mode]
                                ).real
    assert np.allclose(vectorized_expectation(d_d_du, eigenvectors), expected)
