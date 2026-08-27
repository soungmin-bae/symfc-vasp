import numpy as np

from symfc_vasp.gruneisen import vectorized_dphidu


def test_vectorized_dphidu_matches_phono3py_loop_expression():
    rng = np.random.default_rng(12)
    fc3 = rng.normal(size=(4, 4, 4, 3, 3, 3))
    p2s = np.array([1, 3])
    ys = [rng.normal(size=(4, 3, 3, 3)) for _ in p2s]
    actual = vectorized_dphidu(fc3, p2s, lambda nu: ys[nu])
    expected = np.zeros_like(actual)
    for nu, super_index in enumerate(p2s):
        for pi in range(4):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for ell in range(3):
                            expected[nu, pi, i, j, k, ell] = (
                                fc3[super_index, pi, :, i, j, :] * ys[nu][:, :, k, ell]
                            ).sum()
    np.testing.assert_allclose(actual, expected)
