# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Soungmin Bae
# Portions are adapted from phono3py's Gruneisen implementation.
# Copyright (c) 2015-2024, Phonopy. All rights reserved.
# See LICENSES/phono3py-BSD-3-Clause.txt.

"""Accelerated phono3py Gruneisen tensor evaluation.

phono3py 4.x evaluates the expectation value of dD/du with five nested Python
loops.  The tensor contraction is independent across phonon modes and can be
expressed exactly as one NumPy Einstein contraction.  This module preserves
phono3py's dynamical-matrix, NAC, and FC3 handling while replacing only that
algebraic bottleneck.
"""

from __future__ import annotations

import numpy as np


def vectorized_expectation(d_d_du: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """Return ``<e_s|dD/du_ij|e_s>`` for all modes and strain components.

    Parameters follow phono3py's internal convention: ``d_d_du`` has shape
    ``(atom_i, atom_j, cart_i, cart_j, strain_i, strain_j)`` and
    eigenvectors have shape ``(3*natom, mode)``.
    """
    natom = d_d_du.shape[0]
    modes = eigenvectors.reshape(natom, 3, -1)
    return np.einsum(
        "ais,abijuv,bjs->suv",
        modes.conj(),
        d_d_du,
        modes,
        optimize=True,
    ).real


def vectorized_dphidu(fc3: np.ndarray, p2s: np.ndarray, get_y, *, progress=None) -> np.ndarray:
    """Build phono3py's FC3 strain-derivative tensor without Cartesian loops.

    ``Gruneisen._get_dPhidu`` normally evaluates this contraction with seven
    nested Python loops. The expression is exactly one contraction for each
    primitive atom. ``progress`` receives completed and total primitive atoms.
    """
    nprimitive = len(p2s)
    nsuper = fc3.shape[1]
    result = np.empty((nprimitive, nsuper, 3, 3, 3, 3), dtype=float)
    for nu, super_index in enumerate(p2s):
        y = get_y(nu)
        # upstream: sum_{q,r} fc3[s,p,q,i,j,r] * Y[q,r,k,l]
        result[nu] = np.einsum(
            "pqijr,qrkl->pijkl", fc3[super_index], y, optimize=True
        )
        if progress is not None:
            progress(nu + 1, nprimitive)
    return result


def accelerated_gruneisen_class():
    """Build a phono3py-compatible Gruneisen class with vectorized tensors."""
    from phono3py.phonon3.gruneisen import Gruneisen

    class VectorizedGruneisen(Gruneisen):
        def _get_dPhidu(self):
            def report(done: int, total: int) -> None:
                print(
                    f"[gruneisen] FC3 strain-derivative tensor: primitive atom {done}/{total}",
                    flush=True,
                )

            return vectorized_dphidu(
                self._fc3, self._pcell.p2s_map, self._get_Y, progress=report
            )

        def _calculate_at_qpoints(self, qpoints):
            """Upstream calculation with periodic q-point progress reporting."""
            from phonopy.harmonic.dynamical_matrix import DynamicalMatrixNAC

            parameters = []
            frequencies = []
            total = len(qpoints)
            interval = max(1, total // 10)
            label = "band" if self._run_mode == "band" else "mesh"
            for index, q in enumerate(qpoints):
                if index == 0 or (index + 1) % interval == 0 or index + 1 == total:
                    print(f"[gruneisen] {label} q point {index + 1}/{total}", flush=True)
                if isinstance(self._dm, DynamicalMatrixNAC):
                    if (np.abs(q) < 1e-5).all():
                        if self._run_mode == "band":
                            if index > 0:
                                q_direction = qpoints[index] - qpoints[index - 1]
                            elif total > 1:
                                q_direction = qpoints[index + 1] - qpoints[index]
                            else:
                                q_direction = None
                            gamma, omega2 = self._get_gruneisen_tensor(q, nac_q_direction=q_direction)
                        else:
                            gamma, omega2 = self._get_gruneisen_tensor(
                                q, nac_q_direction=self._nac_q_direction
                            )
                    else:
                        gamma, omega2 = self._get_gruneisen_tensor(q, nac_q_direction=q)
                else:
                    gamma, omega2 = self._get_gruneisen_tensor(q)
                parameters.append(gamma)
                frequencies.append(np.sqrt(abs(omega2)) * np.sign(omega2) * self._factor)
            return (
                np.array(parameters, dtype="double", order="C"),
                np.array(frequencies, dtype="double", order="C"),
            )

        def _get_gruneisen_tensor(self, q, nac_q_direction=None):
            if nac_q_direction is None:
                self._dm.run(q)
            else:
                self._dm.run(q, nac_q_direction)
            assert self._dm.dynamical_matrix is not None
            omega2, eigenvectors = np.linalg.eigh(self._dm.dynamical_matrix)
            expectation = vectorized_expectation(self._get_dDdu(q), eigenvectors)
            gamma = np.zeros_like(expectation)
            nonzero = np.abs(omega2) > np.finfo(float).eps
            gamma[nonzero] = -0.5 * expectation[nonzero] / omega2[nonzero, None, None]
            if (np.abs(q) < 1e-5).all():
                gamma[:3] = 0.0
            return gamma, omega2

    return VectorizedGruneisen
