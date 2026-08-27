"""Crystal-system-aware projections for directional Gruneisen tensors.

Raw phono3py tensors are Cartesian tensors in the input-cell frame.  This
module defines a reproducible orthonormal frame from spglib's standard
conventional lattice before choosing reported tensor components.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import spglib
import yaml


_SYSTEM_BY_NUMBER = (
    (2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
    (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"), (230, "cubic"),
)


def crystal_system(number: int) -> str:
    """Return the crystallographic system for an international SG number."""
    for upper, name in _SYSTEM_BY_NUMBER:
        if number <= upper:
            return name
    raise ValueError(f"Invalid space-group number: {number}")


def component_keys(system: str) -> list[str]:
    """Choose the smallest physically useful diagonal/shear component set."""
    if system == "cubic":
        return ["hydro"]
    if system in {"tetragonal", "trigonal", "hexagonal"}:
        return ["ab", "c"]
    if system == "orthorhombic":
        return ["a", "b", "c"]
    return ["a", "b", "c", "shear_ab", "shear_bc", "shear_ca"]


def _standard_frame(lattice: np.ndarray) -> np.ndarray:
    """Build a right-handed orthonormal a,b,c frame from a standard lattice."""
    a, b, c = np.asarray(lattice, dtype=float)
    e_a = a / np.linalg.norm(a)
    e_c = np.cross(a, b)
    e_c /= np.linalg.norm(e_c)
    if np.dot(e_c, c) < 0:
        e_c *= -1
    e_b = np.cross(e_c, e_a)
    return np.column_stack((e_a, e_b, e_c))


def component_config(unitcell, symprec: float) -> dict:
    """Return the conventional-frame component contract for a phonopy cell."""
    structure = (
        np.asarray(unitcell.cell, dtype=float),
        np.asarray(unitcell.scaled_positions, dtype=float),
        np.asarray(unitcell.numbers, dtype=int),
    )
    dataset = spglib.get_symmetry_dataset(structure, symprec=symprec)
    if dataset is None:
        raise ValueError("spglib could not determine symmetry for component projection")
    number = int(dataset.number)
    system = crystal_system(number)
    frame = _standard_frame(np.asarray(dataset.std_lattice, dtype=float))
    return {
        "schema": "symfc-vasp-crystal-components-v1",
        "symmetry": {
            "international": str(dataset.international),
            "number": number,
            "crystal_system": system,
            "symprec": float(symprec),
        },
        "frame": {
            "definition": "right-handed orthonormal frame derived from spglib standard conventional lattice",
            "standard_lattice_A": np.asarray(dataset.std_lattice, dtype=float).tolist(),
            "basis_columns_cartesian": frame.tolist(),
        },
        "components": component_keys(system),
    }


def write_component_config(output: Path, unitcell, symprec: float) -> dict:
    """Write ``analysis_frame.yaml`` consumed by portable analysis plotters."""
    config = component_config(unitcell, symprec)
    with (output / "analysis_frame.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config
