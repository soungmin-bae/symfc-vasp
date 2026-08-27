import numpy as np
import pytest
from phonopy.structure.atoms import PhonopyAtoms

from symfc_vasp.engine import (
    apply_mass_overrides,
    parse_atom_mass_overrides,
    parse_mass_overrides,
)


def test_mass_override_is_applied_by_element_symbol():
    unit = PhonopyAtoms(
        symbols=["Co", "H", "H", "C", "N"],
        cell=np.eye(3),
        scaled_positions=np.zeros((5, 3)),
    )
    summary = apply_mass_overrides(unit, parse_mass_overrides(["H", "2.014"]))
    assert np.allclose(unit.masses[1:3], 2.014)
    assert summary["effective_by_species_amu"]["H"] == 2.014
    assert summary["original_by_species_amu"]["H"] != 2.014


@pytest.mark.parametrize("values", [["H"], ["H", "-1"], ["H", "nan"]])
def test_invalid_mass_override_is_rejected(values):
    with pytest.raises(ValueError):
        parse_mass_overrides(values)


def test_unknown_mass_override_symbol_is_rejected():
    unit = PhonopyAtoms(
        symbols=["H"], cell=np.eye(3), scaled_positions=np.zeros((1, 3))
    )
    with pytest.raises(ValueError, match="not present"):
        apply_mass_overrides(unit, {"D": 2.014})


def test_atom_index_mass_override_takes_precedence_over_symbol():
    unit = PhonopyAtoms(
        symbols=["H", "H"], cell=np.eye(3), scaled_positions=np.zeros((2, 3))
    )
    indexed = parse_atom_mass_overrides(["2", "3.016"])
    summary = apply_mass_overrides(unit, {"H": 2.014}, indexed)
    assert np.allclose(unit.masses, [2.014, 3.016])
    assert summary["atom_overrides_amu"] == {2: 3.016}


@pytest.mark.parametrize("values", [["0", "2.0"], ["1"], ["1", "nan"]])
def test_invalid_atom_index_mass_override_is_rejected(values):
    with pytest.raises(ValueError):
        parse_atom_mass_overrides(values)
