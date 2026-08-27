from argparse import Namespace
from pathlib import Path

import pytest

from symfc_vasp.cli import validate_fit_contract


def test_fit_contract_requires_a_trajectory_or_saved_dataset():
    args = Namespace(
        trajectory=None,
        dataset_npz=None,
        unitcell=None,
        supercell=None,
        dim=None,
        reference_mode="auto",
        output=None,
    )
    with pytest.raises(ValueError, match="provide OUTCAR"):
        validate_fit_contract(args)


def test_outcar_only_contract_defaults_to_trajectory_reference_and_current_directory_output(tmp_path: Path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text("placeholder")
    args = Namespace(
        trajectory=outcar,
        dataset_npz=None,
        unitcell=None,
        supercell=None,
        dim=None,
        reference_mode="auto",
        output=None,
    )
    validate_fit_contract(args)


def test_provided_contract_requires_both_poscars_but_not_dim(tmp_path: Path):
    trajectory = tmp_path / "OUTCAR"
    unitcell = tmp_path / "POSCAR-unitcell"
    supercell = tmp_path / "POSCAR"
    for path in (trajectory, unitcell, supercell):
        path.write_text("placeholder")
    args = Namespace(
        trajectory=trajectory,
        dataset_npz=None,
        unitcell=unitcell,
        supercell=supercell,
        dim=None,
        reference_mode="provided",
        output=tmp_path / "fit",
    )
    validate_fit_contract(args)


def test_unitcell_only_contract_is_valid_and_supercell_only_is_rejected(tmp_path: Path):
    trajectory = tmp_path / "OUTCAR"
    unitcell = tmp_path / "POSCAR-unitcell"
    supercell = tmp_path / "POSCAR"
    for path in (trajectory, unitcell, supercell):
        path.write_text("placeholder")
    args = Namespace(
        trajectory=trajectory,
        dataset_npz=None,
        unitcell=unitcell,
        supercell=None,
        dim=None,
        reference_mode="auto",
        output=tmp_path / "fit",
    )
    validate_fit_contract(args)
    args.unitcell = None
    args.supercell = supercell
    with pytest.raises(ValueError, match="--supercell requires --unitcell"):
        validate_fit_contract(args)
