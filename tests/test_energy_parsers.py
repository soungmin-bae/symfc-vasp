import lzma
import shutil
from pathlib import Path

import numpy as np
import pytest

from symfc_vasp.parsers.outcar import parse_outcar
from symfc_vasp.parsers.vasprun import parse_vasprun


def test_ml_outcar_energy_is_paired_with_selected_force_frames(tmp_path: Path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        " NIONS = 1 ions\n"
        " POSITION TOTAL-FORCE (eV/Angst) (ML)\n---\n0 0 0 1 2 3\n"
        " ML energy  without entropy= -10.25 ML energy(sigma->0) = -10.20\n"
        " POSITION TOTAL-FORCE (eV/Angst) (ML)\n---\n0.1 0 0 4 5 6\n"
        " ML energy  without entropy= -9.75 ML energy(sigma->0) = -9.70\n"
    )
    result = parse_outcar(outcar, np.asarray([1, 0]))
    assert result.energy_field == "ml_energy_without_entropy"
    assert np.allclose(result.energies, [-9.75, -10.25])
    assert np.allclose(result.positions[:, 0, 0], [0.1, 0.0])


def test_requested_outcar_energy_field_must_cover_every_selected_frame(tmp_path: Path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        " NIONS = 1 ions\n"
        " POSITION TOTAL-FORCE (eV/Angst)\n---\n0 0 0 1 2 3\n"
        " energy  without entropy= -2 energy(sigma->0) = -1.9\n"
        " POSITION TOTAL-FORCE (eV/Angst)\n---\n0 0 0 1 2 3\n"
    )
    with pytest.raises(ValueError, match="do not align"):
        parse_outcar(outcar, np.asarray([0, 1]), energy_field="energy_without_entropy")


def test_standard_and_interactive_vasprun_energies(tmp_path: Path):
    standard = tmp_path / "standard.xml"
    standard.write_text(
        "<modeling><calculation><structure><crystal><varray name='basis'>"
        "<v>1 0 0</v><v>0 1 0</v><v>0 0 1</v></varray></crystal>"
        "<varray name='positions'><v>0 0 0</v></varray></structure>"
        "<varray name='forces'><v>1 2 3</v></varray><energy>"
        "<i name='e_fr_energy'>-3.1</i><i name='e_wo_entrp'>-3.0</i>"
        "<i name='e_0_energy'>-2.9</i></energy></calculation></modeling>"
    )
    interactive = tmp_path / "interactive.xml"
    interactive.write_text(
        "<modeling><structure><crystal><varray name='basis'>"
        "<v>1 0 0</v><v>0 1 0</v><v>0 0 1</v></varray></crystal>"
        "<varray name='positions'><v>0 0 0</v></varray></structure>"
        "<varray name='forces'><v>1 2 3</v></varray><energy>"
        "<i name='e_fr_energy'>-4.1</i><i name='e_wo_entrp'>-4.0</i>"
        "<i name='e_0_energy'>-3.9</i></energy></modeling>"
    )
    first = parse_vasprun(standard, np.asarray([0]))
    second = parse_vasprun(interactive, np.asarray([0]))
    assert first.energy_field == second.energy_field == "energy_without_entropy"
    assert np.allclose(first.energies, [-3.0])
    assert np.allclose(second.energies, [-4.0])


def test_bundled_mlff_outcar_exposes_all_selected_energies(tmp_path: Path):
    archive = Path(__file__).parents[1] / "examples" / "CoH3CN6" / "OUTCAR.xz"
    outcar = tmp_path / "OUTCAR"
    with lzma.open(archive, "rb") as source, outcar.open("wb") as target:
        shutil.copyfileobj(source, target)
    selected = np.asarray([0, 50, 199])
    result = parse_outcar(outcar, selected)
    assert result.energies.shape == (3,)
    assert result.energy_field == "ml_energy_without_entropy"
