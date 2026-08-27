from pathlib import Path

import numpy as np
import pytest

from symfc_vasp.parsers.outcar import parse_outcar, parse_outcar_metadata
from symfc_vasp.parsers.vasprun import parse_vasprun


def test_outcar_rejects_variable_cell_history(tmp_path: Path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        """ VRHFIN =Si:
 ions per type = 1
 direct lattice vectors                 reciprocal lattice vectors
 2.0 0.0 0.0  0.5 0.0 0.0
 0.0 2.0 0.0  0.0 0.5 0.0
 0.0 0.0 2.0  0.0 0.0 0.5
 direct lattice vectors                 reciprocal lattice vectors
 2.1 0.0 0.0  0.5 0.0 0.0
 0.0 2.0 0.0  0.0 0.5 0.0
 0.0 0.0 2.0  0.0 0.0 0.5
"""
    )

    with pytest.raises(ValueError, match="variable-cell trajectory detected"):
        parse_outcar_metadata(outcar)


def test_outcar_accepts_mlff_force_blocks(tmp_path: Path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        """ NIONS = 1 ions
 POSITION                                       TOTAL-FORCE (eV/Angst) (ML)
 -----------------------------------------------------------------------------------
 0.1 0.2 0.3  1.0 2.0 3.0
"""
    )

    data = parse_outcar(outcar, np.array([0]))
    np.testing.assert_allclose(data.positions[0, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(data.forces[0, 0], [1.0, 2.0, 3.0])


def test_vasprun_rejects_cell_change_outside_selected_frames(tmp_path: Path):
    xml = tmp_path / "vasprun.xml"
    xml.write_text(
        """<modeling>
<calculation><structure><crystal><varray name="basis"><v>2 0 0</v><v>0 2 0</v><v>0 0 2</v></varray></crystal>
<varray name="positions"><v>0 0 0</v></varray></structure>
<varray name="forces"><v>0 0 0</v></varray></calculation>
<calculation><structure><crystal><varray name="basis"><v>2.1 0 0</v><v>0 2 0</v><v>0 0 2</v></varray></crystal>
<varray name="positions"><v>0 0 0</v></varray></structure>
<varray name="forces"><v>0 0 0</v></varray></calculation>
</modeling>"""
    )

    with pytest.raises(ValueError, match="variable-cell trajectory detected"):
        parse_vasprun(xml, np.array([0]))


def test_vasprun_accepts_fixed_cell_with_roundoff(tmp_path: Path):
    xml = tmp_path / "vasprun.xml"
    xml.write_text(
        """<modeling>
<calculation><structure><crystal><varray name="basis"><v>2 0 0</v><v>0 2 0</v><v>0 0 2</v></varray></crystal>
<varray name="positions"><v>0 0 0</v></varray></structure>
<varray name="forces"><v>0 0 0</v></varray></calculation>
<calculation><structure><crystal><varray name="basis"><v>2.0000001 0 0</v><v>0 2 0</v><v>0 0 2</v></varray></crystal>
<varray name="positions"><v>0 0 0</v></varray></structure>
<varray name="forces"><v>0 0 0</v></varray></calculation>
</modeling>"""
    )

    data = parse_vasprun(xml, np.array([0, 1]))
    assert data.positions.shape == (2, 1, 3)
