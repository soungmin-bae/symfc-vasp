from pathlib import Path

import numpy as np

from symfc_vasp.parsers.outcar import parse_outcar
from symfc_vasp.parsers.vasprun import parse_vasprun


def test_outcar_and_vasprun_share_numerical_contract(tmp_path: Path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(""" NIONS =      2 ions
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
 0.0 0.0 0.0  1.0 2.0 3.0
 1.0 1.0 1.0 -1.0 -2.0 -3.0
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
 0.2 0.0 0.0  4.0 5.0 6.0
 1.2 1.0 1.0 -4.0 -5.0 -6.0
""")
    xml = tmp_path / "vasprun.xml"
    xml.write_text("""<modeling>
<calculation><structure><crystal><varray name="basis"><v>2 0 0</v><v>0 2 0</v><v>0 0 2</v></varray></crystal>
<varray name="positions"><v>0 0 0</v><v>0.5 0.5 0.5</v></varray></structure>
<varray name="forces"><v>1 2 3</v><v>-1 -2 -3</v></varray></calculation>
<calculation><structure><crystal><varray name="basis"><v>2 0 0</v><v>0 2 0</v><v>0 0 2</v></varray></crystal>
<varray name="positions"><v>0.1 0 0</v><v>0.6 0.5 0.5</v></varray></structure>
<varray name="forces"><v>4 5 6</v><v>-4 -5 -6</v></varray></calculation>
</modeling>""")
    indices = np.array([0, 1])
    text = parse_outcar(outcar, indices)
    structured = parse_vasprun(xml, indices)
    assert np.allclose(text.positions, structured.positions, atol=1e-12)
    assert np.allclose(text.forces, structured.forces, atol=1e-12)

