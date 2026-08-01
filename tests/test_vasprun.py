from pathlib import Path

import numpy as np

from symfc_vasp.parsers.vasprun import parse_vasprun


def test_minimal_vasprun(tmp_path: Path):
    xml = tmp_path / "vasprun.xml"
    xml.write_text("""<modeling>
<calculation><structure><crystal><varray name="basis"><v>2 0 0</v><v>0 2 0</v><v>0 0 2</v></varray></crystal>
<varray name="positions"><v>0 0 0</v><v>0.5 0.5 0.5</v></varray></structure>
<varray name="forces"><v>1 2 3</v><v>-1 -2 -3</v></varray></calculation>
<calculation><structure><crystal><varray name="basis"><v>2 0 0</v><v>0 2 0</v><v>0 0 2</v></varray></crystal>
<varray name="positions"><v>0.1 0 0</v><v>0.6 0.5 0.5</v></varray></structure>
<varray name="forces"><v>4 5 6</v><v>-4 -5 -6</v></varray></calculation>
</modeling>""")
    data = parse_vasprun(xml, np.array([0, 1]))
    assert data.positions.shape == (2, 2, 3)
    assert np.allclose(data.positions[1, 0], [0.2, 0, 0])
    assert np.allclose(data.forces[0, 1], [-1, -2, -3])

