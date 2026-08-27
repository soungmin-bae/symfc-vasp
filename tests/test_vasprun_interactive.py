from pathlib import Path

import numpy as np

from symfc_vasp.parsers.vasprun import count_vasprun_frames, parse_vasprun


def test_vasp6_interactive_sibling_structure_and_forces(tmp_path: Path):
    path = tmp_path / "vasprun.xml"
    path.write_text(
        """<modeling>
<structure><crystal><varray name="basis"><v>1 0 0</v><v>0 1 0</v><v>0 0 1</v></varray></crystal><varray name="positions"><v>0 0 0</v></varray></structure>
<varray name="forces"><v>1 2 3</v></varray>
<structure><crystal><varray name="basis"><v>1 0 0</v><v>0 1 0</v><v>0 0 1</v></varray></crystal><varray name="positions"><v>0.5 0 0</v></varray></structure>
<varray name="forces"><v>4 5 6</v></varray>
</modeling>"""
    )
    assert count_vasprun_frames(path) == 2
    data = parse_vasprun(path, np.array([0, 1]))
    assert np.allclose(data.positions[:, 0, 0], [0, 0.5])
    assert np.allclose(data.forces[:, 0], [[1, 2, 3], [4, 5, 6]])
