from pathlib import Path

from .outcar import parse_outcar
from .vasprun import parse_vasprun


def parse_trajectory(
    path: Path,
    indices,
    *,
    cell_tolerance: float = 1e-6,
    energy_field: str = "auto",
):
    name = path.name.lower()
    if name == "outcar" or name.endswith(".outcar"):
        return parse_outcar(path, indices, energy_field=energy_field)
    if name.endswith(".xml"):
        return parse_vasprun(
            path,
            indices,
            cell_tolerance=cell_tolerance,
            energy_field=energy_field,
        )
    raise ValueError(f"cannot infer trajectory format from {path.name}; use OUTCAR or vasprun.xml")
