from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..models import TrajectoryDataset
from .outcar import _select_energy_field


def vasprun_symbols(path: Path) -> tuple[str, ...]:
    """Return the VASP atom order recorded in ``atominfo``."""
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "array" or element.attrib.get("name") != "atoms":
            continue
        symbols = []
        for row in element.findall(".//rc"):
            columns = row.findall("c")
            if columns and columns[0].text:
                symbols.append(columns[0].text.strip())
        element.clear()
        if symbols:
            return tuple(symbols)
    raise ValueError(f"{path} contains no atominfo/atoms records")


def _varray(element, name: str):
    node = element.find(f".//varray[@name='{name}']")
    if node is None:
        return None
    return np.asarray([[float(value) for value in row.text.split()] for row in node.findall("v")])


def _is_standard_calculation_xml(path: Path) -> bool:
    """Distinguish regular VASP MD XML from VASP INTERACTIVE XML cheaply.

    VASP 6 INTERACTIVE runs write a sequence of sibling ``structure`` and
    ``varray name='forces'`` elements instead of wrapping each ionic frame in
    a ``calculation`` element. Both carry the same information, but need
    different streaming boundaries.
    """
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if b"<calculation" in chunk:
                return True
    return False


def _structure_arrays(element):
    scaled = _varray(element, "positions")
    cell = _varray(element, "basis")
    return scaled, cell


def _xml_energy_values(element) -> dict[str, float]:
    values: dict[str, float] = {}
    names = {
        "e_wo_entrp": "energy_without_entropy",
        "e_0_energy": "sigma0",
        "e_fr_energy": "free_energy",
    }
    for node in element.findall(".//i"):
        field = names.get(node.attrib.get("name"))
        if field is not None and node.text:
            values[field] = float(node.text)
    return values


def _interactive_frames(path: Path):
    """Yield frames from VASP 6 MLFF/INTERACTIVE XML without loading it all."""
    pending = None
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "structure":
            scaled, cell = _structure_arrays(element)
            if scaled is not None and cell is not None:
                if pending is not None and pending[1] is not None:
                    yield pending
                pending = [scaled, None, cell, {}]
            element.clear()
        elif element.tag == "varray" and element.attrib.get("name") == "forces":
            if pending is not None:
                force = np.asarray(
                    [[float(value) for value in row.text.split()] for row in element.findall("v")]
                )
                if force.shape == pending[0].shape:
                    pending[1] = force
            element.clear()
        elif element.tag == "energy" and pending is not None:
            pending[3].update(_xml_energy_values(element))
            element.clear()
    if pending is not None and pending[1] is not None:
        yield pending


def count_vasprun_frames(path: Path) -> int:
    if not _is_standard_calculation_xml(path):
        return sum(1 for _ in _interactive_frames(path))
    count = 0
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "calculation":
            positions = _varray(element, "positions")
            forces = _varray(element, "forces")
            if positions is not None and forces is not None:
                count += 1
            element.clear()
    return count


def parse_vasprun(
    path: Path,
    indices: np.ndarray,
    *,
    cell_tolerance: float = 1e-6,
    energy_field: str = "auto",
) -> TrajectoryDataset:
    wanted = {int(index): slot for slot, index in enumerate(indices)}
    positions = forces = cells = None
    iframe = 0
    found = np.zeros(len(indices), dtype=bool)
    energy_records: list[dict[str, float]] = [{} for _ in indices]
    if _is_standard_calculation_xml(path):
        frames = []
        for _, element in ET.iterparse(path, events=("end",)):
            if element.tag != "calculation":
                continue
            scaled = _varray(element, "positions")
            force = _varray(element, "forces")
            cell = _varray(element, "basis")
            if scaled is not None and force is not None:
                frames.append((scaled, force, cell, _xml_energy_values(element)))
            element.clear()
    else:
        frames = _interactive_frames(path)
    reference_cell = None
    for scaled, force, cell, energy_values in frames:
        if cell is None:
            raise ValueError(f"calculation {iframe} has no crystal basis")
        if reference_cell is None:
            reference_cell = np.asarray(cell, dtype=float)
        else:
            deviation = float(np.max(np.abs(cell - reference_cell)))
            if deviation > cell_tolerance:
                deformation = cell @ np.linalg.inv(reference_cell) - np.eye(3)
                volume_change = float(
                    np.linalg.det(cell) / np.linalg.det(reference_cell) - 1.0
                )
                raise ValueError(
                    f"variable-cell trajectory detected in {path.name}: frame {iframe} "
                    f"differs from frame 0 by {deviation:.6g} A "
                    f"(max deformation={np.max(np.abs(deformation)):.6g}, "
                    f"relative volume change={volume_change:.6g}). "
                    "symfc-vasp accepts fixed-cell NVT or fixed-cell IBRION=11 data only"
                )
        slot = wanted.get(iframe)
        if slot is not None:
            if positions is None:
                positions = np.empty((len(indices), len(scaled), 3))
                forces = np.empty_like(positions)
                cells = np.empty((len(indices), 3, 3))
            positions[slot] = scaled @ cell
            forces[slot] = force
            cells[slot] = cell
            energy_records[slot].update(energy_values)
            found[slot] = True
        iframe += 1
    if iframe == 0:
        raise ValueError(f"{path} contains no calculation blocks with both positions and forces")
    if positions is None or not found.all():
        raise ValueError(f"requested frames are absent from {path}; usable frames={iframe}")
    energies, selected_field, metadata = _select_energy_field(
        energy_records, energy_field
    )
    result = TrajectoryDataset(
        positions,
        forces,
        cells,
        indices.copy(),
        path,
        "vasp-xml",
        energies=energies,
        energy_field=selected_field,
        energy_metadata=metadata,
    )
    result.validate()
    return result
