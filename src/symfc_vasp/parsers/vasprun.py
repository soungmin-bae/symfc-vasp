from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..models import TrajectoryDataset


def _varray(element, name: str):
    node = element.find(f".//varray[@name='{name}']")
    if node is None:
        return None
    return np.asarray([[float(value) for value in row.text.split()] for row in node.findall("v")])


def count_vasprun_frames(path: Path) -> int:
    count = 0
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "calculation":
            positions = _varray(element, "positions")
            forces = _varray(element, "forces")
            if positions is not None and forces is not None:
                count += 1
            element.clear()
    return count


def parse_vasprun(path: Path, indices: np.ndarray) -> TrajectoryDataset:
    wanted = {int(index): slot for slot, index in enumerate(indices)}
    positions = forces = cells = None
    iframe = 0
    found = np.zeros(len(indices), dtype=bool)
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "calculation":
            continue
        scaled = _varray(element, "positions")
        force = _varray(element, "forces")
        cell = _varray(element, "basis")
        if scaled is not None and force is not None:
            if cell is None:
                raise ValueError(f"calculation {iframe} has no crystal basis")
            slot = wanted.get(iframe)
            if slot is not None:
                if positions is None:
                    positions = np.empty((len(indices), len(scaled), 3))
                    forces = np.empty_like(positions)
                    cells = np.empty((len(indices), 3, 3))
                positions[slot] = scaled @ cell
                forces[slot] = force
                cells[slot] = cell
                found[slot] = True
            iframe += 1
        element.clear()
    if iframe == 0:
        raise ValueError(f"{path} contains no calculation blocks with both positions and forces")
    if positions is None or not found.all():
        raise ValueError(f"requested frames are absent from {path}; usable frames={iframe}")
    result = TrajectoryDataset(positions, forces, cells, indices.copy(), path, "vasp-xml")
    result.validate()
    return result

