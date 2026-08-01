from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..models import TrajectoryDataset

FORCE_HEADER = re.compile(r"POSITION\s+TOTAL-FORCE\s+\(eV/Angst\)(?:\s+\(ML\))?")


def scan_outcar(path: Path) -> tuple[int, int, int]:
    natom = None
    frames = ml_frames = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            if natom is None and "NIONS" in line:
                match = re.search(r"NIONS\s*=\s*(\d+)", line)
                if match:
                    natom = int(match.group(1))
            if FORCE_HEADER.search(line):
                frames += 1
                ml_frames += int("(ML)" in line)
    if natom is None:
        raise ValueError(f"NIONS was not found in {path}")
    return natom, frames, ml_frames


def parse_outcar(path: Path, indices: np.ndarray) -> TrajectoryDataset:
    natom, total, _ = scan_outcar(path)
    wanted = {int(index): slot for slot, index in enumerate(indices)}
    positions = np.empty((len(indices), natom, 3))
    forces = np.empty_like(positions)
    found = np.zeros(len(indices), dtype=bool)
    iframe = -1
    with path.open(errors="replace") as handle:
        iterator = iter(handle)
        for line in iterator:
            if not FORCE_HEADER.search(line):
                continue
            iframe += 1
            if "---" not in next(iterator, ""):
                raise ValueError(f"malformed force block {iframe}: separator missing")
            slot = wanted.get(iframe)
            for iatom in range(natom):
                fields = next(iterator, "").split()
                if len(fields) < 6:
                    raise ValueError(f"malformed force block {iframe}, atom {iatom}")
                if slot is not None:
                    values = [float(value) for value in fields[:6]]
                    positions[slot, iatom] = values[:3]
                    forces[slot, iatom] = values[3:]
            if slot is not None:
                found[slot] = True
            if found.all():
                break
    if total <= int(indices[-1]) or not found.all():
        raise ValueError(f"requested frames are absent from {path}")
    result = TrajectoryDataset(positions, forces, None, indices.copy(), path, "vasp-outcar")
    result.validate(natom)
    return result

