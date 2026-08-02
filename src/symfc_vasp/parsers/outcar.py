from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import TrajectoryDataset

FORCE_HEADER = re.compile(r"POSITION\s+TOTAL-FORCE\s+\(eV/Angst\)(?:\s+\(ML\))?")


@dataclass(frozen=True)
class OutcarScan:
    natom: int
    frames: int
    ml_frames: int
    requested_nsw: int | None
    spilling_factor_step: int | None
    soft_stop: bool


def scan_outcar_summary(path: Path) -> OutcarScan:
    natom = requested_nsw = spilling_factor_step = None
    frames = ml_frames = 0
    soft_stop = False
    with path.open(errors="replace") as handle:
        for line in handle:
            if natom is None and "NIONS" in line:
                match = re.search(r"NIONS\s*=\s*(\d+)", line)
                if match:
                    natom = int(match.group(1))
            if requested_nsw is None and "NSW" in line:
                match = re.search(r"NSW\s*=\s*(\d+)", line)
                if match:
                    requested_nsw = int(match.group(1))
            if "Spilling factor limit" in line:
                match = re.search(r"ionic step\s+(\d+)", line)
                if match:
                    spilling_factor_step = int(match.group(1))
            if "soft stop encountered" in line.lower():
                soft_stop = True
            if FORCE_HEADER.search(line):
                frames += 1
                ml_frames += int("(ML)" in line)
    if natom is None:
        raise ValueError(f"NIONS was not found in {path}")
    return OutcarScan(
        natom=natom,
        frames=frames,
        ml_frames=ml_frames,
        requested_nsw=requested_nsw,
        spilling_factor_step=spilling_factor_step,
        soft_stop=soft_stop,
    )


def scan_outcar(path: Path) -> tuple[int, int, int]:
    summary = scan_outcar_summary(path)
    return summary.natom, summary.frames, summary.ml_frames


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
