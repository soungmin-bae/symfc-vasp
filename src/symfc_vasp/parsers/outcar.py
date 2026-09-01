from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import TrajectoryDataset

FORCE_HEADER = re.compile(r"POSITION\s+TOTAL-FORCE\s+\(eV/Angst\)(?:\s+\(ML\))?")
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


@dataclass(frozen=True)
class OutcarScan:
    natom: int
    frames: int
    ml_frames: int
    requested_nsw: int | None
    spilling_factor_step: int | None
    soft_stop: bool


@dataclass(frozen=True)
class OutcarMetadata:
    """Structure information recoverable without a companion POSCAR."""

    symbols: tuple[str, ...]
    masses: tuple[float, ...] | None
    cell: np.ndarray
    lattice_records: int


def parse_outcar_metadata(path: Path, cell_tolerance: float = 1e-6) -> OutcarMetadata:
    """Read species order and the fixed simulation cell from an OUTCAR.

    VASP writes lattice-vector fields at a fixed width, so adjacent values can
    appear without whitespace (e.g. ``0.000000000-10.0``).  A numeric regular
    expression is used instead of ``str.split`` for those records.
    """
    vrhfin: list[str] = []
    pomass: list[float] = []
    counts: list[int] | None = None
    cells: list[np.ndarray] = []
    lines = Path(path).read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        match = re.search(r"VRHFIN\s*=\s*([A-Za-z]+)", line)
        if match:
            vrhfin.append(match.group(1))
        match = re.search(r"POMASS\s*=\s*(" + _NUMBER.pattern + r")", line)
        if match:
            pomass.append(float(match.group(1)))
        match = re.search(r"ions per type\s*=\s*(.*)", line)
        if match:
            counts = [int(value) for value in match.group(1).split()]
        if "direct lattice vectors" in line and index + 3 < len(lines):
            try:
                cell = np.asarray(
                    [[float(value) for value in _NUMBER.findall(lines[index + offset])[:3]] for offset in (1, 2, 3)],
                    dtype=float,
                )
            except ValueError:
                continue
            if cell.shape == (3, 3) and abs(float(np.linalg.det(cell))) > 1e-12:
                cells.append(cell)
    if counts is None or not vrhfin:
        raise ValueError("OUTCAR does not contain both VRHFIN and ions-per-type records")
    if len(vrhfin) != len(counts):
        raise ValueError(
            f"OUTCAR species/count mismatch: VRHFIN has {len(vrhfin)} entries, ions-per-type has {len(counts)}"
        )
    if not cells:
        raise ValueError("OUTCAR contains no readable direct lattice vectors")
    cell = cells[0]
    for record, candidate in enumerate(cells[1:], start=2):
        deviation = float(np.max(np.abs(candidate - cell)))
        if deviation > cell_tolerance:
            deformation = candidate @ np.linalg.inv(cell) - np.eye(3)
            volume_change = float(np.linalg.det(candidate) / np.linalg.det(cell) - 1.0)
            raise ValueError(
                "variable-cell trajectory detected in OUTCAR: "
                f"lattice record {record} differs from the first by {deviation:.6g} A "
                f"(max deformation={np.max(np.abs(deformation)):.6g}, "
                f"relative volume change={volume_change:.6g}). "
                "symfc-vasp accepts fixed-cell NVT or fixed-cell IBRION=11 data only"
            )
    symbols = tuple(symbol for symbol, count in zip(vrhfin, counts) for _ in range(count))
    masses = None
    if len(pomass) >= len(counts):
        masses = tuple(
            mass
            for mass, count in zip(pomass[:len(counts)], counts)
            for _ in range(count)
        )
    return OutcarMetadata(
        symbols=symbols,
        masses=masses,
        cell=cell,
        lattice_records=len(cells),
    )


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
