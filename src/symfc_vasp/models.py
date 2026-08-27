from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrajectoryDataset:
    """Common representation returned by every trajectory parser."""

    positions: np.ndarray
    forces: np.ndarray
    cells: np.ndarray | None
    source_indices: np.ndarray
    source_path: Path
    source_format: str
    symbols: tuple[str, ...] | None = None

    def validate(self, natom: int | None = None) -> None:
        if self.positions.ndim != 3 or self.positions.shape[-1] != 3:
            raise ValueError(f"positions must have shape (frames, atoms, 3), got {self.positions.shape}")
        if self.forces.shape != self.positions.shape:
            raise ValueError("positions and forces have different shapes")
        if len(self.source_indices) != len(self.positions):
            raise ValueError("source index count differs from frame count")
        if natom is not None and self.positions.shape[1] != natom:
            raise ValueError(f"trajectory has {self.positions.shape[1]} atoms; expected {natom}")
        if not np.isfinite(self.positions).all() or not np.isfinite(self.forces).all():
            raise ValueError("trajectory contains NaN or Inf")
        if self.cells is not None:
            if self.cells.shape != (len(self.positions), 3, 3):
                raise ValueError(f"cells must have shape (frames, 3, 3), got {self.cells.shape}")
            if not np.isfinite(self.cells).all():
                raise ValueError("cell history contains NaN or Inf")


@dataclass(frozen=True)
class ReferenceResult:
    """Reference structures and mapping used by a force-constant fit."""

    unitcell_path: Path
    supercell_path: Path
    supercell_matrix: np.ndarray
    generated_to_source: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class FitResult:
    """Numerical force constants and provenance produced by symfc."""

    output_dir: Path
    fc2: np.ndarray
    fc3: np.ndarray | None
    reference: ReferenceResult
    diagnostics: dict[str, Any]
    files: tuple[Path, ...]


@dataclass(frozen=True)
class PhononResult:
    """Phonon band arrays and reproducibility files."""

    output_dir: Path
    qpoints: np.ndarray
    distances: np.ndarray
    frequencies: np.ndarray
    labels: tuple[str, ...]
    diagnostics: dict[str, Any]
    files: tuple[Path, ...]


@dataclass(frozen=True)
class GruneisenResult:
    """Band-path and mesh tensor mode-Gruneisen arrays."""

    output_dir: Path
    band_qpoints: np.ndarray
    band_frequencies: np.ndarray
    band_tensors: np.ndarray
    mesh_qpoints: np.ndarray
    mesh_weights: np.ndarray
    mesh_frequencies: np.ndarray
    mesh_tensors: np.ndarray
    diagnostics: dict[str, Any]
    files: tuple[Path, ...]
