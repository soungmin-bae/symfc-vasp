from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

