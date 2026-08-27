#!/usr/bin/env python3
"""Readers and component selectors for phono3py Gruneisen tensors."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import yaml
from phonopy.interface.vasp import read_vasp


COMPONENT_LABELS = {
    "ab": r"$\gamma_{ab}=(\gamma_{xx}+\gamma_{yy})/2$",
    "c": r"$\gamma_c=\gamma_{zz}$",
    "hydro": r"$\gamma_{\mathrm{hydro}}=\mathrm{Tr}(\gamma)/3$",
    "trace": r"$\mathrm{Tr}(\gamma)$",
    "xx": r"$\gamma_{xx}$",
    "yy": r"$\gamma_{yy}$",
    "zz": r"$\gamma_{zz}$",
    "a": r"$\gamma_{aa}$",
    "b": r"$\gamma_{bb}$",
    "shear_ab": r"$\gamma_{ab}$",
    "shear_bc": r"$\gamma_{bc}$",
    "shear_ca": r"$\gamma_{ca}$",
}


def load_analysis_frame(path: Path = Path("analysis_frame.yaml")) -> dict:
    """Load the crystal-frame contract; fall back to legacy raw ab/c axes."""
    if not path.is_file():
        return {"components": ["ab", "c"], "frame": {"basis_columns_cartesian": np.eye(3).tolist()}}
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "frame" not in data:
        raise ValueError(f"Invalid analysis-frame file: {path}")
    return data


def available_components(path: Path = Path("analysis_frame.yaml")) -> list[str]:
    return list(load_analysis_frame(path).get("components", ["ab", "c"]))


def component_label(component: str) -> str:
    if component not in COMPONENT_LABELS:
        raise ValueError(f"Unknown component: {component}")
    return COMPONENT_LABELS[component]


def select_component(tensor: np.ndarray, component: str, frame_path: Path = Path("analysis_frame.yaml")) -> np.ndarray:
    if tensor.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3) tensor, got {tensor.shape}")
    frame = np.asarray(load_analysis_frame(frame_path)["frame"]["basis_columns_cartesian"], dtype=float)
    if frame.shape != (3, 3):
        raise ValueError("analysis_frame.yaml must contain a 3x3 Cartesian basis")
    # Symmetrisation only removes round-off antisymmetry; it does not alter the
    # physical symmetric strain-response tensor.
    tensor = (tensor + np.swapaxes(tensor, -1, -2)) / 2
    tensor = np.einsum("ia,...ij,jb->...ab", frame, tensor, frame)
    if component == "ab":
        return (tensor[..., 0, 0] + tensor[..., 1, 1]) / 2
    if component == "c" or component == "zz":
        return tensor[..., 2, 2]
    if component in {"a", "xx"}:
        return tensor[..., 0, 0]
    if component in {"b", "yy"}:
        return tensor[..., 1, 1]
    if component == "shear_ab":
        return tensor[..., 0, 1]
    if component == "shear_bc":
        return tensor[..., 1, 2]
    if component == "shear_ca":
        return tensor[..., 2, 0]
    trace = np.trace(tensor, axis1=-2, axis2=-1)
    if component == "trace":
        return trace
    if component == "hydro":
        return trace / 3
    raise ValueError(f"Unknown component: {component}")


def valid_mode_mask(frequency: np.ndarray, cutoff: float, policy: str) -> np.ndarray:
    if policy == "stable-only":
        return frequency >= cutoff
    if policy == "abs-frequency":
        return np.abs(frequency) >= cutoff
    raise ValueError(f"Unknown mode policy: {policy}")


def reciprocal_lattice_from_poscar(path: Path) -> np.ndarray:
    cell = np.asarray(read_vasp(str(path)).cell, dtype=float)
    return np.linalg.inv(cell).T


def read_band_yaml(path: Path, component: str):
    data = yaml.safe_load(path.read_text())
    distances, frequencies, gammas = [], [], []
    offset = 0.0
    scalar_error = 0.0
    for segment in data["path"]:
        phonons = segment["phonon"]
        raw_distance = np.asarray([entry["distance"] for entry in phonons], dtype=float)
        x = raw_distance - raw_distance[0] + offset
        freq = np.asarray(
            [[mode["frequency"] for mode in entry["band"]] for entry in phonons],
            dtype=float,
        )
        tensor = np.asarray(
            [[mode["gruneisen_tensor"] for mode in entry["band"]] for entry in phonons],
            dtype=float,
        )
        scalar = np.asarray(
            [[mode["gruneisen"] for mode in entry["band"]] for entry in phonons],
            dtype=float,
        )
        scalar_error = max(scalar_error, float(np.max(np.abs(scalar - select_component(tensor, "hydro")))))
        distances.append(x)
        frequencies.append(freq)
        gammas.append(select_component(tensor, component))
        offset = x[-1]
    if scalar_error > 1e-6:
        raise ValueError(f"Scalar gruneisen != trace/3: max error={scalar_error:g}")
    return distances, frequencies, gammas


def read_mesh_hdf5(path: Path, component: str):
    with h5py.File(path, "r") as h5:
        qpoints = h5["qpoint"][()]
        weights = h5["weight"][()]
        frequencies = h5["frequency"][()]
        scalar = h5["gruneisen"][()]
        tensor = h5["gruneisen_tensor"][()]
        mesh = h5["mesh"][()]
    if tensor.shape != frequencies.shape + (3, 3):
        raise ValueError(f"Tensor/frequency shape mismatch: {tensor.shape}, {frequencies.shape}")
    error = float(np.max(np.abs(scalar - select_component(tensor, "hydro"))))
    if error > 1e-10:
        raise ValueError(f"Scalar gruneisen != trace/3: max error={error:g}")
    return qpoints, weights, frequencies, select_component(tensor, component), mesh


def read_labels(path: Path, npath: int) -> list[str]:
    labels: list[str] = []
    for line in path.read_text().splitlines():
        if line.strip().startswith("BAND_LABELS"):
            labels = line.split("=", 1)[1].split()
            break
    if len(labels) != npath + 1:
        return [f"q{i}" for i in range(npath + 1)]
    return [r"$\Gamma$" if label in {"GM", "GAMMA"} else label.replace("_2", "$_2$") for label in labels]
