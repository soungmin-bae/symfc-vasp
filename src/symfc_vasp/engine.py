#!/usr/bin/env python3
"""Fit finite-temperature FC2/FC3 from a VASP OUTCAR using symfc.

The workflow intentionally uses the positions and Born-Oppenheimer forces
already stored in OUTCAR.  It does not invoke an external force calculator.
The unit-cell symmetry is transferred to the supplied supercell through an
explicit, species-preserving periodic atom map.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import io
import itertools
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


CM1_PER_THZ = 33.35640951981521
SCHEMA_VERSION = "symfc-vasp-run-v2"
from .parsers import parse_trajectory
from .parsers.outcar import parse_outcar_metadata, scan_outcar_summary
from .parsers.vasprun import count_vasprun_frames, vasprun_symbols
from .outcar_reference import build_outcar_reference
from .selection import select_indices
from .reproducibility import (
    write_band_dat,
    write_plain_phonon_band_dat,
    write_band_gnuplot_scripts,
    write_mesh_dat,
    write_mesh_gnuplot_script,
    write_phonon_inputs,
    write_phonopy_yaml,
    write_reproduction_readme,
    write_gruneisen_mesh_yaml,
    write_tensor_plotter_bundle,
    render_tensor_plotter_bundle,
)
from .symmetry_components import write_component_config


def stage(name: str, message: str) -> None:
    """Write a progress message immediately for interactive and batch logs."""
    print(f"[{name}] {message}", flush=True)


def write_phonopy_bandplot_data(output: Path) -> Path:
    """Write the exact two-column gnuplot data emitted by phonopy-bandplot."""
    command = shutil.which("phonopy-bandplot")
    if command is None:
        raise RuntimeError(
            "phonopy-bandplot was not found on PATH. Install phonopy with its CLI entry points "
            "to write phonopy-band.dat."
        )
    result = subprocess.run(
        [command, "--gnuplot", "band.yaml"],
        cwd=output,
        text=True,
        capture_output=True,
        check=False,
    )
    # phonopy 4.4 on macOS can return status 1 after printing a complete
    # --gnuplot dataset. The emitted data is the contract we need here, so
    # accept that case and fail only when no usable dataset was produced.
    if not result.stdout.strip():
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"phonopy-bandplot --gnuplot band.yaml failed: {detail}")
    destination = output / "phonopy-band.dat"
    destination.write_text(result.stdout)
    return destination


class SymfcProgressCapture(io.TextIOBase):
    """Keep the full symfc log while exposing only useful progress by default."""

    _PROGRESS = re.compile(
        r"^(?:Solver_atoms:|Final size of basis set:|Rank of projector:|"
        r"Time \(Basis FC[23]\)|Solver: Calculate|Time \(disp @ compr @ eigvecs\))"
    )

    def __init__(self, path: Path, *, verbose: bool) -> None:
        self._handle = path.open("w")
        self._verbose = verbose
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._handle.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._show(line)
        return len(text)

    def flush(self) -> None:
        if not self._handle.closed:
            self._handle.flush()

    def close(self) -> None:
        if not self.closed:
            if self._buffer:
                self._show(self._buffer)
                self._buffer = ""
            self._handle.close()
        super().close()

    def _show(self, line: str) -> None:
        if self._verbose:
            sys.__stdout__.write(f"{line}\n")
            sys.__stdout__.flush()
        elif self._PROGRESS.match(line.strip()):
            sys.__stdout__.write(f"[symfc] {line}\n")
            sys.__stdout__.flush()


def parse_mass_overrides(values: list[str] | tuple[str, ...] | None) -> dict[str, float]:
    """Parse ``Symbol Mass`` pairs and validate positive finite masses."""
    if not values:
        return {}
    if len(values) % 2:
        raise ValueError("--mass requires Symbol Mass pairs, e.g. --mass H 2.014")
    overrides: dict[str, float] = {}
    for symbol, raw_mass in zip(values[::2], values[1::2]):
        mass = float(raw_mass)
        if not symbol or not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"Invalid atomic mass override: {symbol!r} {raw_mass!r}")
        overrides[str(symbol)] = mass
    return overrides


def parse_atom_mass_overrides(values: list[str] | tuple[str, ...] | None) -> dict[int, float]:
    """Parse one-based primitive atom index and mass pairs."""
    if not values:
        return {}
    if len(values) % 2:
        raise ValueError("--mass-index requires INDEX MASS pairs, e.g. --mass-index 2 2.014")
    overrides: dict[int, float] = {}
    for raw_index, raw_mass in zip(values[::2], values[1::2]):
        index = int(raw_index)
        mass = float(raw_mass)
        if index < 1 or not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"Invalid indexed mass override: {raw_index!r} {raw_mass!r}")
        overrides[index] = mass
    return overrides


def apply_mass_overrides(
    unit,
    overrides: dict[str, float],
    atom_overrides: dict[int, float] | None = None,
) -> dict:
    """Apply isotope masses to a phonopy unit cell and return provenance."""
    symbols = list(unit.symbols)
    original = np.asarray(unit.masses, dtype=float)
    missing = sorted(set(overrides) - set(symbols))
    if missing:
        raise ValueError(f"Mass override symbols not present in unit cell: {', '.join(missing)}")
    effective = np.asarray(
        [overrides.get(symbol, mass) for symbol, mass in zip(symbols, original)],
        dtype=float,
    )
    atom_overrides = atom_overrides or {}
    invalid = sorted(index for index in atom_overrides if index > len(unit))
    if invalid:
        raise ValueError(
            "Mass override primitive atom indices out of range: "
            + ", ".join(map(str, invalid))
        )
    for index, mass in atom_overrides.items():
        effective[index - 1] = mass
    unit.masses = effective
    return {
        "overrides_amu": dict(overrides),
        "atom_overrides_amu": dict(atom_overrides),
        "original_by_species_amu": {
            symbol: float(original[symbols.index(symbol)]) for symbol in dict.fromkeys(symbols)
        },
        "effective_by_species_amu": {
            symbol: float(effective[symbols.index(symbol)]) for symbol in dict.fromkeys(symbols)
        },
    }


def link_force_constant_inputs(fit_dir: Path, output: Path) -> None:
    """Expose FC files in analysis using portable relative symbolic links."""
    for filename in ("FORCE_CONSTANTS", "fc2.hdf5", "fc3.hdf5"):
        source = fit_dir / filename
        link = output / filename
        # Analysis can be regenerated after a new fit in the same directory.
        # Always refresh only these package-owned links so they cannot point at
        # an older force-constant set.
        if source.is_file():
            if source.resolve() == link.resolve():
                continue
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path(os.path.relpath(source, output)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict[str, str]:
    import phonopy
    import phono3py
    import scipy
    import seekpath
    import spglib
    import symfc

    from . import __version__

    return {
        "symfc_vasp": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "phonopy": phonopy.__version__,
        "phono3py": phono3py.__version__,
        "symfc": symfc.__version__,
        "spglib": spglib.__version__,
        "seekpath": seekpath.__version__,
    }


def peak_memory_mib() -> float:
    """Return process peak resident memory using platform-correct units."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def _guard_fit_output(output: Path, source: Path | None, *, force: bool) -> None:
    """Prevent a flat output directory from silently mixing independent fits."""
    summary_path = output / "symfc_summary.yaml"
    if not summary_path.is_file() or source is None:
        return
    previous = yaml.safe_load(summary_path.read_text()) or {}
    old_hash = (previous.get("trajectory") or {}).get("sha256")
    if old_hash and old_hash != sha256(source) and not force:
        raise FileExistsError(
            f"{output} contains a fit from a different input trajectory. "
            "Use --force to replace package-generated fitting outputs."
        )


def _guard_analysis_output(
    output: Path,
    fit_dir: Path,
    *,
    summary_name: str,
    key: str,
    source_name: str,
    force: bool,
) -> None:
    """Reject analysis output left by a different fitted FC file."""
    summary_path = output / summary_name
    source = fit_dir / source_name
    if not summary_path.is_file() or not source.is_file():
        return
    previous = yaml.safe_load(summary_path.read_text()) or {}
    old_hash = (previous.get("inputs") or {}).get(key)
    if old_hash and old_hash != sha256(source) and not force:
        raise FileExistsError(
            f"{output} contains analysis from a different {source_name}. "
            "Use --force to replace package-generated analysis outputs."
        )


def validate_nac_params(nac_params: dict, primitive_atoms: int) -> dict:
    """Validate phonopy-expanded NAC tensors against the active primitive."""
    born = np.asarray(nac_params.get("born"), dtype=float)
    dielectric = np.asarray(nac_params.get("dielectric"), dtype=float)
    if born.shape != (primitive_atoms, 3, 3):
        raise ValueError(
            f"BORN expands to {born.shape}; expected ({primitive_atoms}, 3, 3) "
            "for the active primitive cell"
        )
    if dielectric.shape != (3, 3):
        raise ValueError(f"BORN dielectric tensor has shape {dielectric.shape}; expected (3, 3)")
    if not np.isfinite(born).all() or not np.isfinite(dielectric).all():
        raise ValueError("BORN contains NaN or infinite tensors")
    return {
        "primitive_atoms": primitive_atoms,
        "born_shape": list(born.shape),
        "dielectric_shape": list(dielectric.shape),
    }


def force_constant_symmetry_diagnostics(
    fc2: np.ndarray,
    fc3: np.ndarray | None,
    *,
    sample_atoms: int = 12,
) -> dict:
    """Report exact FC2 and deterministic sampled FC3 permutation residuals."""
    fc2_permutation = 0.0
    for start in range(0, len(fc2), 16):
        block = fc2[start : start + 16]
        counterpart = fc2[:, start : start + 16].transpose(1, 0, 3, 2)
        fc2_permutation = max(
            fc2_permutation, float(np.max(np.abs(block - counterpart)))
        )
    result = {
        "fc2_permutation_max_abs": fc2_permutation,
        "fc3_permutation_sample_max_abs": None,
        "fc3_permutation_sample_atoms": [],
    }
    if fc3 is not None:
        indices = np.unique(
            np.linspace(0, len(fc3) - 1, min(sample_atoms, len(fc3)), dtype=int)
        )
        subset = fc3[np.ix_(indices, indices, indices, range(3), range(3), range(3))]
        swap_ij = subset.transpose(1, 0, 2, 4, 3, 5)
        swap_ik = subset.transpose(2, 1, 0, 5, 4, 3)
        result.update({
            "fc3_permutation_sample_max_abs": float(
                max(np.max(np.abs(subset - swap_ij)), np.max(np.abs(subset - swap_ik)))
            ),
            "fc3_permutation_sample_atoms": indices.tolist(),
        })
    return result


def validate_fit_dataset(displacements: np.ndarray, forces: np.ndarray) -> dict:
    """Validate the minimum numerical contract required by a symfc fit."""
    if displacements.shape != forces.shape or displacements.ndim != 3:
        raise ValueError(
            f"displacement/force arrays must share (frames, atoms, 3), got "
            f"{displacements.shape} and {forces.shape}"
        )
    if len(displacements) < 2:
        raise ValueError(
            "at least two independent position/force frames are required for fitting; "
            f"received {len(displacements)}"
        )
    if not np.isfinite(displacements).all() or not np.isfinite(forces).all():
        raise ValueError("fitting dataset contains NaN or infinite values")
    variance = float(np.var(displacements))
    if variance <= np.finfo(float).eps:
        raise ValueError(
            "selected displacements have zero numerical variance; force constants are not identifiable"
        )
    return {
        "frames": int(len(displacements)),
        "atoms": int(displacements.shape[1]),
        "displacement_variance_A2": variance,
    }


def minimum_image_displacements(delta_frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return shortest Cartesian images for fractional displacement vectors.

    Component-wise fractional wrapping is only sufficient for orthogonal
    cells.  For skewed cells (including hexagonal cells), the shortest image
    can be one of the neighboring lattice translations.  Search the 27
    translations around the component-wrapped image and retain the shortest
    Cartesian vector.
    """
    wrapped = np.asarray(delta_frac, dtype=float) - np.rint(delta_frac)
    best = wrapped @ cell
    best_norm2 = np.einsum("...i,...i->...", best, best)
    for shift in itertools.product((-1.0, 0.0, 1.0), repeat=3):
        if shift == (0.0, 0.0, 0.0):
            continue
        candidate = (wrapped + np.asarray(shift)) @ cell
        norm2 = np.einsum("...i,...i->...", candidate, candidate)
        replace = norm2 < best_norm2
        best = np.where(replace[..., None], candidate, best)
        best_norm2 = np.where(replace, norm2, best_norm2)
    return best


def center_periodic_trajectory(
    frame_frac: np.ndarray,
    initial_center_frac: np.ndarray,
    cell: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Center a periodic trajectory about its intrinsic per-atom mean.

    A Cartesian arithmetic mean of already wrapped displacements is invalid
    when samples straddle a periodic boundary.  Iterate the per-atom center on
    the periodic cell using shortest Cartesian images.  The returned shift is
    the shortest displacement from the supplied reference to the converged
    periodic center.
    """
    initial_center = np.asarray(initial_center_frac, dtype=float)
    phase = np.exp(2j * np.pi * np.asarray(frame_frac, dtype=float))
    resultant = np.mean(phase, axis=0)
    circular_center = (np.angle(resultant) / (2.0 * np.pi)) % 1.0
    # A near-zero circular resultant is genuinely ambiguous; retain the
    # crystallographic reference for that coordinate and let the Cartesian
    # refinement handle the remaining shift.
    center = np.where(np.abs(resultant) > 1e-8, circular_center, initial_center).copy()
    inv_cell = np.linalg.inv(cell)
    for iteration in range(1, max_iterations + 1):
        displacements = minimum_image_displacements(frame_frac - center[None, :, :], cell)
        residual = np.mean(displacements, axis=0)
        center = (center + residual @ inv_cell) % 1.0
        if float(np.max(np.linalg.norm(residual, axis=1))) < tolerance:
            break
    else:
        raise RuntimeError(
            "Periodic trajectory centering did not converge; the selected trajectory may be "
            "diffusive or multimodal rather than harmonic about one structure"
        )
    displacements = minimum_image_displacements(frame_frac - center[None, :, :], cell)
    shift = minimum_image_displacements(center - initial_center, cell)
    return displacements, shift, iteration


def periodic_distance_matrix(frac_a: np.ndarray, frac_b: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = frac_a[:, None, :] - frac_b[None, :, :]
    return np.linalg.norm(minimum_image_displacements(delta, cell), axis=-1)


def build_atom_map(given, generated, tolerance: float) -> tuple[np.ndarray, dict]:
    """Return generated-index -> given-index mapping."""
    if len(given) != len(generated):
        raise ValueError(f"Atom count differs: {len(given)} != {len(generated)}")
    if not np.allclose(given.cell, generated.cell, atol=tolerance, rtol=0):
        raise ValueError("POSCAR-supercell lattice differs from generated dim supercell")
    mapping = np.full(len(generated), -1, dtype=int)
    distances = np.full(len(generated), np.nan)
    species = list(dict.fromkeys(generated.symbols))
    for symbol in species:
        ig = np.flatnonzero(np.asarray(generated.symbols) == symbol)
        iv = np.flatnonzero(np.asarray(given.symbols) == symbol)
        if len(ig) != len(iv):
            raise ValueError(f"Species count differs for {symbol}: {len(iv)} != {len(ig)}")
        cost = periodic_distance_matrix(
            np.asarray(generated.scaled_positions)[ig],
            np.asarray(given.scaled_positions)[iv],
            np.asarray(generated.cell),
        )
        rows, cols = linear_sum_assignment(cost)
        mapping[ig[rows]] = iv[cols]
        distances[ig[rows]] = cost[rows, cols]
    if np.any(mapping < 0) or len(np.unique(mapping)) != len(mapping):
        raise RuntimeError("Atom mapping has missing or duplicate indices")
    if float(np.max(distances)) > tolerance:
        raise ValueError(
            f"Best atom map exceeds tolerance: {np.max(distances):.6g} > {tolerance:.6g} A"
        )
    return mapping, {
        "max_distance_A": float(np.max(distances)),
        "rms_distance_A": float(np.sqrt(np.mean(distances**2))),
        "identity": bool(np.array_equal(mapping, np.arange(len(mapping)))),
        "unique_indices": int(len(np.unique(mapping))),
    }


def infer_supercell_matrix(unitcell, supercell_cell: np.ndarray, supercell_atoms: int, *, tolerance: float) -> tuple[np.ndarray, dict]:
    """Infer the general integer unitcell-to-supercell matrix from lattices.

    This deliberately does not assume a diagonal ``dim``.  A diagonal
    ``--dim`` remains a convenient user assertion, but the lattice relation is
    the authoritative definition of the fitting supercell.
    """
    matrix_float = np.asarray(supercell_cell, dtype=float) @ np.linalg.inv(np.asarray(unitcell.cell, dtype=float))
    matrix = np.rint(matrix_float).astype(int)
    residual = float(np.max(np.abs(matrix_float - matrix)))
    determinant = abs(int(round(np.linalg.det(matrix))))
    expected_atoms = determinant * len(unitcell)
    if residual > tolerance or determinant == 0 or expected_atoms != supercell_atoms:
        raise ValueError(
            "unitcell and trajectory/supercell do not define a valid integer supercell matrix: "
            f"matrix_residual={residual:.3e}, determinant={determinant}, "
            f"unitcell_atoms={len(unitcell)}, supercell_atoms={supercell_atoms}. "
            "Supply a unitcell compatible with the fixed-cell trajectory."
        )
    return matrix, {
        "matrix": matrix.tolist(),
        "residual": residual,
        "determinant": determinant,
        "expected_supercell_atoms": expected_atoms,
    }


def validate_dim_override(dim, matrix: np.ndarray) -> None:
    """Treat --dim as a diagonal consistency assertion, never as the source of truth."""
    if dim is None:
        return
    requested = np.diag(np.asarray(dim, dtype=int))
    if not np.array_equal(requested, matrix):
        raise ValueError(
            f"--dim {' '.join(map(str, dim))} disagrees with inferred supercell matrix "
            f"{matrix.tolist()}. Omit --dim for a general matrix, or provide the matching diagonal replication."
        )


def parse_rc3(value: str) -> float | str:
    """Accept a numeric FC3 cutoff or the reproducible ``auto`` policy."""
    if value.strip().lower() == "auto":
        return "auto"
    cutoff = float(value)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise argparse.ArgumentTypeError("--rc3 must be a positive Angstrom value or 'auto'")
    return cutoff


def resolve_auto_rc3(symfc_atoms, generated, nframes: int, *, use_mkl: bool, minimum_ratio: float) -> tuple[float, dict]:
    """Choose the longest identifiable FC3 cutoff using symmetry-basis size.

    The candidate grid follows actual pair distances rather than a material
    name.  It stops once the number of fitted force equations per irreducible
    FC3 parameter falls below ``minimum_ratio``. This is an identifiability
    guard, not a substitute for an explicit cutoff-convergence study.
    """
    from symfc import Symfc

    frac = np.asarray(generated.scaled_positions)
    distances = periodic_distance_matrix(frac, frac, np.asarray(generated.cell))
    distances[np.diag_indices_from(distances)] = np.inf
    nearest = float(np.min(distances))
    cell_limit = 0.5 * float(np.min(np.linalg.norm(np.asarray(generated.cell), axis=1)))
    first = max(0.5, np.ceil((nearest + 1e-10) * 2.0) / 2.0)
    last = np.floor((cell_limit + 1e-10) * 2.0) / 2.0
    if last < first:
        last = cell_limit
    candidates = np.arange(first, last + 1e-10, 0.5)
    if len(candidates) == 0:
        candidates = np.asarray([cell_limit])
    equations = int(3 * len(generated) * nframes)
    probes: list[dict] = []
    accepted: float | None = None
    for candidate in candidates:
        probe = Symfc(supercell=symfc_atoms, cutoff={3: float(candidate)}, use_mkl=use_mkl, log_level=0)
        probe.compute_basis_set(orders=[3])
        parameters = int(probe.basis_set[3].basis_set.shape[0])
        ratio = float(equations / parameters)
        probes.append({
            "rc3_A": float(candidate), "parameters": parameters,
            "equations": equations, "equations_per_parameter": ratio,
        })
        if ratio >= minimum_ratio:
            accepted = float(candidate)
        else:
            # Parameter count increases monotonically with cutoff. Later
            # candidates cannot restore the identifiability ratio.
            break
    if accepted is None:
        accepted = float(candidates[0])
        reason = "minimum candidate is below the requested identifiability ratio"
    else:
        reason = "largest candidate meeting the requested identifiability ratio"
    return accepted, {
        "policy": "basis-size-identifiability",
        "minimum_equations_per_parameter": float(minimum_ratio),
        "nearest_pair_distance_A": nearest,
        "half_shortest_cell_vector_A": cell_limit,
        "candidates": probes,
        "selected_rc3_A": accepted,
        "selection_reason": reason,
    }


def prepare_outcar_only_dataset(args, output: Path) -> tuple[object, object, np.ndarray, np.ndarray, dict]:
    """Create an FC fitting dataset by reconstructing a reference from a trajectory.

    The reference structure is deliberately reconstructed from selected OUTCAR
    or vasprun.xml frames. No POSCAR, XDATCAR, or phonopy YAML is read on this path.
    """
    trajectory = args.trajectory.resolve()
    is_outcar = trajectory.name.lower() == "outcar" or trajectory.name.lower().endswith(".outcar")
    scan = scan_outcar_summary(trajectory) if is_outcar else None
    total_frames = scan.frames if scan is not None else count_vasprun_frames(trajectory)
    metadata = parse_outcar_metadata(trajectory) if scan is not None else None
    symbols = metadata.symbols if metadata is not None else vasprun_symbols(trajectory)
    natom = scan.natom if scan is not None else len(symbols)
    stage(
        "trajectory",
        f"Scanned {trajectory.name}: {total_frames} position/force frames, {natom} atoms"
        + (f", {scan.ml_frames} ML force blocks" if scan is not None else ""),
    )
    source_indices = select_indices(
        total_frames, skip=args.skip, stop=args.stop, samples=args.samples,
        stride=args.stride, method=args.selection, seed=args.seed,
    )
    stage(
        "selection",
        f"skip={args.skip}, stop={args.stop if args.stop is not None else total_frames}, "
        f"method={args.selection}, requested_samples={args.samples}; "
        f"using {len(source_indices)} frames [{source_indices[0]}..{source_indices[-1]}]",
    )
    stage("reference", "Building the periodic mean structure from the selected trajectory frames")
    dataset = parse_trajectory(trajectory, source_indices)
    dataset.validate(natom)
    # OUTCAR records the fixed lattice in its header, whereas vasprun.xml
    # carries a lattice with every frame.  The lightweight OUTCAR parser only
    # stores positions/forces, so fall back to its metadata cell here.
    trajectory_cell = (
        np.asarray(dataset.cells[0], dtype=float)
        if dataset.cells is not None
        else np.asarray(metadata.cell, dtype=float)
    )
    if args.unitcell is None:
        try:
            unit, generated, mapping, reference = build_outcar_reference(
                outcar=trajectory,
                positions=dataset.positions,
                output=output,
                symprec_max=args.reference_symprec_max,
                map_tolerance=args.reference_map_tolerance,
                symbols=symbols,
                cell=trajectory_cell,
            )
        except ValueError as error:
            # A finite random-displacement ensemble can make spglib choose a
            # formally valid but site-incompatible standard setting. When the
            # original phonopy unitcell is already present beside the OUTCAR,
            # it is a stronger displacement reference than such a failed
            # reconstruction. Use it only after the trajectory-only map has
            # explicitly failed, and record the fallback in the output.
            fallback_unitcell = trajectory.parent / "POSCAR-unitcell"
            if not fallback_unitcell.is_file():
                raise
            stage(
                "reference",
                "WARNING: trajectory-only spglib reference failed atom-map validation; "
                f"falling back to existing {fallback_unitcell.name}",
            )
            fallback_args = copy.copy(args)
            fallback_args.unitcell = fallback_unitcell
            fallback_args._automatic_unitcell_fallback_reason = str(error)
            return prepare_outcar_only_dataset(fallback_args, output)
        validate_dim_override(args.dim, np.asarray(reference["supercell_matrix"], dtype=int))
        spacegroup = reference["selected_spacegroup"]
        matrix = np.asarray(reference["supercell_matrix"], dtype=int)
        mapping_summary = reference["mapping"]
        stage(
            "reference",
            "spglib refinement: periodic trajectory mean -> idealized primitive reference",
        )
        stage(
            "reference",
            f"Selected space group: {spacegroup['international']} (No. {spacegroup['number']}, "
            f"{spacegroup['operations']} operations); selected symprec={reference['selected_symprec_A']:.6g} A "
            f"(scan maximum={args.reference_symprec_max:.6g} A)",
        )
        stage(
            "reference",
            f"Reference relation: {len(unit)}-atom primitive -> {len(generated)}-atom supercell; "
            f"integer matrix={matrix.tolist()}, det={abs(round(np.linalg.det(matrix)))}; "
            f"lattice residual={reference['supercell_matrix_residual']:.3e}",
        )
        stage(
            "reference",
            f"Atom mapping to refined reference: max={mapping_summary['max_distance_A']:.6g} A, "
            f"RMS={mapping_summary['rms_distance_A']:.6g} A "
            f"(acceptance tolerance={mapping_summary['tolerance_A']:.6g} A)",
        )
        stage(
            "reference",
            "Saved POSCAR-mean, POSCAR-unitcell, POSCAR-supercell, SPOSCAR, "
            "supercell_matrix.dat, and symmetry_report.yaml in the output directory",
        )
    else:
        from phonopy import Phonopy
        from phonopy.interface.vasp import read_vasp, write_vasp
        from phonopy.structure.atoms import PhonopyAtoms

        unit = read_vasp(str(args.unitcell))
        matrix, matrix_info = infer_supercell_matrix(
            unit, trajectory_cell, natom, tolerance=max(args.cell_tolerance, 1e-6)
        )
        validate_dim_override(args.dim, matrix)
        phonon = Phonopy(unit, supercell_matrix=matrix, primitive_matrix="P", symprec=args.symprec)
        generated = phonon.supercell
        if not np.allclose(generated.cell, trajectory_cell, atol=args.cell_tolerance, rtol=0):
            raise ValueError("the supplied --unitcell generates a supercell lattice different from the trajectory cell")
        phases = np.exp(2j * np.pi * (dataset.positions @ np.linalg.inv(trajectory_cell)))
        mean_frac = np.mod(np.angle(np.mean(phases, axis=0)) / (2.0 * np.pi), 1.0)
        mean_atoms = PhonopyAtoms(cell=trajectory_cell, scaled_positions=mean_frac, symbols=symbols)
        mapping, map_summary = build_atom_map(mean_atoms, generated, args.reference_map_tolerance)
        write_vasp(str(output / "POSCAR-mean"), mean_atoms)
        # Preserve the supplied reference byte-for-byte in the run output so
        # there is no ambiguity about the structure used for displacements.
        unitcell_target = output / "POSCAR-unitcell"
        if args.unitcell.resolve() != unitcell_target.resolve():
            shutil.copy2(args.unitcell, unitcell_target)
        write_vasp(str(output / "POSCAR-supercell"), mean_atoms)
        write_vasp(str(output / "SPOSCAR"), generated)
        np.savetxt(output / "supercell_matrix.dat", matrix, fmt="%d")
        (output / "generated_to_outcar_index.json").write_text(
            json.dumps({"generated_supercell_index_to_outcar_index": mapping.tolist()}, indent=2) + "\n"
        )
        fallback_reason = getattr(args, "_automatic_unitcell_fallback_reason", None)
        reference = {
            "schema": "symfc-vasp-structure-report-v2",
            "source": (
                "automatic-existing-unitcell-fallback"
                if fallback_reason is not None
                else "supplied-unitcell-plus-trajectory-supercell"
            ),
            "unitcell": str(args.unitcell.resolve()),
            "supercell_matrix": matrix.tolist(),
            "supercell_matrix_residual": matrix_info["residual"],
            "mapping": map_summary,
            "selected_spacegroup": {
                "international": str(phonon.symmetry.dataset.international),
                "number": int(phonon.symmetry.dataset.number),
            },
        }
        if fallback_reason is not None:
            reference["trajectory_only_reconstruction_failure"] = fallback_reason
        (output / "structure_report.yaml").write_text(yaml.safe_dump(reference, sort_keys=False))
        stage(
            "reference",
            f"Using supplied {args.unitcell.name} unchanged as the displacement reference; "
            f"inferred supercell matrix={matrix.tolist()} (residual={matrix_info['residual']:.3e})",
        )
        stage(
            "reference",
            f"{len(unit)}-atom supplied unitcell -> {len(generated)}-atom trajectory supercell; "
            f"atom-map max distance={map_summary['max_distance_A']:.6g} A",
        )
        stage(
            "reference",
            "Saved POSCAR-mean, POSCAR-unitcell, POSCAR-supercell, SPOSCAR, "
            "supercell_matrix.dat, and structure_report.yaml in the output directory",
        )
    cell = np.asarray(generated.cell)
    positions = dataset.positions[:, mapping, :]
    forces = dataset.forces[:, mapping, :]
    frame_frac = positions @ np.linalg.inv(cell)
    ref_frac = np.asarray(generated.scaled_positions)
    displacements = minimum_image_displacements(frame_frac - ref_frac[None, :, :], cell)
    mean_displacement = np.mean(displacements, axis=0)
    mean_force = np.mean(forces, axis=0)
    centering_iterations = 0
    if args.center_selected:
        displacements, mean_displacement, centering_iterations = center_periodic_trajectory(
            frame_frac, ref_frac, cell
        )
        forces = forces - mean_force[None, :, :]
    centering = {
        "enabled": bool(args.center_selected),
        "method": "periodic-intrinsic-mean" if args.center_selected else "none",
        "iterations": centering_iterations,
        "mean_displacement_rms_A": float(np.sqrt(np.mean(mean_displacement**2))),
        "mean_displacement_max_atom_norm_A": float(np.max(np.linalg.norm(mean_displacement, axis=1))),
        "mean_force_rms_eV_per_A": float(np.sqrt(np.mean(mean_force**2))),
        "mean_force_max_atom_norm_eV_per_A": float(np.max(np.linalg.norm(mean_force, axis=1))),
    }
    summary = {
        "trajectory": {
            "path": str(trajectory), "format": dataset.source_format, "sha256": sha256(trajectory),
            "natom": natom, "force_blocks": total_frames,
            "ml_force_blocks": scan.ml_frames if scan is not None else None,
        },
        "selection": {
            "skip": args.skip, "stop": args.stop, "method": args.selection,
            "stride": int(source_indices[1] - source_indices[0]) if len(source_indices) > 1 else None,
            "requested_samples": args.samples, "selected_frames": int(len(source_indices)),
            "first_source_index": int(source_indices[0]), "last_source_index": int(source_indices[-1]),
        },
        "structure": {
            "reference_mode": "trajectory", "unit_atoms": len(unit), "supercell_atoms": len(generated),
            "volume_ratio": float(np.linalg.det(generated.cell) / np.linalg.det(unit.cell)),
            "spacegroup": reference["selected_spacegroup"]["international"],
            "spacegroup_number": reference["selected_spacegroup"]["number"],
            "supercell_matrix": reference["supercell_matrix"],
        },
        "atom_mapping": reference["mapping"],
        "reference": reference,
        "dataset": {
            "displacement_shape": list(displacements.shape), "force_shape": list(forces.shape),
            "max_abs_displacement_A": float(np.max(np.abs(displacements))),
            "max_abs_force_eV_per_A": float(np.max(np.abs(forces))), "centering": centering,
        },
    }
    np.savez_compressed(
        output / "symfc_input.npz", displacements=displacements, forces=forces,
        source_indices=source_indices, generated_to_vasp=mapping,
        symbols=np.asarray(generated.symbols), cell=cell, scaled_positions=ref_frac,
        mean_displacement=mean_displacement, mean_force=mean_force,
    )
    np.savetxt(output / "selected_indices.txt", source_indices, fmt="%d")
    dataset_path = output / "symfc_input.npz"
    stage(
        "dataset",
        f"Wrote {dataset_path.name}: displacements{tuple(displacements.shape)}, "
        f"forces{tuple(forces.shape)}, centered={args.center_selected}, "
        f"{dataset_path.stat().st_size / 1024**2:.2f} MiB",
    )
    return unit, generated, displacements, forces, summary


def prepare_dataset(args, output: Path) -> tuple[object, object, np.ndarray, np.ndarray, dict]:
    from phonopy import Phonopy
    from phonopy.interface.vasp import read_vasp

    mode = getattr(args, "reference_mode", "auto")
    has_provided_reference = (
        args.unitcell is not None
        and args.supercell is not None
        and args.unitcell.is_file()
        and args.supercell.is_file()
    )
    if mode in {"outcar", "trajectory"} or (mode == "auto" and not has_provided_reference):
        return prepare_outcar_only_dataset(args, output)
    if mode == "provided" and not has_provided_reference:
        raise FileNotFoundError("--reference-mode provided requires both --unitcell and --supercell files")
    unit = read_vasp(str(args.unitcell))
    given = read_vasp(str(args.supercell))
    matrix, matrix_info = infer_supercell_matrix(
        unit, np.asarray(given.cell), len(given), tolerance=max(args.cell_tolerance, 1e-6)
    )
    validate_dim_override(args.dim, matrix)
    phonon = Phonopy(
        unit,
        supercell_matrix=matrix,
        primitive_matrix="auto",
        symprec=args.symprec,
    )
    generated = phonon.supercell
    mapping, map_summary = build_atom_map(given, generated, args.map_tolerance)

    cell = np.asarray(generated.cell)
    if args.dataset_npz is not None:
        dataset_path = args.dataset_npz.resolve()
        saved = np.load(dataset_path)
        displacements = np.asarray(saved["displacements"], dtype=float)
        forces = np.asarray(saved["forces"], dtype=float)
        source_indices = np.asarray(saved["source_indices"], dtype=int)
        natom = displacements.shape[1]
        if displacements.shape != forces.shape or displacements.shape[-1] != 3:
            raise ValueError("saved dataset displacement/force shapes are inconsistent")
        if natom != len(generated):
            raise ValueError(f"saved dataset atoms={natom}, generated supercell atoms={len(generated)}")
        if "cell" in saved and not np.allclose(saved["cell"], cell, atol=1e-8, rtol=0):
            raise ValueError("saved dataset cell differs from generated supercell")
        if not np.isfinite(displacements).all() or not np.isfinite(forces).all():
            raise ValueError("saved dataset contains NaN or Inf")
        # A reusable symfc_input.npz may contain a large parent dataset.
        # Reapply selection in its own frame index space so cutoff scans use
        # precisely the same configurations without reparsing OUTCAR.
        parent_frames = len(source_indices)
        try:
            dataset_selection = select_indices(
                parent_frames, skip=args.skip, stop=args.stop,
                samples=args.samples, stride=args.stride,
                method=args.selection, seed=args.seed,
            )
        except ValueError as exc:
            raise ValueError(
                f"saved dataset contains {parent_frames} frames; requested "
                f"skip={args.skip}, stop={args.stop}, samples={args.samples}. {exc}"
            ) from exc
        displacements = displacements[dataset_selection]
        forces = forces[dataset_selection]
        source_indices = source_indices[dataset_selection]
        nframes = parent_frames
        nframes_ml = None
        trajectory_info = {
            "path": str(dataset_path),
            "format": "symfc-input-npz",
            "sha256": sha256(dataset_path),
            "natom": natom,
            "force_blocks": None,
            "ml_force_blocks": None,
        }
    else:
        trajectory = args.trajectory
        if trajectory.name.lower() == "outcar" or trajectory.name.lower().endswith(".outcar"):
            outcar_scan = scan_outcar_summary(trajectory)
            natom, nframes, nframes_ml = (
                outcar_scan.natom, outcar_scan.frames, outcar_scan.ml_frames,
            )
        else:
            nframes = count_vasprun_frames(trajectory)
            natom = len(given)
            nframes_ml = 0
        stage(
            "trajectory",
            f"Scanned {trajectory.name}: {nframes} position/force frames, {natom} atoms",
        )
        if natom != len(given):
            raise ValueError(f"trajectory atoms={natom}, POSCAR-supercell atoms={len(given)}")
        try:
            source_indices = select_indices(
                nframes, skip=args.skip, stop=args.stop, samples=args.samples, stride=args.stride,
                method=args.selection, seed=args.seed,
            )
        except ValueError as exc:
            context = [
                f"trajectory contains {nframes} force/position frames",
                f"requested skip={args.skip}, stop={args.stop}, samples={args.samples}",
            ]
            if trajectory.name.lower() == "outcar" or trajectory.name.lower().endswith(".outcar"):
                if outcar_scan.requested_nsw is not None:
                    context.append(f"VASP NSW={outcar_scan.requested_nsw}")
                if outcar_scan.spilling_factor_step is not None:
                    context.append(
                        "VASP stopped because the MLFF spilling-factor limit was exceeded "
                        f"at ionic step {outcar_scan.spilling_factor_step}; this trajectory is not "
                        "a reliable production dataset"
                    )
                elif outcar_scan.soft_stop:
                    context.append("VASP encountered a soft stop before completing the trajectory")
            raise ValueError(f"{exc}. " + "; ".join(context)) from exc
        stage(
            "selection",
            f"skip={args.skip}, stop={args.stop if args.stop is not None else nframes}, "
            f"method={args.selection}, requested_samples={args.samples}; "
            f"using {len(source_indices)} frames [{source_indices[0]}..{source_indices[-1]}]",
        )
        dataset = parse_trajectory(trajectory, source_indices)
        dataset.validate(natom)
        positions, forces = dataset.positions, dataset.forces
        if dataset.cells is not None:
            deviation = float(np.max(np.abs(dataset.cells - dataset.cells[0])))
            if deviation > args.cell_tolerance:
                raise ValueError(f"trajectory is not fixed-cell: max cell deviation={deviation:.6g} A")
        positions = positions[:, mapping, :]
        forces = forces[:, mapping, :]
        inv_cell = np.linalg.inv(cell)
        frame_frac = positions @ inv_cell
        ref_frac = np.asarray(generated.scaled_positions)
        delta_frac = frame_frac - ref_frac[None, :, :]
        displacements = minimum_image_displacements(delta_frac, cell)
        trajectory_info = {
            "path": str(trajectory.resolve()), "format": dataset.source_format,
            "sha256": sha256(trajectory), "natom": natom,
            "force_blocks": nframes, "ml_force_blocks": nframes_ml,
        }

    stage(
        "reference",
        "Using the supplied --unitcell and --supercell without averaging or spglib refinement",
    )
    stage(
        "reference",
        f"Provided {len(unit)}-atom unitcell -> generated {len(generated)}-atom supercell; "
        f"atom-map max distance={map_summary['max_distance_A']:.6g} A",
    )
    from phonopy.interface.vasp import write_vasp

    # Preserve supplied reference files byte-for-byte in the run record.
    shutil.copy2(args.unitcell, output / "POSCAR-unitcell")
    write_vasp(str(output / "POSCAR-supercell"), given)
    write_vasp(str(output / "SPOSCAR"), generated)
    np.savetxt(output / "supercell_matrix.dat", matrix, fmt="%d")
    structure_report = {
        "schema": "symfc-vasp-structure-report-v2",
        "source": "provided-poscars",
        "refinement": "none",
        "unitcell": str(args.unitcell.resolve()),
        "supercell": str(args.supercell.resolve()),
        "supercell_matrix": matrix.tolist(),
        "supercell_matrix_residual": matrix_info["residual"],
        "unitcell_atoms": len(unit),
        "generated_supercell_atoms": len(generated),
        "atom_mapping": map_summary,
    }
    (output / "structure_report.yaml").write_text(yaml.safe_dump(structure_report, sort_keys=False))
    stage(
        "reference",
        "Saved POSCAR-unitcell, POSCAR-supercell, SPOSCAR, supercell_matrix.dat, and structure_report.yaml "
        "in the output directory",
    )

    mean_displacement = np.mean(displacements, axis=0)
    mean_force = np.mean(forces, axis=0)
    centering_iterations = 0
    if args.center_selected and not args.dataset_npz:
        displacements, mean_displacement, centering_iterations = center_periodic_trajectory(
            frame_frac, ref_frac, cell
        )
    centering = {
        "enabled": bool(args.center_selected),
        "method": "periodic-intrinsic-mean" if args.center_selected and not args.dataset_npz else "arithmetic",
        "iterations": centering_iterations,
        "mean_displacement_rms_A": float(np.sqrt(np.mean(mean_displacement**2))),
        "mean_displacement_max_atom_norm_A": float(np.max(np.linalg.norm(mean_displacement, axis=1))),
        "mean_force_rms_eV_per_A": float(np.sqrt(np.mean(mean_force**2))),
        "mean_force_max_atom_norm_eV_per_A": float(np.max(np.linalg.norm(mean_force, axis=1))),
    }
    if args.center_selected:
        if args.dataset_npz:
            displacements = displacements - mean_displacement[None, :, :]
        forces = forces - mean_force[None, :, :]

    summary = {
        "trajectory": trajectory_info,
        "selection": {
            "skip": args.skip,
            "stop": args.stop,
            "method": args.selection,
            "stride": int(source_indices[1] - source_indices[0]) if len(source_indices) > 1 else None,
            "requested_samples": args.samples,
            "selected_frames": int(len(source_indices)),
            "first_source_index": int(source_indices[0]),
            "last_source_index": int(source_indices[-1]),
        },
        "structure": {
            "unitcell": str(args.unitcell.resolve()),
            "supercell": str(args.supercell.resolve()),
            "supercell_matrix": matrix.tolist(),
            "unit_atoms": len(unit),
            "supercell_atoms": len(generated),
            "volume_ratio": float(np.linalg.det(generated.cell) / np.linalg.det(unit.cell)),
            "spacegroup": phonon.symmetry.dataset.international,
            "spacegroup_number": int(phonon.symmetry.dataset.number),
        },
        "atom_mapping": map_summary,
        "dataset": {
            "displacement_shape": list(displacements.shape),
            "force_shape": list(forces.shape),
            "max_abs_displacement_A": float(np.max(np.abs(displacements))),
            "max_abs_force_eV_per_A": float(np.max(np.abs(forces))),
            "centering": centering,
        },
    }
    np.savez_compressed(
        output / "symfc_input.npz",
        displacements=displacements,
        forces=forces,
        source_indices=source_indices,
        generated_to_vasp=mapping,
        symbols=np.asarray(generated.symbols),
        cell=np.asarray(generated.cell),
        scaled_positions=np.asarray(generated.scaled_positions),
        mean_displacement=mean_displacement,
        mean_force=mean_force,
    )
    np.savetxt(output / "selected_indices.txt", source_indices, fmt="%d")
    dataset_path = output / "symfc_input.npz"
    stage(
        "dataset",
        f"Wrote {dataset_path.name}: displacements{tuple(displacements.shape)}, "
        f"forces{tuple(forces.shape)}, centered={args.center_selected}, "
        f"{dataset_path.stat().st_size / 1024**2:.2f} MiB",
    )
    return unit, generated, displacements, forces, summary


def force_metrics(
    displacements: np.ndarray,
    forces: np.ndarray,
    fc2: np.ndarray,
    fc3: np.ndarray | None,
    max_samples: int,
) -> dict:
    # Full FC3 contraction is expensive. Use a deterministic, uniformly spaced
    # validation subset while all configurations still enter the actual fit.
    nvalidate = min(len(displacements), max_samples)
    selected = np.unique(np.linspace(0, len(displacements) - 1, nvalidate, dtype=int))
    displacements = displacements[selected]
    forces = forces[selected]
    sse = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    count = 0
    max_abs = 0.0
    for start in range(0, len(displacements), 20):
        u = displacements[start : start + 20]
        y = forces[start : start + 20]
        pred = -np.einsum("ijab,sjb->sia", fc2, u, optimize=True)
        if fc3 is not None:
            pred -= 0.5 * np.einsum("ijkabc,sjb,skc->sia", fc3, u, u, optimize=True)
        residual = y - pred
        sse += float(np.sum(residual**2))
        sum_y += float(np.sum(y))
        sum_y2 += float(np.sum(y**2))
        count += y.size
        max_abs = max(max_abs, float(np.max(np.abs(residual))))
    mean_y = sum_y / count
    sst = sum_y2 - count * mean_y**2
    # A single selected frame with centering enabled has exactly zero force
    # variance.  A fit can still be written for diagnostic purposes, but R2
    # is mathematically undefined rather than zero or one.
    r2 = None if sst <= np.finfo(float).eps * max(sum_y2, 1.0) else float(1.0 - sse / sst)
    return {
        "r2": r2,
        "r2_status": "undefined: validation force variance is zero" if r2 is None else "defined",
        "rmse_eV_per_A": float(np.sqrt(sse / count)),
        "residual_std_eV_per_A": float(np.sqrt(sse / max(count - 1, 1))),
        "max_abs_residual_eV_per_A": max_abs,
        "equations": count,
        "validation_samples": int(len(selected)),
        "validation_sample_indices": selected.tolist(),
    }


def fit(args) -> Path:
    from phonopy.file_IO import write_FORCE_CONSTANTS
    from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
    from symfc import Symfc
    from symfc.utils.utils import SymfcAtoms

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = args.trajectory.resolve() if args.trajectory is not None else (
        args.dataset_npz.resolve() if args.dataset_npz is not None else None
    )
    _guard_fit_output(output, source, force=bool(getattr(args, "force", False)))
    resolved_reference_mode = args.reference_mode
    if resolved_reference_mode == "auto":
        if args.unitcell and args.supercell:
            resolved_reference_mode = "provided"
        elif args.unitcell:
            resolved_reference_mode = "unitcell-inferred-supercell"
        else:
            resolved_reference_mode = "trajectory"
    if resolved_reference_mode == "outcar":
        resolved_reference_mode = "trajectory"
    request = {
        "schema": SCHEMA_VERSION,
        "trajectory": str(args.trajectory.resolve()) if args.trajectory else None,
        "dataset_npz": str(args.dataset_npz.resolve()) if args.dataset_npz else None,
        "input_sha256": sha256(source) if source is not None else None,
        "reference": {
            "requested_mode": args.reference_mode,
            "resolved_mode": resolved_reference_mode,
            "unitcell": str(args.unitcell.resolve()) if args.unitcell else None,
            "supercell": str(args.supercell.resolve()) if args.supercell else None,
            "dim": list(args.dim) if args.dim else None,
        },
        "selection": {
            "method": args.selection,
            "skip": args.skip,
            "stop": args.stop,
            "samples": args.samples,
            "stride": args.stride,
            "seed": args.seed,
            "center_selected": args.center_selected,
        },
        "force_constants": {
            "orders": list(args.order),
            "rc2_A": args.rc2,
            "rc3_A": args.rc3,
            "symprec": args.symprec,
            "map_tolerance_A": args.map_tolerance,
            "batch_size": args.batch_size,
        },
        "output": str(output),
    }
    with (output / "fit_request.yaml").open("w") as handle:
        yaml.safe_dump(request, handle, sort_keys=False)
    stage("fit", "Resolved input contract (also saved as fit_request.yaml):")
    print(yaml.safe_dump(request, sort_keys=False), end="", flush=True)
    stage("fit", f"Preparing selected trajectory data in {output}")
    unit, generated, u, f, summary = prepare_dataset(args, output)
    dataset_validation = validate_fit_dataset(u, f)
    displacement_variance = dataset_validation["displacement_variance_A2"]
    symfc_atoms = SymfcAtoms(
        numbers=np.asarray(generated.numbers),
        scaled_positions=np.asarray(generated.scaled_positions),
        cell=np.asarray(generated.cell),
    )
    sparse_mkl_available = False
    try:
        import sparse_dot_mkl  # noqa: F401

        sparse_mkl_available = True
    except ImportError:
        pass
    use_mkl = bool(args.use_mkl and sparse_mkl_available)
    orders = sorted(set(args.order))
    cutoff: dict[int, float] = {}
    if args.rc2 is not None:
        cutoff[2] = args.rc2
    rc3_auto_summary = None
    if 3 in orders:
        if args.rc3 == "auto":
            stage("fit", "Selecting FC3 cutoff automatically from symmetry-basis identifiability")
            rc3, rc3_auto_summary = resolve_auto_rc3(
                symfc_atoms, generated, len(u), use_mkl=use_mkl,
                minimum_ratio=args.rc3_auto_min_ratio,
            )
            cutoff[3] = rc3
            stage(
                "fit",
                f"FC3 auto selected rc3={rc3:.3g} A "
                f"({rc3_auto_summary['selection_reason']}; "
                f"minimum equations/parameter={args.rc3_auto_min_ratio:g})",
            )
        else:
            cutoff[3] = args.rc3
    request["force_constants"]["rc2_A"] = args.rc2
    request["force_constants"]["rc3_A"] = cutoff.get(3)
    request["force_constants"]["rc3_requested"] = args.rc3
    if rc3_auto_summary is not None:
        request["force_constants"]["rc3_auto"] = rc3_auto_summary
    with (output / "fit_request.yaml").open("w") as handle:
        yaml.safe_dump(request, handle, sort_keys=False)
    start = time.time()
    stage("fit", f"Fitting force constants of orders {orders} from {len(u)} configurations")
    solver_log = output / "symfc_solver.log"
    stage(
        "fit",
        f"symfc detailed log: {solver_log.name} "
        f"({'raw terminal output enabled' if args.verbose else 'use --verbose for raw terminal output'})",
    )
    with SymfcProgressCapture(solver_log, verbose=bool(args.verbose)) as capture:
        with contextlib.redirect_stdout(capture):
            solver = Symfc(
                supercell=symfc_atoms,
                displacements=u,
                forces=f,
                cutoff=cutoff,
                use_mkl=use_mkl,
                log_level=2,
            )
            solver.run(orders=orders, is_compact_fc=False, batch_size=args.batch_size)
    elapsed = time.time() - start
    fc2 = np.asarray(solver.force_constants[2])
    fc3 = np.asarray(solver.force_constants[3]) if 3 in orders else None
    if not np.isfinite(fc2).all() or (fc3 is not None and not np.isfinite(fc3).all()):
        raise RuntimeError("symfc produced non-finite force constants")

    write_FORCE_CONSTANTS(fc2, filename=str(output / "FORCE_CONSTANTS"))
    write_fc2_to_hdf5(fc2, filename=str(output / "fc2.hdf5"))
    if fc3 is not None:
        write_fc3_to_hdf5(fc3, filename=str(output / "fc3.hdf5"))
    else:
        # A later FC2-only fit must not leave an old FC3 available for an
        # accidental Gruneisen calculation in the same flat output directory.
        stale_fc3 = output / "fc3.hdf5"
        if stale_fc3.is_file() or stale_fc3.is_symlink():
            stale_fc3.unlink()
    # Store the exact structures used by postprocessing.
    from phonopy.interface.vasp import write_vasp

    # Reference preparation already records the unitcell. Do not rewrite a
    # user-supplied POSCAR after fitting, since it is the displacement contract.
    if not (output / "POSCAR-unitcell").is_file():
        write_vasp(str(output / "POSCAR-unitcell"), unit)
    write_vasp(str(output / "SPOSCAR"), generated)

    summary["software"] = versions()
    summary["schema"] = SCHEMA_VERSION
    summary["parallel"] = {
        "use_mkl_requested": bool(args.use_mkl),
        "sparse_dot_mkl_available": sparse_mkl_available,
        "use_mkl_effective": use_mkl,
        "solver_log": solver_log.name,
        "verbose": bool(args.verbose),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
    }
    basis_parameters = {
        str(order): int(solver.basis_set[order].basis_set.shape[0]) for order in orders
    }
    equations = int(3 * len(generated) * len(u))
    summary["fit"] = {
        "rc2_A": args.rc2,
        "rc3_A": cutoff.get(3),
        "rc3_requested": args.rc3,
        "rc3_auto": rc3_auto_summary,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "fc2_shape": list(fc2.shape),
        "fc3_shape": list(fc3.shape) if fc3 is not None else None,
        "selected_frames": int(len(u)),
        "dataset_validation": dataset_validation,
        "displacement_variance_A2": displacement_variance,
        "force_equations": equations,
        "basis_parameters": basis_parameters,
        "equations_per_parameter": {
            order: float(equations / parameters)
            for order, parameters in basis_parameters.items()
        },
        "fc2_max_translational_drift": float(np.max(np.abs(np.sum(fc2, axis=1)))),
        "fc3_max_translational_drift_j": float(np.max(np.abs(np.sum(fc3, axis=1)))) if fc3 is not None else None,
        "fc3_max_translational_drift_k": float(np.max(np.abs(np.sum(fc3, axis=2)))) if fc3 is not None else None,
        "permutation": force_constant_symmetry_diagnostics(fc2, fc3),
        "in_sample_reconstruction": force_metrics(u, f, fc2, fc3, args.metric_samples),
        "peak_memory_MiB": peak_memory_mib(),
    }
    with (output / "symfc_summary.yaml").open("w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    products = "FC2 and FC3" if fc3 is not None else "FC2"
    stage("fit", f"{products} written to {output} in {elapsed:.1f} s")
    return output


def seekpath_segments(unit, points_per_segment: int, *, bridge_discontinuities: bool = True):
    """Return a display-ready Seekpath band path.

    Seekpath can return consecutive path pieces whose end and start points are
    different (for example H2 -> H0 in rhombohedral cells).  Phonopy's
    conventional band path representation includes those connecting pieces so
    that the result has one unambiguous sequence of special-point labels.  Add
    those bridges by default; callers can still request the raw Seekpath pieces
    when that distinction is needed.
    """
    import seekpath

    structure = (
        np.asarray(unit.cell),
        np.asarray(unit.scaled_positions),
        np.asarray(unit.numbers),
    )
    path = seekpath.get_path(structure)
    segments = []
    labels = []
    previous_end = None
    previous_label = None
    for start, end in path["path"]:
        q0 = np.asarray(path["point_coords"][start], dtype=float)
        q1 = np.asarray(path["point_coords"][end], dtype=float)
        if bridge_discontinuities and previous_end is not None and not np.allclose(
            previous_end, q0, atol=1e-10, rtol=0
        ):
            segments.append(np.linspace(previous_end, q0, points_per_segment))
            labels.append((previous_label.replace("GAMMA", "Γ"), start.replace("GAMMA", "Γ")))
        segments.append(np.linspace(q0, q1, points_per_segment))
        labels.append((start.replace("GAMMA", "Γ"), end.replace("GAMMA", "Γ")))
        previous_end = q1
        previous_label = end
    return segments, labels


def flatten_band(gr, labels):
    gammas = gr.get_gruneisen_parameters()
    frequencies = gr._frequencies
    distances = gr._band_distances
    rows = []
    boundaries = [0.0]
    offset = 0.0
    for iseg, (gseg, fseg, dseg) in enumerate(zip(gammas, frequencies, distances)):
        shifted = np.asarray(dseg) + offset
        for iq, distance in enumerate(shifted):
            for mode in range(fseg.shape[1]):
                tensor = np.asarray(gseg[iq, mode])
                rows.append(
                    [
                        iseg,
                        iq,
                        distance,
                        mode + 1,
                        fseg[iq, mode],
                        tensor[0, 0],
                        tensor[1, 1],
                        tensor[2, 2],
                        (tensor[0, 0] + tensor[1, 1]) / 2,
                        np.trace(tensor) / 3,
                    ]
                )
        offset += float(dseg[-1])
        boundaries.append(offset)
    return np.asarray(rows), np.asarray(boundaries), labels


def write_band_tsv(path: Path, rows: np.ndarray) -> None:
    header = "segment q_index distance mode frequency_THz gamma_xx gamma_yy gamma_zz gamma_a gamma_trace_over_3"
    np.savetxt(path, rows, header=header, fmt=["%d", "%d", "%.10g", "%d"] + ["%.10g"] * 6)


def special_ticks(boundaries, labels):
    tick_labels = [labels[0][0]]
    for previous, current in zip(labels[:-1], labels[1:]):
        tick_labels.append(previous[1] if previous[1] == current[0] else f"{previous[1]}|{current[0]}")
    tick_labels.append(labels[-1][1])
    return boundaries, tick_labels


def plot_band_results(rows, boundaries, labels, output: Path, gmin: float, gmax: float, fmin_cm1: float, fmax_cm1: float, cutoff: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    ticks, tick_labels = special_ticks(boundaries, labels)
    components = [(5, r"$\gamma_{xx}$"), (6, r"$\gamma_{yy}$"), (7, r"$\gamma_{zz}$"), (9, r"$\mathrm{Tr}(\gamma)/3$")]
    valid = np.abs(rows[:, 4]) >= cutoff

    np.savetxt(
        output / "phonon_dispersion.tsv",
        rows[:, :5],
        header="segment q_index distance mode frequency_THz",
        fmt=["%d", "%d", "%.10g", "%d", "%.10g"],
    )
    stage("plot", "Writing phonon dispersion plot")
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for segment in np.unique(rows[:, 0].astype(int)):
        seg_rows = rows[rows[:, 0] == segment]
        for mode in np.unique(seg_rows[:, 3].astype(int)):
            data = seg_rows[seg_rows[:, 3] == mode]
            ax.plot(data[:, 2], data[:, 4] * CM1_PER_THZ, color="black", lw=0.75)
    ax.axhline(0, color="0.45", lw=0.8, ls="--")
    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.set_ylim(fmin_cm1, fmax_cm1)
    ax.set_ylabel(r"Frequency (cm$^{-1}$)")
    ax.set_xlabel("High-symmetry q path")
    ax.set_xticks(*special_ticks(boundaries, labels))
    for x in boundaries:
        ax.axvline(x, color="0.82", lw=0.7)
    fig.savefig(output / "phonon_dispersion.pdf")
    fig.savefig(output / "phonon_dispersion.png", dpi=180)
    plt.close(fig)

    stage("plot", "Writing q-resolved mode-Gruneisen plot")
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    if not gmin < 0 < gmax:
        raise ValueError("Gruneisen color range must straddle zero")
    # Keep zero white even for asymmetric display limits such as [-60, 20].
    # A linear Normalize would place zero at 75% of this range and incorrectly
    # render weakly negative modes red.
    norm = TwoSlopeNorm(vmin=gmin, vcenter=0.0, vmax=gmax)
    cmap = plt.get_cmap("bwr")
    for ax, (column, title) in zip(axes, components):
        colors = cmap(norm(np.clip(rows[:, column], gmin, gmax)))
        ax.scatter(rows[valid, 2], rows[valid, column], c=colors[valid], s=5, linewidths=0, rasterized=True)
        ax.axhline(0, color="0.45", lw=0.8, ls="--")
        ax.set_ylim(gmin, gmax)
        ax.set_ylabel("Mode Grüneisen parameter")
        ax.set_title(title)
        for x in boundaries:
            ax.axvline(x, color="0.82", lw=0.7)
    axes[-1].set_xticks(ticks, tick_labels)
    axes[-1].set_xlabel("High-symmetry q path")
    fig.savefig(output / "mode_gruneisen_q_resolved.pdf")
    fig.savefig(output / "mode_gruneisen_q_resolved.png", dpi=180)
    plt.close(fig)

    stage("plot", "Writing mode-Gruneisen overlay on the phonon dispersion")
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    for ax, (column, title) in zip(axes, components):
        for segment in np.unique(rows[:, 0].astype(int)):
            seg_rows = rows[rows[:, 0] == segment]
            for mode in np.unique(seg_rows[:, 3].astype(int)):
                data = seg_rows[seg_rows[:, 3] == mode]
                mask = np.abs(data[:, 4]) >= cutoff
                data = data[mask]
                if len(data) < 2:
                    continue
                points = np.column_stack([data[:, 2], data[:, 4] * CM1_PER_THZ])
                line_segments = np.stack([points[:-1], points[1:]], axis=1)
                values = (data[:-1, column] + data[1:, column]) / 2
                ax.add_collection(LineCollection(line_segments, cmap=cmap, norm=norm, array=values, linewidth=0.9))
        ax.axhline(0, color="0.45", lw=0.8, ls="--")
        ax.set_xlim(boundaries[0], boundaries[-1])
        ax.set_ylim(fmin_cm1, fmax_cm1)
        ax.set_ylabel(r"Frequency (cm$^{-1}$)")
        ax.set_title(title)
        for x in boundaries:
            ax.axvline(x, color="0.82", lw=0.7)
    axes[-1].set_xticks(ticks, tick_labels)
    axes[-1].set_xlabel("High-symmetry q path")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(scalar, ax=axes, label="Mode Grüneisen parameter", fraction=0.025)
    fig.savefig(output / "mode_gruneisen_on_phonon_dispersion.pdf")
    fig.savefig(output / "mode_gruneisen_on_phonon_dispersion.png", dpi=180)
    plt.close(fig)


def write_and_plot_mesh(output, unit, qpoints, weights, frequencies, tensors, mesh, gmin, gmax, cutoff):
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh_tag = "x".join(str(int(value)) for value in mesh)
    trace = np.trace(tensors, axis1=2, axis2=3) / 3
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(unit.cell)).T
    qnorm = np.linalg.norm(qpoints @ reciprocal, axis=1)
    with h5py.File(output / f"gruneisen_qmesh_{mesh_tag}.hdf5", "w") as h5:
        h5["mesh"] = np.asarray(mesh, dtype=int)
        h5["qpoint"] = qpoints
        h5["weight"] = weights
        h5["frequency_THz"] = frequencies
        # phono3py-compatible aliases used by the self-contained plotters.
        h5["frequency"] = frequencies
        h5["gruneisen_tensor"] = tensors
        h5["gruneisen"] = trace
        h5["gamma_xx"] = tensors[:, :, 0, 0]
        h5["gamma_yy"] = tensors[:, :, 1, 1]
        h5["gamma_zz"] = tensors[:, :, 2, 2]
        h5["gamma_trace_over_3"] = trace
    rows = []
    for iq, (qpoint, weight) in enumerate(zip(qpoints, weights)):
        for mode in range(frequencies.shape[1]):
            rows.append(
                [iq, *qpoint, qnorm[iq], weight, mode + 1, frequencies[iq, mode],
                 tensors[iq, mode, 0, 0], tensors[iq, mode, 1, 1],
                 tensors[iq, mode, 2, 2], trace[iq, mode]]
            )
    rows = np.asarray(rows)
    np.savetxt(
        output / f"gruneisen_qmesh_{mesh_tag}.tsv",
        rows,
        header="q_index qx qy qz qnorm_A^-1 weight mode frequency_THz gamma_xx gamma_yy gamma_zz gamma_trace_over_3",
        fmt=["%d"] + ["%.10g"] * 4 + ["%d", "%d"] + ["%.10g"] * 5,
    )
    components = [(8, r"$\gamma_{xx}$"), (9, r"$\gamma_{yy}$"), (10, r"$\gamma_{zz}$"), (11, r"$\mathrm{Tr}(\gamma)/3$")]
    valid = np.abs(rows[:, 7]) >= cutoff
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True, constrained_layout=True)
    for ax, (column, title) in zip(axes, components):
        ax.scatter(rows[valid, 4], rows[valid, column], s=4, alpha=0.45, linewidths=0, rasterized=True)
        ax.axhline(0, color="0.45", lw=0.8, ls="--")
        ax.set_ylim(gmin, gmax)
        ax.set_ylabel("Mode Grüneisen parameter")
        ax.set_title(title)
    axes[-1].set_xlabel(r"$|q|$ ($\AA^{-1}$)")
    fig.savefig(output / f"mode_gruneisen_qmesh_{mesh_tag}.pdf")
    fig.savefig(output / f"mode_gruneisen_qmesh_{mesh_tag}.png", dpi=180)
    plt.close(fig)
    return rows


def phonon(args, fit_dir: Path | None = None) -> Path:
    """Calculate an FC2-only phonon band structure without phono3py."""
    from phonopy import Phonopy
    from phonopy.file_IO import parse_BORN, parse_FORCE_CONSTANTS
    from phonopy.interface.vasp import read_vasp

    started = time.time()
    fit_dir = (fit_dir or args.fit_dir).resolve()
    output = args.analysis_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _guard_analysis_output(
        output, fit_dir, summary_name="phonon_summary.yaml",
        key="FORCE_CONSTANTS_sha256", source_name="FORCE_CONSTANTS",
        force=bool(getattr(args, "force", False)),
    )
    unit = read_vasp(str(fit_dir / "POSCAR-unitcell"))
    mass_summary = apply_mass_overrides(
        unit,
        parse_mass_overrides(args.mass),
        parse_atom_mass_overrides(getattr(args, "mass_index", None)),
    )
    reference_matrix = fit_dir / "supercell_matrix.dat"
    if not reference_matrix.is_file():
        # Read previous releases without forcing a migration of completed runs.
        reference_matrix = fit_dir / "reference" / "supercell_matrix.dat"
    if reference_matrix.is_file():
        matrix = np.loadtxt(reference_matrix, dtype=int)
        primitive_matrix = "P"
    else:
        if args.dim is None:
            raise FileNotFoundError(
                f"{fit_dir} has no supercell_matrix.dat; provide --dim NA NB NC "
                "only for a legacy diagonal-supercell result"
            )
        matrix = np.diag(args.dim)
        primitive_matrix = "auto"
    phonon = Phonopy(unit, supercell_matrix=matrix, primitive_matrix=primitive_matrix, symprec=args.symprec)
    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(fit_dir / "FORCE_CONSTANTS"))
    born_source = Path(args.born).resolve() if args.born is not None else fit_dir / "BORN"
    has_nac = born_source.is_file()
    nac_validation = None
    if has_nac:
        nac_params = parse_BORN(phonon.primitive, symprec=args.symprec, filename=born_source)
        from phonopy.physical_units import get_calculator_physical_units

        nac_params.setdefault("factor", get_calculator_physical_units("vasp").nac_factor)
        nac_validation = validate_nac_params(nac_params, len(phonon.primitive))
        phonon.nac_params = nac_params
        target_born = output / "BORN"
        if born_source.resolve() != target_born.resolve():
            shutil.copy2(born_source, target_born)
        stage("nac", f"Applying non-analytical correction from {born_source}")
    elif args.born is not None:
        raise FileNotFoundError(f"BORN file does not exist: {born_source}")
    segments, labels = seekpath_segments(unit, args.band_points)
    stage("phonon", f"Calculating {sum(len(segment) for segment in segments)} q points from FC2")
    connections = [
        bool(np.allclose(segments[index][-1], segments[index + 1][0], atol=1e-10, rtol=0))
        for index in range(len(segments) - 1)
    ] + [False]
    # The Phonopy 4 non-legacy plot accepts labels only at discontinuities.
    # Our bridged Seekpath route is deliberately one continuous high-symmetry
    # path, so use the legacy renderer: it accepts one label for every path
    # boundary and retains the conventional single-axis band plot.
    plot_labels = [labels[0][0], *(end for _, end in labels)]
    phonon.run_band_structure(
        segments,
        path_connections=connections,
        labels=plot_labels,
        is_legacy_plot=True,
    )
    phonon.write_yaml_band_structure(filename=str(output / "band.yaml"))
    stage("phonon", "Writing phonopy-band.dat with phonopy-bandplot --gnuplot")
    write_phonopy_bandplot_data(output)
    figure = phonon.plot_band_structure()
    figure.savefig(output / "band.pdf")
    figure.savefig(output / "band.png", dpi=180)
    import matplotlib.pyplot as plt

    plt.close("all")
    band = phonon.get_band_structure_dict()
    write_plain_phonon_band_dat(
        output / "phonon_band.dat", band["distances"], band["frequencies"]
    )
    groups: list[list[np.ndarray]] = [[segments[0][0], segments[0][-1]]]
    label_groups: list[list[str]] = [[labels[0][0], labels[0][1]]]
    for index, segment in enumerate(segments[1:], start=1):
        if connections[index - 1]:
            groups[-1].append(segment[-1])
            label_groups[-1].append(labels[index][1])
        else:
            groups.append([segment[0], segment[-1]])
            label_groups.append([labels[index][0], labels[index][1]])
    qpath = ", ".join(
        " ".join(f"{value:.10g}" for point in group for value in point) for group in groups
    )
    label_text = " ".join(label for group in label_groups for label in group)
    matrix_text = " ".join(str(int(value)) for value in np.asarray(matrix).reshape(-1))
    mass_text = " ".join(f"{float(value):.10g}" for value in unit.masses)
    band_conf = (
        "# phonopy input generated by symfc-vasp phonon\n"
        "FORCE_CONSTANTS = READ\n"
        + ("NAC = .TRUE.\n" if has_nac else "#NAC = .TRUE.\n")
        + f"DIM = {matrix_text}\nMASS = {mass_text}\n"
        + f"BAND = {qpath}\nBAND_POINTS = {args.band_points}\nBAND_LABELS = {label_text}\n"
        + "BAND_CONNECTION = .TRUE.\n"
    )
    (output / "band.conf").write_text(band_conf)
    phonon.save(filename=str(output / "phonopy_disp.yaml"), settings={"force_constants": True})
    link_force_constant_inputs(fit_dir, output)
    unitcell_target = output / "POSCAR-unitcell"
    if (fit_dir / "POSCAR-unitcell").resolve() != unitcell_target.resolve():
        shutil.copy2(fit_dir / "POSCAR-unitcell", unitcell_target)
    summary = {
        "schema": SCHEMA_VERSION,
        "source": "symfc-vasp phonon",
        "band_points": args.band_points,
        "segments": len(segments),
        "spacegroup": phonon.symmetry.dataset.international,
        "spacegroup_number": int(phonon.symmetry.dataset.number),
        "supercell_matrix": np.asarray(matrix, dtype=int).tolist(),
        "frequency_min_THz": float(np.min(np.concatenate(band["frequencies"]))),
        "frequency_max_THz": float(np.max(np.concatenate(band["frequencies"]))),
        "imaginary_mode_points": int(
            np.count_nonzero(np.concatenate(band["frequencies"]) < 0)
        ),
        "gamma_frequencies_THz": np.asarray(
            phonon.get_frequencies([0, 0, 0]), dtype=float
        ).tolist(),
        "gamma_acoustic_frequencies_THz": np.sort(
            np.asarray(phonon.get_frequencies([0, 0, 0]), dtype=float)
        )[:3].tolist(),
        "nac": {
            "enabled": has_nac,
            "born": str(born_source) if has_nac else None,
            "validation": nac_validation,
        },
        "masses": mass_summary,
        "inputs": {
            "FORCE_CONSTANTS_sha256": sha256(fit_dir / "FORCE_CONSTANTS"),
            "POSCAR_unitcell_sha256": sha256(fit_dir / "POSCAR-unitcell"),
            "BORN_sha256": sha256(born_source) if has_nac else None,
        },
        "timing": {
            "elapsed_seconds": float(time.time() - started),
            "peak_memory_MiB": peak_memory_mib(),
        },
    }
    with (output / "phonon_summary.yaml").open("w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    stage("phonon", f"Wrote band.yaml and band.pdf in {output}")
    return output

def postprocess(
    args,
    fit_dir: Path | None = None,
    *,
    do_band: bool = True,
    do_mesh: bool = True,
) -> Path:
    from phonopy import Phonopy
    from phonopy.file_IO import parse_BORN
    from phonopy.interface.vasp import read_vasp
    from phono3py import Phono3py
    from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
    from .gruneisen import accelerated_gruneisen_class

    started = time.time()
    fit_dir = (fit_dir or args.fit_dir).resolve()
    output = args.analysis_output.resolve()
    fc3_path = fit_dir / "fc3.hdf5"
    if not fc3_path.is_file():
        raise FileNotFoundError(
            f"FC3 file is required for mode-Gruneisen analysis: {fc3_path}. "
            "Run `symfc-vasp fit OUTCAR --fc3` first."
        )
    output.mkdir(parents=True, exist_ok=True)
    _guard_analysis_output(
        output, fit_dir, summary_name="analysis_summary.yaml",
        key="fc3_sha256", source_name="fc3.hdf5",
        force=bool(getattr(args, "force", False)),
    )
    link_force_constant_inputs(fit_dir, output)
    unit = read_vasp(str(fit_dir / "POSCAR-unitcell"))
    unitcell_target = output / "POSCAR-unitcell"
    if (fit_dir / "POSCAR-unitcell").resolve() != unitcell_target.resolve():
        shutil.copy2(fit_dir / "POSCAR-unitcell", unitcell_target)
    mass_summary = apply_mass_overrides(
        unit,
        parse_mass_overrides(args.mass),
        parse_atom_mass_overrides(getattr(args, "mass_index", None)),
    )
    if mass_summary["overrides_amu"] or mass_summary["atom_overrides_amu"]:
        stage(
            "mass",
            "Applying isotope mass overrides (amu): "
            f"species={mass_summary['overrides_amu']}, atoms={mass_summary['atom_overrides_amu']}",
        )
    reference_matrix = fit_dir / "supercell_matrix.dat"
    if reference_matrix.is_file():
        dim = np.loadtxt(reference_matrix, dtype=int)
        primitive_matrix = "P"
    else:
        if args.dim is None:
            raise FileNotFoundError(
                f"{fit_dir} has no supercell_matrix.dat; provide --dim NA NB NC "
                "only for a legacy diagonal-supercell result"
            )
        dim = np.diag(args.dim)
        primitive_matrix = "auto"
    ph3 = Phono3py(unit, supercell_matrix=dim, primitive_matrix=primitive_matrix, symprec=args.symprec)
    nac_params = None
    nac_validation = None
    requested_born = getattr(args, "born", None)
    born_source = Path(requested_born).resolve() if requested_born is not None else fit_dir / "BORN"
    if born_source.is_file():
        born_source = born_source.resolve()
        # BORN is expressed for the primitive cell and may contain only
        # symmetry-inequivalent Born-charge tensors, as in phonopy's format.
        nac_params = parse_BORN(ph3.primitive, symprec=args.symprec, filename=born_source)
        # parse_BORN returns tensors only. Phonopy requires the calculator's
        # electrostatic unit-conversion factor when NAC is assigned through
        # its Python API (the CLI supplies this implicitly for VASP).
        from phonopy.physical_units import get_calculator_physical_units
        nac_params.setdefault("factor", get_calculator_physical_units("vasp").nac_factor)
        nac_validation = validate_nac_params(nac_params, len(ph3.primitive))
        ph3.nac_params = nac_params
        born_target = output / "BORN"
        if born_source != born_target.resolve():
            shutil.copy2(born_source, born_target)
        stage("nac", f"Applying non-analytical correction from {born_source}")
    elif requested_born is not None:
        raise FileNotFoundError(f"BORN file does not exist: {born_source}")
    fc2 = read_fc2_from_hdf5(filename=str(fit_dir / "fc2.hdf5"))
    fc3 = read_fc3_from_hdf5(filename=str(fc3_path))
    ph3.fc2 = fc2
    ph3.fc3 = fc3
    write_phonopy_yaml(
        output, unit, dim, fc2, symprec=args.symprec, nac_params=nac_params,
    )

    stage("gruneisen", "Preparing the FC3 strain-derivative tensor (this can take time for large FC3 fits)")
    gr = accelerated_gruneisen_class()(
        fc2=fc2, fc3=fc3, supercell=ph3.supercell, primitive=ph3.primitive,
        nac_params=nac_params, symprec=args.symprec,
    )
    summary_path = output / "analysis_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text()) or {} if summary_path.is_file() else {}
    summary["software"] = versions()
    summary["schema"] = SCHEMA_VERSION
    summary["masses"] = mass_summary
    summary["nac"] = {
        "enabled": nac_params is not None,
        "born": str(born_source) if nac_params is not None else None,
        "validation": nac_validation,
    }
    summary["inputs"] = {
        "fc2_sha256": sha256(fit_dir / "fc2.hdf5"),
        "fc3_sha256": sha256(fc3_path),
        "POSCAR_unitcell_sha256": sha256(fit_dir / "POSCAR-unitcell"),
        "BORN_sha256": sha256(born_source) if nac_params is not None else None,
    }

    if do_band:
        stage("phonon", "Building the high-symmetry q path with seekpath")
        segments, labels = seekpath_segments(unit, args.band_points)
        stage("phonon", f"Calculating {sum(len(segment) for segment in segments)} band-path q points")
        gr.set_band_structure(segments)
        gr.run()
        rows, boundaries, labels = flatten_band(gr, labels)
        stage("gruneisen", "Writing the band-path tensor mode-Gruneisen data")
        write_band_tsv(output / "mode_gruneisen_qpath.tsv", rows)
        write_band_dat(output / "phonon_band.dat", rows)
        write_phonon_inputs(
            output, segments, labels, dim, args.band_points, args.mesh,
            unit.masses, has_nac=nac_params is not None,
        )
        write_band_gnuplot_scripts(
            output, boundaries, labels, args.gmin, args.gmax,
            args.fmin_cm1, args.fmax_cm1, args.frequency_cutoff,
        )
        write_reproduction_readme(output, args.mesh)
        gr.write(filename=str(output / "gruneisen_band"))
        # Keep the former dashed filename for existing downstream scripts.
        shutil.copy2(output / "gruneisen_band.yaml", output / "gruneisen-band.yaml")
        plot_band_results(
            rows, boundaries, labels, output, args.gmin, args.gmax,
            args.fmin_cm1, args.fmax_cm1, args.frequency_cutoff,
        )
        summary["band"] = {
            "segments": len(segments),
            "points_per_segment": args.band_points,
            "rows": int(len(rows)),
            "frequency_cutoff_THz": args.frequency_cutoff,
            "gruneisen_display_range": [args.gmin, args.gmax],
        }

    if do_mesh:
        # Match phono3py's CLI behavior by reducing the mesh with primitive
        # point-group operations.
        mesh_tag = "x".join(str(int(value)) for value in args.mesh)
        stage("gruneisen", f"Calculating the {mesh_tag} q-mesh tensor")
        # phono3py 4.x accepts the Symmetry object directly, while 3.x used
        # rotations and an explicit Gamma-centering flag.
        mesh_parameters = inspect.signature(gr.set_sampling_mesh).parameters
        if "primitive_symmetry" in mesh_parameters:
            gr.set_sampling_mesh(args.mesh, primitive_symmetry=ph3.primitive_symmetry)
        else:
            gr.set_sampling_mesh(
                args.mesh,
                rotations=ph3.primitive_symmetry.pointgroup_operations,
                is_gamma_center=True,
            )
        gr.run()
        tensors = np.asarray(gr.get_gruneisen_parameters())
        qpoints = np.asarray(gr._qpoints)
        weights = np.asarray(gr._weights)
        frequencies = np.asarray(gr._frequencies)
        stage("gruneisen", f"Writing {len(qpoints)} irreducible q points")
        np.savez_compressed(
            output / f"gruneisen_qmesh_{mesh_tag}.npz",
            qpoints=qpoints,
            weights=weights,
            frequencies_THz=frequencies,
            gruneisen_tensor=tensors,
            gamma_xx=tensors[:, :, 0, 0],
            gamma_yy=tensors[:, :, 1, 1],
            gamma_zz=tensors[:, :, 2, 2],
            gamma_trace_over_3=np.trace(tensors, axis1=2, axis2=3) / 3,
        )
        stage("plot", f"Writing {mesh_tag} q-mesh mode-Gruneisen plot")
        mesh_rows = write_and_plot_mesh(
            output, unit, qpoints, weights, frequencies, tensors, args.mesh,
            args.gmin, args.gmax, args.frequency_cutoff,
        )
        write_gruneisen_mesh_yaml(
            output, unit, args.mesh, qpoints, weights, frequencies, tensors,
        )
        # ``mesh`` can be rerun independently after an older ``band`` run.
        # Preserve backward compatibility while guaranteeing the underscore
        # filename consumed by the tensor plotter bundle.
        legacy_band_yaml = output / "gruneisen-band.yaml"
        tensor_band_yaml = output / "gruneisen_band.yaml"
        if not tensor_band_yaml.is_file() and legacy_band_yaml.is_file():
            shutil.copy2(legacy_band_yaml, tensor_band_yaml)
        component_config = write_component_config(output, unit, args.symprec)
        stage(
            "plot",
            "Using %s conventional-frame components: %s"
            % (component_config["symmetry"]["crystal_system"], ", ".join(component_config["components"])),
        )
        write_tensor_plotter_bundle(output, args.mesh, component_config["components"])
        canonical_mesh = output / "gruneisen_mesh.hdf5"
        tagged_mesh = output / f"gruneisen_qmesh_{mesh_tag}.hdf5"
        if canonical_mesh.is_symlink() or canonical_mesh.exists():
            canonical_mesh.unlink()
        canonical_mesh.symlink_to(tagged_mesh.name)
        stage("plot", "Rendering self-contained tensor Gruneisen plotter outputs")
        render_tensor_plotter_bundle(
            output, args.mesh, args.gmin, args.gmax,
            args.fmin_cm1, args.fmax_cm1, args.frequency_cutoff,
            component_config["components"],
        )
        write_mesh_dat(output / f"gruneisen_qmesh_{mesh_tag}.dat", mesh_rows)
        write_mesh_gnuplot_script(
            output, args.mesh, args.gmin, args.gmax, args.frequency_cutoff,
        )
        write_reproduction_readme(output, args.mesh)
        summary["mesh"] = {
            "mesh": list(args.mesh),
            "irreducible_qpoints": int(len(qpoints)),
            "modes": int(frequencies.shape[1]),
            "tensor_shape": list(tensors.shape),
        }
    summary["reproducibility"] = {
        "phonopy_yaml": "phonopy_disp.yaml",
        "phonopy_input": "band.conf" if do_band else None,
        "phono3py_band_input": "phono3py-gruneisen-band.conf" if do_band else None,
        "phono3py_mesh_input": "phono3py-gruneisen-mesh.conf" if do_band else None,
        "gnuplot_terminal_default": "pdfcairo",
        "gnuplot_terminal_override_example": "gnuplot -e 'plot_terminal=\"qt\"' SCRIPT.gp",
    }
    summary["timing"] = {
        "elapsed_seconds": float(time.time() - started),
        "peak_memory_MiB": peak_memory_mib(),
    }

    with summary_path.open("w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    completed = "band and mesh" if do_band and do_mesh else "band" if do_band else "mesh"
    stage("analysis", f"Completed {completed} outputs in {output}")
    return output
