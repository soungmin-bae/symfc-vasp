"""Build a symmetric phonopy reference directly from a fixed-cell OUTCAR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import spglib
import yaml
from scipy.optimize import linear_sum_assignment

from .parsers.outcar import parse_outcar_metadata


def _periodic_mean(frames: np.ndarray) -> np.ndarray:
    phase = np.exp(2j * np.pi * np.asarray(frames, dtype=float))
    return np.mod(np.angle(np.mean(phase, axis=0)) / (2.0 * np.pi), 1.0)


def _minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Shortest Cartesian images, including the neighbouring translations."""
    wrapped = np.asarray(delta, dtype=float) - np.rint(delta)
    candidates = []
    for shift in np.ndindex(3, 3, 3):
        shift = np.asarray(shift, dtype=float) - 1.0
        candidates.append((wrapped + shift) @ cell)
    stacked = np.stack(candidates, axis=0)
    norms = np.einsum("s...i,s...i->s...", stacked, stacked)
    return np.take_along_axis(stacked, np.argmin(norms, axis=0)[None, ..., None], axis=0)[0]


def _species_numbers(symbols: tuple[str, ...]) -> np.ndarray:
    from phonopy.structure.atoms import get_atomic_data

    symbol_map = get_atomic_data().symbol_map
    return np.asarray([symbol_map[symbol] for symbol in symbols], dtype=int)


def _align_origin(
    generated_frac: np.ndarray, mean_frac: np.ndarray, generated_symbols: np.ndarray,
    source_symbols: np.ndarray, cell: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Find the rigid fractional translation between spglib and VASP origins."""
    first_symbol = next(iter(dict.fromkeys(generated_symbols.tolist())))
    ig = np.flatnonzero(generated_symbols == first_symbol)
    isource = np.flatnonzero(source_symbols == first_symbol)
    best_shift = np.zeros(3)
    best_score = float("inf")
    for source_index in isource:
        for generated_index in ig:
            shift = (mean_frac[source_index] - generated_frac[generated_index]) % 1.0
            score = 0.0
            for symbol in dict.fromkeys(generated_symbols.tolist()):
                left = np.flatnonzero(generated_symbols == symbol)
                right = np.flatnonzero(source_symbols == symbol)
                delta = (generated_frac[left, None, :] + shift) - mean_frac[right][None, :, :]
                cost = np.linalg.norm(_minimum_image(delta, cell), axis=-1)
                rows, cols = linear_sum_assignment(cost)
                score = max(score, float(np.max(cost[rows, cols])))
            if score < best_score:
                best_score = score
                best_shift = shift
    return best_shift, best_score


def _write_poscar(path: Path, atoms, comment: str) -> None:
    from phonopy.interface.vasp import write_vasp

    write_vasp(str(path), atoms)
    lines = path.read_text().splitlines()
    lines[0] = comment
    path.write_text("\n".join(lines) + "\n")


def _symprec_grid(maximum: float) -> list[float]:
    if maximum <= 0:
        raise ValueError("--reference-symprec-max must be positive")
    # A random-displacement trajectory retains small finite-sample offsets in
    # its periodic mean. Values above 0.1 A are often needed to recover the
    # parent space group while still being far below an inter-site distance.
    base = [
        1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2,
        5e-2, 7e-2, 1e-1, 1.5e-1, 2e-1, 3e-1,
    ]
    grid = [value for value in base if value <= maximum * (1 + 1e-12)]
    if not grid or grid[-1] < maximum:
        grid.append(float(maximum))
    return sorted(set(grid))


def build_outcar_reference(
    *,
    outcar: Path,
    positions: np.ndarray,
    output: Path,
    symprec_max: float,
    map_tolerance: float,
    symbols: tuple[str, ...] | None = None,
    cell: np.ndarray | None = None,
) -> tuple[object, object, np.ndarray, dict]:
    """Return a symmetry-projected unit cell, supercell, map, and manifest.

    ``positions`` retain VASP's original atom order.  The returned mapping is
    phonopy-supercell index -> OUTCAR index and is therefore applied to every
    trajectory frame before fitting.
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    if symbols is None or cell is None:
        metadata = parse_outcar_metadata(outcar)
        symbols = metadata.symbols
        cell = metadata.cell
        lattice_records = metadata.lattice_records
    else:
        lattice_records = None
    cell = np.asarray(cell, dtype=float)
    if positions.ndim != 3 or positions.shape[1] != len(symbols):
        raise ValueError("trajectory positions do not match recovered species metadata")
    mean_frac = _periodic_mean(np.asarray(positions) @ np.linalg.inv(cell))
    numbers = _species_numbers(symbols)
    scan: list[dict] = []
    candidates: list[dict] = []
    best_mapping_residual = np.inf
    for symprec in _symprec_grid(symprec_max):
        dataset = spglib.get_symmetry_dataset((cell, mean_frac, numbers), symprec=symprec)
        if dataset is None:
            continue
        record = {
            "symprec_A": float(symprec),
            "international": str(dataset.international),
            "number": int(dataset.number),
            "operations": int(len(dataset.rotations)),
        }
        try:
            standardized = spglib.standardize_cell(
                (cell, mean_frac, numbers), to_primitive=True,
                no_idealize=False, symprec=symprec,
            )
            if standardized is None:
                raise ValueError("standardization failed")
            unit_lattice, unit_frac, unit_numbers = standardized
            matrix_float = cell @ np.linalg.inv(unit_lattice)
            matrix = np.rint(matrix_float).astype(int)
            matrix_residual = float(np.max(np.abs(matrix_float - matrix)))
            determinant = abs(int(round(np.linalg.det(matrix))))
            if matrix_residual > 1e-6 or determinant * len(unit_numbers) != len(symbols):
                raise ValueError(
                    f"non-integer cell relation (residual={matrix_residual:.3e}, det={determinant})"
                )
            unit = PhonopyAtoms(
                cell=unit_lattice, scaled_positions=unit_frac, numbers=unit_numbers
            )
            phonon = Phonopy(
                unit, supercell_matrix=matrix, primitive_matrix="P", symprec=symprec
            )
            generated = phonon.supercell
            if not np.allclose(generated.cell, cell, atol=1e-6, rtol=0):
                raise ValueError("generated supercell lattice differs from trajectory")
            generated_frac = np.asarray(generated.scaled_positions)
            source_symbols = np.asarray(symbols)
            generated_symbols = np.asarray(generated.symbols)
            origin_shift, origin_residual = _align_origin(
                generated_frac, mean_frac, generated_symbols, source_symbols, cell
            )
            generated_frac = (generated_frac + origin_shift) % 1.0
            mapping = np.full(len(generated), -1, dtype=int)
            distances = np.full(len(generated), np.nan)
            for symbol in dict.fromkeys(generated_symbols.tolist()):
                ig = np.flatnonzero(generated_symbols == symbol)
                isource = np.flatnonzero(source_symbols == symbol)
                if len(ig) != len(isource):
                    raise ValueError(f"species count differs for {symbol}")
                delta = generated_frac[ig][:, None, :] - mean_frac[isource][None, :, :]
                cost = np.linalg.norm(_minimum_image(delta, cell), axis=-1)
                rows, cols = linear_sum_assignment(cost)
                mapping[ig[rows]] = isource[cols]
                distances[ig[rows]] = cost[rows, cols]
            max_distance = float(np.max(distances))
            best_mapping_residual = min(best_mapping_residual, max_distance)
            if np.any(mapping < 0) or len(np.unique(mapping)) != len(mapping):
                raise ValueError("atom map is incomplete")
            if max_distance > map_tolerance:
                raise ValueError(
                    f"atom-map residual {max_distance:.4f} A exceeds {map_tolerance:.4f} A"
                )
            generated = PhonopyAtoms(
                cell=generated.cell,
                scaled_positions=generated_frac,
                symbols=generated.symbols,
            )
            signature = (
                int(dataset.number), len(unit), tuple(int(value) for value in matrix.reshape(-1))
            )
            record.update({
                "valid": True,
                "unitcell_atoms": int(len(unit)),
                "supercell_matrix": matrix.tolist(),
                "matrix_residual": matrix_residual,
                "mapping_max_distance_A": max_distance,
                "mapping_rms_distance_A": float(np.sqrt(np.mean(distances**2))),
            })
            candidates.append({
                "dataset": dataset, "symprec": float(symprec), "unit": unit,
                "generated": generated, "matrix": matrix,
                "matrix_residual": matrix_residual, "mapping": mapping,
                "distances": distances, "origin_shift": origin_shift,
                "origin_residual": float(origin_residual), "signature": signature,
            })
        except (ValueError, np.linalg.LinAlgError) as exc:
            record.update({"valid": False, "rejection": str(exc)})
        scan.append(record)
    if not candidates:
        raise ValueError(
            "no symmetry candidate passed integer-cell and atom-map validation"
            + (
                f"; best atom-map residual={best_mapping_residual:.4f} A"
                if np.isfinite(best_mapping_residual) else ""
            )
        )
    counts: dict[tuple, int] = {}
    for candidate in candidates:
        counts[candidate["signature"]] = counts.get(candidate["signature"], 0) + 1
    stable = [candidate for candidate in candidates if counts[candidate["signature"]] >= 2]
    pool = stable or candidates
    selected = max(
        pool,
        key=lambda candidate: (
            len(candidate["dataset"].rotations),
            counts[candidate["signature"]],
            -candidate["symprec"],
        ),
    )
    chosen = selected["dataset"]
    chosen_symprec = selected["symprec"]
    unit = selected["unit"]
    generated = selected["generated"]
    matrix = selected["matrix"]
    matrix_residual = selected["matrix_residual"]
    mapping = selected["mapping"]
    distances = selected["distances"]
    origin_shift = selected["origin_shift"]
    origin_residual = selected["origin_residual"]

    mean_atoms = PhonopyAtoms(cell=cell, scaled_positions=mean_frac, symbols=symbols)
    _write_poscar(output / "POSCAR-mean", mean_atoms, "Periodic mean structure recovered from trajectory")
    _write_poscar(output / "POSCAR-unitcell", unit, "spglib-symmetrized primitive reference from trajectory")
    _write_poscar(output / "POSCAR-supercell", generated, "spglib-symmetrized supercell reference from trajectory")
    _write_poscar(output / "SPOSCAR", generated, "Phonopy supercell reference from trajectory")
    np.savetxt(output / "supercell_matrix.dat", matrix, fmt="%d")
    (output / "generated_to_outcar_index.json").write_text(
        json.dumps({"generated_supercell_index_to_outcar_index": mapping.tolist()}, indent=2) + "\n"
    )
    manifest = {
        "schema": "symfc-vasp-symmetry-report-v2",
        "source": "trajectory-periodic-mean",
        "trajectory": str(Path(outcar).resolve()),
        "natom": len(symbols),
        "lattice_records": lattice_records,
        "symmetry_scan": scan,
        "selection_policy": {
            "requirements": [
                "integer primitive-to-supercell relation",
                "species-preserving one-to-one atom map",
                "mapping residual within tolerance",
            ],
            "stable_plateau_minimum": 2,
            "stable_candidate_used": bool(stable),
            "selected_signature_occurrences": int(counts[selected["signature"]]),
            "ranking": "operations, plateau length, smallest symprec",
        },
        "selected_symprec_A": float(chosen_symprec),
        "selected_spacegroup": {"international": str(chosen.international), "number": int(chosen.number), "operations": int(len(chosen.rotations))},
        "unitcell_atoms": int(len(unit)),
        "supercell_matrix": matrix.tolist(),
        "supercell_matrix_residual": matrix_residual,
        "mapping": {
            "max_distance_A": float(np.max(distances)),
            "rms_distance_A": float(np.sqrt(np.mean(distances ** 2))),
            "tolerance_A": float(map_tolerance),
            "origin_shift_fractional": origin_shift.tolist(),
            "origin_alignment_residual_A": float(origin_residual),
        },
    }
    (output / "symmetry_report.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return unit, generated, mapping, manifest
