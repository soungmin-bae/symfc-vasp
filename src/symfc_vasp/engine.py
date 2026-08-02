#!/usr/bin/env python3
"""Fit finite-temperature FC2/FC3 from a VASP OUTCAR using symfc.

The workflow intentionally uses the positions and Born-Oppenheimer forces
already stored in OUTCAR.  It does not invoke an external force calculator.
The unit-cell symmetry is transferred to the supplied supercell through an
explicit, species-preserving periodic atom map.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


CM1_PER_THZ = 33.35640951981521
from .parsers import parse_trajectory
from .parsers.outcar import scan_outcar_summary
from .parsers.vasprun import count_vasprun_frames
from .selection import select_indices
from .reproducibility import (
    write_band_dat,
    write_band_gnuplot_scripts,
    write_mesh_dat,
    write_mesh_gnuplot_script,
    write_phonon_inputs,
    write_phonopy_yaml,
    write_reproduction_readme,
)


def stage(name: str, message: str) -> None:
    """Write a progress message immediately for interactive and batch logs."""
    print(f"[{name}] {message}", flush=True)


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


def apply_mass_overrides(unit, overrides: dict[str, float]) -> dict:
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
    unit.masses = effective
    return {
        "overrides_amu": dict(overrides),
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
        if source.is_file() and not link.exists() and not link.is_symlink():
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
    import symfc

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "phonopy": phonopy.__version__,
        "phono3py": phono3py.__version__,
        "symfc": symfc.__version__,
    }


def periodic_distance_matrix(frac_a: np.ndarray, frac_b: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = frac_a[:, None, :] - frac_b[None, :, :]
    delta -= np.rint(delta)
    return np.linalg.norm(delta @ cell, axis=-1)


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


def prepare_dataset(args, output: Path) -> tuple[object, object, np.ndarray, np.ndarray, dict]:
    from phonopy import Phonopy
    from phonopy.interface.vasp import read_vasp

    unit = read_vasp(str(args.unitcell))
    given = read_vasp(str(args.supercell))
    dim = np.diag(args.dim)
    phonon = Phonopy(
        unit,
        supercell_matrix=dim,
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
        nframes = int(source_indices[-1]) + 1
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
        if natom != len(given):
            raise ValueError(f"trajectory atoms={natom}, POSCAR-supercell atoms={len(given)}")
        try:
            source_indices = select_indices(
                nframes, skip=args.skip, samples=args.samples, stride=args.stride,
                method=args.selection, seed=args.seed,
            )
        except ValueError as exc:
            context = [
                f"trajectory contains {nframes} force/position frames",
                f"requested skip={args.skip}, samples={args.samples}",
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
        delta_frac -= np.rint(delta_frac)
        displacements = delta_frac @ cell
        trajectory_info = {
            "path": str(trajectory.resolve()), "format": dataset.source_format,
            "sha256": sha256(trajectory), "natom": natom,
            "force_blocks": nframes, "ml_force_blocks": nframes_ml,
        }

    mean_displacement = np.mean(displacements, axis=0)
    mean_force = np.mean(forces, axis=0)
    centering = {
        "enabled": bool(args.center_selected),
        "mean_displacement_rms_A": float(np.sqrt(np.mean(mean_displacement**2))),
        "mean_displacement_max_atom_norm_A": float(np.max(np.linalg.norm(mean_displacement, axis=1))),
        "mean_force_rms_eV_per_A": float(np.sqrt(np.mean(mean_force**2))),
        "mean_force_max_atom_norm_eV_per_A": float(np.max(np.linalg.norm(mean_force, axis=1))),
    }
    if args.center_selected:
        displacements = displacements - mean_displacement[None, :, :]
        forces = forces - mean_force[None, :, :]

    summary = {
        "trajectory": trajectory_info,
        "selection": {
            "skip": args.skip,
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
            "dim": list(args.dim),
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
    return {
        "r2": float(1.0 - sse / sst),
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
    stage("fit", f"Preparing selected trajectory data in {output}")
    output.mkdir(parents=True, exist_ok=True)
    unit, generated, u, f, summary = prepare_dataset(args, output)
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
    cutoff = {2: args.rc2}
    if 3 in orders:
        cutoff[3] = args.rc3
    start = time.time()
    stage("fit", f"Fitting force constants of orders {orders} from {len(u)} configurations")
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
    # Store the exact structures used by postprocessing.
    from phonopy.interface.vasp import write_vasp

    write_vasp(str(output / "POSCAR-unitcell"), unit)
    write_vasp(str(output / "SPOSCAR"), generated)

    summary["software"] = versions()
    summary["parallel"] = {
        "use_mkl_requested": bool(args.use_mkl),
        "sparse_dot_mkl_available": sparse_mkl_available,
        "use_mkl_effective": use_mkl,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
    }
    summary["fit"] = {
        "rc2_A": args.rc2,
        "rc3_A": args.rc3,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "fc2_shape": list(fc2.shape),
        "fc3_shape": list(fc3.shape) if fc3 is not None else None,
        "fc2_max_translational_drift": float(np.max(np.abs(np.sum(fc2, axis=1)))),
        "fc3_max_translational_drift_j": float(np.max(np.abs(np.sum(fc3, axis=1)))) if fc3 is not None else None,
        "fc3_max_translational_drift_k": float(np.max(np.abs(np.sum(fc3, axis=2)))) if fc3 is not None else None,
        "force_reconstruction": force_metrics(u, f, fc2, fc3, args.metric_samples),
    }
    with (output / "symfc_summary.yaml").open("w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    stage("fit", f"FC2/FC3 written to {output} in {elapsed:.1f} s")
    return output


def seekpath_segments(unit, points_per_segment: int):
    import seekpath

    structure = (
        np.asarray(unit.cell),
        np.asarray(unit.scaled_positions),
        np.asarray(unit.numbers),
    )
    path = seekpath.get_path(structure)
    segments = []
    labels = []
    for start, end in path["path"]:
        q0 = np.asarray(path["point_coords"][start], dtype=float)
        q1 = np.asarray(path["point_coords"][end], dtype=float)
        segments.append(np.linspace(q0, q1, points_per_segment))
        labels.append((start.replace("GAMMA", "Γ"), end.replace("GAMMA", "Γ")))
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
        h5["gruneisen_tensor"] = tensors
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

def postprocess(
    args,
    fit_dir: Path | None = None,
    *,
    do_band: bool = True,
    do_mesh: bool = True,
) -> Path:
    from phonopy import Phonopy
    from phonopy.interface.vasp import read_vasp
    from phono3py import Phono3py
    from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
    from phono3py.phonon3.gruneisen import Gruneisen

    fit_dir = (fit_dir or args.fit_dir).resolve()
    output = args.analysis_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    link_force_constant_inputs(fit_dir, output)
    unit = read_vasp(str(fit_dir / "POSCAR-unitcell"))
    shutil.copy2(fit_dir / "POSCAR-unitcell", output / "POSCAR-unitcell")
    mass_summary = apply_mass_overrides(unit, parse_mass_overrides(args.mass))
    if mass_summary["overrides_amu"]:
        stage("mass", f"Applying isotope mass overrides (amu): {mass_summary['overrides_amu']}")
    dim = np.diag(args.dim)
    ph3 = Phono3py(unit, supercell_matrix=dim, primitive_matrix="auto", symprec=args.symprec)
    fc2 = read_fc2_from_hdf5(filename=str(fit_dir / "fc2.hdf5"))
    fc3 = read_fc3_from_hdf5(filename=str(fit_dir / "fc3.hdf5"))
    ph3.fc2 = fc2
    ph3.fc3 = fc3
    write_phonopy_yaml(
        output, unit, args.dim, fc2, symprec=args.symprec,
    )

    gr = Gruneisen(fc2=fc2, fc3=fc3, supercell=ph3.supercell, primitive=ph3.primitive, symprec=args.symprec)
    summary_path = output / "analysis_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text()) or {} if summary_path.is_file() else {}
    summary["software"] = versions()
    summary["masses"] = mass_summary

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
            output, segments, labels, args.dim, args.band_points, args.mesh,
            unit.masses,
        )
        write_band_gnuplot_scripts(
            output, boundaries, labels, args.gmin, args.gmax,
            args.fmin_cm1, args.fmax_cm1, args.frequency_cutoff,
        )
        write_reproduction_readme(output, args.mesh)
        gr.write(filename=str(output / "gruneisen-band"))
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

    with summary_path.open("w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    completed = "band and mesh" if do_band and do_mesh else "band" if do_band else "mesh"
    stage("analysis", f"Completed {completed} outputs in {output}")
    return output


def render_existing(args) -> Path:
    """Regenerate band plots from an existing mode_gruneisen_qpath.tsv."""
    from phonopy.interface.vasp import read_vasp
    from phono3py.file_IO import read_fc2_from_hdf5

    fit_dir = args.fit_dir.resolve()
    output = args.analysis_output.resolve()
    link_force_constant_inputs(fit_dir, output)
    shutil.copy2(fit_dir / "POSCAR-unitcell", output / "POSCAR-unitcell")
    rows = np.loadtxt(output / "mode_gruneisen_qpath.tsv")
    unit = read_vasp(str(fit_dir / "POSCAR-unitcell"))
    mass_summary = apply_mass_overrides(unit, parse_mass_overrides(args.mass))
    fc2 = read_fc2_from_hdf5(filename=str(fit_dir / "fc2.hdf5"))
    write_phonopy_yaml(
        output, unit, args.dim, fc2, symprec=args.symprec,
    )
    segments, labels = seekpath_segments(unit, args.band_points)
    segment_ids = np.unique(rows[:, 0].astype(int))
    boundaries = [float(np.min(rows[rows[:, 0] == segment_ids[0], 2]))]
    boundaries.extend(float(np.max(rows[rows[:, 0] == segment, 2])) for segment in segment_ids)
    plot_band_results(
        rows, np.asarray(boundaries), labels, output,
        args.gmin, args.gmax, args.fmin_cm1, args.fmax_cm1, args.frequency_cutoff,
    )
    write_band_dat(output / "phonon_band.dat", rows)
    write_band_gnuplot_scripts(
        output, boundaries, labels, args.gmin, args.gmax,
        args.fmin_cm1, args.fmax_cm1, args.frequency_cutoff,
    )
    write_phonon_inputs(
        output, segments, labels, args.dim, args.band_points, args.mesh,
        unit.masses,
    )
    mesh_tag = "x".join(str(int(value)) for value in args.mesh)
    mesh_tsv = output / f"gruneisen_qmesh_{mesh_tag}.tsv"
    if mesh_tsv.is_file():
        mesh_rows = np.loadtxt(mesh_tsv)
        write_mesh_dat(output / f"gruneisen_qmesh_{mesh_tag}.dat", mesh_rows)
        write_mesh_gnuplot_script(
            output, args.mesh, args.gmin, args.gmax, args.frequency_cutoff,
        )
    write_reproduction_readme(output, args.mesh)
    summary_path = output / "analysis_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text()) or {} if summary_path.is_file() else {}
    summary["masses"] = mass_summary
    summary["reproducibility"] = {
        "phonopy_yaml": "phonopy_disp.yaml",
        "phonopy_input": "band.conf",
        "phono3py_band_input": "phono3py-gruneisen-band.conf",
        "phono3py_mesh_input": "phono3py-gruneisen-mesh.conf",
        "gnuplot_terminal_default": "pdfcairo",
        "gnuplot_terminal_override_example": "gnuplot -e 'plot_terminal=\"qt\"' SCRIPT.gp",
    }
    with summary_path.open("w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print(f"Band plots regenerated in {output}")
    return output


def add_common_fit(parser):
    parser.add_argument("--trajectory", type=Path, default=Path("OUTCAR"))
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        help="Reuse a validated symfc_input.npz instead of reparsing the trajectory.",
    )
    parser.add_argument("--supercell", type=Path, default=Path("POSCAR-supercell"))
    parser.add_argument("--unitcell", type=Path, default=Path("POSCAR-unitcell"))
    parser.add_argument("--dim", nargs=3, type=int, default=(2, 2, 2), metavar=("NA", "NB", "NC"))
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--selection", choices=("stride", "uniform", "random"), default="stride")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--center-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fit fluctuations around the selected finite-temperature mean by subtracting "
            "mean displacement and mean force (default: enabled; disable with "
            "--no-center-selected)."
        ),
    )
    parser.add_argument("--order", nargs="+", type=int, choices=(2, 3), default=(2, 3))
    parser.add_argument("--rc2", type=float, default=7.0)
    parser.add_argument("--rc3", type=float, default=4.0)
    parser.add_argument("--symprec", type=float, default=1e-5)
    parser.add_argument("--map-tolerance", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--metric-samples",
        type=int,
        default=5,
        help="Uniformly spaced frames used for explicit FC2+FC3 force reconstruction (all selected frames are still fitted).",
    )
    parser.add_argument("--use-mkl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("fit"))


def add_analysis(parser):
    parser.add_argument("--fit-dir", type=Path, default=Path("fit"))
    parser.add_argument("--analysis-output", type=Path, default=Path("analysis"))
    parser.add_argument("--dim", nargs=3, type=int, default=(2, 2, 2), metavar=("NA", "NB", "NC"))
    parser.add_argument("--symprec", type=float, default=1e-5)
    parser.add_argument("--band-points", type=int, default=21)
    parser.add_argument("--mesh", nargs=3, type=int, default=(11, 11, 11), metavar=("NQ1", "NQ2", "NQ3"))
    parser.add_argument("--gmin", type=float, default=-60.0)
    parser.add_argument("--gmax", type=float, default=20.0)
    parser.add_argument("--frequency-cutoff", type=float, default=0.05)
    parser.add_argument("--fmin-cm1", type=float, default=-100.0)
    parser.add_argument("--fmax-cm1", type=float, default=2300.0)
    add_mass_overrides(parser)


def add_mass_overrides(parser):
    parser.add_argument(
        "--mass",
        nargs="+",
        metavar=("SYMBOL", "AMU"),
        help="Override isotope masses as Symbol Mass pairs, e.g. --mass H 2.014.",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit", help="Parse OUTCAR and fit FC2/FC3")
    add_common_fit(fit_parser)
    post_parser = sub.add_parser("postprocess", help="Calculate phonons and Grüneisen parameters")
    add_analysis(post_parser)
    render_parser = sub.add_parser("render", help="Regenerate band plots from an existing q-path TSV")
    add_analysis(render_parser)
    all_parser = sub.add_parser("all", help="Run fitting and postprocessing")
    add_common_fit(all_parser)
    # Avoid duplicate --dim/--symprec while adding analysis-only arguments.
    all_parser.add_argument("--analysis-output", type=Path, default=Path("analysis"))
    all_parser.add_argument("--band-points", type=int, default=21)
    all_parser.add_argument("--mesh", nargs=3, type=int, default=(11, 11, 11), metavar=("NQ1", "NQ2", "NQ3"))
    all_parser.add_argument("--gmin", type=float, default=-60.0)
    all_parser.add_argument("--gmax", type=float, default=20.0)
    all_parser.add_argument("--frequency-cutoff", type=float, default=0.05)
    all_parser.add_argument("--fmin-cm1", type=float, default=-100.0)
    all_parser.add_argument("--fmax-cm1", type=float, default=2300.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "fit":
        fit(args)
    elif args.command == "postprocess":
        postprocess(args)
    elif args.command == "render":
        render_existing(args)
    else:
        fit_dir = fit(args)
        postprocess(args, fit_dir=fit_dir)


if __name__ == "__main__":
    main()
