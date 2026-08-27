"""Typed public API for VASP trajectory force-constant workflows.

The command-line interface is intentionally an adapter over this module.  No
public function accepts an ``argparse.Namespace``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import h5py
import numpy as np
import yaml

from . import engine
from .models import FitResult, GruneisenResult, PhononResult, ReferenceResult, TrajectoryDataset
from .parsers import parse_trajectory
from .parsers.outcar import parse_outcar_metadata, scan_outcar_summary
from .parsers.vasprun import count_vasprun_frames, vasprun_symbols
from .selection import select_indices


SelectionMethod = Literal["stride", "uniform", "random"]
ReferenceMode = Literal["auto", "trajectory", "provided"]


@dataclass(frozen=True)
class TrajectoryConfig:
    path: Path
    skip: int = 0
    stop: int | None = None
    samples: int | None = None
    stride: int | None = None
    selection: SelectionMethod = "stride"
    seed: int = 0
    cell_tolerance: float = 1e-6


@dataclass(frozen=True)
class ReferenceConfig:
    mode: ReferenceMode = "auto"
    unitcell: Path | None = None
    supercell: Path | None = None
    dim: tuple[int, int, int] | None = None
    symprec: float = 1e-5
    symmetry_scan_max: float = 0.3
    reference_map_tolerance: float = 1.0
    supplied_map_tolerance: float = 1e-4


@dataclass(frozen=True)
class FitConfig:
    trajectory: TrajectoryConfig | None = None
    dataset_npz: Path | None = None
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    output_dir: Path = Path(".")
    orders: tuple[int, ...] = (2,)
    rc2: float | None = None
    rc3: float | Literal["auto"] = "auto"
    rc3_auto_min_ratio: float = 8.0
    center_selected: bool = True
    batch_size: int = 20
    metric_samples: int = 5
    use_mkl: bool = True
    verbose: bool = False
    overwrite: bool = False

    @classmethod
    def from_namespace(cls, args) -> "FitConfig":
        trajectory = None
        if getattr(args, "trajectory", None) is not None:
            trajectory = TrajectoryConfig(
                path=Path(args.trajectory), skip=args.skip, stop=args.stop,
                samples=args.samples, stride=args.stride, selection=args.selection,
                seed=args.seed, cell_tolerance=args.cell_tolerance,
            )
        mode = "trajectory" if args.reference_mode == "outcar" else args.reference_mode
        return cls(
            trajectory=trajectory,
            dataset_npz=getattr(args, "dataset_npz", None),
            reference=ReferenceConfig(
                mode=mode, unitcell=args.unitcell, supercell=args.supercell,
                dim=tuple(args.dim) if args.dim else None, symprec=args.symprec,
                symmetry_scan_max=args.reference_symprec_max,
                reference_map_tolerance=args.reference_map_tolerance,
                supplied_map_tolerance=args.map_tolerance,
            ),
            output_dir=args.output, orders=tuple(args.order), rc2=args.rc2,
            rc3=args.rc3, rc3_auto_min_ratio=args.rc3_auto_min_ratio,
            center_selected=args.center_selected, batch_size=args.batch_size,
            metric_samples=args.metric_samples, use_mkl=args.use_mkl,
            verbose=args.verbose, overwrite=getattr(args, "force", False),
        )


@dataclass(frozen=True)
class AnalysisConfig:
    fit_dir: Path = Path(".")
    output_dir: Path = Path(".")
    dim: tuple[int, int, int] | None = None
    symprec: float = 1e-5
    band_points: int = 21
    mesh: tuple[int, int, int] = (11, 11, 11)
    gruneisen_range: tuple[float, float] = (-60.0, 20.0)
    frequency_cutoff_thz: float = 0.05
    frequency_range_cm1: tuple[float, float] = (-100.0, 2300.0)
    born: Path | None = None
    mass_overrides: dict[str, float] = field(default_factory=dict)
    atom_mass_overrides: dict[int, float] = field(default_factory=dict)
    overwrite: bool = False

    @classmethod
    def from_namespace(cls, args, *, fit_dir: Path | None = None) -> "AnalysisConfig":
        return cls(
            fit_dir=Path(fit_dir or args.fit_dir), output_dir=Path(args.analysis_output),
            dim=tuple(args.dim) if args.dim else None, symprec=args.symprec,
            band_points=args.band_points, mesh=tuple(args.mesh),
            gruneisen_range=(args.gmin, args.gmax),
            frequency_cutoff_thz=args.frequency_cutoff,
            frequency_range_cm1=(args.fmin_cm1, args.fmax_cm1), born=args.born,
            mass_overrides=engine.parse_mass_overrides(args.mass),
            atom_mass_overrides=engine.parse_atom_mass_overrides(
                getattr(args, "mass_index", None)
            ),
            overwrite=getattr(args, "force", False),
        )


@dataclass(frozen=True)
class WorkflowConfig:
    fit: FitConfig
    analysis: AnalysisConfig


def _fit_namespace(config: FitConfig) -> SimpleNamespace:
    trajectory = config.trajectory
    reference = config.reference
    return SimpleNamespace(
        trajectory=trajectory.path if trajectory else None,
        dataset_npz=config.dataset_npz,
        unitcell=reference.unitcell,
        supercell=reference.supercell,
        reference_mode=reference.mode,
        reference_symprec_max=reference.symmetry_scan_max,
        reference_map_tolerance=reference.reference_map_tolerance,
        dim=reference.dim,
        skip=trajectory.skip if trajectory else 0,
        stop=trajectory.stop if trajectory else None,
        samples=trajectory.samples if trajectory else None,
        stride=trajectory.stride if trajectory else None,
        selection=trajectory.selection if trajectory else "stride",
        seed=trajectory.seed if trajectory else 0,
        cell_tolerance=trajectory.cell_tolerance if trajectory else 1e-6,
        center_selected=config.center_selected,
        order=config.orders,
        fc3=3 in config.orders,
        rc2=config.rc2,
        rc3=config.rc3,
        rc3_auto_min_ratio=config.rc3_auto_min_ratio,
        symprec=reference.symprec,
        map_tolerance=reference.supplied_map_tolerance,
        batch_size=config.batch_size,
        metric_samples=config.metric_samples,
        use_mkl=config.use_mkl,
        verbose=config.verbose,
        output=config.output_dir,
        force=config.overwrite,
    )


def _analysis_namespace(config: AnalysisConfig) -> SimpleNamespace:
    masses: list[str] = []
    for symbol, mass in config.mass_overrides.items():
        masses.extend((symbol, str(mass)))
    indexed: list[str] = []
    for index, mass in config.atom_mass_overrides.items():
        indexed.extend((str(index), str(mass)))
    return SimpleNamespace(
        fit_dir=config.fit_dir, analysis_output=config.output_dir,
        dim=config.dim, symprec=config.symprec, band_points=config.band_points,
        mesh=config.mesh, gmin=config.gruneisen_range[0],
        gmax=config.gruneisen_range[1],
        frequency_cutoff=config.frequency_cutoff_thz,
        fmin_cm1=config.frequency_range_cm1[0],
        fmax_cm1=config.frequency_range_cm1[1], born=config.born,
        mass=masses or None, mass_index=indexed or None,
        force=config.overwrite,
    )


def read_trajectory(config: TrajectoryConfig) -> TrajectoryDataset:
    """Read a deterministic selection from a fixed-cell VASP trajectory."""
    path = config.path.resolve()
    is_outcar = path.name.lower() == "outcar" or path.name.lower().endswith(".outcar")
    if is_outcar:
        scan = scan_outcar_summary(path)
        total = scan.frames
        symbols = parse_outcar_metadata(path).symbols
    else:
        total = count_vasprun_frames(path)
        symbols = vasprun_symbols(path)
    indices = select_indices(
        total, skip=config.skip, stop=config.stop, samples=config.samples,
        stride=config.stride, method=config.selection, seed=config.seed,
    )
    dataset = parse_trajectory(path, indices)
    dataset = TrajectoryDataset(
        dataset.positions, dataset.forces, dataset.cells, dataset.source_indices,
        dataset.source_path, dataset.source_format, symbols,
    )
    dataset.validate(len(symbols))
    if dataset.cells is not None:
        deviation = float(np.max(np.abs(dataset.cells - dataset.cells[0])))
        if deviation > config.cell_tolerance:
            raise ValueError(
                f"trajectory is not fixed-cell: max cell deviation={deviation:.6g} A"
            )
    return dataset


def build_reference(config: FitConfig) -> ReferenceResult:
    """Prepare the exact reference and fitting dataset without running symfc."""
    args = _fit_namespace(config)
    output = config.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    engine.prepare_dataset(args, output)
    return _read_reference(output)


def fit_force_constants(config: FitConfig) -> FitResult:
    """Fit FC2 or FC2+FC3 and return numerical arrays plus provenance."""
    output = engine.fit(_fit_namespace(config))
    return _read_fit_result(output)


def calculate_phonons(config: AnalysisConfig) -> PhononResult:
    """Calculate a phonopy band structure from a completed FC2 fit."""
    output = engine.phonon(_analysis_namespace(config))
    return _read_phonon_result(output)


def calculate_gruneisen(config: AnalysisConfig) -> GruneisenResult:
    """Calculate FC3-based tensor mode-Gruneisen band and mesh data."""
    output = engine.postprocess(_analysis_namespace(config), do_band=True, do_mesh=True)
    return _read_gruneisen_result(output)


def run_workflow(config: WorkflowConfig) -> tuple[FitResult, PhononResult, GruneisenResult | None]:
    """Run fitting and analysis through the same API used by the CLI."""
    fit = fit_force_constants(config.fit)
    analysis = AnalysisConfig(**{**config.analysis.__dict__, "fit_dir": fit.output_dir})
    phonon = calculate_phonons(analysis)
    gruneisen = calculate_gruneisen(analysis) if fit.fc3 is not None else None
    return fit, phonon, gruneisen


def _hdf5_array(path: Path, preferred: tuple[str, ...]) -> np.ndarray:
    with h5py.File(path) as handle:
        for key in preferred:
            if key in handle:
                return np.asarray(handle[key])
        datasets: list[np.ndarray] = []
        handle.visititems(lambda _name, value: datasets.append(np.asarray(value)) if isinstance(value, h5py.Dataset) else None)
    if not datasets:
        raise ValueError(f"no numerical dataset found in {path}")
    return datasets[0]


def _owned_files(output: Path) -> tuple[Path, ...]:
    return tuple(sorted((path for path in output.iterdir() if path.is_file() or path.is_symlink()), key=lambda path: path.name))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {} if path.is_file() else {}


def _read_reference(output: Path) -> ReferenceResult:
    matrix = np.loadtxt(output / "supercell_matrix.dat", dtype=int).reshape(3, 3)
    mapping_path = output / "generated_to_outcar_index.json"
    if mapping_path.is_file():
        import json
        mapping = np.asarray(json.loads(mapping_path.read_text())["generated_supercell_index_to_outcar_index"], dtype=int)
    else:
        with np.load(output / "symfc_input.npz") as saved:
            mapping = np.asarray(saved["generated_to_vasp"], dtype=int)
    report = _read_yaml(output / "symmetry_report.yaml") or _read_yaml(output / "structure_report.yaml")
    return ReferenceResult(
        output / "POSCAR-unitcell", output / "POSCAR-supercell", matrix, mapping, report,
    )


def _read_fit_result(output: Path) -> FitResult:
    output = output.resolve()
    fc2 = _hdf5_array(output / "fc2.hdf5", ("fc2", "force_constants"))
    fc3_path = output / "fc3.hdf5"
    fc3 = _hdf5_array(fc3_path, ("fc3", "force_constants")) if fc3_path.is_file() else None
    return FitResult(output, fc2, fc3, _read_reference(output), _read_yaml(output / "symfc_summary.yaml"), _owned_files(output))


def _read_phonon_result(output: Path) -> PhononResult:
    output = output.resolve()
    data = _read_yaml(output / "band.yaml")
    phonons = data.get("phonon", [])
    qpoints = np.asarray([entry["q-position"] for entry in phonons], dtype=float)
    distances = np.asarray([entry.get("distance", np.nan) for entry in phonons], dtype=float)
    frequencies = np.asarray([[mode["frequency"] for mode in entry["band"]] for entry in phonons], dtype=float)
    labels = tuple(entry.get("label", "") for entry in phonons if entry.get("label"))
    return PhononResult(output, qpoints, distances, frequencies, labels, _read_yaml(output / "phonon_summary.yaml"), _owned_files(output))


def _read_gruneisen_result(output: Path) -> GruneisenResult:
    output = output.resolve()
    band = _read_yaml(output / "gruneisen_band.yaml")
    if "path" in band:
        phonons = [entry for segment in band["path"] for entry in segment["phonon"]]
    else:
        phonons = band.get("phonon", [])
    band_qpoints = np.asarray([entry["q-position"] for entry in phonons], dtype=float)
    band_frequencies = np.asarray([[mode["frequency"] for mode in entry["band"]] for entry in phonons], dtype=float)
    band_tensors = np.asarray([[mode["gruneisen_tensor"] for mode in entry["band"]] for entry in phonons], dtype=float)
    mesh_path = output / "gruneisen_mesh.hdf5"
    with h5py.File(mesh_path) as handle:
        mesh_qpoints = np.asarray(handle["qpoint"])
        mesh_weights = np.asarray(handle["weight"])
        mesh_frequencies = np.asarray(handle["frequency"])
        mesh_tensors = np.asarray(handle["gruneisen_tensor"])
    return GruneisenResult(
        output, band_qpoints, band_frequencies, band_tensors,
        mesh_qpoints, mesh_weights, mesh_frequencies, mesh_tensors,
        _read_yaml(output / "analysis_summary.yaml"), _owned_files(output),
    )
