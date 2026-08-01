from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from . import __version__
from . import engine
from .parsers.outcar import scan_outcar
from .parsers.vasprun import count_vasprun_frames
from .selection import select_indices


def add_inputs(parser):
    engine.add_common_fit(parser)


def add_post(parser):
    engine.add_analysis(parser)


def add_config(parser):
    parser.add_argument(
        "--config",
        type=Path,
        help="Load settings from a previous run.yaml; explicit CLI options override the file.",
    )


def _append(tokens: list[str], option: str, value) -> None:
    if value is None:
        return
    tokens.append(option)
    if isinstance(value, (list, tuple)):
        tokens.extend(str(item) for item in value)
    else:
        tokens.append(str(value))


def _config_tokens(command: str, path: Path) -> list[str]:
    """Translate run.yaml into CLI defaults that explicit arguments may override."""
    data = yaml.safe_load(path.read_text()) or {}
    tokens: list[str] = []
    fit_commands = {"inspect", "extract", "fit", "run"}
    analysis_commands = {"phonon", "band", "mesh", "gruneisen", "plot", "run"}
    if command in fit_commands:
        _append(tokens, "--trajectory", data.get("trajectory"))
        _append(tokens, "--dataset-npz", data.get("dataset_npz"))
        _append(tokens, "--unitcell", data.get("unitcell"))
        _append(tokens, "--supercell", data.get("supercell"))
        _append(tokens, "--dim", data.get("dim"))
        selection = data.get("selection", {})
        _append(tokens, "--selection", selection.get("method"))
        _append(tokens, "--skip", selection.get("skip"))
        _append(tokens, "--samples", selection.get("samples"))
        _append(tokens, "--stride", selection.get("stride"))
        _append(tokens, "--seed", selection.get("seed"))
        if selection.get("center_selected") is not None:
            tokens.append("--center-selected" if selection["center_selected"] else "--no-center-selected")
        fc = data.get("force_constants", {})
        _append(tokens, "--order", fc.get("orders"))
        _append(tokens, "--rc2", fc.get("rc2_A"))
        _append(tokens, "--rc3", fc.get("rc3_A"))
        _append(tokens, "--symprec", fc.get("symprec"))
        _append(tokens, "--map-tolerance", fc.get("map_tolerance_A"))
        _append(tokens, "--batch-size", fc.get("batch_size"))
        _append(tokens, "--metric-samples", fc.get("metric_samples"))
        if fc.get("use_mkl") is not None:
            tokens.append("--use-mkl" if fc["use_mkl"] else "--no-use-mkl")
    if command in analysis_commands:
        if command != "run":
            _append(tokens, "--dim", data.get("dim"))
        analysis = data.get("analysis", {})
        _append(tokens, "--band-points", analysis.get("band_points"))
        _append(tokens, "--mesh", analysis.get("mesh"))
        _append(tokens, "--frequency-cutoff", analysis.get("frequency_cutoff_THz"))
        limits = analysis.get("gruneisen_plot_range")
        if limits:
            _append(tokens, "--gmin", limits[0])
            _append(tokens, "--gmax", limits[1])
        frequency_limits = analysis.get("frequency_plot_range_cm1")
        if frequency_limits:
            _append(tokens, "--fmin-cm1", frequency_limits[0])
            _append(tokens, "--fmax-cm1", frequency_limits[1])
        if command != "run":
            _append(tokens, "--symprec", data.get("force_constants", {}).get("symprec"))
    return tokens


def expand_config_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    command = argv[0]
    config_path = None
    for index, token in enumerate(argv[1:], start=1):
        if token == "--config" and index + 1 < len(argv):
            config_path = Path(argv[index + 1])
            break
        if token.startswith("--config="):
            config_path = Path(token.split("=", 1)[1])
            break
    if config_path is None:
        return argv
    return [command, *_config_tokens(command, config_path), *argv[1:]]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="symfc-vasp",
        description="Fit symfc FC2/FC3 and calculate phonopy/phono3py finite-temperature phonons.",
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("inspect", "Inspect trajectory and resolve frame selection"),
        ("extract", "Extract selected trajectory and atom mapping"),
        ("fit", "Fit FC2 or FC2+FC3 with symfc"),
    ):
        item = commands.add_parser(name, help=help_text)
        add_config(item)
        add_inputs(item)
    for name, help_text in (
        ("phonon", "Calculate phonon and band-path analysis from fitted FCs (alias of band)"),
        ("band", "Calculate phonon dispersion and band-path mode-Gruneisen plots"),
        ("mesh", "Calculate q-mesh mode-Gruneisen data and plots"),
        ("gruneisen", "Calculate tensor mode-Gruneisen band and mesh data"),
        ("plot", "Regenerate plots from existing q-path data"),
    ):
        item = commands.add_parser(name, help=help_text)
        add_config(item)
        add_post(item)
    run = commands.add_parser("run", help="Run extraction, fitting, phonon and Gruneisen stages")
    add_config(run)
    add_inputs(run)
    run.add_argument("--analysis-output", type=Path, default=Path("analysis"))
    run.add_argument("--band-points", type=int, default=21)
    run.add_argument("--mesh", nargs=3, type=int, default=(11, 11, 11))
    run.add_argument("--gmin", type=float, default=-60.0)
    run.add_argument("--gmax", type=float, default=20.0)
    run.add_argument("--frequency-cutoff", type=float, default=0.05)
    run.add_argument("--fmin-cm1", type=float, default=-100.0)
    run.add_argument("--fmax-cm1", type=float, default=2300.0)
    status = commands.add_parser("status", help="Report outputs present in a run directory")
    status.add_argument("run_dir", type=Path, nargs="?", default=Path("."))
    validate = commands.add_parser("validate", help="Validate a completed run")
    validate.add_argument("run_dir", type=Path, nargs="?", default=Path("."))
    validate.add_argument("--reference", type=Path)
    return root


def _frame_count(path: Path) -> tuple[int, int | None]:
    if path.name.lower() == "outcar" or path.name.lower().endswith(".outcar"):
        natom, frames, _ = scan_outcar(path)
        return frames, natom
    return count_vasprun_frames(path), None


def inspect(args) -> dict:
    total, natom = _frame_count(args.trajectory)
    indices = select_indices(
        total, skip=args.skip, samples=args.samples, stride=args.stride,
        method=args.selection, seed=args.seed,
    )
    result = {
        "trajectory": str(args.trajectory.resolve()),
        "total_frames": total,
        "natom": natom,
        "selection": {
            "method": args.selection,
            "skip": args.skip,
            "samples": len(indices),
            "first": int(indices[0]),
            "last": int(indices[-1]),
            "intervals": sorted(set(np.diff(indices).tolist())) if len(indices) > 1 else [],
        },
    }
    print(yaml.safe_dump(result, sort_keys=False), end="")
    return result


def validate(run_dir: Path, reference: Path | None = None) -> dict:
    fit_dir = run_dir / "force_constants"
    analysis = run_dir / "analysis"
    run_config = {}
    if (run_dir / "run.yaml").is_file():
        run_config = yaml.safe_load((run_dir / "run.yaml").read_text()) or {}
    orders = run_config.get("force_constants", {}).get("orders", [2, 3])
    mesh = run_config.get("analysis", {}).get("mesh", [11, 11, 11])
    mesh_tag = "x".join(str(value) for value in mesh)
    required = [
        fit_dir / "FORCE_CONSTANTS", fit_dir / "fc2.hdf5",
        fit_dir / "selected_indices.txt", fit_dir / "symfc_summary.yaml",
        analysis / "phonon_dispersion.tsv", analysis / "mode_gruneisen_qpath.tsv",
        analysis / f"gruneisen_qmesh_{mesh_tag}.hdf5",
        analysis / "mode_gruneisen_q_resolved.pdf", analysis / "mode_gruneisen_q_resolved.png",
        analysis / "mode_gruneisen_on_phonon_dispersion.pdf", analysis / "mode_gruneisen_on_phonon_dispersion.png",
        analysis / f"mode_gruneisen_qmesh_{mesh_tag}.pdf", analysis / f"mode_gruneisen_qmesh_{mesh_tag}.png",
    ]
    if 3 in orders:
        required.append(fit_dir / "fc3.hdf5")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    indices = np.loadtxt(fit_dir / "selected_indices.txt", dtype=int) if not missing else np.array([])
    arrays_finite = True
    if not missing:
        arrays_finite = bool(np.isfinite(np.loadtxt(analysis / "mode_gruneisen_qpath.tsv")).all())
    comparison = None
    if reference is not None and (reference / "analysis/mode_gruneisen_qpath.tsv").is_file() and not missing:
        current = np.loadtxt(analysis / "mode_gruneisen_qpath.tsv")
        expected = np.loadtxt(reference / "analysis/mode_gruneisen_qpath.tsv")
        comparison = {
            "shape_equal": current.shape == expected.shape,
            "max_abs_difference": float(np.max(np.abs(current - expected))) if current.shape == expected.shape else None,
        }
    selection = run_config.get("selection", {})
    requested_samples = selection.get("samples")
    requested_skip = selection.get("skip", 0)
    expected_contract = bool(
        len(indices) > 0
        and (requested_samples is None or len(indices) == requested_samples)
        and indices[0] >= requested_skip
        and len(np.unique(indices)) == len(indices)
        and np.all(np.diff(indices) > 0)
    )
    passed = not missing and arrays_finite and expected_contract
    result = {
        "passed": passed,
        "missing": missing,
        "arrays_finite": arrays_finite,
        "selected_frames": int(len(indices)),
        "first_index": int(indices[0]) if len(indices) else None,
        "last_index": int(indices[-1]) if len(indices) else None,
        "validation_selection_contract": expected_contract,
        "reference_comparison": comparison,
    }
    with (run_dir / "FINAL_VALIDATION.yaml").open("w") as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    print(yaml.safe_dump(result, sort_keys=False), end="")
    return result


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(expand_config_argv(raw_argv))
    if args.command == "inspect":
        inspect(args)
    elif args.command == "extract":
        args.output.mkdir(parents=True, exist_ok=True)
        engine.prepare_dataset(args, args.output.resolve())
    elif args.command == "fit":
        engine.fit(args)
    elif args.command in ("phonon", "band"):
        engine.postprocess(args, do_band=True, do_mesh=False)
    elif args.command == "mesh":
        engine.postprocess(args, do_band=False, do_mesh=True)
    elif args.command == "gruneisen":
        engine.postprocess(args, do_band=True, do_mesh=True)
    elif args.command == "plot":
        engine.render_existing(args)
    elif args.command == "run":
        run_dir = args.output.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "run.yaml").open("w") as handle:
            yaml.safe_dump(
                {
                    "trajectory": str(args.trajectory.resolve()),
                    "dataset_npz": str(args.dataset_npz.resolve()) if args.dataset_npz else None,
                    "unitcell": str(args.unitcell.resolve()),
                    "supercell": str(args.supercell.resolve()),
                    "dim": list(args.dim),
                    "selection": {
                        "method": args.selection, "skip": args.skip,
                        "samples": args.samples, "stride": args.stride, "seed": args.seed,
                        "center_selected": args.center_selected,
                    },
                    "force_constants": {
                        "orders": list(args.order), "rc2_A": args.rc2, "rc3_A": args.rc3,
                        "symprec": args.symprec, "map_tolerance_A": args.map_tolerance,
                        "batch_size": args.batch_size, "metric_samples": args.metric_samples,
                        "use_mkl": args.use_mkl,
                    },
                    "analysis": {
                        "band_points": args.band_points, "mesh": list(args.mesh),
                        "frequency_cutoff_THz": args.frequency_cutoff,
                        "gruneisen_plot_range": [args.gmin, args.gmax],
                        "frequency_plot_range_cm1": [args.fmin_cm1, args.fmax_cm1],
                    },
                },
                handle,
                sort_keys=False,
            )
        fit_dir = run_dir / "force_constants"
        args.output = fit_dir
        args.analysis_output = run_dir / "analysis"
        engine.fit(args)
        engine.postprocess(args, fit_dir=fit_dir, do_band=True, do_mesh=True)
        engine.stage("validate", "Checking finite arrays, frame selection, and required outputs")
        validate(run_dir)
    elif args.command == "status":
        files = sorted(str(path.relative_to(args.run_dir)) for path in args.run_dir.rglob("*") if path.is_file())
        print(json.dumps({"run_dir": str(args.run_dir.resolve()), "files": files}, indent=2))
    elif args.command == "validate":
        result = validate(args.run_dir.resolve(), args.reference.resolve() if args.reference else None)
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
