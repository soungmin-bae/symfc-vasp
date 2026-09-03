"""Small command-line interface for fitting and analysing VASP trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import __version__, engine
from .api import AnalysisConfig, FitConfig, WorkflowConfig, calculate_gruneisen, calculate_phonons, fit_force_constants, run_workflow
from .cli_options import add_analysis, add_common_fit, add_mass_overrides


def add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="Read optional run.yaml defaults; explicit command-line options take precedence.",
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
    """Translate the small supported subset of run.yaml into CLI defaults."""
    data = yaml.safe_load(path.read_text()) or {}
    tokens: list[str] = []
    if command in {"fit", "full"}:
        _append(tokens, "--trajectory", data.get("trajectory"))
        _append(tokens, "--dataset-npz", data.get("dataset_npz"))
        _append(tokens, "--unitcell", data.get("unitcell"))
        _append(tokens, "--supercell", data.get("supercell"))
        _append(tokens, "--reference-mode", data.get("reference_mode"))
        _append(tokens, "--dim", data.get("dim"))
        selection = data.get("selection", {}) or {}
        _append(tokens, "--selection", selection.get("method"))
        _append(tokens, "--skip", selection.get("skip"))
        _append(tokens, "--stop", selection.get("stop"))
        _append(tokens, "--samples", selection.get("samples"))
        _append(tokens, "--stride", selection.get("stride"))
        _append(tokens, "--seed", selection.get("seed"))
        if selection.get("center_selected") is not None:
            tokens.append("--center-selected" if selection["center_selected"] else "--no-center-selected")
        fc = data.get("force_constants", {}) or {}
        if 3 in fc.get("orders", []):
            tokens.append("--fc3")
        _append(tokens, "--rc2", fc.get("rc2_A"))
        if 3 in fc.get("orders", []):
            _append(tokens, "--rc3", fc.get("rc3_A"))
        _append(tokens, "--symprec", fc.get("symprec"))
        _append(tokens, "--map-tolerance", fc.get("map_tolerance_A"))
        _append(tokens, "--batch-size", fc.get("batch_size"))
        _append(tokens, "--metric-samples", fc.get("metric_samples"))
        energy = data.get("effective_energy", {}) or {}
        _append(tokens, "--energy-field", energy.get("field"))
        _append(tokens, "--energy-bootstrap-samples", energy.get("bootstrap_samples"))
        _append(tokens, "--energy-block-size", energy.get("block_size"))
        if energy.get("enabled") is not None:
            tokens.append("--effective-energy-offset" if energy["enabled"] else "--no-effective-energy-offset")
        if fc.get("use_mkl") is False:
            tokens.append("--no-mkl")

    if command in {"phonon", "gruneisen", "full"}:
        if command != "full":
            _append(tokens, "--dim", data.get("dim"))
            _append(tokens, "--symprec", (data.get("force_constants", {}) or {}).get("symprec"))
        mass_values: list[object] = []
        for symbol, mass in (data.get("mass_overrides", {}) or {}).items():
            mass_values.extend((symbol, mass))
        _append(tokens, "--mass", mass_values or None)
        atom_mass_values: list[object] = []
        for index, mass in (data.get("atom_mass_overrides", {}) or {}).items():
            atom_mass_values.extend((index, mass))
        _append(tokens, "--mass-index", atom_mass_values or None)
        analysis = data.get("analysis", {}) or {}
        nac = data.get("nac", {}) or {}
        _append(tokens, "--born", nac.get("born"))
        _append(tokens, "--band-points", analysis.get("band_points"))
        _append(tokens, "--mesh", analysis.get("mesh"))
        _append(tokens, "--frequency-cutoff", analysis.get("frequency_cutoff_THz"))
        _append(tokens, "--tmin", analysis.get("thermal_min_temperature_K"))
        _append(tokens, "--tmax", analysis.get("thermal_max_temperature_K"))
        _append(tokens, "--tstep", analysis.get("thermal_temperature_step_K"))
        limits = analysis.get("gruneisen_plot_range")
        if limits:
            _append(tokens, "--gmin", limits[0])
            _append(tokens, "--gmax", limits[1])
        frequency_limits = analysis.get("frequency_plot_range_cm1")
        if frequency_limits:
            _append(tokens, "--fmin-cm1", frequency_limits[0])
            _append(tokens, "--fmax-cm1", frequency_limits[1])
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
    return argv if config_path is None else [command, *_config_tokens(command, config_path), *argv[1:]]


def validate_fit_contract(args) -> None:
    errors: list[str] = []
    trajectory = getattr(args, "trajectory", None)
    dataset = getattr(args, "dataset_npz", None)
    unitcell = getattr(args, "unitcell", None)
    supercell = getattr(args, "supercell", None)
    mode = getattr(args, "reference_mode", "auto")
    if trajectory is None and dataset is None:
        errors.append("provide OUTCAR/vasprun.xml or --trajectory FILE (or --dataset-npz FILE)")
    if trajectory is not None and dataset is not None:
        errors.append("use either a trajectory or --dataset-npz, not both")
    if trajectory is not None and not trajectory.is_file():
        errors.append(f"trajectory file does not exist: {trajectory}")
    if dataset is not None and not dataset.is_file():
        errors.append(f"dataset file does not exist: {dataset}")
    if mode == "provided" and (unitcell is None or supercell is None):
        errors.append("--reference-mode provided requires both --unitcell and --supercell")
    if supercell is not None and unitcell is None:
        errors.append("--supercell requires --unitcell")
    for label, path in (("unitcell", unitcell), ("supercell", supercell)):
        if path is not None and not path.is_file():
            errors.append(f"{label} file does not exist: {path}")
    if getattr(args, "energy_bootstrap_samples", 0) < 0:
        errors.append("--energy-bootstrap-samples must be non-negative")
    if getattr(args, "energy_block_size", None) is not None and args.energy_block_size < 1:
        errors.append("--energy-block-size must be positive")
    if errors:
        message = ["Invalid fitting input:", *(f"  - {error}" for error in errors), "", "Examples:",
                   "  symfc-vasp fit OUTCAR", "  symfc-vasp fit vasprun.xml --fc3",
                   "  symfc-vasp full OUTCAR --fc3"]
        raise ValueError("\n".join(message))


def validate_analysis_contract(args) -> None:
    errors: list[str] = []
    if any(value < 1 for value in args.mesh):
        errors.append("--mesh values must be positive")
    if args.band_points < 2:
        errors.append("--band-points must be at least 2")
    if args.tstep <= 0:
        errors.append("--tstep must be positive")
    if args.tmax < args.tmin:
        errors.append("--tmax must be greater than or equal to --tmin")
    if errors:
        raise ValueError("Invalid analysis input:\n" + "\n".join(f"  - {item}" for item in errors))


def _add_full_analysis(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--analysis-output", type=Path, default=Path("."), help="Analysis directory (default: current directory).")
    parser.add_argument("--band-points", type=int, default=21)
    parser.add_argument("--mesh", nargs=3, type=int, default=(11, 11, 11), metavar=("NQ1", "NQ2", "NQ3"))
    parser.add_argument("--gmin", type=float, default=-60.0)
    parser.add_argument("--gmax", type=float, default=20.0)
    parser.add_argument("--frequency-cutoff", type=float, default=0.05)
    parser.add_argument("--fmin-cm1", type=float, default=-100.0)
    parser.add_argument("--fmax-cm1", type=float, default=2300.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=1000.0)
    parser.add_argument("--tstep", type=float, default=10.0)
    parser.add_argument("--born", type=Path, help="phonopy-format BORN file for NAC; copied to analysis.")
    add_mass_overrides(parser)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="symfc-vasp",
        description="Fit symfc force constants from VASP trajectories and postprocess with phonopy/phono3py.",
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)

    fit = commands.add_parser(
        "fit", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Fit FC2 (or FC2+FC3) and evaluate aligned snapshot energies.",
        epilog=("Examples:\n  symfc-vasp fit OUTCAR\n  symfc-vasp fit vasprun.xml --fc3\n"
                "  symfc-vasp fit OUTCAR --unitcell POSCAR-unitcell --samples 3000 --selection uniform\n\n"
                "Writes FC files and, when energies are available, the TDEP-style U0_eff diagnostic.\n"
                "The concise progress log is shown\n"
                "in the terminal; use --verbose to also stream the complete symfc solver log."),
    )
    add_config(fit)
    add_common_fit(fit)

    phonon = commands.add_parser(
        "phonon", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Create FC2 phonon band, DOS, and harmonic thermal properties.",
        epilog=("Examples:\n  symfc-vasp phonon .\n  symfc-vasp phonon fit-dir --analysis-output analysis\n\n"
                "Use '.' explicitly for the current directory. It overwrites package-generated\n"
                "phonopy outputs and writes band, DOS, and thermal-property data and plots."),
    )
    add_config(phonon)
    add_analysis(phonon)

    gruneisen = commands.add_parser(
        "gruneisen", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Create FC2+FC3 band-path and q-mesh mode-Gruneisen outputs in the current directory.",
        epilog=("Examples:\n  symfc-vasp gruneisen .\n  symfc-vasp gruneisen fit-dir --mesh 21 21 21\n\n"
                "Requires fc3.hdf5 from `symfc-vasp fit ... --fc3`. It refreshes the package-generated\n"
                "band, mesh, YAML, table, and plot files in the current directory."),
    )
    add_config(gruneisen)
    add_analysis(gruneisen)

    full = commands.add_parser(
        "full", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Fit and create energy/phonon outputs; add --fc3 for Gruneisen outputs.",
        epilog=("Examples:\n  symfc-vasp full OUTCAR\n  symfc-vasp full OUTCAR --fc3 --mesh 11 11 11\n"
                "  symfc-vasp full vasprun.xml --unitcell POSCAR-unitcell --born BORN --mass H 2.014\n\n"
                "Without --fc3 this writes FC2, phonon band, DOS, and harmonic thermal properties.\n"
                "With --fc3 it also writes band-path and q-mesh mode-Gruneisen data and plots."),
    )
    add_config(full)
    add_common_fit(full)
    _add_full_analysis(full)
    return root


def _apply_positionals(args) -> None:
    trajectory = getattr(args, "trajectory_input", None)
    if trajectory is not None:
        args.trajectory = trajectory
    fit_dir = getattr(args, "fit_dir_input", None)
    if fit_dir is not None:
        args.fit_dir = fit_dir


def _normalise_orders(args) -> None:
    args.order = (2, 3) if getattr(args, "fc3", False) else (2,)


def _write_full_config(args, fit_dir: Path) -> None:
    payload = {
        "trajectory": str(args.trajectory.resolve()) if args.trajectory else None,
        "dataset_npz": str(args.dataset_npz.resolve()) if args.dataset_npz else None,
        "unitcell": str(args.unitcell.resolve()) if args.unitcell else None,
        "supercell": str(args.supercell.resolve()) if args.supercell else None,
        "dim": list(args.dim) if args.dim else None,
        "selection": {"method": args.selection, "skip": args.skip, "stop": args.stop, "samples": args.samples,
                      "stride": args.stride, "seed": args.seed, "center_selected": args.center_selected},
        "force_constants": {"orders": list(args.order), "rc2_A": args.rc2, "rc3_A": args.rc3 if args.fc3 else None,
                            "symprec": args.symprec, "map_tolerance_A": args.map_tolerance,
                            "batch_size": args.batch_size, "metric_samples": args.metric_samples,
                            "use_mkl": args.use_mkl},
        "effective_energy": {
            "enabled": args.effective_energy_offset,
            "field": args.energy_field,
            "bootstrap_samples": args.energy_bootstrap_samples,
            "block_size": args.energy_block_size,
        },
        "mass_overrides": engine.parse_mass_overrides(args.mass),
        "atom_mass_overrides": engine.parse_atom_mass_overrides(
            getattr(args, "mass_index", None)
        ),
        "analysis": {"band_points": args.band_points, "mesh": list(args.mesh), "frequency_cutoff_THz": args.frequency_cutoff,
                     "thermal_min_temperature_K": args.tmin, "thermal_max_temperature_K": args.tmax,
                     "thermal_temperature_step_K": args.tstep,
                     "gruneisen_plot_range": [args.gmin, args.gmax], "frequency_plot_range_cm1": [args.fmin_cm1, args.fmax_cm1]},
        "nac": {"born": str(args.born.resolve()) if args.born else None},
    }
    with (fit_dir / "run.yaml").open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def main(argv=None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        parser().parse_args(["--help"])
    # A bare command never starts an expensive calculation. Use an explicit
    # positional '.' for the flat current-directory postprocessing workflow.
    if raw_argv in (["fit"], ["phonon"], ["gruneisen"], ["full"]):
        parser().parse_args([raw_argv[0], "--help"])
    args = parser().parse_args(expand_config_argv(raw_argv))
    _apply_positionals(args)

    if args.command in {"fit", "full"}:
        try:
            validate_fit_contract(args)
        except ValueError as exc:
            parser().error(str(exc))
        _normalise_orders(args)
    if args.command in {"phonon", "gruneisen", "full"}:
        try:
            validate_analysis_contract(args)
        except ValueError as exc:
            parser().error(str(exc))

    if args.command == "fit":
        fit_force_constants(FitConfig.from_namespace(args))
    elif args.command == "phonon":
        calculate_phonons(AnalysisConfig.from_namespace(args))
    elif args.command == "gruneisen":
        calculate_gruneisen(AnalysisConfig.from_namespace(args))
    elif args.command == "full":
        fit_dir = args.output.resolve()
        fit_dir.mkdir(parents=True, exist_ok=True)
        _write_full_config(args, fit_dir)
        fit_config = FitConfig.from_namespace(args)
        analysis_config = AnalysisConfig.from_namespace(args, fit_dir=fit_dir)
        run_workflow(WorkflowConfig(fit=fit_config, analysis=analysis_config))
        if not args.fc3:
            engine.stage("full", "FC2 phonon workflow complete; use --fc3 to add mode-Gruneisen analysis")


if __name__ == "__main__":
    main()
