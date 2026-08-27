"""Argument definitions for the thin command-line adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import parse_rc3


def add_common_fit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "trajectory_input", nargs="?", type=Path, metavar="OUTCAR_OR_VASPRUN",
        help="Positional shorthand for --trajectory OUTCAR or --trajectory vasprun.xml.",
    )
    parser.add_argument(
        "--trajectory", type=Path,
        help="VASP OUTCAR or vasprun.xml. Required unless --dataset-npz is supplied.",
    )
    parser.add_argument(
        "--dataset-npz", type=Path,
        help=(
            "Reuse a validated symfc_input.npz instead of reparsing the trajectory. "
            "Selection options address frames in that saved dataset."
        ),
    )
    parser.add_argument(
        "--supercell", type=Path,
        help="VASP POSCAR for the fixed-cell MD or random-displacement supercell.",
    )
    parser.add_argument(
        "--unitcell", type=Path,
        help=(
            "VASP POSCAR used unchanged as the displacement and phonon reference unit cell. "
            "Without --supercell, its integer relation to the trajectory cell is inferred."
        ),
    )
    parser.add_argument(
        "--reference-mode", choices=("auto", "provided", "trajectory", "outcar"),
        default="auto",
        help=(
            "Reference source. auto reconstructs from trajectory unless --unitcell is supplied. "
            "provided requires both POSCAR files. 'outcar' is a compatibility alias."
        ),
    )
    parser.add_argument(
        "--reference-symprec-max", type=float, default=0.3,
        help="Largest spglib tolerance scanned during trajectory reference reconstruction (A).",
    )
    parser.add_argument(
        "--reference-map-tolerance", type=float, default=1.0,
        help="Maximum trajectory-mean to reconstructed-reference atom-map distance (A).",
    )
    parser.add_argument(
        "--dim", nargs=3, type=int, metavar=("NA", "NB", "NC"),
        help="Optional diagonal assertion; omit for a general integer supercell matrix.",
    )
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--stop", type=int, help="Exclusive source-frame stop index.")
    parser.add_argument(
        "--samples", "--nstep", dest="samples", type=int,
        help="Number of post-skip frames. Default: every available frame.",
    )
    parser.add_argument("--stride", type=int)
    parser.add_argument(
        "--selection", choices=("stride", "uniform", "random"), default="stride",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--center-selected", action=argparse.BooleanOptionalAction, default=True,
        help="Subtract selected mean displacement and force before fitting (default: enabled).",
    )
    parser.add_argument(
        "--fc3", action="store_true",
        help="Fit third-order force constants in addition to the default FC2 fit.",
    )
    parser.add_argument(
        "--rc2", type=float,
        help="Optional FC2 cutoff in A. Default: full symfc FC2 basis.",
    )
    parser.add_argument(
        "--rc3", type=parse_rc3, default="auto",
        help="FC3 cutoff in A or auto (default with --fc3).",
    )
    parser.add_argument("--rc3-auto-min-ratio", type=float, default=8.0)
    parser.add_argument("--symprec", type=float, default=1e-5)
    parser.add_argument("--map-tolerance", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--metric-samples", type=int, default=5,
        help="Deterministic frames used for in-sample force reconstruction diagnostics.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-mkl", dest="use_mkl", action="store_false", default=True,
        help="Disable optional sparse_dot_mkl acceleration.",
    )
    parser.add_argument("--use-mkl", dest="use_mkl", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-use-mkl", dest="use_mkl", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument(
        "--force", action="store_true",
        help="Replace package-generated fit outputs when the input hash differs.",
    )


def add_analysis(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "fit_dir_input", nargs="?", type=Path, metavar="FIT_DIR",
        help="Positional shorthand for --fit-dir; use '.' for the current directory.",
    )
    parser.add_argument("--fit-dir", type=Path, default=Path("."))
    parser.add_argument("--analysis-output", type=Path, default=Path("."))
    parser.add_argument(
        "--dim", nargs=3, type=int, metavar=("NA", "NB", "NC"),
        help="Legacy fallback when FIT_DIR has no supercell_matrix.dat.",
    )
    parser.add_argument("--symprec", type=float, default=1e-5)
    parser.add_argument("--band-points", type=int, default=21)
    parser.add_argument("--mesh", nargs=3, type=int, default=(11, 11, 11), metavar=("NQ1", "NQ2", "NQ3"))
    parser.add_argument("--gmin", type=float, default=-60.0)
    parser.add_argument("--gmax", type=float, default=20.0)
    parser.add_argument("--frequency-cutoff", type=float, default=0.05)
    parser.add_argument("--fmin-cm1", type=float, default=-100.0)
    parser.add_argument("--fmax-cm1", type=float, default=2300.0)
    parser.add_argument(
        "--born", type=Path,
        help="phonopy-format BORN file for non-analytical correction.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Replace package-generated analysis when fitted input hashes differ.",
    )
    add_mass_overrides(parser)


def add_mass_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mass", nargs="+", metavar=("SYMBOL", "AMU"),
        help="Override element masses, e.g. --mass H 2.014.",
    )
    parser.add_argument(
        "--mass-index", nargs="+", metavar=("INDEX", "AMU"),
        help="Override one-based primitive atom masses; takes precedence over --mass.",
    )
