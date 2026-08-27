#!/usr/bin/env python3
"""Render one phono3py mode-Gruneisen tensor component in the first BZ."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Voronoi
from phonopy.interface.vasp import read_vasp
from phonopy.structure.symmetry import Symmetry

from gruneisen_tensor_io import (
    component_label,
    read_mesh_hdf5,
    reciprocal_lattice_from_poscar,
    valid_mode_mask,
)


def fold_to_first_bz(qfrac, reciprocal):
    shifts = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=float)
    candidates = qfrac[:, None, :] + shifts[None, :, :]
    cartesian = candidates @ reciprocal
    closest = np.argmin(np.linalg.norm(cartesian, axis=2), axis=1)
    return candidates[np.arange(len(qfrac)), closest]


def unique_mean(qfrac, gamma):
    rounded = np.round(qfrac, decimals=9)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=gamma)
    counts = np.bincount(inverse)
    return unique, sums / counts


def bz_edges(reciprocal):
    grid = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=float)
    points = grid @ reciprocal
    origin = int(np.where(np.all(grid == 0, axis=1))[0][0])
    vor = Voronoi(points)
    edges = []
    for pair, ridge in zip(vor.ridge_points, vor.ridge_vertices, strict=True):
        if origin in pair and -1 not in ridge:
            edges.append(vor.vertices[np.asarray([*ridge, ridge[0]])])
    return edges


def add_bz_wireframe(figure, reciprocal):
    for polygon in bz_edges(reciprocal):
        for start, end in zip(polygon[:-1], polygon[1:], strict=True):
            figure.add_trace(
                go.Scatter3d(
                    x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                    mode="lines", line={"color": "black", "width": 5},
                    hoverinfo="skip", showlegend=False,
                )
            )


def zero_centred_colorscale(gmin, gmax):
    zero = -gmin / (gmax - gmin)
    return [[0.0, "#08306b"], [zero, "#ffffff"], [1.0, "#b2182b"]]


def main(default_component: str | None = None, default_output: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, default=Path("gruneisen_mesh.hdf5"))
    parser.add_argument("--poscar", type=Path, default=Path("POSCAR-unitcell"))
    parser.add_argument(
        "--component",
        default=default_component or "ab",
    )
    parser.add_argument("--output", type=Path, default=Path(default_output or "gruneisen-mesh-tensor.html"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--gmin", type=float, default=-15.0)
    parser.add_argument("--gmax", type=float, default=15.0)
    parser.add_argument("--cutoff", type=float, default=0.05, help="THz")
    parser.add_argument(
        "--mode-policy", choices=("stable-only", "abs-frequency"), default="stable-only"
    )
    parser.add_argument("--marker-size", type=float, default=3.2)
    parser.add_argument(
        "--static", choices=("png", "pdf", "both"), default=None,
        help="optional slow Kaleido export; HTML and CSV are always written",
    )
    args = parser.parse_args()
    if not args.gmin < 0 < args.gmax:
        parser.error("--gmin must be negative and --gmax must be positive")

    qfrac, weights, frequency, gamma_mode, mesh = read_mesh_hdf5(args.hdf5, args.component)
    valid = valid_mode_mask(frequency, args.cutoff, args.mode_policy) & np.isfinite(gamma_mode)
    counts = valid.sum(axis=1)
    keep = counts > 0
    gamma = np.divide(
        np.where(valid, gamma_mode, 0).sum(axis=1), counts,
        out=np.full(len(counts), np.nan), where=counts > 0,
    )[keep]
    qfrac = qfrac[keep]
    reciprocal = reciprocal_lattice_from_poscar(args.poscar)

    # phono3py Gruneisen mesh output is point-group reduced. Expand its stars.
    # ab, c, hydro, and trace are invariant under this trigonal point group.
    rotations = Symmetry(read_vasp(str(args.poscar))).reciprocal_operations
    qfrac = np.vstack([qfrac @ rotation.T for rotation in rotations])
    gamma = np.tile(gamma, len(rotations))
    qfrac -= np.floor(qfrac + 0.5)
    qfrac, gamma = unique_mean(qfrac, gamma)
    qfrac = fold_to_first_bz(qfrac, reciprocal)
    qcart = qfrac @ reciprocal

    csv_path = args.csv or args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        csv_path, np.column_stack((qfrac, qcart, gamma)), delimiter=",",
        header="qx_reduced,qy_reduced,qz_reduced,qx_cart,qy_cart,qz_cart,gamma_mode_mean",
        comments="",
    )

    figure = go.Figure()
    add_bz_wireframe(figure, reciprocal)
    figure.add_trace(
        go.Scatter3d(
            x=qcart[:, 0], y=qcart[:, 1], z=qcart[:, 2], mode="markers", showlegend=False,
            marker={
                "size": args.marker_size, "color": np.clip(gamma, args.gmin, args.gmax),
                "cmin": args.gmin, "cmax": args.gmax,
                "colorscale": zero_centred_colorscale(args.gmin, args.gmax), "opacity": 0.9,
                "colorbar": {"title": component_label(args.component)},
            },
            customdata=np.column_stack((qfrac, gamma)),
            hovertemplate=(
                "q=(%{customdata[0]:.4f}, %{customdata[1]:.4f}, %{customdata[2]:.4f})"
                "<br>mean gamma=%{customdata[3]:.4f}<extra></extra>"
            ),
        )
    )
    axis = {"title": "reciprocal coordinate", "showbackground": False, "showgrid": False}
    figure.update_layout(
        title=f"{mesh[0]}x{mesh[1]}x{mesh[2]} mesh: {component_label(args.component)}",
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 55}, width=980, height=820,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(args.output, include_plotlyjs="cdn", full_html=True)
    if args.static:
        formats = ("png", "pdf") if args.static == "both" else (args.static,)
        for image_format in formats:
            try:
                figure.write_image(
                    args.output.with_suffix(f".{image_format}"),
                    width=1200,
                    height=950,
                    scale=1,
                )
            except Exception as exc:
                print(f"Static {image_format.upper()} was not written: {exc}")
    print(f"Wrote {len(qfrac)} BZ points to {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
