#!/usr/bin/env python3
"""Plot one component of a phono3py mode-Gruneisen tensor on a band path."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from gruneisen_tensor_io import (
    component_label,
    read_band_yaml,
    read_labels,
    valid_mode_mask,
)


CM1_PER_THZ = 33.35640952


def add_coloured_branches(ax, x, frequency, gamma, norm, cutoff, policy):
    for mode in range(frequency.shape[1]):
        y = frequency[:, mode]
        g = gamma[:, mode]
        valid = np.isfinite(y) & np.isfinite(g) & valid_mode_mask(y, cutoff, policy)
        if valid.sum() < 2:
            continue
        points = np.column_stack((x, y * CM1_PER_THZ)).reshape(-1, 1, 2)
        segments = np.concatenate((points[:-1], points[1:]), axis=1)
        values = 0.5 * (g[:-1] + g[1:])
        keep = valid[:-1] & valid[1:]
        collection = LineCollection(
            segments[keep], cmap="bwr", norm=norm, linewidth=1.15, alpha=0.92
        )
        collection.set_array(np.clip(values[keep], norm.vmin, norm.vmax))
        ax.add_collection(collection)


def main(default_component: str | None = None, default_output: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=Path("gruneisen_band.yaml"))
    parser.add_argument("--band-conf", type=Path, default=Path("band.conf"))
    parser.add_argument("--component", default=default_component or "ab")
    parser.add_argument("--output", type=Path, default=Path(default_output or "gruneisen-band-tensor.pdf"))
    parser.add_argument("--gmin", type=float, default=-15.0)
    parser.add_argument("--gmax", type=float, default=15.0)
    parser.add_argument("--fmin", type=float, default=-100.0, help="cm^-1")
    parser.add_argument("--fmax", type=float, default=2500.0, help="cm^-1")
    parser.add_argument("--cutoff", type=float, default=0.05, help="THz")
    parser.add_argument(
        "--mode-policy", choices=("stable-only", "abs-frequency"), default="stable-only"
    )
    args = parser.parse_args()
    if not args.gmin < 0 < args.gmax:
        parser.error("--gmin must be negative and --gmax must be positive")

    xs, frequencies, gammas = read_band_yaml(args.yaml, args.component)
    labels = read_labels(args.band_conf, len(xs))
    norm = TwoSlopeNorm(vmin=args.gmin, vcenter=0.0, vmax=args.gmax)
    fig, ax = plt.subplots(figsize=(12.5, 5.9), constrained_layout=True)
    for x, freq, gamma in zip(xs, frequencies, gammas, strict=True):
        add_coloured_branches(ax, x, freq, gamma, norm, args.cutoff, args.mode_policy)

    boundaries = [xs[0][0], *[x[-1] for x in xs]]
    for boundary in boundaries:
        ax.axvline(boundary, color="0.78", linewidth=0.9, zorder=0)
    ax.axhline(0, color="0.45", linestyle="--", linewidth=1.0)
    ax.set(xlim=(boundaries[0], boundaries[-1]), ylim=(args.fmin, args.fmax))
    ax.set_xticks(boundaries, labels)
    ax.set_xlabel("High-symmetry path")
    ax.set_ylabel(r"Frequency (cm$^{-1}$)")
    ax.set_title(f"0 K FC2/FC3 mode Gruneisen tensor: {component_label(args.component)}")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="bwr")
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.015)
    colorbar.set_label(component_label(args.component))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    if args.output.suffix.lower() == ".pdf":
        fig.savefig(args.output.with_suffix(".png"), dpi=220)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
