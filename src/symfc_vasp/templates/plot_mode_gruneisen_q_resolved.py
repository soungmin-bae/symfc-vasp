#!/usr/bin/env python3
"""Plot ab- and c-axis tensor mode-Gruneisen parameters along the band q path."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from gruneisen_tensor_io import (
    available_components,
    component_label,
    read_band_yaml,
    read_labels,
    valid_mode_mask,
)


def add_panel(ax, xs, frequencies, gammas, component, norm, cutoff, policy):
    handle = None
    for x, frequency, gamma in zip(xs, frequencies, gammas, strict=True):
        xx = np.broadcast_to(x[:, None], gamma.shape)
        valid = (
            np.isfinite(gamma)
            & np.isfinite(frequency)
            & valid_mode_mask(frequency, cutoff, policy)
        )
        handle = ax.scatter(
            xx[valid],
            np.clip(gamma[valid], norm.vmin, norm.vmax),
            c=np.clip(gamma[valid], norm.vmin, norm.vmax),
            cmap="bwr",
            norm=norm,
            s=11,
            linewidths=0,
            alpha=0.76,
        )
    boundaries = [xs[0][0], *[x[-1] for x in xs]]
    for boundary in boundaries:
        ax.axvline(boundary, color="0.80", linewidth=0.8, zorder=0)
    ax.axhline(0, color="0.42", linestyle="--", linewidth=1.0)
    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.set_ylim(norm.vmin, norm.vmax)
    ax.set_title(component_label(component), fontsize=14)
    ax.set_xlabel("High-symmetry path")
    return handle, boundaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=Path("gruneisen_band.yaml"))
    parser.add_argument("--band-conf", type=Path, default=Path("band.conf"))
    parser.add_argument("--output", type=Path, default=Path("mode_gruneisen_q_resolved.pdf"))
    parser.add_argument("--gmin", type=float, default=-15.0)
    parser.add_argument("--gmax", type=float, default=15.0)
    parser.add_argument("--cutoff", type=float, default=0.05, help="THz")
    parser.add_argument(
        "--mode-policy", choices=("stable-only", "abs-frequency"), default="stable-only"
    )
    args = parser.parse_args()
    if not args.gmin < 0 < args.gmax:
        parser.error("--gmin must be negative and --gmax must be positive")

    components = available_components()
    datasets = [read_band_yaml(args.yaml, component) for component in components]
    labels = read_labels(args.band_conf, len(datasets[0][0]))
    norm = TwoSlopeNorm(vmin=args.gmin, vcenter=0, vmax=args.gmax)
    fig, axes = plt.subplots(len(components), 1, figsize=(12.5, 4.5 * len(components)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    handles = []
    for ax, dataset, component in zip(axes, datasets, components, strict=True):
        handle, boundaries = add_panel(
            ax, *dataset, component, norm, args.cutoff, args.mode_policy
        )
        handles.append(handle)
        ax.set_ylabel("Mode Gruneisen parameter")
    axes[-1].set_xticks(boundaries, labels)
    fig.suptitle("0 K FC2/FC3 tensor mode Gruneisen parameters along the q path", fontsize=18)
    colorbar = fig.colorbar(handles[-1], ax=axes, pad=0.015)
    colorbar.set_label("Mode Gruneisen parameter")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    if args.output.suffix.lower() == ".pdf":
        fig.savefig(args.output.with_suffix(".png"), dpi=220)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
