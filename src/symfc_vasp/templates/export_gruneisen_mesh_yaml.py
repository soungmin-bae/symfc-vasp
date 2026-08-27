#!/usr/bin/env python3
"""Stream a phono3py Gruneisen mesh HDF5 file to tensor YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from gruneisen_tensor_io import reciprocal_lattice_from_poscar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, default=Path("gruneisen_mesh.hdf5"))
    parser.add_argument("--output", type=Path, default=Path("gruneisen_mesh.yaml"))
    parser.add_argument("--poscar", type=Path, default=Path("POSCAR-unitcell"))
    args = parser.parse_args()
    reciprocal = reciprocal_lattice_from_poscar(args.poscar)
    with h5py.File(args.hdf5, "r") as h5, args.output.open("w") as out:
        mesh = h5["mesh"][()]
        qpoints = h5["qpoint"]
        weights = h5["weight"]
        frequencies = h5["frequency"]
        scalar = h5["gruneisen"]
        tensors = h5["gruneisen_tensor"]
        out.write(f"mesh: [ {mesh[0]:5d}, {mesh[1]:5d}, {mesh[2]:5d} ]\n")
        out.write(f"nqpoint: {len(qpoints)}\n")
        out.write("reciprocal_lattice:\n")
        for row in reciprocal:
            out.write("- [ %15.10f, %15.10f, %15.10f ]\n" % tuple(row))
        out.write("phonon:\n")
        for iq in range(len(qpoints)):
            out.write("- q-position: [ %10.7f, %10.7f, %10.7f ]\n" % tuple(qpoints[iq]))
            out.write(f"  multiplicity: {int(weights[iq])}\n")
            out.write("  band:\n")
            for ib, (freq, gamma, tensor) in enumerate(
                zip(frequencies[iq], scalar[iq], tensors[iq], strict=True), start=1
            ):
                out.write(f"  - # {ib}\n")
                out.write(f"    frequency: {freq:15.10f}\n")
                out.write(f"    gruneisen: {gamma:15.10f}\n")
                out.write("    gruneisen_tensor:\n")
                for row in tensor:
                    out.write("    - [ %10.7f, %10.7f, %10.7f ]\n" % tuple(row))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
