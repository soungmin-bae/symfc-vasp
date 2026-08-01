#!/usr/bin/env python3
"""Audit a completed vasp-symfc production directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--expected-frames", type=int, default=20000)
    args = parser.parse_args()
    root = args.root.resolve()
    fit = root / f"fit_N{args.expected_frames:05d}"
    analysis = root / f"analysis_N{args.expected_frames:05d}"
    checks: dict[str, dict] = {}

    with (fit / "symfc_summary.yaml").open() as handle:
        summary = yaml.safe_load(handle)
    checks["trajectory_coverage"] = {
        "passed": summary["selection"]["selected_frames"] == args.expected_frames
        and summary["selection"]["first_source_index"] == 0
        and summary["selection"]["last_source_index"] == args.expected_frames - 1,
        "selected_frames": summary["selection"]["selected_frames"],
        "source_range": [summary["selection"]["first_source_index"], summary["selection"]["last_source_index"]],
    }
    checks["atom_mapping"] = {
        "passed": summary["atom_mapping"]["unique_indices"] == 128
        and summary["atom_mapping"]["max_distance_A"] < 1e-4,
        **summary["atom_mapping"],
    }
    checks["force_constants"] = {
        "passed": summary["fit"]["fc2_max_translational_drift"] < 1e-8
        and summary["fit"]["fc3_max_translational_drift_j"] < 1e-8
        and (fit / "fc2.hdf5").stat().st_size > 0
        and (fit / "fc3.hdf5").stat().st_size > 0,
        "fc2_drift": summary["fit"]["fc2_max_translational_drift"],
        "fc3_drift_j": summary["fit"]["fc3_max_translational_drift_j"],
        "fit_metrics": summary["fit"]["force_reconstruction"],
    }
    with h5py.File(analysis / "gruneisen_qmesh_11x11x11.hdf5") as h5:
        frequencies = h5["frequency_THz"][:]
        tensors = h5["gruneisen_tensor"][:]
        weights = h5["weight"][:]
        qpoints = h5["qpoint"][:]
    checks["qmesh"] = {
        "passed": bool(np.isfinite(frequencies).all()
        and np.isfinite(tensors).all()
        and int(np.sum(weights)) == 11**3
        and tensors.shape == (len(qpoints), 48, 3, 3)),
        "irreducible_qpoints": len(qpoints),
        "weight_sum": int(np.sum(weights)),
        "frequency_range_THz": [float(np.min(frequencies)), float(np.max(frequencies))],
        "tensor_range": [float(np.min(tensors)), float(np.max(tensors))],
    }
    band = np.loadtxt(analysis / "mode_gruneisen_qpath.tsv")
    checks["band_path"] = {
        "passed": bool(band.shape == (10080, 10) and np.isfinite(band).all()),
        "rows": int(len(band)),
        "segments": int(len(np.unique(band[:, 0]))),
        "modes": int(len(np.unique(band[:, 3]))),
        "frequency_range_THz": [float(np.min(band[:, 4])), float(np.max(band[:, 4]))],
        "minimum_frequency_cm-1": float(np.min(band[:, 4]) * 33.35640951981521),
    }
    plot_names = [
        "phonon_dispersion.png",
        "mode_gruneisen_q_resolved.png",
        "mode_gruneisen_on_phonon_dispersion.png",
        "mode_gruneisen_qmesh_11x11x11.png",
    ]
    plot_details = {}
    plots_pass = True
    for name in plot_names:
        path = analysis / name
        image = np.asarray(Image.open(path).convert("RGB"))
        variance = float(np.var(image))
        passed = path.stat().st_size > 10000 and variance > 1.0
        plots_pass &= passed
        plot_details[name] = {"bytes": path.stat().st_size, "pixel_variance": variance, "passed": passed}
    checks["plots"] = {"passed": plots_pass, "files": plot_details}
    passed = all(item["passed"] for item in checks.values())
    result = {
        "passed": passed,
        "root": str(root),
        "fit_job_id": 3007113,
        "postprocess_job_id": 3007114,
        "checks": checks,
    }
    with (root / "FINAL_VALIDATION.yaml").open("w") as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    print(yaml.safe_dump(result, sort_keys=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
