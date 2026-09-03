"""TDEP-style effective harmonic energy offset and trajectory diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def harmonic_energies(
    displacements: np.ndarray,
    fc2: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return ``0.5 * u.T @ Phi @ u`` in eV for every snapshot."""
    u = np.asarray(displacements, dtype=float)
    phi = np.asarray(fc2, dtype=float)
    natom = u.shape[1]
    if u.ndim != 3 or u.shape[2] != 3:
        raise ValueError(f"displacements must have shape (frames, atoms, 3), got {u.shape}")
    if phi.shape != (natom, natom, 3, 3):
        raise ValueError(
            "full FC2 shape is incompatible with displacements: "
            f"fc2={phi.shape}, displacements={u.shape}"
        )
    matrix = phi.transpose(0, 2, 1, 3).reshape(3 * natom, 3 * natom)
    flat = u.reshape(len(u), 3 * natom)
    result = np.empty(len(flat), dtype=float)
    for start in range(0, len(flat), chunk_size):
        block = flat[start : start + chunk_size]
        result[start : start + len(block)] = 0.5 * np.einsum(
            "si,si->s", block @ matrix, block, optimize=True
        )
    return result


def integrated_autocorrelation_time(values: np.ndarray) -> tuple[float, int]:
    """Estimate statistical inefficiency with Geyer's initial-positive pairs."""
    values = np.asarray(values, dtype=float)
    nvalue = len(values)
    if nvalue < 3 or float(np.var(values)) <= np.finfo(float).eps:
        return 1.0, 0
    centered = values - np.mean(values)
    nfft = 1 << (2 * nvalue - 1).bit_length()
    spectrum = np.fft.rfft(centered, nfft)
    covariance = np.fft.irfft(spectrum * spectrum.conjugate(), nfft)[:nvalue]
    covariance /= np.arange(nvalue, 0, -1)
    correlation = covariance / covariance[0]
    positive_sum = 0.0
    stop_lag = 0
    for first in range(1, nvalue - 1, 2):
        pair = float(correlation[first] + correlation[first + 1])
        if pair <= 0:
            break
        positive_sum += pair
        stop_lag = first + 1
    return max(1.0, 1.0 + 2.0 * positive_sum), stop_lag


def _two_gaussian_fit(values: np.ndarray) -> dict[str, Any] | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 20 or np.std(values) <= np.finfo(float).eps:
        return None
    means = np.asarray(np.quantile(values, [0.3, 0.7]), dtype=float)
    variances = np.repeat(float(np.var(values)), 2)
    weights = np.asarray([0.5, 0.5], dtype=float)
    previous = -np.inf
    log_likelihood = previous
    for _ in range(500):
        logs = []
        for weight, mean, variance in zip(weights, means, variances, strict=True):
            variance = max(float(variance), 1e-16)
            logs.append(
                np.log(max(float(weight), 1e-16))
                - 0.5 * (np.log(2 * np.pi * variance) + (values - mean) ** 2 / variance)
            )
        maximum = np.maximum(logs[0], logs[1])
        denominator = maximum + np.log(
            np.exp(logs[0] - maximum) + np.exp(logs[1] - maximum)
        )
        responsibility = np.exp(logs[0] - denominator)
        counts = np.asarray([responsibility.sum(), len(values) - responsibility.sum()])
        if np.min(counts) <= np.finfo(float).eps * len(values):
            return None
        weights = counts / len(values)
        means = np.asarray(
            [
                np.sum(responsibility * values) / counts[0],
                np.sum((1 - responsibility) * values) / counts[1],
            ]
        )
        variances = np.asarray(
            [
                np.sum(responsibility * (values - means[0]) ** 2) / counts[0],
                np.sum((1 - responsibility) * (values - means[1]) ** 2) / counts[1],
            ]
        )
        log_likelihood = float(np.sum(denominator))
        if abs(log_likelihood - previous) <= 1e-10 * max(1.0, abs(log_likelihood)):
            break
        previous = log_likelihood
    order = np.argsort(means)
    weights = weights[order]
    means = means[order]
    standard_deviations = np.sqrt(variances[order])
    variance = max(float(np.var(values)), 1e-16)
    one_log_likelihood = float(
        np.sum(-0.5 * (np.log(2 * np.pi * variance) + (values - np.mean(values)) ** 2 / variance))
    )
    bic_one = 2 * np.log(len(values)) - 2 * one_log_likelihood
    bic_two = 5 * np.log(len(values)) - 2 * log_likelihood
    pooled = np.sqrt(np.mean(standard_deviations**2))
    separation = float(abs(means[1] - means[0]) / pooled) if pooled > 0 else np.inf
    return {
        "weights": weights.tolist(),
        "means_eV": means.tolist(),
        "standard_deviations_eV": standard_deviations.tolist(),
        "separation_pooled_sigma": separation,
        "delta_BIC_one_minus_two": float(bic_one - bic_two),
        "descriptive_only": True,
    }


def effective_energy_statistics(
    residuals: np.ndarray,
    source_indices: np.ndarray,
    *,
    bootstrap_samples: int = 200,
    block_size: int | None = None,
    seed: int = 0,
) -> tuple[dict[str, Any], list[str]]:
    residuals = np.asarray(residuals, dtype=float)
    source_indices = np.asarray(source_indices, dtype=int)
    nvalue = len(residuals)
    if nvalue == 0:
        raise ValueError("effective energy requires at least one snapshot")
    standard_deviation = float(np.std(residuals, ddof=1)) if nvalue > 1 else 0.0
    naive_sem = standard_deviation / np.sqrt(nvalue)
    tau, stop_lag = integrated_autocorrelation_time(residuals)
    effective_samples = max(1.0, min(float(nvalue), nvalue / tau))
    if block_size is None:
        block_size = min(max(1, int(np.ceil(tau))), max(1, nvalue // 8))
    if block_size < 1 or block_size > nvalue:
        raise ValueError(f"block size must satisfy 1 <= block_size <= {nvalue}")
    source_step = (
        float(np.median(np.diff(source_indices))) if nvalue > 1 else None
    )
    nblocks = nvalue // block_size
    if nblocks >= 2:
        trimmed = residuals[: nblocks * block_size]
        block_means = trimmed.reshape(nblocks, block_size).mean(axis=1)
        block_sem = float(np.std(block_means, ddof=1) / np.sqrt(nblocks))
        if bootstrap_samples > 0:
            rng = np.random.default_rng(seed)
            bootstrap_means = np.mean(
                rng.choice(block_means, size=(bootstrap_samples, nblocks), replace=True),
                axis=1,
            )
            bootstrap_std = float(np.std(bootstrap_means, ddof=1))
            bootstrap_interval = np.quantile(bootstrap_means, [0.025, 0.975]).tolist()
        else:
            bootstrap_std = None
            bootstrap_interval = None
    else:
        block_means = np.asarray([np.mean(residuals)])
        block_sem = None
        bootstrap_std = None
        bootstrap_interval = None

    quarter = max(1, nvalue // 4)
    first_mean = float(np.mean(residuals[:quarter]))
    last_mean = float(np.mean(residuals[-quarter:]))
    drift = last_mean - first_mean
    drift_sigma = abs(drift) / standard_deviation if standard_deviation > 0 else 0.0
    mixture = _two_gaussian_fit(residuals)
    mixture_like = bool(
        mixture
        and mixture["delta_BIC_one_minus_two"] > 10
        and mixture["separation_pooled_sigma"] > 1.5
        and min(mixture["weights"]) > 0.1
    )
    warnings: list[str] = []
    if effective_samples < max(10.0, 0.05 * nvalue):
        warnings.append(
            "residual series is strongly autocorrelated; use block uncertainty, not the naive SEM"
        )
    if drift_sigma >= 0.5:
        warnings.append(
            "residual series is non-stationary across the selected window; first/last-quarter means differ substantially"
        )
    if mixture_like:
        warnings.append(
            "residual distribution is bimodal-like or strongly non-Gaussian; no automatic basin splitting was performed"
        )
    return {
        "num_snapshots": nvalue,
        "mean_eV_supercell": float(np.mean(residuals)),
        "standard_deviation_eV_supercell": standard_deviation,
        "naive_standard_error_eV_supercell": float(naive_sem),
        "minimum_eV_supercell": float(np.min(residuals)),
        "maximum_eV_supercell": float(np.max(residuals)),
        "integrated_autocorrelation_time_selected_frames": float(tau),
        "autocorrelation_positive_sequence_stop_lag": int(stop_lag),
        "effective_sample_size": float(effective_samples),
        "block": {
            "size_selected_frames": int(block_size),
            "median_source_frame_step": source_step,
            "approximate_source_frame_span": (
                float(block_size * source_step) if source_step is not None else None
            ),
            "num_complete_blocks": int(nblocks),
            "standard_error_eV_supercell": block_sem,
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_standard_deviation_eV_supercell": bootstrap_std,
            "bootstrap_95_percent_interval_eV_supercell": bootstrap_interval,
        },
        "stationarity": {
            "first_quarter_mean_eV_supercell": first_mean,
            "last_quarter_mean_eV_supercell": last_mean,
            "last_minus_first_eV_supercell": drift,
            "absolute_shift_over_residual_std": float(drift_sigma),
        },
        "two_gaussian_diagnostic": mixture,
        "source_index_range": [int(source_indices[0]), int(source_indices[-1])],
    }, warnings


def calculate_effective_energy_offset(
    *,
    output: Path,
    displacements: np.ndarray,
    energies: np.ndarray,
    fc2: np.ndarray,
    source_indices: np.ndarray,
    energy_field: str,
    energy_metadata: dict[str, Any],
    primitive_cells_per_supercell: int,
    bootstrap_samples: int = 200,
    block_size: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    output = Path(output)
    harmonic = harmonic_energies(displacements, fc2)
    energies = np.asarray(energies, dtype=float)
    if energies.shape != harmonic.shape:
        raise ValueError(
            f"energy/displacement frame mismatch: energies={energies.shape}, harmonic={harmonic.shape}"
        )
    residuals = energies - harmonic
    statistics, residual_warnings = effective_energy_statistics(
        residuals,
        source_indices,
        bootstrap_samples=bootstrap_samples,
        block_size=block_size,
        seed=seed,
    )
    potential_statistics, potential_warnings = effective_energy_statistics(
        energies,
        source_indices,
        bootstrap_samples=bootstrap_samples,
        block_size=block_size,
        seed=seed,
    )
    warnings = [
        *(f"potential energy: {warning}" for warning in potential_warnings),
        *(f"effective-energy residual: {warning}" for warning in residual_warnings),
    ]
    if np.std(energies) > 0 and np.std(harmonic) > 0:
        correlation = float(np.corrcoef(energies, harmonic)[0, 1])
    else:
        correlation = None
    primitive_cells = max(1, int(primitive_cells_per_supercell))
    value = statistics["mean_eV_supercell"]
    result = {
        "schema": "symfc-vasp-effective-energy-v1",
        "effective_energy_offset": {
            "value_eV_supercell": value,
            "value_eV_primitive_cell": value / primitive_cells,
            "definition": "mean(E_potential - 0.5 * u.T @ fc2 @ u)",
        },
        "statistics": statistics,
        "potential_energy_statistics": potential_statistics,
        "energy_source": {
            "field": energy_field,
            **energy_metadata,
        },
        "model": {
            "fc2_file": "fc2.hdf5",
            "displacements_file": "symfc_input.npz",
            "primitive_cells_per_supercell": primitive_cells,
            "energy_harmonic_energy_correlation": correlation,
            "linear_term_included": False,
        },
        "warnings": warnings,
    }
    with (output / "tdep_energy_offset.yaml").open("w") as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    rows = np.column_stack((source_indices, energies, harmonic, residuals))
    np.savetxt(
        output / "tdep_energy_residuals.tsv",
        rows,
        header="source_index E_potential_eV E_harmonic_eV residual_eV",
        fmt=["%d", "%.12g", "%.12g", "%.12g"],
        delimiter="\t",
    )
    _plot_energy_diagnostics(
        output, source_indices, energies, harmonic, residuals, statistics
    )
    return result


def _plot_energy_diagnostics(
    output: Path,
    source_indices: np.ndarray,
    energies: np.ndarray,
    harmonic: np.ndarray,
    residuals: np.ndarray,
    statistics: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(source_indices, energies, color="#4d77a8", linewidth=0.45, rasterized=True)
    axes[0, 0].set_ylabel("Potential energy (eV/supercell)")
    axes[0, 0].set_title("Potential-energy time series")
    axes[0, 1].plot(source_indices, harmonic, color="#7b6ba8", linewidth=0.45, rasterized=True)
    axes[0, 1].set_ylabel("Harmonic energy (eV/supercell)")
    axes[0, 1].set_title("FC2 harmonic-energy time series")
    axes[1, 0].plot(source_indices, residuals, color="0.2", linewidth=0.45, rasterized=True)
    axes[1, 0].axhline(statistics["mean_eV_supercell"], color="#b2182b", linewidth=1)
    axes[1, 0].set_xlabel("Source frame index")
    axes[1, 0].set_ylabel(r"$E - E_{harm}$ (eV/supercell)")
    axes[1, 0].set_title("Effective-energy residual time series")
    axes[1, 1].hist(residuals, bins="auto", color="#4d77a8", alpha=0.85)
    axes[1, 1].axvline(statistics["mean_eV_supercell"], color="#b2182b", linewidth=1)
    axes[1, 1].set_xlabel(r"$E - E_{harm}$ (eV/supercell)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Residual distribution")
    figure.savefig(output / "tdep_energy_diagnostics.pdf")
    figure.savefig(output / "tdep_energy_diagnostics.png", dpi=180)
    plt.close(figure)
