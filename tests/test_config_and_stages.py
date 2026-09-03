from pathlib import Path

import yaml

from symfc_vasp.cli import expand_config_argv, parser


def write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "trajectory": "/case/OUTCAR",
                "unitcell": "/case/POSCAR-unitcell",
                "supercell": "/case/POSCAR",
                "dim": [2, 2, 2],
                "mass_overrides": {"H": 2.014},
                "selection": {
                    "method": "uniform",
                    "skip": 5000,
                    "samples": 3000,
                    "stride": None,
                    "seed": 0,
                    "center_selected": True,
                },
                "force_constants": {
                    "orders": [2, 3],
                    "rc2_A": 7.0,
                    "rc3_A": 4.0,
                    "symprec": 1e-5,
                    "batch_size": 20,
                },
                "effective_energy": {
                    "enabled": True,
                    "field": "energy_without_entropy",
                    "bootstrap_samples": 50,
                    "block_size": 25,
                },
                "analysis": {
                    "band_points": 21,
                    "mesh": [11, 11, 11],
                    "frequency_cutoff_THz": 0.05,
                    "gruneisen_plot_range": [-10, 20],
                    "frequency_plot_range_cm1": [-800, 2300],
                    "thermal_min_temperature_K": 50,
                    "thermal_max_temperature_K": 500,
                    "thermal_temperature_step_K": 25,
                },
            },
            sort_keys=False,
        )
    )


def test_run_yaml_is_reusable_and_cli_wins(tmp_path):
    config = tmp_path / "run.yaml"
    write_config(config)
    argv = expand_config_argv(
        ["full", "--config", str(config), "--samples", "4000", "--output", "rerun"]
    )
    args = parser().parse_args(argv)
    assert args.trajectory == Path("/case/OUTCAR")
    assert args.samples == 4000
    assert args.skip == 5000
    assert args.center_selected is True
    assert args.mesh == [11, 11, 11]
    assert args.gmin == -10
    assert args.mass == ["H", "2.014"]
    assert args.output == Path("rerun")
    assert args.energy_field == "energy_without_entropy"
    assert args.effective_energy_offset is True
    assert args.energy_bootstrap_samples == 50
    assert args.energy_block_size == 25
    assert args.tmin == 50
    assert args.tmax == 500
    assert args.tstep == 25


def test_split_stage_parsers_load_relevant_config(tmp_path):
    config = tmp_path / "run.yaml"
    write_config(config)
    fit = parser().parse_args(expand_config_argv(["fit", "--config", str(config)]))
    gruneisen = parser().parse_args(
        expand_config_argv(["gruneisen", "--config", str(config), "--fit-dir", "fc"])
    )
    assert fit.samples == 3000
    assert fit.fc3 is True
    assert fit.rc3 == 4.0
    assert gruneisen.band_points == 21
    assert gruneisen.mass == ["H", "2.014"]
    assert gruneisen.fmin_cm1 == -800
    assert gruneisen.mesh == [11, 11, 11]


def test_centering_is_enabled_by_default_and_can_be_disabled():
    default = parser().parse_args(["fit"])
    disabled = parser().parse_args(["fit", "--no-center-selected"])
    assert default.center_selected is True
    assert disabled.center_selected is False
    assert default.fc3 is False


def test_cli_mass_override_wins_over_run_yaml(tmp_path):
    config = tmp_path / "run.yaml"
    write_config(config)
    argv = expand_config_argv(
        ["gruneisen", "--config", str(config), "--mass", "H", "3.0"]
    )
    args = parser().parse_args(argv)
    assert args.mass == ["H", "3.0"]
