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
                "analysis": {
                    "band_points": 21,
                    "mesh": [11, 11, 11],
                    "frequency_cutoff_THz": 0.05,
                    "gruneisen_plot_range": [-10, 20],
                    "frequency_plot_range_cm1": [-800, 2300],
                },
            },
            sort_keys=False,
        )
    )


def test_run_yaml_is_reusable_and_cli_wins(tmp_path):
    config = tmp_path / "run.yaml"
    write_config(config)
    argv = expand_config_argv(
        ["run", "--config", str(config), "--samples", "4000", "--output", "rerun"]
    )
    args = parser().parse_args(argv)
    assert args.trajectory == Path("/case/OUTCAR")
    assert args.samples == 4000
    assert args.skip == 5000
    assert args.center_selected is True
    assert args.mesh == [11, 11, 11]
    assert args.gmin == -10
    assert args.output == Path("rerun")


def test_split_stage_parsers_load_relevant_config(tmp_path):
    config = tmp_path / "run.yaml"
    write_config(config)
    fit = parser().parse_args(expand_config_argv(["fit", "--config", str(config)]))
    band = parser().parse_args(
        expand_config_argv(["band", "--config", str(config), "--fit-dir", "fc"])
    )
    mesh = parser().parse_args(
        expand_config_argv(["mesh", "--config", str(config), "--fit-dir", "fc"])
    )
    assert fit.samples == 3000
    assert fit.rc3 == 4.0
    assert band.band_points == 21
    assert band.fmin_cm1 == -800
    assert mesh.mesh == [11, 11, 11]


def test_centering_is_enabled_by_default_and_can_be_disabled():
    default = parser().parse_args(["fit"])
    disabled = parser().parse_args(["fit", "--no-center-selected"])
    assert default.center_selected is True
    assert disabled.center_selected is False
