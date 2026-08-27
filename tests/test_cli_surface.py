from symfc_vasp.cli import parser


def test_public_cli_is_small_and_fc2_is_the_default():
    root = parser()
    commands = next(action for action in root._actions if action.dest == "command")
    assert set(commands.choices) == {"fit", "phonon", "gruneisen", "full"}
    assert root.parse_args(["fit", "OUTCAR"]).fc3 is False
    assert root.parse_args(["fit", "OUTCAR", "--fc3"]).fc3 is True


def test_postprocessing_defaults_to_current_directory():
    args = parser().parse_args(["phonon"])
    assert str(args.fit_dir) == "."
    assert str(args.analysis_output) == "."
