from subprocess import CompletedProcess

from symfc_vasp.engine import write_phonopy_bandplot_data


def test_phonopy_bandplot_output_is_saved_verbatim(tmp_path, monkeypatch):
    monkeypatch.setattr("symfc_vasp.engine.shutil.which", lambda _: "/fake/phonopy-bandplot")
    expected = "# End points of segments:\n0.000000 1.000000\n"
    monkeypatch.setattr(
        "symfc_vasp.engine.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, stdout=expected, stderr=""),
    )
    assert write_phonopy_bandplot_data(tmp_path) == tmp_path / "phonopy-band.dat"
    assert (tmp_path / "phonopy-band.dat").read_text() == expected


def test_phonopy_bandplot_nonzero_status_with_data_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr("symfc_vasp.engine.shutil.which", lambda _: "/fake/phonopy-bandplot")
    expected = "0.000000 1.000000\n"
    monkeypatch.setattr(
        "symfc_vasp.engine.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 1, stdout=expected, stderr=""),
    )
    write_phonopy_bandplot_data(tmp_path)
    assert (tmp_path / "phonopy-band.dat").read_text() == expected
