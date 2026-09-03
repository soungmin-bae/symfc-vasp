import numpy as np

from symfc_vasp.engine import write_phonopy_bandplot_data


def test_phonopy_bandplot_output_is_written_in_process(tmp_path):
    distances = [np.array([0.0, 0.5]), np.array([0.5, 1.0])]
    frequencies = [
        np.array([[1.0, 10.0], [2.0, 20.0]]),
        np.array([[2.0, 20.0], [3.0, 30.0]]),
    ]
    path = write_phonopy_bandplot_data(tmp_path, distances, frequencies)
    assert path == tmp_path / "phonopy-band.dat"
    text = path.read_text()
    assert text.startswith(
        "# End points of segments:\n#   0.00000000 0.50000000 1.00000000 \n"
    )
    assert "0.000000 1.000000\n0.500000 2.000000\n\n" in text
    assert "0.000000 10.000000\n0.500000 20.000000\n\n" in text


def test_phonopy_bandplot_rejects_incompatible_shapes(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "incompatible shapes"):
        write_phonopy_bandplot_data(
            tmp_path,
            [np.array([0.0, 1.0])],
            [np.array([[1.0], [2.0], [3.0]])],
        )
