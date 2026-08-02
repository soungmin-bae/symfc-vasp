from pathlib import Path

from symfc_vasp.parsers.outcar import scan_outcar_summary


def test_scan_reports_mlff_spilling_factor_soft_stop(tmp_path: Path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        " NIONS = 128 ions\n"
        " NSW = 100000\n"
        " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
        " Spilling factor limit (0.9) was exceeded in ionic step 344 (1 atom).\n"
        " soft stop encountered!  aborting job ...\n"
    )

    summary = scan_outcar_summary(outcar)

    assert summary.natom == 128
    assert summary.frames == 1
    assert summary.requested_nsw == 100000
    assert summary.spilling_factor_step == 344
    assert summary.soft_stop is True
