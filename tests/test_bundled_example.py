import lzma
import shutil
from pathlib import Path

from symfc_vasp.parsers.outcar import parse_outcar_metadata, scan_outcar_summary


def test_bundled_outcar_is_a_fixed_cell_mlff_trajectory(tmp_path: Path):
    archive = (
        Path(__file__).parents[1] / "examples" / "CoH3CN6" / "OUTCAR.xz"
    )
    outcar = tmp_path / "OUTCAR"
    with lzma.open(archive, "rb") as source, outcar.open("wb") as target:
        shutil.copyfileobj(source, target)

    summary = scan_outcar_summary(outcar)
    metadata = parse_outcar_metadata(outcar)

    assert summary.frames == 200
    assert summary.natom == 128
    assert summary.ml_frames == 200
    assert len(metadata.symbols) == 128
    assert metadata.lattice_records >= summary.frames
