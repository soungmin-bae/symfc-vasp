"""Write portable phonopy/phono3py inputs and gnuplot-ready analysis files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


CM1_PER_THZ = 33.35640951981521


def write_phonopy_yaml(
    output: Path,
    unitcell,
    dim,
    force_constants: np.ndarray,
    *,
    symprec: float,
) -> Path:
    """Write a self-contained phonopy YAML file containing the fitted FC2."""
    from phonopy import Phonopy

    phonon = Phonopy(
        unitcell,
        supercell_matrix=np.diag(np.asarray(dim, dtype=int)),
        primitive_matrix="auto",
        symprec=symprec,
    )
    phonon.force_constants = np.asarray(force_constants, dtype=float)
    path = output / "phonopy_disp.yaml"
    phonon.save(filename=path, settings={"force_constants": True})
    return path


def _connected_path_groups(segments, labels):
    """Combine adjacent seekpath segments and retain real path breaks."""
    groups = []
    for segment, (start_label, end_label) in zip(segments, labels):
        start = np.asarray(segment[0], dtype=float)
        end = np.asarray(segment[-1], dtype=float)
        if groups and np.allclose(groups[-1]["points"][-1], start, atol=1e-10, rtol=0):
            groups[-1]["points"].append(end)
            groups[-1]["labels"].append(end_label)
        else:
            groups.append({"points": [start, end], "labels": [start_label, end_label]})
    return groups


def _qpath_text(segments, labels) -> str:
    groups = _connected_path_groups(segments, labels)
    return ", ".join(
        " ".join(f"{value:.10g}" for point in group["points"] for value in point)
        for group in groups
    )


def _label_text(segments, labels) -> str:
    values = [label for group in _connected_path_groups(segments, labels) for label in group["labels"]]
    return " ".join(value.replace("Γ", "GAMMA") for value in values)


def _gnuplot_xtics(boundaries, labels) -> str:
    tick_labels = [labels[0][0]]
    for previous, current in zip(labels[:-1], labels[1:]):
        tick_labels.append(
            previous[1] if previous[1] == current[0] else f"{previous[1]}|{current[0]}"
        )
    tick_labels.append(labels[-1][1])
    values = []
    for label, position in zip(tick_labels, boundaries):
        escaped = label.replace('"', '\\"')
        values.append(f'"{escaped}" {position:.12g}')
    return "set xtics (" + ", ".join(values) + ")"


def _terminal_block(output_name: str, width: int, height: int) -> str:
    return f'''# Set plot_terminal="qt" for an interactive window.
plot_terminal = exists("plot_terminal") ? plot_terminal : "pdf"
if (plot_terminal eq "qt") {{
    set terminal qt size {width},{height} enhanced font "Helvetica,10"
    unset output
}} else {{
    set terminal pdfcairo enhanced color size {width / 100.0:.2f}in,{height / 100.0:.2f}in font "Helvetica,10"
    set output "{output_name}"
}}
'''


def write_band_dat(path: Path, rows: np.ndarray) -> None:
    """Write one blank-line-separated block for each segment and phonon mode."""
    with path.open("w") as handle:
        handle.write(
            "# distance frequency_cm-1 gamma_xx gamma_yy gamma_zz "
            "gamma_trace_over_3 segment mode q_index frequency_THz\n"
        )
        segments = np.unique(rows[:, 0].astype(int))
        modes = np.unique(rows[:, 3].astype(int))
        for segment in segments:
            for mode in modes:
                data = rows[(rows[:, 0] == segment) & (rows[:, 3] == mode)]
                for row in data:
                    values = (
                        row[2], row[4] * CM1_PER_THZ, row[5], row[6], row[7], row[9],
                        int(row[0]), int(row[3]), int(row[1]), row[4],
                    )
                    handle.write(
                        "%.12g %.12g %.12g %.12g %.12g %.12g %d %d %d %.12g\n"
                        % values
                    )
                handle.write("\n\n")


def write_mesh_dat(path: Path, rows: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write(
            "# qnorm_A-1 gamma_xx gamma_yy gamma_zz gamma_trace_over_3 "
            "frequency_cm-1 frequency_THz weight qx qy qz mode q_index\n"
        )
        for row in rows:
            values = (
                row[4], row[8], row[9], row[10], row[11], row[7] * CM1_PER_THZ,
                row[7], int(row[5]), row[1], row[2], row[3], int(row[6]), int(row[0]),
            )
            handle.write(
                "%.12g %.12g %.12g %.12g %.12g %.12g %.12g %d "
                "%.12g %.12g %.12g %d %d\n" % values
            )


def write_phonon_inputs(
    output: Path,
    segments,
    labels,
    dim,
    band_points: int,
    mesh,
    masses,
) -> None:
    path_text = _qpath_text(segments, labels)
    label_text = _label_text(segments, labels)
    dim_text = " ".join(str(int(value)) for value in dim)
    mesh_text = " ".join(str(int(value)) for value in mesh)
    mass_text = " ".join(f"{float(value):.10g}" for value in masses)
    legacy_phono3py_config = output / "phono3py-gruneisen.conf"
    if legacy_phono3py_config.is_file():
        legacy_phono3py_config.unlink()
    common = (
        f"DIM = {dim_text}\n"
        f"MASS = {mass_text}\n"
        f"BAND = {path_text}\n"
        f"BAND_POINTS = {int(band_points)}\n"
        f"BAND_LABELS = {label_text}\n"
        "#BAND_CONNECTION = .TRUE.\n"
        "#EIGENVECTORS = .TRUE.\n"
    )
    (output / "band.conf").write_text(
        "# phonopy input generated by symfc-vasp\n"
        "FORCE_CONSTANTS = READ\n" + common
    )
    (output / "phono3py-gruneisen-band.conf").write_text(
        "# phono3py band-path input generated by symfc-vasp\n"
        "GRUNEISEN = .TRUE.\n" + common
    )
    (output / "phono3py-gruneisen-mesh.conf").write_text(
        "# phono3py q-mesh input generated by symfc-vasp\n"
        "GRUNEISEN = .TRUE.\n"
        f"MESH = {mesh_text}\n"
        f"DIM = {dim_text}\n"
        f"MASS = {mass_text}\n"
    )
    command = '''#!/usr/bin/env bash
set -euo pipefail

# phonopy_disp.yaml contains the cell, supercell matrix, masses, and FC2.
phonopy -p band.conf -s
phonopy-bandplot --gnuplot band.yaml > phonopy-band.dat
phono3py phono3py-gruneisen-band.conf -c POSCAR-unitcell --fc2 --fc3
phono3py phono3py-gruneisen-mesh.conf -c POSCAR-unitcell --fc2 --fc3
'''
    command_path = output / "run_phonopy_phono3py.sh"
    command_path.write_text(command)
    command_path.chmod(0o755)


def write_band_gnuplot_scripts(
    output: Path,
    boundaries,
    labels,
    gmin: float,
    gmax: float,
    fmin_cm1: float,
    fmax_cm1: float,
    cutoff_thz: float,
) -> None:
    xtics = _gnuplot_xtics(boundaries, labels)
    verticals = "\n".join(
        f"set arrow from first {x:.12g}, graph 0 to first {x:.12g}, graph 1 nohead lc rgb '#cccccc' lw 0.7 back"
        for x in boundaries
    )
    common = f'''set border lw 1
set tics in
set xtics mirror
set ytics mirror
set grid noxtics noytics
set xzeroaxis lt 2 lc rgb "#777777"
set xrange [{boundaries[0]:.12g}:{boundaries[-1]:.12g}]
{xtics}
{verticals}
'''
    phonon = _terminal_block("phonon_dispersion_gnuplot.pdf", 1000, 550) + common + f'''set yrange [{fmin_cm1:.12g}:{fmax_cm1:.12g}]
set xlabel "High-symmetry q path"
set ylabel "Frequency (cm^{{-1}})"
plot "phonon_band.dat" using 1:2 with lines lc rgb "black" lw 0.75 notitle
'''
    (output / "plot_phonon_dispersion.gp").write_text(phonon)

    component_titles = ["gamma_xx", "gamma_yy", "gamma_zz", "Tr(gamma)/3"]
    component_columns = [3, 4, 5, 6]
    qresolved = _terminal_block("mode_gruneisen_q_resolved_gnuplot.pdf", 1100, 1200)
    qresolved += common + f"gmin={gmin:.12g}\ngmax={gmax:.12g}\nfcut={cutoff_thz:.12g}\n"
    qresolved += (
        "set palette defined (gmin '#0000ff', 0 '#ffffff', gmax '#ff0000')\n"
        "set cbrange [gmin:gmax]\nunset colorbox\n"
        "set multiplot layout 4,1 rowsfirst\nset yrange [gmin:gmax]\nunset xlabel\n"
    )
    for index, (title, column) in enumerate(zip(component_titles, component_columns)):
        qresolved += f'set title "{title}"\nset ylabel "Mode Gruneisen parameter"\n'
        if index == 3:
            qresolved += 'set xlabel "High-symmetry q path"\n'
        else:
            qresolved += "set format x \"\"\n"
        qresolved += (
            f'plot "phonon_band.dat" using 1:(abs($10)>=fcut?${column}:1/0):{column} '
            "with points pt 7 ps 0.18 lc palette notitle\n"
        )
        if index == 2:
            qresolved += "set format x \"%g\"\n"
    qresolved += "unset multiplot\n"
    (output / "plot_mode_gruneisen_q_resolved.gp").write_text(qresolved)

    overlay = _terminal_block("mode_gruneisen_on_phonon_dispersion_gnuplot.pdf", 1100, 1200)
    overlay += common + (
        f"gmin={gmin:.12g}\ngmax={gmax:.12g}\nfcut={cutoff_thz:.12g}\n"
        f"set yrange [{fmin_cm1:.12g}:{fmax_cm1:.12g}]\n"
        "set palette defined (gmin '#0000ff', 0 '#ffffff', gmax '#ff0000')\n"
        "set cbrange [gmin:gmax]\nset colorbox\n"
        "set multiplot layout 4,1 rowsfirst\nunset xlabel\n"
    )
    for index, (title, column) in enumerate(zip(component_titles, component_columns)):
        overlay += f'set title "{title}"\nset ylabel "Frequency (cm^{{-1}})"\n'
        if index == 3:
            overlay += 'set xlabel "High-symmetry q path"\n'
        else:
            overlay += "set format x \"\"\n"
        overlay += (
            f'plot "phonon_band.dat" using 1:(abs($10)>=fcut?$2:1/0):{column} '
            "with lines lc palette lw 0.8 notitle\n"
        )
        if index == 2:
            overlay += "set format x \"%g\"\n"
    overlay += "unset multiplot\n"
    (output / "plot_mode_gruneisen_on_phonon_dispersion.gp").write_text(overlay)


def write_mesh_gnuplot_script(
    output: Path,
    mesh,
    gmin: float,
    gmax: float,
    cutoff_thz: float,
) -> None:
    mesh_tag = "x".join(str(int(value)) for value in mesh)
    script = _terminal_block(f"mode_gruneisen_qmesh_{mesh_tag}_gnuplot.pdf", 900, 1100)
    script += f'''set border lw 1
set tics in
set xtics mirror
set ytics mirror
set xzeroaxis lt 2 lc rgb "#777777"
gmin={gmin:.12g}
gmax={gmax:.12g}
fcut={cutoff_thz:.12g}
set yrange [gmin:gmax]
set multiplot layout 4,1 rowsfirst
unset xlabel
'''
    for index, (title, column) in enumerate(
        zip(["gamma_xx", "gamma_yy", "gamma_zz", "Tr(gamma)/3"], [2, 3, 4, 5])
    ):
        script += f'set title "{title}"\nset ylabel "Mode Gruneisen parameter"\n'
        if index == 3:
            script += 'set xlabel "|q| (A^{{-1}})"\n'
        else:
            script += "set format x \"\"\n"
        script += (
            f'plot "gruneisen_qmesh_{mesh_tag}.dat" using 1:(abs($7)>=fcut?${column}:1/0) '
            "with points pt 7 ps 0.16 lc rgb '#315efb' notitle\n"
        )
        if index == 2:
            script += "set format x \"%g\"\n"
    script += "unset multiplot\n"
    (output / "plot_mode_gruneisen_qmesh.gp").write_text(script)


def write_reproduction_readme(output: Path, mesh) -> None:
    mesh_tag = "x".join(str(int(value)) for value in mesh)
    (output / "README_REPRODUCE.md").write_text(
        f'''# Reproducing the phonon and mode-Gruneisen analysis

The `.dat` files are plain ASCII and contain column definitions on their first
comment line. Blank lines separate phonon branches and path segments.

Regenerate the four figures with gnuplot:

```bash
gnuplot plot_phonon_dispersion.gp
gnuplot plot_mode_gruneisen_q_resolved.gp
gnuplot plot_mode_gruneisen_on_phonon_dispersion.gp
gnuplot plot_mode_gruneisen_qmesh.gp
```

For an interactive window, replace the first command by, for example:

```bash
gnuplot -e 'plot_terminal="qt"' plot_phonon_dispersion.gp
```

The q-path data are in `phonon_band.dat`; the mesh data are in
`gruneisen_qmesh_{mesh_tag}.dat`. `phonopy_disp.yaml` contains the unit cell,
supercell matrix, effective masses, and fitted FC2. `band.conf` and
`phono3py-gruneisen-band.conf` and `phono3py-gruneisen-mesh.conf` record the
external phonopy/phono3py settings.
To rerun those programs, link or copy `FORCE_CONSTANTS`, `fc2.hdf5`, and
`fc3.hdf5` into this directory and execute `run_phonopy_phono3py.sh`.
'''
    )
