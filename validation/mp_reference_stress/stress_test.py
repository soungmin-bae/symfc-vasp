#!/usr/bin/env python3
"""Reproducible Materials Project stress test for reference reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import spglib
import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = HERE / "artifacts"
CRYSTAL_SYSTEMS = (
    "Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal",
    "Trigonal", "Hexagonal", "Cubic",
)
MATRICES = {
    "diag-2x3x7": np.diag([2, 3, 7]),
    "nondiagonal-unimodular": np.array([[1, -1, 0], [0, 1, 0], [1, 1, 1]]),
}


def _structure_sha256(structure) -> str:
    payload = json.dumps(structure.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _is_molecular(structure) -> tuple[bool, list[int]]:
    from pymatgen.analysis.dimensionality import get_structure_components
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import JmolNN

    graph = StructureGraph.from_local_env_strategy(structure, JmolNN(tol=0.2))
    components = get_structure_components(graph)
    dimensions = [int(component["dimensionality"]) for component in components]
    return bool(dimensions) and all(value == 0 for value in dimensions), dimensions


def _doc_record(doc, *, molecular: bool, dimensions: list[int]) -> dict:
    symmetry = doc.symmetry
    return {
        "material_id": str(doc.material_id),
        "formula": str(doc.formula_pretty),
        "nsites": int(doc.nsites),
        "density_g_cm3": float(doc.density),
        "mp_spacegroup": {
            "symbol": str(symmetry.symbol),
            "number": int(symmetry.number),
            "crystal_system": str(symmetry.crystal_system.value),
        },
        "molecular_0d": bool(molecular),
        "component_dimensions": dimensions,
        "structure_sha256": _structure_sha256(doc.structure),
    }


def _fetch_by_ids(mpr, ids: list[str]) -> list:
    docs = []
    for start in range(0, len(ids), 200):
        docs.extend(mpr.materials.summary.search(
            material_ids=ids[start:start + 200], deprecated=False,
            fields=[
                "material_id", "formula_pretty", "structure", "nsites",
                "density", "symmetry",
            ],
            chunk_size=200,
        ))
    return docs


def prepare(args: argparse.Namespace) -> None:
    from monty.serialization import dumpfn
    from mp_api.client import MPRester

    if not 0.0 <= args.molecular_fraction <= 1.0:
        raise ValueError("--molecular-fraction must be between 0 and 1")
    output = args.artifacts.resolve()
    structures_dir = output / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with MPRester(mute_progress_bars=True) as mpr:
        molecular_candidates = mpr.materials.summary.search(
            density=(0.0, 3.5), num_sites=(2, args.max_primitive_sites),
            energy_above_hull=(0.0, 0.2), deprecated=False,
            fields=[
                "material_id", "formula_pretty", "structure", "nsites",
                "density", "symmetry",
            ],
            chunk_size=1000, num_chunks=args.molecular_chunks,
        )
        molecular_records: list[tuple[object, bool, list[int]]] = []
        for index, doc in enumerate(molecular_candidates, start=1):
            if not doc.structure.is_ordered:
                continue
            try:
                molecular, dimensions = _is_molecular(doc.structure)
            except Exception:
                molecular, dimensions = False, []
            if molecular:
                molecular_records.append((doc, molecular, dimensions))
            if index % 250 == 0:
                print(
                    f"[prepare] topology {index}/{len(molecular_candidates)}; "
                    f"molecular={len(molecular_records)}",
                    flush=True,
                )

        rng.shuffle(molecular_records)
        molecular_target = min(
            len(molecular_records), round(args.count * args.molecular_fraction)
        )
        chosen = molecular_records[:molecular_target]
        chosen_ids = {str(doc.material_id) for doc, _, _ in chosen}

        remaining = args.count - len(chosen)
        if remaining:
            metadata = []
            per_system = max(300, (remaining * 3 + 6) // 7)
            for system in CRYSTAL_SYSTEMS:
                docs = mpr.materials.summary.search(
                    crystal_system=system, num_sites=(1, args.max_primitive_sites),
                    energy_above_hull=(0.0, 0.2), deprecated=False,
                    fields=["material_id"], chunk_size=per_system, num_chunks=1,
                )
                ids = [str(doc.material_id) for doc in docs if str(doc.material_id) not in chosen_ids]
                rng.shuffle(ids)
                metadata.extend(ids[:per_system])
            metadata = list(dict.fromkeys(metadata))
            rng.shuffle(metadata)
            general_docs = _fetch_by_ids(mpr, metadata[:max(remaining * 2, remaining)])
            for doc in general_docs:
                if len(chosen) == args.count:
                    break
                if not doc.structure.is_ordered:
                    continue
                try:
                    molecular, dimensions = _is_molecular(doc.structure)
                except Exception:
                    molecular, dimensions = False, []
                chosen.append((doc, molecular, dimensions))

        db_version = mpr.get_database_version()

    if len(chosen) != args.count:
        raise RuntimeError(f"selected {len(chosen)} structures; requested {args.count}")
    rng.shuffle(chosen)
    entries = []
    for doc, molecular, dimensions in chosen:
        material_id = str(doc.material_id)
        dumpfn(doc.structure, structures_dir / f"{material_id}.json.gz")
        entries.append(_doc_record(doc, molecular=molecular, dimensions=dimensions))
    manifest = {
        "schema": "symfc-vasp-mp-reference-stress-v1",
        "seed": int(args.seed),
        "count": len(entries),
        "molecular_0d_count": sum(entry["molecular_0d"] for entry in entries),
        "mp_database_version": str(db_version) if db_version else None,
        "selection": {
            "energy_above_hull_eV_atom": [0.0, 0.2],
            "max_primitive_sites": int(args.max_primitive_sites),
            "molecular_target_fraction": float(args.molecular_fraction),
            "molecular_classifier": "pymatgen JmolNN(tol=0.2), all components 0D",
        },
        "entries": entries,
    }
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(
        f"[prepare] wrote {len(entries)} structures; "
        f"molecular={manifest['molecular_0d_count']} to {output}",
        flush=True,
    )


def _canonical_primitive(structure):
    from pymatgen.core import Element
    from phonopy.structure.atoms import PhonopyAtoms

    cell = (
        np.asarray(structure.lattice.matrix), np.asarray(structure.frac_coords),
        np.asarray([site.specie.Z for site in structure], dtype=int),
    )
    standardized = spglib.standardize_cell(
        cell, to_primitive=True, no_idealize=False, symprec=0.1
    )
    if standardized is None:
        raise ValueError("Materials Project structure could not be standardized")
    lattice, positions, numbers = standardized
    dataset = spglib.get_symmetry_dataset(standardized, symprec=1e-5)
    if dataset is None:
        raise ValueError("standardized Materials Project primitive has no strict symmetry")
    return (
        PhonopyAtoms(
            cell=lattice,
            scaled_positions=positions,
            numbers=numbers,
            masses=np.asarray(
                [float(Element.from_Z(int(number)).atomic_mass) for number in numbers]
            ),
        ),
        dataset,
    )


def _run_entry(payload: tuple[dict, str, int, float, str]) -> list[dict]:
    from monty.serialization import loadfn
    from phonopy import Phonopy
    from symfc_vasp.outcar_reference import build_outcar_reference

    entry, structures_dir, frames, amplitude, failures_dir = payload
    material_id = entry["material_id"]
    try:
        structure = loadfn(Path(structures_dir) / f"{material_id}.json.gz")
        primitive, expected_dataset = _canonical_primitive(structure)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        failure = Path(failures_dir) / material_id
        failure.mkdir(parents=True, exist_ok=True)
        (failure / "error.txt").write_text(error + "\n")
        return [
            {
                "material_id": material_id,
                "matrix": matrix_name,
                "passed": False,
                "error": error,
                "molecular_0d": bool(entry["molecular_0d"]),
                "elapsed_seconds": 0.0,
            }
            for matrix_name in MATRICES
        ]
    results = []
    for matrix_name, matrix in MATRICES.items():
        started = time.perf_counter()
        seed_payload = f"{material_id}:{matrix_name}:20260831".encode()
        seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        temporary_path = Path(tempfile.mkdtemp(prefix="symfc-vasp-reference-"))
        try:
            supercell = Phonopy(
                primitive, supercell_matrix=matrix, primitive_matrix="P"
            ).supercell
            positions = np.asarray(supercell.positions)[None, :, :] + rng.uniform(
                -amplitude, amplitude, size=(frames, len(supercell), 3)
            )
            unit, generated, mapping, report = build_outcar_reference(
                outcar=temporary_path / "OUTCAR",
                positions=positions,
                output=temporary_path,
                symprec_max=0.3,
                map_tolerance=1.0,
                symbols=tuple(supercell.symbols),
                cell=np.asarray(supercell.cell),
                masses=tuple(float(value) for value in supercell.masses),
            )
            recovered = spglib.get_symmetry_dataset(
                (unit.cell, unit.scaled_positions, unit.numbers), symprec=1e-5
            )
            checks = {
                "primitive_search_success": bool(
                    report.get("primitive_search_success")
                ),
                "strict_spacegroup": bool(
                    recovered is not None
                    and int(recovered.number) == int(expected_dataset.number)
                ),
                "primitive_atom_count": len(unit) == len(primitive),
                "supercell_atom_count": len(generated) == len(supercell),
                "determinant": abs(round(np.linalg.det(report["supercell_matrix"])))
                == abs(round(np.linalg.det(matrix))),
                "one_to_one_mapping": sorted(mapping.tolist())
                == list(range(len(supercell))),
                "cell_reproduced": bool(
                    np.allclose(generated.cell, supercell.cell, atol=1e-6, rtol=0)
                ),
            }
            passed = all(checks.values())
            if not passed:
                failure = Path(failures_dir) / material_id / matrix_name
                failure.mkdir(parents=True, exist_ok=True)
                shutil.copy2(temporary_path / "symmetry_report.yaml", failure)
            result = {
                "material_id": material_id,
                "matrix": matrix_name,
                "passed": passed,
                "checks": checks,
                "expected_spacegroup": int(expected_dataset.number),
                "recovered_spacegroup": (
                    int(recovered.number) if recovered is not None else None
                ),
                "primitive_atoms": len(primitive),
                "supercell_atoms": len(supercell),
                "mapping_max_distance_A": report["mapping"]["max_distance_A"],
                "selected_symprec_A": report["selected_symprec_A"],
            }
        except Exception as exc:
            failure = Path(failures_dir) / material_id / matrix_name
            failure.mkdir(parents=True, exist_ok=True)
            report_path = temporary_path / "symmetry_report.yaml"
            if report_path.is_file():
                shutil.copy2(report_path, failure)
            (failure / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            result = {
                "material_id": material_id,
                "matrix": matrix_name,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "expected_spacegroup": int(expected_dataset.number),
                "primitive_atoms": len(primitive),
            }
        finally:
            shutil.rmtree(temporary_path, ignore_errors=True)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["molecular_0d"] = bool(entry["molecular_0d"])
        results.append(result)
    return results


def run(args: argparse.Namespace) -> None:
    manifest = yaml.safe_load((args.artifacts / "manifest.yaml").read_text())
    entries = manifest["entries"][:args.limit] if args.limit else manifest["entries"]
    results_path = args.artifacts / "results.jsonl"
    completed: set[tuple[str, str]] = set()
    if results_path.is_file() and not args.restart:
        for line in results_path.read_text().splitlines():
            result = json.loads(line)
            completed.add((result["material_id"], result["matrix"]))
    elif args.restart:
        results_path.unlink(missing_ok=True)
        shutil.rmtree(args.artifacts / "failures", ignore_errors=True)

    todo = [
        entry for entry in entries
        if any((entry["material_id"], name) not in completed for name in MATRICES)
    ]
    payloads = [
        (
            entry, str(args.artifacts / "structures"), args.frames,
            args.amplitude, str(args.artifacts / "failures"),
        )
        for entry in todo
    ]
    passed = failed = 0
    with results_path.open("a") as stream, ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {executor.submit(_run_entry, payload): payload[0] for payload in payloads}
        for index, future in enumerate(as_completed(futures), start=1):
            for result in future.result():
                key = (result["material_id"], result["matrix"])
                if key in completed:
                    continue
                stream.write(json.dumps(result, sort_keys=True) + "\n")
                stream.flush()
                completed.add(key)
                if result["passed"]:
                    passed += 1
                else:
                    failed += 1
            if index % 10 == 0 or index == len(futures):
                print(
                    f"[run] structures={index}/{len(futures)} "
                    f"new_pass={passed} new_fail={failed}", flush=True
                )


def summary(args: argparse.Namespace) -> None:
    manifest = yaml.safe_load((args.artifacts / "manifest.yaml").read_text())
    results = [
        json.loads(line) for line in (args.artifacts / "results.jsonl").read_text().splitlines()
    ]
    failures = [result for result in results if not result["passed"]]
    report = {
        "schema": "symfc-vasp-mp-reference-stress-summary-v1",
        "structures_requested": manifest["count"],
        "molecular_structures": manifest["molecular_0d_count"],
        "cases_completed": len(results),
        "cases_expected": manifest["count"] * len(MATRICES),
        "cases_passed": len(results) - len(failures),
        "cases_failed": len(failures),
        "all_passed": len(results) == manifest["count"] * len(MATRICES)
        and not failures,
        "failures_by_matrix": {
            name: sum(result["matrix"] == name for result in failures)
            for name in MATRICES
        },
        "failures": [
            {
                "material_id": result["material_id"],
                "matrix": result["matrix"],
                "error": result.get("error"),
                "checks": result.get("checks"),
            }
            for result in failures
        ],
    }
    (args.artifacts / "summary.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    print(yaml.safe_dump(report, sort_keys=False))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    commands = result.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--count", type=int, default=1000)
    prepare_parser.add_argument("--seed", type=int, default=20260831)
    prepare_parser.add_argument("--max-primitive-sites", type=int, default=24)
    prepare_parser.add_argument("--molecular-chunks", type=int, default=3)
    prepare_parser.add_argument("--molecular-fraction", type=float, default=0.7)
    prepare_parser.set_defaults(handler=prepare)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    run_parser.add_argument("--frames", type=int, default=200)
    run_parser.add_argument("--amplitude", type=float, default=0.3)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--restart", action="store_true")
    run_parser.set_defaults(handler=run)
    summary_parser = commands.add_parser("summary")
    summary_parser.set_defaults(handler=summary)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.artifacts.mkdir(parents=True, exist_ok=True)
    arguments.handler(arguments)
