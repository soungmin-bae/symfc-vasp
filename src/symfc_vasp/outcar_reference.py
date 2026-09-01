"""Build a symmetric phonopy reference directly from a fixed-cell OUTCAR."""

from __future__ import annotations

import json
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import spglib
import yaml
from scipy.spatial import cKDTree

from .parsers.outcar import parse_outcar_metadata


def _periodic_mean(
    frames: np.ndarray, cell: np.ndarray | None = None,
) -> np.ndarray:
    """Average wrapped coordinates, using the cell metric when available."""
    frames = np.asarray(frames, dtype=float)
    if cell is None:
        phase = np.exp(2j * np.pi * frames)
        return np.mod(np.angle(np.mean(phase, axis=0)) / (2.0 * np.pi), 1.0)
    cell = np.asarray(cell, dtype=float)
    reference = frames[0]
    unwrapped_cart = _minimum_image(frames - reference[None, :, :], cell)
    mean_cart = reference @ cell + np.mean(unwrapped_cart, axis=0)
    return np.mod(mean_cart @ np.linalg.inv(cell), 1.0)


def _minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Shortest Cartesian images, including the neighbouring translations."""
    wrapped = np.asarray(delta, dtype=float) - np.rint(delta)
    candidates = []
    for shift in np.ndindex(3, 3, 3):
        shift = np.asarray(shift, dtype=float) - 1.0
        candidates.append((wrapped + shift) @ cell)
    stacked = np.stack(candidates, axis=0)
    norms = np.einsum("s...i,s...i->s...", stacked, stacked)
    return np.take_along_axis(stacked, np.argmin(norms, axis=0)[None, ..., None], axis=0)[0]


def _translation_invariant_projection_distances(
    projected_frac: np.ndarray, reference_frac: np.ndarray, cell: np.ndarray,
) -> np.ndarray:
    """Measure internal projection distortion after removing rigid translation."""
    displacement = _minimum_image(projected_frac - reference_frac, cell)
    displacement -= np.mean(displacement, axis=0)
    return np.linalg.norm(displacement, axis=-1)


def _species_numbers(symbols: tuple[str, ...]) -> np.ndarray:
    from phonopy.structure.atoms import get_atomic_data

    symbol_map = get_atomic_data().symbol_map
    return np.asarray([symbol_map[symbol] for symbol in symbols], dtype=int)


def _periodic_trees(
    frac: np.ndarray, symbols: np.ndarray, cell: np.ndarray,
) -> dict[str, tuple[cKDTree, np.ndarray]]:
    """Build species-resolved Cartesian trees with neighbouring images."""
    shifts = np.asarray(list(np.ndindex(3, 3, 3)), dtype=float) - 1.0
    shift_cart = shifts @ cell
    trees: dict[str, tuple[cKDTree, np.ndarray]] = {}
    for symbol in dict.fromkeys(symbols.tolist()):
        indices = np.flatnonzero(symbols == symbol)
        cart = frac[indices] @ cell
        images = (cart[None, :, :] + shift_cart[:, None, :]).reshape(-1, 3)
        trees[symbol] = (cKDTree(images), np.tile(indices, len(shifts)))
    return trees


def _nearest_species_map(
    generated_frac: np.ndarray,
    generated_symbols: np.ndarray,
    source_symbols: np.ndarray,
    cell: np.ndarray,
    trees: dict[str, tuple[cKDTree, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a one-to-one nearest map, or ``None`` when sites collide."""
    mapping = np.full(len(generated_frac), -1, dtype=int)
    distances = np.full(len(generated_frac), np.nan)
    generated_cart = np.mod(generated_frac, 1.0) @ cell
    for symbol in dict.fromkeys(generated_symbols.tolist()):
        left = np.flatnonzero(generated_symbols == symbol)
        right = np.flatnonzero(source_symbols == symbol)
        if len(left) != len(right):
            return None
        tree, image_to_source = trees[symbol]
        values, image_indices = tree.query(generated_cart[left], k=1)
        matched = image_to_source[np.asarray(image_indices, dtype=int)]
        if len(np.unique(matched)) != len(matched):
            return None
        mapping[left] = matched
        distances[left] = values
    if np.any(mapping < 0) or len(np.unique(mapping)) != len(mapping):
        return None
    return mapping, distances


def _align_origin(
    generated_frac: np.ndarray, mean_frac: np.ndarray, generated_symbols: np.ndarray,
    source_symbols: np.ndarray, cell: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Find the rigid fractional translation between spglib and VASP origins."""
    counts = {
        symbol: int(np.count_nonzero(generated_symbols == symbol))
        for symbol in dict.fromkeys(generated_symbols.tolist())
    }
    anchor_symbol = min(counts, key=counts.get)
    generated_anchor = int(np.flatnonzero(generated_symbols == anchor_symbol)[0])
    source_anchors = np.flatnonzero(source_symbols == anchor_symbol)
    trees = _periodic_trees(mean_frac, source_symbols, cell)
    best_shift = np.zeros(3)
    best_score = float("inf")
    for source_index in source_anchors:
        shift = (mean_frac[source_index] - generated_frac[generated_anchor]) % 1.0
        result = _nearest_species_map(
            generated_frac + shift, generated_symbols, source_symbols, cell, trees
        )
        if result is None:
            continue
        _, distances = result
        score = float(np.max(distances))
        if score < best_score:
            best_score = score
            best_shift = shift
    if not np.isfinite(best_score):
        raise ValueError("could not find a one-to-one species-preserving origin alignment")
    return best_shift, best_score


def _symmetrize_mean_positions(
    frac: np.ndarray,
    numbers: np.ndarray,
    cell: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a noisy periodic mean onto symmetry in the original cell frame."""
    symbols = numbers.astype(str)
    current = np.asarray(frac, dtype=float)
    for _ in range(2):
        trees = _periodic_trees(current, symbols, cell)
        accumulated = np.zeros_like(current)
        counts = np.zeros(len(current), dtype=int)
        for rotation, translation in zip(rotations, translations, strict=True):
            transformed = np.mod(current @ rotation.T + translation, 1.0)
            result = _nearest_species_map(
                transformed, symbols, symbols, cell, trees
            )
            if result is None:
                raise ValueError("symmetry operation does not give a one-to-one atom map")
            mapping, _ = result
            for source_index, target_index in enumerate(mapping):
                delta = transformed[source_index] - current[target_index]
                cart = _minimum_image(delta[None, :], cell)[0]
                accumulated[target_index] += current[target_index] + cart @ np.linalg.inv(cell)
                counts[target_index] += 1
        if np.any(counts == 0):
            raise ValueError("symmetry projection left an unmapped atom")
        current = np.mod(accumulated / counts[:, None], 1.0)
    inversion = -np.eye(3, dtype=int)
    if any(np.array_equal(rotation, inversion) for rotation in rotations):
        # Keep inversion projections origin-sensitive. Removing an arbitrary
        # shift here can manufacture a centrosymmetric parent for a genuinely
        # polar structure (for example P4mm -> P4/mmm).
        distortion = np.linalg.norm(_minimum_image(current - frac, cell), axis=-1)
    else:
        # A non-centrosymmetric projection may choose a different but
        # equivalent origin. The species-preserving alignment below handles
        # that rigid shift, so it is not structural distortion.
        distortion = _translation_invariant_projection_distances(current, frac, cell)
    return current, distortion


def _operation_set_is_subset(
    lower: dict, upper: dict, *, tolerance_A: float = 0.05,
) -> bool:
    """Return whether two candidates form an affine subgroup in one cell."""
    lower_rotations = lower.get("common_rotations")
    lower_translations = lower.get("common_translations")
    upper_rotations = upper.get("common_rotations")
    upper_translations = upper.get("common_translations")
    common_cell = lower.get("common_cell")
    if any(
        value is None
        for value in (
            lower_rotations,
            lower_translations,
            upper_rotations,
            upper_translations,
            common_cell,
        )
    ):
        return False
    for rotation, translation in zip(
        lower_rotations, lower_translations, strict=True
    ):
        matched = False
        for upper_rotation, upper_translation in zip(
            upper_rotations, upper_translations, strict=True
        ):
            if not np.array_equal(rotation, upper_rotation):
                continue
            delta = translation - upper_translation
            distance = np.linalg.norm(
                _minimum_image(delta[None, :], common_cell)[0]
            )
            if distance <= tolerance_A:
                matched = True
                break
        if not matched:
            return False
    return True


def _rotation_set_is_subset(lower: dict, upper: dict) -> bool:
    """Return whether the lower point operations occur in the upper group."""
    lower_rotations = {
        tuple(np.asarray(rotation, dtype=int).ravel())
        for rotation in lower["dataset"].rotations
    }
    upper_rotations = {
        tuple(np.asarray(rotation, dtype=int).ravel())
        for rotation in upper["dataset"].rotations
    }
    return lower_rotations.issubset(upper_rotations)


def _common_translation_count(candidate: dict) -> int:
    rotations = candidate.get("common_rotations")
    if rotations is None:
        return 0
    identity = np.eye(3, dtype=int)
    return sum(np.array_equal(rotation, identity) for rotation in rotations)


def _primitive_translation_count(candidate: dict) -> int:
    identity = np.eye(3, dtype=int)
    return sum(
        np.array_equal(rotation, identity)
        for rotation in candidate["dataset"].rotations
    )


def _write_poscar(path: Path, atoms, comment: str) -> None:
    from phonopy.interface.vasp import write_vasp

    write_vasp(str(path), atoms)
    lines = path.read_text().splitlines()
    lines[0] = comment
    path.write_text("\n".join(lines) + "\n")


def _symprec_grid(maximum: float) -> list[float]:
    if maximum <= 0:
        raise ValueError("--reference-symprec-max must be positive")
    # A random-displacement trajectory retains small finite-sample offsets in
    # its periodic mean. Values above 0.1 A are often needed to recover the
    # parent space group while still being far below an inter-site distance.
    base = [
        1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2,
        5e-2, 7e-2, 1e-1, 1.2e-1, 1.4e-1, 1.5e-1, 1.8e-1,
        2e-1, 2.5e-1, 3e-1,
    ]
    grid = [value for value in base if value <= maximum * (1 + 1e-12)]
    if not grid or grid[-1] < maximum:
        grid.append(float(maximum))
    return sorted(set(grid))


def _affine_operation_subgroups(
    rotations: np.ndarray, translations: np.ndarray,
) -> list[tuple[int, ...]]:
    """Return proper affine subgroups generated by up to three operations.

    Rotation-only subgroups cannot distinguish primitive and centred space
    groups that share a point group.  Keeping each fractional translation with
    its rotation recovers those alternatives without guessing a conventional
    setting.
    """
    rotations = np.asarray(rotations, dtype=int)
    translations = np.mod(np.asarray(translations, dtype=float), 1.0)

    def key(rotation: np.ndarray, translation: np.ndarray) -> tuple:
        wrapped = np.mod(translation, 1.0)
        wrapped[np.isclose(wrapped, 1.0, atol=1e-8)] = 0.0
        return (*rotation.ravel().tolist(), *np.round(wrapped, 8).tolist())

    operation_index = {
        key(rotation, translation): index
        for index, (rotation, translation) in enumerate(
            zip(rotations, translations, strict=True)
        )
    }
    identity_key = key(np.eye(3, dtype=int), np.zeros(3))
    identity = operation_index.get(identity_key)
    if identity is None:
        return []

    def compose(left: int, right: int) -> int | None:
        # Fractional row vectors transform as x' = x R.T + t.
        rotation = rotations[right] @ rotations[left]
        translation = translations[left] @ rotations[right].T + translations[right]
        return operation_index.get(key(rotation, translation))

    def closure(generators: tuple[int, ...]) -> tuple[int, ...] | None:
        members = {identity, *generators}
        changed = True
        while changed:
            changed = False
            current = tuple(members)
            for left in current:
                for right in (*current, *generators):
                    product = compose(left, right)
                    if product is None:
                        return None
                    if product not in members:
                        members.add(product)
                        changed = True
        return tuple(sorted(members))

    nonidentity = tuple(index for index in range(len(rotations)) if index != identity)
    subgroups: set[tuple[int, ...]] = set()
    frontier: set[tuple[int, ...]] = set()
    if len(rotations) > 64:
        return []
    for generator in nonidentity:
        members = closure((generator,))
        if members is not None and 1 < len(members) < len(rotations):
            subgroups.add(members)
            frontier.add(members)
    # Crystallographic point groups and their centred affine extensions are
    # generated by at most three independent operations in the cases relevant
    # here. Expand unique closures instead of enumerating all generator tuples.
    for _ in range(2):
        next_frontier: set[tuple[int, ...]] = set()
        for members in frontier:
            for generator in nonidentity:
                if generator in members:
                    continue
                expanded = closure((*members, generator))
                if (
                    expanded is not None
                    and 1 < len(expanded) < len(rotations)
                    and expanded not in subgroups
                ):
                    subgroups.add(expanded)
                    next_frontier.add(expanded)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(subgroups, key=lambda members: (len(members), members))


def _rotation_operation_subgroups(rotations: np.ndarray) -> list[tuple[int, ...]]:
    """Return inexpensive cyclic point-operation subsets for large supercells."""
    identity = np.eye(3, dtype=int)
    unique = {tuple(np.asarray(rotation, dtype=int).ravel()) for rotation in rotations}
    rotation_subgroups: set[tuple[tuple[int, ...], ...]] = set()
    for rotation_key in unique:
        generator = np.asarray(rotation_key, dtype=int).reshape(3, 3)
        if np.array_equal(generator, identity):
            continue
        current = identity.copy()
        closure: set[tuple[int, ...]] = set()
        for _ in range(12):
            key = tuple(current.ravel())
            if key in closure:
                break
            closure.add(key)
            current = current @ generator
        if np.array_equal(current, identity) and 1 < len(closure) < len(unique):
            rotation_subgroups.add(tuple(sorted(closure)))
    result = []
    for subgroup in rotation_subgroups:
        members = set(subgroup)
        indices = tuple(
            index
            for index, rotation in enumerate(rotations)
            if tuple(np.asarray(rotation, dtype=int).ravel()) in members
        )
        if 1 < len(indices) < len(rotations):
            result.append(indices)
    return result


def _rotation_order(rotation: np.ndarray) -> int:
    """Return the finite order of a crystallographic rotation."""
    current = np.eye(3, dtype=int)
    rotation = np.asarray(rotation, dtype=int)
    for order in range(1, 13):
        current = rotation @ current
        if np.array_equal(current, np.eye(3, dtype=int)):
            return order
    return 0


def _proper_rotation_orders(rotations: np.ndarray) -> frozenset[int]:
    """Return finite orders represented by orientation-preserving rotations."""
    return frozenset(
        _rotation_order(rotation)
        for rotation in rotations
        if round(np.linalg.det(rotation)) == 1
    )


def _fold_mean_to_candidate_unit(
    candidate: dict, mean_frac: np.ndarray, cell: np.ndarray,
) -> np.ndarray:
    """Average translation-equivalent supercell sites in a candidate unit cell."""
    unit = candidate["unit"]
    generated = candidate["generated"]
    mapping = candidate["mapping"]
    unit_lattice = np.asarray(unit.cell)
    unit_frac = np.asarray(unit.scaled_positions)
    unit_symbols = np.asarray(unit.symbols)
    origin_shift_cart = np.asarray(candidate["origin_shift"]) @ cell
    generated_unit_frac = np.mod(
        (np.asarray(generated.positions) - origin_shift_cart)
        @ np.linalg.inv(unit_lattice),
        1.0,
    )
    trees = _periodic_trees(unit_frac, unit_symbols, unit_lattice)
    site_indices = np.full(len(generated), -1, dtype=int)
    generated_symbols = np.asarray(generated.symbols)
    for symbol in dict.fromkeys(generated_symbols.tolist()):
        indices = np.flatnonzero(generated_symbols == symbol)
        tree, image_to_source = trees[symbol]
        distances, image_indices = tree.query(
            generated_unit_frac[indices] @ unit_lattice, k=1
        )
        if float(np.max(distances)) > 1e-5:
            raise ValueError("generated supercell does not fold onto candidate unit sites")
        site_indices[indices] = image_to_source[np.asarray(image_indices, dtype=int)]
    if np.any(site_indices < 0):
        raise ValueError("candidate unit-site assignment is incomplete")

    source_unit_frac = np.mod(
        (mean_frac[np.asarray(mapping, dtype=int)] @ cell - origin_shift_cart)
        @ np.linalg.inv(unit_lattice),
        1.0,
    )
    accumulated = np.zeros((len(unit), 3), dtype=float)
    counts = np.zeros(len(unit), dtype=int)
    for generated_index, unit_index in enumerate(site_indices):
        delta = _minimum_image(
            (source_unit_frac[generated_index] - unit_frac[unit_index])[None, :],
            unit_lattice,
        )[0]
        accumulated[unit_index] += delta
        counts[unit_index] += 1
    determinant = abs(round(np.linalg.det(candidate["matrix"])))
    if np.any(counts != determinant):
        raise ValueError(
            "translation folding did not assign one copy per supercell translation"
        )
    return np.mod(
        unit_frac + (accumulated / counts[:, None]) @ np.linalg.inv(unit_lattice),
        1.0,
    )


def _materialize_candidate(
    *,
    unit_lattice: np.ndarray,
    unit_frac: np.ndarray,
    unit_numbers: np.ndarray,
    dataset,
    symprec: float,
    cell: np.ndarray,
    mean_frac: np.ndarray,
    symbols: tuple[str, ...],
    masses_by_number: dict[int, float],
    map_tolerance: float,
    supercell_projection_distances: np.ndarray,
    primitive_projection_distances: np.ndarray,
) -> dict:
    """Build and map a strict primitive candidate in the trajectory frame."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    matrix_float = (cell @ np.linalg.inv(unit_lattice)).T
    matrix = np.rint(matrix_float).astype(int)
    matrix_residual = float(np.max(np.abs(matrix_float - matrix)))
    determinant = abs(int(round(np.linalg.det(matrix))))
    if matrix_residual > 1e-6 or determinant * len(unit_numbers) != len(symbols):
        raise ValueError(
            f"non-integer cell relation (residual={matrix_residual:.3e}, det={determinant})"
        )
    unit = PhonopyAtoms(
        cell=unit_lattice,
        scaled_positions=unit_frac,
        numbers=unit_numbers,
        masses=np.asarray(
            [masses_by_number[int(number)] for number in unit_numbers]
        ),
    )
    phonon = Phonopy(
        unit, supercell_matrix=matrix, primitive_matrix="P", symprec=1e-5
    )
    generated = phonon.supercell
    if not np.allclose(generated.cell, cell, atol=1e-6, rtol=0):
        raise ValueError("generated supercell lattice differs from trajectory")
    generated_frac = np.asarray(generated.scaled_positions)
    source_symbols = np.asarray(symbols)
    generated_symbols = np.asarray(generated.symbols)
    origin_shift, origin_residual = _align_origin(
        generated_frac, mean_frac, generated_symbols, source_symbols, cell
    )
    generated_frac = (generated_frac + origin_shift) % 1.0
    mapped = _nearest_species_map(
        generated_frac, generated_symbols, source_symbols, cell,
        _periodic_trees(mean_frac, source_symbols, cell),
    )
    if mapped is None:
        raise ValueError("atom map is not one-to-one")
    mapping, distances = mapped
    max_distance = float(np.max(distances))
    if max_distance > map_tolerance:
        raise ValueError(
            f"atom-map residual {max_distance:.4f} A exceeds {map_tolerance:.4f} A"
        )
    generated = PhonopyAtoms(
        cell=generated.cell, scaled_positions=generated_frac,
        symbols=generated.symbols, masses=generated.masses,
    )
    common_symmetry = spglib.get_symmetry(
        (
            cell,
            generated_frac,
            _species_numbers(tuple(generated.symbols)),
        ),
        symprec=1e-5,
    )
    signature = (
        int(dataset.number), len(unit), determinant
    )
    return {
        "dataset": dataset,
        "symprec": float(symprec),
        "unit": unit,
        "generated": generated,
        "matrix": matrix,
        "matrix_residual": matrix_residual,
        "mapping": mapping,
        "distances": distances,
        "origin_shift": origin_shift,
        "origin_residual": float(origin_residual),
        "signature": signature,
        "projection_distances": supercell_projection_distances,
        "primitive_projection_distances": primitive_projection_distances,
        "common_rotations": (
            np.asarray(common_symmetry["rotations"])
            if common_symmetry is not None else None
        ),
        "common_translations": (
            np.asarray(common_symmetry["translations"])
            if common_symmetry is not None else None
        ),
        "common_cell": np.asarray(cell),
    }


def build_outcar_reference(
    *,
    outcar: Path,
    positions: np.ndarray,
    output: Path,
    symprec_max: float,
    map_tolerance: float,
    symbols: tuple[str, ...] | None = None,
    cell: np.ndarray | None = None,
    masses: tuple[float, ...] | None = None,
) -> tuple[object, object, np.ndarray, dict]:
    """Return a symmetry-projected unit cell, supercell, map, and manifest.

    ``positions`` retain VASP's original atom order.  The returned mapping is
    phonopy-supercell index -> OUTCAR index and is therefore applied to every
    trajectory frame before fitting.
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    if symbols is None or cell is None:
        metadata = parse_outcar_metadata(outcar)
        symbols = metadata.symbols
        masses = metadata.masses
        cell = metadata.cell
        lattice_records = metadata.lattice_records
    else:
        lattice_records = None
    cell = np.asarray(cell, dtype=float)
    if positions.ndim != 3 or positions.shape[1] != len(symbols):
        raise ValueError("trajectory positions do not match recovered species metadata")
    fractional_frames = np.asarray(positions) @ np.linalg.inv(cell)
    mean_frac = _periodic_mean(fractional_frames, cell)
    if len(fractional_frames) > 1:
        fluctuations = _minimum_image(
            fractional_frames - mean_frac[None, :, :], cell
        )
        atom_mean_sem = np.std(fluctuations, axis=0, ddof=1) / np.sqrt(
            len(fractional_frames)
        )
        atom_mean_sem_norm = np.linalg.norm(atom_mean_sem, axis=1)
        mean_sem_95 = float(np.quantile(atom_mean_sem_norm, 0.95))
    else:
        mean_sem_95 = 0.0
    # A higher-symmetry candidate is defensible only when its displacement
    # from the trajectory mean is comparable to the finite-sampling error of
    # that mean. This prevents a large symprec from forcing a nearby but false
    # supergroup while retaining noisy means from random-displacement data.
    distortion_budget = min(
        float(map_tolerance), max(0.075, 3.7 * mean_sem_95)
    )
    numbers = _species_numbers(symbols)
    if masses is None:
        try:
            masses = tuple(
                float(value)
                for value in PhonopyAtoms(
                    cell=cell,
                    scaled_positions=mean_frac,
                    symbols=symbols,
                ).masses
            )
        except RuntimeError as exc:
            raise ValueError(
                "atomic masses are unavailable; provide masses explicitly or "
                "use an OUTCAR containing POMASS records"
            ) from exc
    if len(masses) != len(symbols):
        raise ValueError("atomic masses do not match the recovered atom count")
    masses_by_number: dict[int, float] = {}
    for number, mass in zip(numbers, masses):
        previous = masses_by_number.setdefault(int(number), float(mass))
        if not np.isclose(previous, mass, atol=1e-8, rtol=0):
            raise ValueError(
                "isotope-dependent masses within one chemical species require "
                "an explicit primitive-cell reference"
            )
    scan: list[dict] = []
    candidates: list[dict] = []
    detection_cache: dict[tuple, tuple[dict, dict, list[dict]]] = {}
    best_mapping_residual = np.inf
    for symprec in _symprec_grid(symprec_max):
        dataset = spglib.get_symmetry_dataset((cell, mean_frac, numbers), symprec=symprec)
        if dataset is None:
            continue
        record = {
            "symprec_A": float(symprec),
            "international": str(dataset.international),
            "number": int(dataset.number),
            "operations": int(len(dataset.rotations)),
        }
        detection_key = (
            int(dataset.number), dataset.rotations.tobytes(),
            np.round(dataset.translations, decimals=10).tobytes(),
        )
        if detection_key in detection_cache:
            fields, cached_candidate, cached_promotions = detection_cache[detection_key]
            record.update(deepcopy(fields))
            candidate = copy(cached_candidate)
            candidate["symprec"] = float(symprec)
            candidate["scan_symprec"] = float(symprec)
            candidates.append(candidate)
            for cached_promotion in cached_promotions:
                promoted_copy = copy(cached_promotion)
                promoted_copy["scan_symprec"] = float(symprec)
                candidates.append(promoted_copy)
            scan.append(record)
            continue
        try:
            projected_frac, supercell_projection_distances = _symmetrize_mean_positions(
                mean_frac, numbers, cell, dataset.rotations, dataset.translations
            )
            standardized = spglib.standardize_cell(
                (cell, projected_frac, numbers), to_primitive=True,
                no_idealize=True, symprec=symprec,
            )
            if standardized is None:
                raise ValueError("standardization failed")
            unit_lattice, unit_frac, unit_numbers = standardized
            strict_before_reprojection = spglib.get_symmetry_dataset(
                (unit_lattice, unit_frac, unit_numbers), symprec=1e-5
            )
            preprojection_candidate = None
            if strict_before_reprojection is not None:
                preprojection_frac, preprojection_distances = (
                    _symmetrize_mean_positions(
                        unit_frac, unit_numbers, unit_lattice,
                        strict_before_reprojection.rotations,
                        strict_before_reprojection.translations,
                    )
                )
                preprojection_strict = spglib.get_symmetry_dataset(
                    (unit_lattice, preprojection_frac, unit_numbers),
                    symprec=1e-5,
                )
                if preprojection_strict is not None:
                    preprojection_candidate = _materialize_candidate(
                        unit_lattice=unit_lattice,
                        unit_frac=preprojection_frac,
                        unit_numbers=unit_numbers,
                        dataset=preprojection_strict, symprec=symprec,
                        cell=cell, mean_frac=mean_frac, symbols=symbols,
                        masses_by_number=masses_by_number,
                        map_tolerance=map_tolerance,
                        supercell_projection_distances=(
                            supercell_projection_distances
                        ),
                        primitive_projection_distances=preprojection_distances,
                    )
            primitive_dataset = spglib.get_symmetry_dataset(
                (unit_lattice, unit_frac, unit_numbers), symprec=symprec
            )
            if primitive_dataset is None:
                raise ValueError("primitive symmetry search failed")
            unit_frac, primitive_projection_distances = _symmetrize_mean_positions(
                unit_frac, unit_numbers, unit_lattice,
                primitive_dataset.rotations, primitive_dataset.translations,
            )
            projected_dataset = spglib.get_symmetry_dataset(
                (unit_lattice, unit_frac, unit_numbers), symprec=1e-5
            )
            if projected_dataset is None:
                raise ValueError("primitive symmetry projection failed at 1e-5 A")
            candidate = _materialize_candidate(
                unit_lattice=unit_lattice, unit_frac=unit_frac,
                unit_numbers=unit_numbers, dataset=projected_dataset,
                symprec=symprec, cell=cell, mean_frac=mean_frac,
                symbols=symbols, masses_by_number=masses_by_number,
                map_tolerance=map_tolerance,
                supercell_projection_distances=supercell_projection_distances,
                primitive_projection_distances=primitive_projection_distances,
            )
            candidate["direct_scan_candidate"] = True
            candidate["scan_symprec"] = float(symprec)
            distances = candidate["distances"]
            max_distance = float(np.max(distances))
            best_mapping_residual = min(best_mapping_residual, max_distance)
            record.update({
                "valid": True,
                "detected_supercell_spacegroup": {
                    "international": str(dataset.international),
                    "number": int(dataset.number),
                    "operations": int(len(dataset.rotations)),
                },
                "international": str(projected_dataset.international),
                "number": int(projected_dataset.number),
                "operations": int(len(projected_dataset.rotations)),
                "unitcell_atoms": int(len(candidate["unit"])),
                "supercell_matrix": candidate["matrix"].tolist(),
                "matrix_residual": candidate["matrix_residual"],
                "mapping_max_distance_A": max_distance,
                "mapping_rms_distance_A": float(np.sqrt(np.mean(distances**2))),
                "projection_max_distance_A": float(
                    np.max(supercell_projection_distances)
                ),
                "projection_rms_distance_A": float(
                    np.sqrt(np.mean(supercell_projection_distances**2))
                ),
                "primitive_projection_max_distance_A": float(
                    np.max(primitive_projection_distances)
                ),
            })
            candidates.append(candidate)

            promotions = []
            promoted_candidates = []
            folded_records = []
            determinant = abs(round(np.linalg.det(candidate["matrix"])))
            if determinant > 1:
                try:
                    folded_frac = _fold_mean_to_candidate_unit(
                        candidate, mean_frac, cell
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    folded_frac = None
                    record["translation_folded_primitive_rejection"] = str(exc)
                if folded_frac is not None:
                    for folded_symprec in _symprec_grid(symprec_max):
                        folded_dataset = spglib.get_symmetry_dataset(
                            (unit_lattice, folded_frac, unit_numbers),
                            symprec=folded_symprec,
                        )
                        if folded_dataset is None or len(folded_dataset.rotations) < 2:
                            continue
                        try:
                            folded_projected, folded_projection_distances = (
                                _symmetrize_mean_positions(
                                    folded_frac, unit_numbers, unit_lattice,
                                    folded_dataset.rotations,
                                    folded_dataset.translations,
                                )
                            )
                            folded_strict = spglib.get_symmetry_dataset(
                                (unit_lattice, folded_projected, unit_numbers),
                                symprec=1e-5,
                            )
                            if folded_strict is None:
                                continue
                            folded_candidate = _materialize_candidate(
                                unit_lattice=unit_lattice,
                                unit_frac=folded_projected,
                                unit_numbers=unit_numbers,
                                dataset=folded_strict,
                                symprec=folded_symprec,
                                cell=cell,
                                mean_frac=mean_frac,
                                symbols=symbols,
                                masses_by_number=masses_by_number,
                                map_tolerance=map_tolerance,
                                supercell_projection_distances=(
                                    supercell_projection_distances
                                ),
                                primitive_projection_distances=(
                                    folded_projection_distances
                                ),
                            )
                        except (
                            ValueError, np.linalg.LinAlgError, spglib.SpglibError
                        ):
                            continue
                        if folded_candidate["signature"] == candidate["signature"]:
                            continue
                        folded_candidate["folded_from"] = int(
                            projected_dataset.number
                        )
                        folded_candidate["scan_symprec"] = float(symprec)
                        candidates.append(folded_candidate)
                        promoted_candidates.append(folded_candidate)
                        folded_records.append({
                            "symprec_A": float(folded_symprec),
                            "international": str(folded_strict.international),
                            "number": int(folded_strict.number),
                            "operations": int(len(folded_strict.rotations)),
                            "mapping_max_distance_A": float(
                                np.max(folded_candidate["distances"])
                            ),
                        })
            if folded_records:
                record["translation_folded_primitive_candidates"] = folded_records
            if (
                preprojection_candidate is not None
                and preprojection_candidate["signature"] != candidate["signature"]
            ):
                preprojection_candidate["preprojection_of"] = int(
                    projected_dataset.number
                )
                preprojection_candidate["scan_symprec"] = float(symprec)
                candidates.append(preprojection_candidate)
                promoted_candidates.append(preprojection_candidate)
                record["strict_before_primitive_reprojection"] = {
                    "international": str(
                        preprojection_candidate["dataset"].international
                    ),
                    "number": int(preprojection_candidate["dataset"].number),
                    "operations": int(
                        len(preprojection_candidate["dataset"].rotations)
                    ),
                    "mapping_max_distance_A": float(
                        np.max(preprojection_candidate["distances"])
                    ),
                }
            subgroup_records = []
            if max_distance > 0.75 * distortion_budget:
                operation_subgroups = _affine_operation_subgroups(
                    dataset.rotations, dataset.translations
                ) or _rotation_operation_subgroups(dataset.rotations)
                for operation_indices in operation_subgroups:
                    operation_indices = np.asarray(operation_indices, dtype=int)
                    try:
                        subgroup_frac, subgroup_projection_distances = (
                            _symmetrize_mean_positions(
                                mean_frac, numbers, cell,
                                dataset.rotations[operation_indices],
                                dataset.translations[operation_indices],
                            )
                        )
                        subgroup_standard = spglib.standardize_cell(
                            (cell, subgroup_frac, numbers), to_primitive=True,
                            no_idealize=True, symprec=1e-5,
                        )
                        if subgroup_standard is None:
                            continue
                        subgroup_lattice, subgroup_unit_frac, subgroup_numbers = (
                            subgroup_standard
                        )
                        subgroup_dataset = spglib.get_symmetry_dataset(
                            (subgroup_lattice, subgroup_unit_frac, subgroup_numbers),
                            symprec=1e-5,
                        )
                        if subgroup_dataset is None:
                            continue
                        subgroup_unit_frac, subgroup_primitive_distances = (
                            _symmetrize_mean_positions(
                                subgroup_unit_frac, subgroup_numbers,
                                subgroup_lattice, subgroup_dataset.rotations,
                                subgroup_dataset.translations,
                            )
                        )
                        subgroup_strict = spglib.get_symmetry_dataset(
                            (subgroup_lattice, subgroup_unit_frac, subgroup_numbers),
                            symprec=1e-5,
                        )
                        if subgroup_strict is None:
                            continue
                        subgroup_candidate = _materialize_candidate(
                            unit_lattice=subgroup_lattice,
                            unit_frac=subgroup_unit_frac,
                            unit_numbers=subgroup_numbers,
                            dataset=subgroup_strict, symprec=symprec,
                            cell=cell, mean_frac=mean_frac, symbols=symbols,
                            masses_by_number=masses_by_number,
                            map_tolerance=map_tolerance,
                            supercell_projection_distances=(
                                subgroup_projection_distances
                            ),
                            primitive_projection_distances=(
                                subgroup_primitive_distances
                            ),
                        )
                    except (ValueError, np.linalg.LinAlgError, spglib.SpglibError):
                        continue
                    if subgroup_candidate["signature"] == candidate["signature"]:
                        continue
                    subgroup_candidate["subgroup_of"] = int(projected_dataset.number)
                    subgroup_candidate["scan_symprec"] = float(symprec)
                    candidates.append(subgroup_candidate)
                    promoted_candidates.append(subgroup_candidate)
                    subgroup_records.append({
                        "international": str(subgroup_strict.international),
                        "number": int(subgroup_strict.number),
                        "operations": int(len(subgroup_strict.rotations)),
                        "generator_closure_operations": int(
                            len(operation_indices)
                        ),
                        "mapping_max_distance_A": float(
                            np.max(subgroup_candidate["distances"])
                        ),
                    })
            if subgroup_records:
                record["affine_space_group_subgroups"] = subgroup_records
            candidate_is_nontrivial = (
                abs(round(np.linalg.det(candidate["matrix"]))) > 1
                or len(projected_dataset.rotations) > 1
            )
            for promotion_symprec in (
                _symprec_grid(symprec_max) if candidate_is_nontrivial else []
            ):
                promoted_dataset = spglib.get_symmetry_dataset(
                    (unit_lattice, unit_frac, unit_numbers),
                    symprec=promotion_symprec,
                )
                if (
                    promoted_dataset is None
                    or int(promoted_dataset.number) == int(projected_dataset.number)
                ):
                    continue
                promoted_standard = spglib.standardize_cell(
                    (unit_lattice, unit_frac, unit_numbers), to_primitive=True,
                    no_idealize=True, symprec=promotion_symprec,
                )
                if promoted_standard is None or len(promoted_standard[1]) != len(unit_numbers):
                    continue
                promoted_frac, promoted_distances = _symmetrize_mean_positions(
                    unit_frac, unit_numbers, unit_lattice,
                    promoted_dataset.rotations, promoted_dataset.translations,
                )
                promoted_strict = spglib.get_symmetry_dataset(
                    (unit_lattice, promoted_frac, unit_numbers), symprec=1e-5
                )
                if (
                    promoted_strict is None
                    or int(promoted_strict.number) != int(promoted_dataset.number)
                ):
                    continue
                promoted = _materialize_candidate(
                    unit_lattice=unit_lattice, unit_frac=promoted_frac,
                    unit_numbers=unit_numbers, dataset=promoted_strict,
                    symprec=promotion_symprec, cell=cell, mean_frac=mean_frac,
                    symbols=symbols, masses_by_number=masses_by_number,
                    map_tolerance=map_tolerance,
                    supercell_projection_distances=supercell_projection_distances,
                    primitive_projection_distances=promoted_distances,
                )
                promoted["promoted_from"] = int(projected_dataset.number)
                promoted["scan_symprec"] = float(symprec)
                candidates.append(promoted)
                promoted_candidates.append(promoted)
                promotions.append({
                    "symprec_A": float(promotion_symprec),
                    "international": str(promoted_strict.international),
                    "number": int(promoted_strict.number),
                    "mapping_max_distance_A": float(
                        np.max(promoted["distances"])
                    ),
                })
            if promotions:
                record["point_symmetry_promotions"] = promotions
            detection_cache[detection_key] = (
                deepcopy({key: value for key, value in record.items() if key != "symprec_A"}),
                candidate,
                promoted_candidates,
            )
        except (ValueError, np.linalg.LinAlgError, spglib.SpglibError) as exc:
            record.update({"valid": False, "rejection": str(exc)})
        scan.append(record)
    if not candidates:
        failure_manifest = {
            "schema": "symfc-vasp-symmetry-report-v3",
            "source": "trajectory-periodic-mean",
            "trajectory": str(Path(outcar).resolve()),
            "natom": len(symbols),
            "primitive_search_success": False,
            "symmetry_scan": scan,
            "best_mapping_residual_A": (
                float(best_mapping_residual)
                if np.isfinite(best_mapping_residual) else None
            ),
        }
        (output / "symmetry_report.yaml").write_text(
            yaml.safe_dump(failure_manifest, sort_keys=False)
        )
        raise ValueError(
            "no symmetry candidate passed integer-cell and atom-map validation"
            + (
                f"; best atom-map residual={best_mapping_residual:.4f} A"
                if np.isfinite(best_mapping_residual) else ""
            )
        )
    tolerance_occurrences: dict[tuple, set[float]] = {}
    direct_tolerance_occurrences: dict[tuple, set[float]] = {}
    direct_projection_by_signature: dict[tuple, float] = {}
    for candidate in candidates:
        scan_symprec = round(
            float(candidate.get("scan_symprec", candidate["symprec"])), 12
        )
        tolerance_occurrences.setdefault(candidate["signature"], set()).add(
            scan_symprec
        )
        if candidate.get("direct_scan_candidate", False):
            direct_tolerance_occurrences.setdefault(
                candidate["signature"], set()
            ).add(scan_symprec)
            direct_projection_by_signature[candidate["signature"]] = min(
                direct_projection_by_signature.get(candidate["signature"], np.inf),
                float(np.max(candidate["projection_distances"])),
            )
    counts = {
        signature: len(tolerances)
        for signature, tolerances in tolerance_occurrences.items()
    }
    statistically_plausible = [
        candidate
        for candidate in candidates
        if float(np.max(candidate["distances"])) <= distortion_budget
        and float(np.max(candidate["projection_distances"]))
        <= distortion_budget
    ]
    if not statistically_plausible:
        statistically_plausible = candidates
    stable = [
        candidate
        for candidate in statistically_plausible
        if counts[candidate["signature"]] >= 2
    ]
    generated_baseline_residual_limit = max(0.055, 2.5 * mean_sem_95)
    # Translation recovery compares the maximum over every supercell site
    # with an atom-wise 95th-percentile SEM. It therefore needs a modest
    # multiple-site allowance. Keep this separate from the stricter point and
    # inversion enrichment gate, where the same allowance can create a false
    # centrosymmetric supergroup.
    translation_baseline_residual_limit = max(0.060, 2.75 * mean_sem_95)
    translation_baseline_rms_limit = max(0.030, 1.35 * mean_sem_95)

    def translation_residual_supported(candidate: dict) -> bool:
        maximum = float(np.max(candidate["distances"]))
        rms = float(np.sqrt(np.mean(candidate["distances"] ** 2)))
        return maximum <= translation_baseline_residual_limit or (
            maximum <= distortion_budget
            and rms <= translation_baseline_rms_limit
        )

    single_scan_translation_candidates = [
        candidate
        for candidate in statistically_plausible
        if candidate.get("direct_scan_candidate", False)
        and abs(round(np.linalg.det(candidate["matrix"]))) > 1
        and translation_residual_supported(candidate)
        and float(np.max(candidate["projection_distances"]))
        <= distortion_budget
    ]
    single_scan_point_candidates = [
        candidate
        for candidate in statistically_plausible
        if candidate.get("direct_scan_candidate", False)
        and len(candidate["dataset"].rotations) > 1
        and len(direct_tolerance_occurrences.get(candidate["signature"], ())) == 1
        and float(np.max(candidate["distances"]))
        <= max(0.05, 2.0 * mean_sem_95)
        and float(np.max(candidate["projection_distances"]))
        <= distortion_budget
    ]
    single_scan_point_signatures = {
        candidate["signature"] for candidate in single_scan_point_candidates
    }
    noise_supported_translation_candidates = [
        candidate
        for candidate in stable
        if candidate.get("direct_scan_candidate", False)
        and abs(round(np.linalg.det(candidate["matrix"]))) > 1
        and translation_residual_supported(candidate)
        and float(np.max(candidate["projection_distances"]))
        <= distortion_budget
    ]
    stable_or_all = list(stable)
    stable_signatures = {candidate["signature"] for candidate in stable_or_all}
    stable_or_all.extend(
        candidate
        for candidate in single_scan_translation_candidates
        if candidate["signature"] not in stable_signatures
    )
    stable_signatures = {candidate["signature"] for candidate in stable_or_all}
    stable_or_all.extend(
        candidate
        for candidate in single_scan_point_candidates
        if candidate["signature"] not in stable_signatures
    )
    if not stable_or_all:
        stable_or_all = statistically_plausible
    high_confidence_baseline_pool = [
        candidate
        for candidate in stable_or_all
        if int(candidate["dataset"].number) == 1
        or float(np.max(candidate["distances"]))
        <= generated_baseline_residual_limit
        or any(
            candidate is supported
            for supported in noise_supported_translation_candidates
        )
    ]
    if high_confidence_baseline_pool:
        stable_or_all = high_confidence_baseline_pool
    determinant_groups: dict[int, list[dict]] = {}
    for candidate in stable_or_all:
        determinant = abs(round(np.linalg.det(candidate["matrix"])))
        determinant_groups.setdefault(determinant, []).append(candidate)
    eligible_determinants = [
        determinant for determinant, group in determinant_groups.items()
        if determinant > 1
        or any(len(candidate["dataset"].rotations) > 1 for candidate in group)
    ]
    translation_determinants = [
        determinant
        for determinant in eligible_determinants
        if determinant > 1
        and any(
            candidate.get("direct_scan_candidate", False)
            for candidate in determinant_groups[determinant]
        )
    ]
    if translation_determinants:
        eligible_determinants = translation_determinants
    if eligible_determinants:
        selected_determinant = min(
            eligible_determinants,
            key=lambda determinant: min(
                candidate["symprec"] for candidate in determinant_groups[determinant]
            ),
        )
        pool = determinant_groups[selected_determinant]
    else:
        selected_determinant = min(determinant_groups)
        pool = determinant_groups[selected_determinant]
    nontrivial_pool = [
        candidate
        for candidate in pool
        if len(candidate["dataset"].rotations) > 1
    ] or pool
    baseline_operations = min(
        len(candidate["dataset"].rotations) for candidate in nontrivial_pool
    )
    baseline_pool = [
        candidate
        for candidate in nontrivial_pool
        if len(candidate["dataset"].rotations) == baseline_operations
    ]
    selected = max(
        baseline_pool,
        key=lambda candidate: (
            counts[candidate["signature"]],
            -float(np.sqrt(np.mean(candidate["distances"] ** 2))),
            -candidate["symprec"],
        ),
    )
    enrichment_baseline = selected
    stable_all = [
        candidate
        for candidate in candidates
        if counts[candidate["signature"]] >= 2
        and abs(round(np.linalg.det(candidate["matrix"])))
        == selected_determinant
    ]
    best_by_signature: dict[tuple, dict] = {}
    for candidate in stable_all:
        signature = candidate["signature"]
        incumbent = best_by_signature.get(signature)
        candidate_projection = float(
            np.max(candidate["projection_distances"])
        )
        candidate_mapping = float(np.max(candidate["distances"]))
        candidate_quality = (
            max(candidate_projection, candidate_mapping),
            candidate_projection,
            candidate_mapping,
            float(candidate["symprec"]),
        )
        if incumbent is not None:
            incumbent_projection = float(
                np.max(incumbent["projection_distances"])
            )
            incumbent_mapping = float(np.max(incumbent["distances"]))
            incumbent_quality = (
                max(incumbent_projection, incumbent_mapping),
                incumbent_projection,
                incumbent_mapping,
                float(incumbent["symprec"]),
            )
        else:
            incumbent_quality = None
        if incumbent_quality is None or candidate_quality < incumbent_quality:
            best_by_signature[signature] = candidate
    enrichment_trace: list[dict] = []
    for candidate in sorted(
        best_by_signature.values(),
        key=lambda item: (
            len(item["dataset"].rotations),
            float(np.max(item["distances"])),
        ),
    ):
        current_operations = len(selected["dataset"].rotations)
        candidate_operations = len(candidate["dataset"].rotations)
        if candidate_operations <= current_operations:
            continue
        selected_parent = (
            selected.get("subgroup_of")
            or selected.get("preprojection_of")
            or selected.get("folded_from")
            or selected.get("promoted_from")
        )
        direct_scan_occurrences = len(
            direct_tolerance_occurrences.get(candidate["signature"], ())
        )
        direct_scan_residual_limit = max(0.05, 2.0 * mean_sem_95)
        direct_scan_supported = (
            direct_scan_occurrences >= 2
            and float(np.max(candidate["distances"]))
            <= direct_scan_residual_limit
        )
        low_residual_subgroup_supported = (
            float(np.max(candidate["distances"])) <= direct_scan_residual_limit
        )
        operation_supergroup_supported = (
            current_operations >= 4
            and _operation_set_is_subset(selected, candidate)
        )
        baseline_operations_count = len(
            enrichment_baseline["dataset"].rotations
        )
        baseline_operation_supergroup_supported = (
            baseline_operations_count >= 2
            and candidate_operations >= 4 * baseline_operations_count
            and float(np.max(candidate["distances"]))
            <= direct_scan_residual_limit
            and _operation_set_is_subset(enrichment_baseline, candidate)
        )
        selected_common_operations = (
            len(selected["common_rotations"])
            if selected.get("common_rotations") is not None else 0
        )
        candidate_common_operations = (
            len(candidate["common_rotations"])
            if candidate.get("common_rotations") is not None else 0
        )
        translation_lattice_preserved = (
            _primitive_translation_count(candidate)
            == _primitive_translation_count(selected)
            and (
                candidate_common_operations > selected_common_operations
                or (
                    direct_scan_occurrences >= 2
                    and translation_residual_supported(candidate)
                )
            )
        )
        parent_supported = (
            int(candidate["dataset"].number) in {
                selected.get("subgroup_of"),
                selected.get("preprojection_of"),
                selected.get("folded_from"),
            }
            or candidate.get("promoted_from")
            == int(selected["dataset"].number)
            or (
                candidate.get("promoted_from") is not None
                and candidate.get("promoted_from")
                == selected.get("promoted_from")
            )
            or operation_supergroup_supported
            or baseline_operation_supergroup_supported
        )
        single_scan_point_supported = (
            candidate["signature"] in single_scan_point_signatures
        )
        current_proper_rotation_orders = _proper_rotation_orders(
            selected["dataset"].rotations
        )
        candidate_proper_rotation_orders = _proper_rotation_orders(
            candidate["dataset"].rotations
        )
        direct_point_supergroup_supported = (
            direct_scan_occurrences >= 2
            and direct_projection_by_signature.get(
                candidate["signature"], np.inf
            )
            <= max(0.060, 2.75 * mean_sem_95)
            and _rotation_set_is_subset(selected, candidate)
        )
        direct_rotation_family_supergroup_supported = (
            direct_scan_occurrences >= 2
            and float(np.max(candidate["distances"])) <= distortion_budget
            and float(np.max(candidate["projection_distances"]))
            <= distortion_budget
            and not candidate_proper_rotation_orders.issubset(
                current_proper_rotation_orders
            )
        )
        if (
            selected_parent is not None
            and int(candidate["dataset"].number) != int(selected_parent)
            and not direct_scan_supported
            and not low_residual_subgroup_supported
            and not operation_supergroup_supported
            and not baseline_operation_supergroup_supported
            and not direct_point_supergroup_supported
            and not direct_rotation_family_supergroup_supported
        ):
            enrichment_trace.append({
                "from_spacegroup": int(selected["dataset"].number),
                "to_spacegroup": int(candidate["dataset"].number),
                "from_operations": int(current_operations),
                "to_operations": int(candidate_operations),
                "from_subgroup_of": selected.get("subgroup_of"),
                "from_preprojection_of": selected.get("preprojection_of"),
                "from_folded_from": selected.get("folded_from"),
                "from_promoted_from": selected.get("promoted_from"),
                "to_direct_scan_occurrences": direct_scan_occurrences,
                "direct_scan_residual_limit_A": direct_scan_residual_limit,
                "accepted": False,
                "reason": "sibling subgroup; compare generated subgroup only with its direct parent",
            })
            continue
        current_residual = float(np.max(selected["distances"]))
        candidate_residual = float(np.max(candidate["distances"]))
        current_rms_residual = float(
            np.sqrt(np.mean(selected["distances"] ** 2))
        )
        candidate_rms_residual = float(
            np.sqrt(np.mean(candidate["distances"] ** 2))
        )
        added_operations = candidate_operations - current_operations
        inversion = -np.eye(3, dtype=int)
        current_has_inversion = any(
            np.array_equal(rotation, inversion)
            for rotation in selected["dataset"].rotations
        )
        candidate_has_inversion = any(
            np.array_equal(rotation, inversion)
            for rotation in candidate["dataset"].rotations
        )
        inversion_projection_limit = max(0.060, 2.75 * mean_sem_95)
        direct_projection = direct_projection_by_signature.get(
            candidate["signature"], np.inf
        )
        direct_inversion_supported = (
            direct_scan_occurrences >= 2
            and direct_projection <= inversion_projection_limit
        )
        current_max_rotation_order = max(
            _rotation_order(rotation)
            for rotation in selected["dataset"].rotations
        )
        candidate_max_rotation_order = max(
            _rotation_order(rotation)
            for rotation in candidate["dataset"].rotations
        )
        rotation_enhanced_inversion_supported = (
            direct_scan_occurrences >= 2
            and bool(
                candidate_proper_rotation_orders
                - current_proper_rotation_orders
            )
            and (
                current_operations == 1
                or max(
                    candidate_proper_rotation_orders
                    - current_proper_rotation_orders
                ) >= 3
            )
        )
        if (
            candidate_residual <= distortion_budget
            and candidate_max_rotation_order >= 3
            and current_max_rotation_order < candidate_max_rotation_order
        ):
            minimum_allowance = 0.05
        elif candidate_has_inversion and not current_has_inversion:
            parent_spacegroup = int(candidate["dataset"].number)
            minimum_allowance = (
                0.025
                if parent_spacegroup in {
                    selected.get("subgroup_of"),
                    selected.get("preprojection_of"),
                    selected.get("folded_from"),
                }
                else 0.02
            )
        else:
            minimum_allowance = 0.03
        operation_cost = (
            0.00675 if candidate_residual <= distortion_budget else 0.0055
        )
        incremental_allowance = max(
            minimum_allowance, operation_cost * added_operations
        )
        incremental_rms_residual = (
            candidate_rms_residual - current_rms_residual
        )
        rms_allowance = max(0.010, 0.46 * mean_sem_95)
        accepted = (
            candidate_residual <= distortion_budget
            and float(np.max(candidate["projection_distances"]))
            <= distortion_budget
            and translation_lattice_preserved
            and (
                direct_scan_occurrences >= 2
                or parent_supported
                or single_scan_point_supported
            )
            and (
                current_operations > 1
                or candidate_residual <= generated_baseline_residual_limit
                or direct_inversion_supported
                or rotation_enhanced_inversion_supported
            )
            and (
                not (candidate_has_inversion and not current_has_inversion)
                or candidate_residual <= generated_baseline_residual_limit
                or (
                    counts[selected["signature"]] == 1
                    and direct_scan_occurrences >= 2
                )
                or direct_inversion_supported
                or rotation_enhanced_inversion_supported
            )
            and (
                candidate_residual - current_residual <= incremental_allowance
                or (
                    direct_scan_occurrences >= 2
                    and incremental_rms_residual <= rms_allowance
                )
                or (
                    candidate_has_inversion
                    and not current_has_inversion
                    and (
                        direct_inversion_supported
                        or rotation_enhanced_inversion_supported
                    )
                )
            )
        )
        enrichment_trace.append({
            "from_spacegroup": int(selected["dataset"].number),
            "to_spacegroup": int(candidate["dataset"].number),
            "from_operations": int(current_operations),
            "to_operations": int(candidate_operations),
            "from_residual_A": current_residual,
            "to_residual_A": candidate_residual,
            "incremental_residual_A": candidate_residual - current_residual,
            "allowance_A": incremental_allowance,
            "incremental_rms_residual_A": incremental_rms_residual,
            "rms_allowance_A": rms_allowance,
            "inversion_projection_limit_A": inversion_projection_limit,
            "direct_scan_projection_A": (
                float(direct_projection) if np.isfinite(direct_projection) else None
            ),
            "direct_inversion_supported": bool(direct_inversion_supported),
            "rotation_enhanced_inversion_supported": bool(
                rotation_enhanced_inversion_supported
            ),
            "direct_point_supergroup_supported": bool(
                direct_point_supergroup_supported
            ),
            "direct_rotation_family_supergroup_supported": bool(
                direct_rotation_family_supergroup_supported
            ),
            "from_proper_rotation_orders": sorted(
                current_proper_rotation_orders
            ),
            "to_proper_rotation_orders": sorted(
                candidate_proper_rotation_orders
            ),
            "to_direct_scan_occurrences": direct_scan_occurrences,
            "parent_supported": bool(parent_supported),
            "operation_supergroup_supported": bool(
                operation_supergroup_supported
            ),
            "baseline_operation_supergroup_supported": bool(
                baseline_operation_supergroup_supported
            ),
            "from_common_operations": (
                len(selected["common_rotations"])
                if selected.get("common_rotations") is not None else None
            ),
            "to_common_operations": (
                len(candidate["common_rotations"])
                if candidate.get("common_rotations") is not None else None
            ),
            "from_common_translations": _common_translation_count(selected),
            "to_common_translations": _common_translation_count(candidate),
            "from_primitive_translations": _primitive_translation_count(
                selected
            ),
            "to_primitive_translations": _primitive_translation_count(
                candidate
            ),
            "translation_lattice_preserved": bool(
                translation_lattice_preserved
            ),
            "single_scan_point_supported": bool(single_scan_point_supported),
            "from_subgroup_of": selected.get("subgroup_of"),
            "from_preprojection_of": selected.get("preprojection_of"),
            "from_folded_from": selected.get("folded_from"),
            "from_promoted_from": selected.get("promoted_from"),
            "accepted": bool(accepted),
        })
        if accepted:
            selected = candidate
    chosen = selected["dataset"]
    chosen_symprec = selected["symprec"]
    unit = selected["unit"]
    generated = selected["generated"]
    matrix = selected["matrix"]
    matrix_residual = selected["matrix_residual"]
    mapping = selected["mapping"]
    distances = selected["distances"]
    origin_shift = selected["origin_shift"]
    origin_residual = selected["origin_residual"]
    projection_distances = selected["projection_distances"]
    primitive_projection_distances = selected["primitive_projection_distances"]

    mean_atoms = PhonopyAtoms(
        cell=cell,
        scaled_positions=mean_frac,
        symbols=symbols,
        masses=masses,
    )
    _write_poscar(output / "POSCAR-mean", mean_atoms, "Periodic mean structure recovered from trajectory")
    _write_poscar(output / "POSCAR-unitcell", unit, "spglib-symmetrized primitive reference from trajectory")
    _write_poscar(output / "POSCAR-supercell", generated, "spglib-symmetrized supercell reference from trajectory")
    _write_poscar(output / "SPOSCAR", generated, "Phonopy supercell reference from trajectory")
    np.savetxt(output / "supercell_matrix.dat", matrix, fmt="%d")
    (output / "generated_to_outcar_index.json").write_text(
        json.dumps({"generated_supercell_index_to_outcar_index": mapping.tolist()}, indent=2) + "\n"
    )
    manifest = {
        "schema": "symfc-vasp-symmetry-report-v3",
        "source": "trajectory-periodic-mean",
        "trajectory": str(Path(outcar).resolve()),
        "natom": len(symbols),
        "primitive_search_success": True,
        "lattice_records": lattice_records,
        "symmetry_scan": scan,
        "selection_policy": {
            "requirements": [
                "integer primitive-to-supercell relation",
                "species-preserving one-to-one atom map",
                "mapping residual within tolerance",
            ],
            "stable_plateau_minimum": 2,
            "stable_candidate_used": bool(stable),
            "single_scan_translation_candidate_used": bool(
                any(
                    selected is candidate
                    for candidate in single_scan_translation_candidates
                )
            ),
            "noise_supported_translation_candidate_used": bool(
                any(
                    selected is candidate
                    for candidate in noise_supported_translation_candidates
                )
            ),
            "single_scan_point_candidate_used": bool(
                any(
                    selected is candidate
                    for candidate in single_scan_point_candidates
                )
            ),
            "generated_baseline_residual_limit_A": (
                generated_baseline_residual_limit
            ),
            "translation_baseline_residual_limit_A": (
                translation_baseline_residual_limit
            ),
            "translation_baseline_rms_limit_A": (
                translation_baseline_rms_limit
            ),
            "trajectory_mean_sem_95_A": mean_sem_95,
            "maximum_symmetry_projection_A": distortion_budget,
            "incremental_supergroup_allowance": (
                "delta residual <= max(minimum, cost * delta operations); "
                "minimum=0.05 A when adding a stable order-3-or-higher rotation "
                "within the mean-noise budget, 0.025 A when rejoining an affine "
                "subgroup to its inversion-containing parent, 0.02 A for other "
                "inversion additions, and 0.03 A otherwise; stable direct-scan "
                "supergroups within both mapping and projection noise budgets "
                "may also pass when RMS residual growth <= max(0.010 A, "
                "0.46 * mean SEM95); newly added inversion also requires the "
                "generated-baseline residual limit unless its lower-symmetry "
                "baseline occurs at only one scan tolerance, or the inversion "
                "is directly repeated with projection <= max(0.060 A, "
                "2.75 * mean SEM95), or it directly and repeatedly restores a "
                "higher rotation order; the global maximum projection budget "
                "is max(0.075 A, 3.7 * mean SEM95); cost=0.00675 A "
                "within the mean-noise budget and 0.0055 A outside"
            ),
            "selected_supercell_determinant": int(selected_determinant),
            "selected_signature_occurrences": int(counts[selected["signature"]]),
            "supergroup_enrichment_trace": enrichment_trace,
            "ranking": (
                "finite-sampling-compatible distortion, first stable nontrivial "
                "translation lattice, minimum stable nontrivial point subgroup, "
                "then operation-normalized incremental-distortion enrichment"
            ),
        },
        "selected_symprec_A": float(chosen_symprec),
        "selected_spacegroup": {"international": str(chosen.international), "number": int(chosen.number), "operations": int(len(chosen.rotations))},
        "unitcell_atoms": int(len(unit)),
        "supercell_matrix": matrix.tolist(),
        "supercell_matrix_residual": matrix_residual,
        "mapping": {
            "max_distance_A": float(np.max(distances)),
            "rms_distance_A": float(np.sqrt(np.mean(distances ** 2))),
            "tolerance_A": float(map_tolerance),
            "origin_shift_fractional": origin_shift.tolist(),
            "origin_alignment_residual_A": float(origin_residual),
        },
        "symmetry_projection": {
            "supercell_mean_max_distance_A": float(np.max(projection_distances)),
            "supercell_mean_rms_distance_A": float(
                np.sqrt(np.mean(projection_distances ** 2))
            ),
            "primitive_max_distance_A": float(
                np.max(primitive_projection_distances)
            ),
            "primitive_rms_distance_A": float(
                np.sqrt(np.mean(primitive_projection_distances ** 2))
            ),
            "validation_symprec_A": 1e-5,
        },
    }
    (output / "symmetry_report.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return unit, generated, mapping, manifest
