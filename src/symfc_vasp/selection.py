from __future__ import annotations

import numpy as np


def select_indices(
    total: int,
    *,
    skip: int = 0,
    stop: int | None = None,
    samples: int | None = None,
    stride: int | None = None,
    method: str = "stride",
    seed: int = 0,
) -> np.ndarray:
    """Select source frame indices after equilibration.

    ``stride`` is the default and intentionally starts at ``skip``. When only
    ``samples`` is supplied, an exact integer stride is inferred if possible.
    """
    if stop is None:
        stop = total
    if not 0 < stop <= total:
        raise ValueError(f"stop must satisfy 0 < stop <= {total}")
    if not 0 <= skip < stop:
        raise ValueError(f"skip must satisfy 0 <= skip < {total}")
    available = stop - skip
    if samples is not None and not 0 < samples <= available:
        raise ValueError(f"samples must satisfy 0 < samples <= {available}")
    if method == "stride":
        if stride is None:
            if samples is None:
                stride = 1
            elif available % samples:
                raise ValueError(
                    f"{available} post-skip frames cannot produce exactly {samples} samples "
                    "with an integer stride; provide --stride or use --selection uniform"
                )
            else:
                stride = available // samples
        if stride <= 0:
            raise ValueError("stride must be positive")
        indices = np.arange(skip, stop, stride, dtype=int)
        if samples is not None:
            if len(indices) != samples:
                raise ValueError(
                    f"stride={stride} selects {len(indices)} frames, not requested {samples}"
                )
            indices = indices[:samples]
        return indices
    if samples is None:
        raise ValueError(f"selection method {method!r} requires --samples")
    if method == "uniform":
        return np.linspace(skip, stop - 1, samples, dtype=int)
    if method == "random":
        return np.sort(np.random.default_rng(seed).choice(np.arange(skip, stop), samples, replace=False))
    raise ValueError(f"unsupported selection method: {method}")
