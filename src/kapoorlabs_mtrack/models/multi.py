"""Joint N-microtubule model for fitting in collision / crossing regions.

When a single segmentation label contains more than one microtubule
(detected upstream by a skeleton-endpoint count of 4, 6, ...), the
fitter needs a forward model that sums the contributions of every
microtubule in the crop. We share a single background scalar across
all microtubules in the crop (one camera bias / autofluorescence floor
per image region) but keep every other parameter per-MT.

Parameter layout for ``N`` microtubules::

    a_concat[0  ..  7]   MT 1: start_x, start_y, end_x, end_y,
                                ds, curvature, inflection, amplitude
    a_concat[8  .. 15]   MT 2: same fields
    ...
    a_concat[-1]         shared background

So ``len(a_concat) == 8 * N + 1``.

This module reuses the single-MT ``val`` / ``jac`` from
``spline_third_order`` — each per-MT block is the same Java-faithful
analytic model, just summed. No new math.
"""

from __future__ import annotations

import numpy as np

from . import spline_third_order as _stm

# Params per microtubule in the joint vector (no background -- shared).
PER_MT = 8


def n_params(n_mt: int) -> int:
    """Length of the joint parameter vector for ``n_mt`` microtubules."""
    return PER_MT * n_mt + 1


def split(a_concat: np.ndarray, n_mt: int) -> tuple[list[np.ndarray], float]:
    """Split joint vector into per-MT 8-vectors + shared background scalar."""
    per_mt = [
        np.asarray(a_concat[i * PER_MT : (i + 1) * PER_MT], dtype=float)
        for i in range(n_mt)
    ]
    bg = float(a_concat[-1])
    return per_mt, bg


def _to_single_a(a_mt8: np.ndarray) -> np.ndarray:
    """Embed an 8-vector MT block into the 9-vector single-MT layout with bg=0.

    The single-MT ``val`` returns ``amp * Etotal + bg`` -- by passing
    bg=0 we get ``amp * Etotal`` which is the MT's contribution. The
    shared joint background is added once at the joint level.
    """
    return np.concatenate([a_mt8, [0.0]])


def val(
    x: np.ndarray, a_concat: np.ndarray, b: np.ndarray, n_mt: int
) -> np.ndarray:
    """Joint model intensity at coordinate(s) ``x``.

    Returns array of shape ``x.shape[:-1]``.
    """
    a_concat = np.asarray(a_concat, dtype=float)
    per_mt, bg = split(a_concat, n_mt)
    out = np.full(x.shape[:-1], bg, dtype=float)
    for a_i in per_mt:
        out = out + _stm.val(x, _to_single_a(a_i), b)
    return out


def jac(
    x: np.ndarray, a_concat: np.ndarray, b: np.ndarray, n_mt: int
) -> np.ndarray:
    """Analytic Jacobian of :func:`val` w.r.t. ``a_concat``.

    Block-diagonal in the per-MT columns (each MT's parameters only
    affect its own contribution), with a constant ``1`` final column
    for the shared background. Returns shape
    ``x.shape[:-1] + (8*N + 1,)``.
    """
    a_concat = np.asarray(a_concat, dtype=float)
    per_mt, _bg = split(a_concat, n_mt)
    blocks = []
    for a_i in per_mt:
        j_i = _stm.jac(x, _to_single_a(a_i), b)  # (..., 9)
        # Drop the single-MT bg column (last) -- bg is shared at joint level.
        blocks.append(j_i[..., :PER_MT])
    # Shared bg column: dval/dbg = 1 everywhere.
    bg_col = np.ones(x.shape[:-1] + (1,))
    blocks.append(bg_col)
    return np.concatenate(blocks, axis=-1)


def pack(per_mt_a8: list[np.ndarray], bg: float) -> np.ndarray:
    """Build a joint vector from per-MT 8-vectors + shared background."""
    arr = [np.asarray(a, dtype=float) for a in per_mt_a8]
    for a in arr:
        if a.shape != (PER_MT,):
            raise ValueError(
                f"each MT vector must be length {PER_MT}, got {a.shape}"
            )
    return np.concatenate(arr + [np.array([float(bg)])])
