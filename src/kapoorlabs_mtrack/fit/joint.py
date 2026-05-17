"""Joint LM fit of N microtubules sharing a crop and a background.

Thin wrapper analogous to :mod:`fit.lm` but for the multi-MT model.
Reuses the same Java-faithful analytic Jacobian (via
``models.multi.jac``) and the same scipy back-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from ..models import multi

# Per-MT endpoint indices in each 8-vector block (start_x, start_y, end_x, end_y).
_PER_MT_ENDPOINT_INDICES = (0, 1, 2, 3)


@dataclass
class JointFitResult:
    """Output of :func:`fit_endpoints_joint`."""

    a_concat: np.ndarray  # refined 8N+1 vector
    per_mt: list[np.ndarray]  # list of per-MT 8-vectors
    background: float
    starts: np.ndarray  # (N, 2)
    ends: np.ndarray  # (N, 2)
    cost: float
    success: bool
    nfev: int
    message: str


def _endpoint_indices_joint(n_mt: int) -> tuple[int, ...]:
    """Indices of all endpoint coordinates in the joint vector."""
    out: list[int] = []
    for i in range(n_mt):
        base = i * multi.PER_MT
        for k in _PER_MT_ENDPOINT_INDICES:
            out.append(base + k)
    return tuple(out)


def _default_bounds_joint(
    seed_a: np.ndarray, n_mt: int, image_shape: tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    lo = np.empty_like(seed_a)
    hi = np.empty_like(seed_a)
    for i in range(n_mt):
        base = i * multi.PER_MT
        lo[base : base + 4] = 0.0
        hi[base : base + 4] = [float(w), float(h), float(w), float(h)]
        lo[base + 4] = 1e-3  # ds > 0
        hi[base + 4] = 10.0
        lo[base + 5 : base + 7] = -1.0
        hi[base + 5 : base + 7] = 1.0
        lo[base + 7] = 1e-3  # amp > 0
        hi[base + 7] = np.inf
    lo[-1] = -np.inf  # shared bg unconstrained
    hi[-1] = np.inf
    # Defensive: widen if the seed itself is outside.
    lo = np.minimum(lo, seed_a - 1e-6)
    hi = np.maximum(hi, seed_a + 1e-6)
    return lo, hi


def _build_residuals_and_jac(
    image: np.ndarray,
    b: np.ndarray,
    n_mt: int,
    weights: Optional[np.ndarray] = None,
    jac_mode: str = "analytic",
    fd_eps: float = 1.0e-3,
):
    h, w = image.shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    flat_coords = np.stack([jj, ii], axis=-1).reshape(-1, 2).astype(float)
    flat_image = image.reshape(-1).astype(float)
    wflat = (
        np.ones_like(flat_image)
        if weights is None
        else weights.reshape(-1).astype(float)
    )

    def residuals(a: np.ndarray) -> np.ndarray:
        pred = multi.val(flat_coords, a, b, n_mt)
        return wflat * (pred - flat_image)

    endpoint_idx = _endpoint_indices_joint(n_mt)

    def _fd_col(a: np.ndarray, k: int) -> np.ndarray:
        ap = a.copy()
        am = a.copy()
        ap[k] += fd_eps
        am[k] -= fd_eps
        return (
            multi.val(flat_coords, ap, b, n_mt)
            - multi.val(flat_coords, am, b, n_mt)
        ) / (2.0 * fd_eps)

    if jac_mode == "analytic":

        def jacobian(a):
            return wflat[:, None] * multi.jac(flat_coords, a, b, n_mt)

    elif jac_mode == "hybrid":

        def jacobian(a):
            j = multi.jac(flat_coords, a, b, n_mt)
            for k in endpoint_idx:
                j[:, k] = _fd_col(a, k)
            return wflat[:, None] * j

    elif jac_mode == "numeric":

        def jacobian(a):
            n = a.shape[0]
            j = np.empty((flat_coords.shape[0], n))
            for k in range(n):
                j[:, k] = _fd_col(a, k)
            return wflat[:, None] * j

    else:
        raise ValueError(
            f"jac_mode must be 'analytic', 'hybrid', or 'numeric', got {jac_mode!r}"
        )

    return residuals, jacobian


def fit_endpoints_joint(
    image: np.ndarray,
    seed_a_concat: np.ndarray,
    b: np.ndarray,
    n_mt: int,
    weights: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = "auto",
    jac_mode: str = "analytic",
    max_nfev: int = 1000,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
) -> JointFitResult:
    """Joint LM fit of ``n_mt`` microtubules sharing one crop.

    See :mod:`models.multi` for the parameter layout
    (``8*n_mt + 1`` long, last entry is shared background).
    Pass per-MT endpoint seeds packed via ``models.multi.pack``.
    """
    seed_a = np.asarray(seed_a_concat, dtype=float)
    b = np.asarray(b, dtype=float)
    expected = multi.n_params(n_mt)
    if seed_a.shape != (expected,):
        raise ValueError(
            f"seed_a_concat must be length {expected} for n_mt={n_mt}, "
            f"got {seed_a.shape}"
        )

    residuals, jacobian = _build_residuals_and_jac(
        image, b, n_mt, weights=weights, jac_mode=jac_mode
    )

    if bounds is None:
        result = least_squares(
            residuals,
            seed_a,
            jac=jacobian,
            method="lm",
            max_nfev=max_nfev,
            xtol=xtol,
            ftol=ftol,
        )
    else:
        if bounds == "auto":
            bounds = _default_bounds_joint(seed_a, n_mt, image.shape)
        result = least_squares(
            residuals,
            seed_a,
            jac=jacobian,
            bounds=bounds,
            method="trf",
            max_nfev=max_nfev,
            xtol=xtol,
            ftol=ftol,
        )

    a = result.x
    per_mt, bg = multi.split(a, n_mt)
    starts = np.array([p[0:2] for p in per_mt])
    ends = np.array([p[2:4] for p in per_mt])
    return JointFitResult(
        a_concat=a,
        per_mt=per_mt,
        background=bg,
        starts=starts,
        ends=ends,
        cost=float(result.cost),
        success=bool(result.success),
        nfev=int(result.nfev),
        message=result.message,
    )
