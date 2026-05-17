"""Thin scipy LM wrapper over the analytic spline-Gaussian model.

The model exposes ``val`` and ``jac`` (analytic, hand-derived); this
module just packages them for :func:`scipy.optimize.least_squares` and
extracts the refined endpoint coordinates from the result.

We default to ``method='trf'`` (trust-region reflective) because it
accepts box bounds, which matter here -- ``amplitude`` and ``ds`` must
stay positive, the endpoints should stay near the seed, and curvature /
inflection should stay near zero to avoid the LM solver wandering off
into ill-conditioned regions. Set ``bounds=None`` to use plain
unconstrained LM (``method='lm'``) for a closer match to the Java
solver's behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from ..models import jac as model_jac
from ..models import val as model_val

# Indices 0..3 are start/end coordinates. The Java analytic gradient
# treats these approximately (only the endpoint-Gaussian contribution,
# not the swept-curve coupling), which leaves a 1-2 px bias in the
# fitted endpoints. Hybrid mode overrides these four columns with
# central differences while keeping the exact analytic columns for the
# shape parameters (ds, curvature, inflection, amp, background).
_ENDPOINT_INDICES = (0, 1, 2, 3)


@dataclass
class FitResult:
    """Output of :func:`fit_endpoints`."""

    a: np.ndarray  # refined 9-vector
    start: np.ndarray  # a[0:2]
    end: np.ndarray  # a[2:4]
    cost: float  # final 0.5 * sum(residuals^2)
    success: bool
    nfev: int
    message: str


def _build_residuals_and_jac(
    image: np.ndarray,
    b: np.ndarray,
    weights: Optional[np.ndarray] = None,
    jac_mode: str = "analytic",
    fd_eps: float = 1.0e-3,
):
    """Return (residuals_fn, jac_fn) closures over the pixel grid.

    ``jac_mode`` is one of ``"analytic"``, ``"hybrid"``, ``"numeric"``:

    - ``analytic``: Java-faithful; fastest; approximate endpoint columns.
    - ``hybrid``: analytic shape columns, central-difference endpoint
      columns. ~4 extra ``val`` evaluations per LM iteration; sub-pixel
      endpoint recovery.
    - ``numeric``: central-difference for every column. Slowest;
      reference behaviour.
    """
    h, w = image.shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([jj, ii], axis=-1).astype(float)  # (H, W, 2)
    flat_coords = coords.reshape(-1, 2)
    flat_image = image.reshape(-1).astype(float)
    if weights is None:
        wflat = np.ones_like(flat_image)
    else:
        wflat = weights.reshape(-1).astype(float)

    def residuals(a: np.ndarray) -> np.ndarray:
        pred = model_val(flat_coords, a, b)
        return wflat * (pred - flat_image)

    def _fd_col(a: np.ndarray, k: int) -> np.ndarray:
        ap = a.copy()
        am = a.copy()
        ap[k] += fd_eps
        am[k] -= fd_eps
        return (
            model_val(flat_coords, ap, b) - model_val(flat_coords, am, b)
        ) / (2.0 * fd_eps)

    if jac_mode == "analytic":

        def jacobian(a: np.ndarray) -> np.ndarray:
            j = model_jac(flat_coords, a, b)  # (Npix, Nparams)
            return wflat[:, None] * j

    elif jac_mode == "hybrid":

        def jacobian(a: np.ndarray) -> np.ndarray:
            j = model_jac(flat_coords, a, b)
            for k in _ENDPOINT_INDICES:
                j[:, k] = _fd_col(a, k)
            return wflat[:, None] * j

    elif jac_mode == "numeric":

        def jacobian(a: np.ndarray) -> np.ndarray:
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


def _default_bounds(
    seed_a: np.ndarray, image_shape: tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Sane box bounds keyed off the seed and image dimensions."""
    h, w = image_shape
    lo = np.array(
        [
            0.0,
            0.0,  # start in image
            0.0,
            0.0,  # end in image
            1e-3,  # ds > 0
            -1.0,
            -1.0,  # curvature, inflection -- mild
            1e-3,  # amplitude > 0
            -np.inf,  # background unconstrained
        ]
    )
    hi = np.array(
        [
            float(w),
            float(h),
            float(w),
            float(h),
            10.0,  # ds upper bound (pixels per step)
            1.0,
            1.0,
            np.inf,
            np.inf,
        ]
    )
    # Defensive: if the seed itself is outside our box, widen.
    lo = np.minimum(lo, seed_a - 1e-6)
    hi = np.maximum(hi, seed_a + 1e-6)
    return lo, hi


def fit_endpoints(
    image: np.ndarray,
    seed_a: np.ndarray,
    b: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = "auto",
    jac_mode: str = "analytic",
    max_nfev: int = 500,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
) -> FitResult:
    """Refine the spline-Gaussian parameters against ``image``.

    Args:
        image: 2-D array of observed intensities (H, W). Pixel ``(i, j)``
            is treated as coordinate ``(j, i)`` to match the renderer.
        seed_a: initial 9-vector (see ``models`` for layout). The
            endpoint seeds should already be within ~1 px of the true
            line (the Hough / RANSAC stage in the original pipeline
            provides this) -- the analytic-gradient LM has a limited
            basin of attraction past that.
        b: fixed 2-vector ``1 / sigma**2`` for the imaging PSF. This is
            a property of the microscope, **not** an optimisation
            target -- the solver never touches it.
        weights: optional per-pixel weights (e.g. ``1 / sqrt(image+1)``
            for approximate Poisson weighting). ``None`` = unit weights.
        bounds: ``"auto"`` builds box bounds from the seed + image, a
            ``(lo, hi)`` tuple is used directly, ``None`` runs
            unconstrained LM (``method='lm'``).
        jac_mode: ``"analytic"`` (default, fastest, Java-faithful with
            ~1-2 px endpoint bias), ``"hybrid"`` (analytic shape +
            numeric endpoint columns, sub-pixel endpoint recovery),
            or ``"numeric"`` (central differences for every column).
        max_nfev: max function evaluations.
        xtol, ftol: scipy LM convergence tolerances.
    """
    seed_a = np.asarray(seed_a, dtype=float)
    b = np.asarray(b, dtype=float)
    residuals, jacobian = _build_residuals_and_jac(
        image, b, weights, jac_mode=jac_mode
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
            bounds = _default_bounds(seed_a, image.shape)
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
    return FitResult(
        a=a,
        start=a[0:2].copy(),
        end=a[2:4].copy(),
        cost=float(result.cost),
        success=bool(result.success),
        nfev=int(result.nfev),
        message=result.message,
    )
