"""3rd-order Gaussian-spline microtubule model.

Port of ``LineModels/GaussianSplinethirdorder.java`` from the original
MTrack project. The microtubule is modelled as a 3rd-order polynomial
curve through fixed start and end points, with curvature and inflection
coefficients, sampled at step length ``ds`` along the tangent. At each
sample the imaging PSF is evaluated as an anisotropic 2-D Gaussian and
the contributions are summed -- the start and end points carry a single
extra Gaussian each (``Estart``, ``Eend``) so the curve has full
intensity at its endpoints.

The free parameter vector ``a`` and fixed vector ``b`` follow the same
layout as the Java implementation so the hand-derived analytic gradients
port directly. See ``models/__init__.py`` for the layout.

The endpoint partials (``a[0..2*ndims-1]``) are kept as the same fast
approximation used by the Java code -- they ignore the second-order
dependence of the swept curve on the endpoint and use only the gradient
of the endpoint Gaussian itself. The shape partials (``ds``,
``curvature``, ``inflection``) are exact and re-walk the curve to
accumulate per-step contributions.
"""

from __future__ import annotations

import numpy as np

FCTEPS = 1.0e-30


def _tangent_slope(
    x_now: float, slope_base: float, curvature: float, inflection: float
) -> float:
    """Polynomial derivative at the current curve x: m + 2 C x + 3 I x^2."""
    return (
        slope_base + 2.0 * curvature * x_now + 3.0 * inflection * x_now * x_now
    )


def _slope_base(
    start: np.ndarray, end: np.ndarray, curvature: float, inflection: float
) -> float:
    """Effective polynomial slope between start and end.

    Mirrors the Java expression for ``slope`` -- the linear coefficient
    of a cubic ``y(x) = y0 + slope x + C x^2 + I x^3`` constrained to
    pass through ``(x0, y0)`` and ``(x1, y1)``.
    """
    dx = end[0] - start[0] + FCTEPS
    return (
        (end[1] - start[1]) / dx
        - curvature * (end[0] + start[0])
        - inflection * (start[0] ** 2 + end[0] ** 2 + start[0] * end[0])
    )


def _walk_terminated(now: np.ndarray, end: np.ndarray, slope: float) -> bool:
    """Java termination guard for the forward walk along the curve.

    The original combined ``or``/``and`` precedence is preserved (Java
    evaluates ``a || b && c`` as ``a || (b && c)``).
    """
    if now[0] >= end[0] or (now[1] >= end[1] and slope >= 0):
        return True
    if now[0] >= end[0] or (now[1] <= end[1] and slope < 0):
        return True
    return False


def walk_curve(a: np.ndarray, max_steps: int = 100_000) -> np.ndarray:
    """Sample the 3rd-order spline curve from start to end.

    Returns the array of points visited by the Java ``Esum`` walk loop.
    The start and end points themselves are *not* included -- they are
    handled separately by ``Estart`` / ``Eend``.
    """
    ndims = 2
    start = a[0:ndims].astype(float).copy()
    end = a[ndims : 2 * ndims].astype(float)
    ds = abs(a[2 * ndims])
    curvature = a[2 * ndims + 1]
    inflection = a[2 * ndims + 2]

    slope = _slope_base(start, end, curvature, inflection)
    now = start.copy()
    pts: list[np.ndarray] = []
    for _ in range(max_steps):
        tan = _tangent_slope(now[0], slope, curvature, inflection)
        denom = np.sqrt(1.0 + tan * tan)
        dx = ds / denom
        dy = tan * dx
        now = now + np.array([dx, dy])
        pts.append(now.copy())
        slope = _slope_base(now, end, curvature, inflection)
        if _walk_terminated(now, end, slope):
            break
    if not pts:
        return np.empty((0, ndims))
    return np.asarray(pts)


def _gauss(x: np.ndarray, center: np.ndarray, b: np.ndarray) -> np.ndarray:
    """exp(-Σ b[d] (x[d] - center[d])^2). Broadcasts on the leading axes."""
    diff = x - center
    return np.exp(-np.sum(b * diff * diff, axis=-1))


def _e_start(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _gauss(x, a[0:2], b)


def _e_end(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _gauss(x, a[2:4], b)


def _e_sum(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sum of Gaussians along the swept curve, excluding the endpoints."""
    pts = walk_curve(a)
    if pts.shape[0] == 0:
        return np.zeros(x.shape[:-1])
    # x has shape (..., 2); pts has shape (N, 2). Broadcast to (..., N, 2).
    diff = x[..., None, :] - pts  # (..., N, 2)
    expo = np.sum(b * diff * diff, axis=-1)  # (..., N)
    return np.sum(np.exp(-expo), axis=-1)  # (...,)


def _e_total(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _e_start(x, a, b) + _e_sum(x, a, b) + _e_end(x, a, b)


def val(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Model intensity at coordinate ``x``.

    ``x`` may be a single coordinate ``(2,)`` or a batch ``(..., 2)``.
    Returns an array of shape ``x.shape[:-1]``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    amp = a[7]
    bg = a[8]
    return amp * _e_total(x, a, b) + bg


def _accumulate_shape_partial(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    dxvec_deriv_fn,
) -> np.ndarray:
    """Generic walked-curve partial for ds / curvature / inflection.

    ``dxvec_deriv_fn(now, end, ds, curvature, inflection, slope, tan, dxvec)``
    returns the 2-vector ``d(curve_position)/d(parameter)`` at the current
    step. The Java code accumulates ``count * dsum * exp(-sum)`` along the
    walk where ``dsum = Σ 2 b[d] (x[d] - now[d]) * dxvec_deriv[d]``.
    Returns array of shape ``x.shape[:-1]``.
    """
    ndims = 2
    start = a[0:ndims].astype(float).copy()
    end = a[ndims : 2 * ndims].astype(float)
    ds = abs(a[2 * ndims])
    curvature = a[2 * ndims + 1]
    inflection = a[2 * ndims + 2]

    slope = _slope_base(start, end, curvature, inflection)
    now = start.copy()

    out = np.zeros(x.shape[:-1])
    count = 1
    for _ in range(100_000):
        tan = _tangent_slope(now[0], slope, curvature, inflection)
        denom = np.sqrt(1.0 + tan * tan)
        dxvec = np.array([ds / denom, tan * ds / denom])
        dxvec_deriv = dxvec_deriv_fn(
            now, end, ds, curvature, inflection, slope, tan, dxvec
        )
        now = now + dxvec
        # gradient contribution at the current curve point
        diff = x - now  # (..., 2)
        sum_b_d2 = np.sum(b * diff * diff, axis=-1)
        dsum = 2.0 * np.sum(b * diff * dxvec_deriv, axis=-1)
        out = out + count * dsum * np.exp(-sum_b_d2)
        count += 1
        slope = _slope_base(now, end, curvature, inflection)
        if _walk_terminated(now, end, slope):
            break
    return out


def _eds(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Partial of Esum w.r.t. ``ds`` (curve step length)."""

    def deriv(now, end, ds, curvature, inflection, slope, tan, dxvec):
        return np.array(
            [1.0 / np.sqrt(1.0 + tan * tan), tan / np.sqrt(1.0 + tan * tan)]
        )

    return _accumulate_shape_partial(x, a, b, deriv)


def _edc(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Partial of Esum w.r.t. ``curvature`` (the 2nd-order coefficient)."""

    def deriv(now, end, ds, curvature, inflection, slope, tan, dxvec):
        d_slope_d_curv = -(end[0] + now[0]) + 2.0 * now[0]
        denom_pow = (1.0 + tan * tan) ** 1.5
        dx_db = -ds * tan * d_slope_d_curv / denom_pow
        return np.array([dx_db, tan * dx_db + d_slope_d_curv * dxvec[0]])

    return _accumulate_shape_partial(x, a, b, deriv)


def _ed_inflection(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Partial of Esum w.r.t. ``inflection`` (the 3rd-order coefficient)."""

    def deriv(now, end, ds, curvature, inflection, slope, tan, dxvec):
        d_slope_d_infl = (
            -(now[0] ** 2 + end[0] ** 2 + now[0] * end[0]) + 3.0 * now[0] ** 2
        )
        denom_pow = (1.0 + tan * tan) ** 1.5
        dx_dc = -ds * tan * d_slope_d_infl / denom_pow
        return np.array([dx_dc, tan * dx_dc + d_slope_d_infl * dxvec[0]])

    return _accumulate_shape_partial(x, a, b, deriv)


def jac(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Analytic Jacobian d(val)/d(a) at coordinate(s) ``x``.

    Returns shape ``x.shape[:-1] + (len(a),)``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    ndims = 2
    amp = a[7]

    out = np.zeros(x.shape[:-1] + (a.shape[0],))

    e_start = _e_start(x, a, b)
    e_end = _e_end(x, a, b)
    e_total = _e_total(x, a, b)

    # k in [0, ndims): partial w.r.t. start coordinate -- approximate
    # (matches Java: just the gradient of the Estart Gaussian).
    for k in range(ndims):
        out[..., k] = 2.0 * b[k] * (x[..., k] - a[k]) * amp * e_start

    # k in [ndims, 2*ndims): partial w.r.t. end coordinate -- approximate.
    for k in range(ndims, 2 * ndims):
        dim = k - ndims
        out[..., k] = 2.0 * b[dim] * (x[..., dim] - a[k]) * amp * e_end

    # Exact partials w.r.t. ds, curvature, inflection.
    out[..., 2 * ndims] = amp * _eds(x, a, b)
    out[..., 2 * ndims + 1] = amp * _edc(x, a, b)
    out[..., 2 * ndims + 2] = amp * _ed_inflection(x, a, b)

    # Amplitude and background.
    out[..., 2 * ndims + 3] = e_total
    out[..., 2 * ndims + 4] = 1.0

    return out
