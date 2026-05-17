"""Numerical-vs-analytic Jacobian comparison.

The MTrack model has hand-derived analytic gradients for speed. For the
shape parameters (ds, curvature, inflection, amplitude, background) the
gradients are exact and should agree with central-difference numerical
gradients to high precision. For the endpoint parameters
(start / end coordinates) the Java implementation deliberately keeps a
faster approximation -- so we expect agreement only on the order of the
PSF-induced coupling, not machine precision.

The :func:`check_jacobian` helper returns a per-parameter report that
makes that distinction obvious instead of throwing on any disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class JacobianReport:
    """Per-parameter comparison of analytic vs numeric Jacobian columns."""

    param_index: int
    max_abs_err: float
    max_rel_err: float
    analytic_norm: float
    numeric_norm: float

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"p[{self.param_index:2d}]  "
            f"|ana|={self.analytic_norm:11.4e}  "
            f"|num|={self.numeric_norm:11.4e}  "
            f"abs_err={self.max_abs_err:10.3e}  "
            f"rel_err={self.max_rel_err:10.3e}"
        )


def numeric_jacobian(
    val_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1.0e-5,
) -> np.ndarray:
    """Central-difference Jacobian of ``val_fn`` w.r.t. ``a``.

    Returns shape ``x.shape[:-1] + (len(a),)`` -- matching the analytic
    :func:`models.spline_third_order.jac` output shape.

    Step size ``eps`` is *not* scaled per-parameter; pass a custom scale
    by transforming ``a`` before calling if you have very different
    magnitudes.
    """
    a = np.asarray(a, dtype=float)
    out = np.zeros(np.shape(val_fn(x, a, b)) + (a.shape[0],))
    for k in range(a.shape[0]):
        ap = a.copy()
        am = a.copy()
        ap[k] += eps
        am[k] -= eps
        out[..., k] = (val_fn(x, ap, b) - val_fn(x, am, b)) / (2.0 * eps)
    return out


def check_jacobian(
    val_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    jac_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1.0e-5,
) -> list[JacobianReport]:
    """Compare analytic ``jac_fn`` against central-difference of ``val_fn``.

    Returns one :class:`JacobianReport` per parameter. Caller decides
    which tolerances are acceptable for which parameters -- the helper
    does not throw.
    """
    a = np.asarray(a, dtype=float)
    analytic = jac_fn(x, a, b)
    numeric = numeric_jacobian(val_fn, x, a, b, eps=eps)

    reports: list[JacobianReport] = []
    for k in range(a.shape[0]):
        ac = analytic[..., k]
        nc = numeric[..., k]
        abs_err = np.max(np.abs(ac - nc))
        denom = np.maximum(np.abs(ac), np.abs(nc))
        denom = np.where(denom < 1e-12, 1.0, denom)  # avoid 0/0 on dead pixels
        rel_err = np.max(np.abs(ac - nc) / denom)
        reports.append(
            JacobianReport(
                param_index=k,
                max_abs_err=float(abs_err),
                max_rel_err=float(rel_err),
                analytic_norm=float(np.linalg.norm(ac)),
                numeric_norm=float(np.linalg.norm(nc)),
            )
        )
    return reports
