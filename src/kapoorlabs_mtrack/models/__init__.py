"""Analytic microtubule line models.

Each model exposes:

- ``val(x, a, b)`` -- model intensity at pixel coordinate ``x``
- ``jac(x, a, b)`` -- gradient w.r.t. the parameter vector ``a`` at ``x``

``a`` is the *free* parameter vector that the Levenberg-Marquardt solver
refines; ``b`` holds *fixed* per-axis ``1 / sigma**2`` of the imaging PSF.

``b`` is a property of the microscope (PSF), **not** something the
solver touches -- pass it in once per dataset and leave it alone.

The layout of ``a`` for the 3rd-order spline model (ndims=2) is::

    a[0..1]   start point (x0, y0)             -- free
    a[2..3]   end   point (x1, y1)             -- free
    a[4]      ds  (curve step length)          -- free; |ds| is used
    a[5]      curvature  (2nd-order coef)      -- free
    a[6]      inflection (3rd-order coef)      -- free
    a[7]      amplitude                        -- free
    a[8]      background                       -- free

This mirrors ``GaussianSplinethirdorder.java`` from the original imglib2
implementation so the analytic gradients there port one-for-one.
"""

from . import multi
from .spline_third_order import jac, val, walk_curve

__all__ = ["val", "jac", "walk_curve", "multi"]
