"""Render the analytic spline-Gaussian model on a pixel grid plus noise.

The renderer reuses the exact same ``models.spline_third_order.val``
function the fitter sees, so simulated images and fit residuals share a
forward model -- no chance of a hidden discrepancy between simulation
and inference.
"""

from __future__ import annotations

import numpy as np

from ..models import val


def render_curve_image(
    a: np.ndarray, b: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Render the 3rd-order spline model on a ``shape = (H, W)`` grid.

    The returned image is the clean (noise-free) intensity expected
    under the parameters ``a`` (free) and ``b`` (fixed PSF widths).
    Coordinate convention matches the Java original: pixel ``(i, j)`` is
    indexed as ``x = (j, i)``  -- column first, row second -- so that
    ``a[0]`` and ``a[2]`` (the x components of start/end) move along the
    image's horizontal axis.
    """
    h, w = shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([jj, ii], axis=-1).astype(float)  # (H, W, 2)
    return val(coords, a, b)


def add_shot_noise(
    image: np.ndarray,
    read_noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Poisson shot noise + optional Gaussian read noise.

    ``image`` is interpreted directly as expected photon counts: each
    pixel becomes ``Poisson(image[i, j]) + Normal(0, read_noise_sigma)``.
    Returns ``float`` so subsequent fitting need not worry about clipping.
    """
    if rng is None:
        rng = np.random.default_rng()
    clean = np.clip(image, 0.0, None)
    out = rng.poisson(clean).astype(float)
    if read_noise_sigma > 0:
        out = out + rng.normal(0.0, read_noise_sigma, size=out.shape)
    return out
