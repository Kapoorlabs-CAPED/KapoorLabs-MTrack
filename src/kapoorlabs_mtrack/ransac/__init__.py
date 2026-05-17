"""RANSAC segmentation of microtubule kymographs.

Independent of the spline-Gaussian optimizer in ``models`` / ``fit`` --
this subpackage operates on *kymograph* images (one MT, x = position,
y = time) rather than per-frame 2-D images. The math is a clean
re-port of ``caped_ai_mtrack.RansacModels`` and ``.Fits``, stripped of
PNG-writing side effects, the ``vollseg`` segmentation dependency,
and the ``caped_ai_tabulour`` table widget.

Public API:

- ``LinearFunction`` / ``QuadraticFunction`` / ``PolynomialFunction``
  -- RANSAC inlier models exposing ``fit``, ``predict``, ``distance``,
  ``residuals``.
- ``Ransac`` -- single-model RANSAC with ``extract_multiple_lines``
  for sequential segment discovery.
- ``ComboRansac`` -- two-pass quadratic-then-linear extractor; finds
  curved + linear motion patterns in the same kymograph.
- ``classify_segments`` / ``dynamic_instability`` -- given fitted
  linear segments, label each as growth / shrinkage / pause and
  compute catastrophe + rescue frequencies plus mean rates.
- ``extract_kymograph_points`` -- threshold + skeletonise a kymograph
  image into a ``(t, x)`` point cloud ready for RANSAC.
"""

from .dynamics import (
    DynamicInstability,
    Segment,
    classify_segments,
    dynamic_instability,
)
from .fits import ComboRansac, Ransac
from .kymograph_extract import extract_kymograph_points
from .models import (
    GeneralizedFunction,
    LinearFunction,
    PolynomialFunction,
    QuadraticFunction,
)

__all__ = [
    "GeneralizedFunction",
    "LinearFunction",
    "QuadraticFunction",
    "PolynomialFunction",
    "Ransac",
    "ComboRansac",
    "Segment",
    "DynamicInstability",
    "classify_segments",
    "dynamic_instability",
    "extract_kymograph_points",
]
