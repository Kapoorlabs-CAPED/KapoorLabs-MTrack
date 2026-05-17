"""Levenberg-Marquardt fitting of the spline-Gaussian model."""

from .joint import JointFitResult, fit_endpoints_joint
from .lm import FitResult, fit_endpoints

__all__ = [
    "fit_endpoints",
    "FitResult",
    "fit_endpoints_joint",
    "JointFitResult",
]
