"""Levenberg-Marquardt fitting of the spline-Gaussian model."""

from .lm import FitResult, fit_endpoints

__all__ = ["fit_endpoints", "FitResult"]
