"""Sanity check: analytic Jacobian vs numerical central differences.

The shape parameters (ds, curvature, inflection, amplitude, background)
should agree to ~1e-6 with eps=1e-5. The endpoint partials are a known
approximation in the Java original and only agree to ~PSF coupling
scale -- we just print them and assert they are at least finite.
"""

from __future__ import annotations

import numpy as np

from kapoorlabs_mtrack.models import jac, val
from kapoorlabs_mtrack.verify import check_jacobian


def _make_params():
    sigma = np.array([2.0, 2.0])
    b = 1.0 / (sigma * sigma)
    a = np.array(
        [
            5.0,  # start x
            10.0,  # start y
            25.0,  # end x
            18.0,  # end y
            0.8,  # ds (step length along curve)
            0.01,  # curvature
            0.001,  # inflection
            100.0,  # amplitude
            5.0,  # background
        ]
    )
    return a, b


def test_jacobian_against_finite_difference():
    a, b = _make_params()
    # A handful of probe pixels: one near the start, one mid-curve, one near end.
    x = np.array([[6.0, 10.5], [15.0, 14.0], [24.0, 17.5]])

    reports = check_jacobian(val, jac, x, a, b, eps=1e-5)

    print()
    print("param   meaning           analytic vs numeric")
    print("-" * 76)
    labels = [
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "ds",
        "curvature",
        "inflection",
        "amplitude",
        "background",
    ]
    for r, label in zip(reports, labels):
        print(f"{label:11s}  {r}")

    # Shape partials -- exact analytic. Compare absolute error against
    # the gradient column norm (per-pixel rel_err is noisy on near-zero
    # pixels where round-off dominates).
    shape_param_indices = {4, 5, 6, 7, 8}
    for r in reports:
        if r.param_index in shape_param_indices:
            scale = max(r.analytic_norm, 1.0)
            assert (
                r.max_abs_err / scale < 1e-4
            ), f"shape partial p[{r.param_index}] disagrees: {r}"

    # Endpoint partials -- approximation; just sanity-check finiteness.
    for r in reports:
        if r.param_index not in shape_param_indices:
            assert np.isfinite(r.max_abs_err)
            assert np.isfinite(r.max_rel_err)


if __name__ == "__main__":
    test_jacobian_against_finite_difference()
    print("\nOK")
