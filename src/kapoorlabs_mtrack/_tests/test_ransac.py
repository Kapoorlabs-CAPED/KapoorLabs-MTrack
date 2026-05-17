"""RANSAC + dynamic-instability tests.

Synthetic kymograph contains three known linear segments forming
a growth → shrinkage → growth pattern, i.e. one catastrophe and one
rescue. We render the kymograph, run RANSAC, classify segments, and
verify the recovered counts and the per-state rates.
"""

from __future__ import annotations

import numpy as np
from skimage.draw import line

from kapoorlabs_mtrack.ransac import (
    LinearFunction,
    QuadraticFunction,
    Ransac,
    classify_segments,
    dynamic_instability,
    extract_kymograph_points,
)


def _render_kymograph_with_segments(segments_xy):
    """Draw thick lines on a blank canvas to simulate a kymograph.

    ``segments_xy`` is a list of (start_xy, end_xy) in (x_pos, y_time).
    """
    img = np.zeros((120, 80), dtype=np.uint8)
    for (x0, t0), (x1, t1) in segments_xy:
        rr, cc = line(int(t0), int(x0), int(t1), int(x1))
        rr = np.clip(rr, 0, img.shape[0] - 1)
        cc = np.clip(cc, 0, img.shape[1] - 1)
        img[rr, cc] = 255
        # Make it 3 px wide so the skeleton recovers cleanly.
        for dx in (-1, 1):
            for dy in (-1, 1):
                rrn = np.clip(rr + dy, 0, img.shape[0] - 1)
                ccn = np.clip(cc + dx, 0, img.shape[1] - 1)
                img[rrn, ccn] = 255
    return img


def test_linear_function_distance_matches_geometric_formula():
    pts = [(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)]  # y = x
    lf = LinearFunction(pts, degree=1)
    lf.fit()
    # m ≈ 1, b ≈ 0 -> distance from (0, 1) to y=x is sqrt(2)/2
    assert abs(lf.distance((0.0, 1.0)) - (1.0 / np.sqrt(2.0))) < 1e-6


def test_quadratic_function_distance_finite_on_offset_point():
    # y = x^2; check distance from a point slightly above the parabola.
    pts = [(0.0, 0.0), (1.0, 1.0), (4.0, 2.0), (9.0, 3.0)]
    qf = QuadraticFunction(pts, degree=2)
    qf.fit()
    d = qf.distance((5.0, 2.0))  # somewhere above the parabola
    assert d > 0 and np.isfinite(d)


def test_ransac_recovers_three_growth_shrink_growth_segments():
    # Geometry: growth (t 0-30 covers x 10-50, slope +1.3 px/frame),
    # shrinkage (t 30-70 covers x 50-10, slope -1.0), growth (t 70-110
    # covers x 10-50, slope +1.0). Times along axis 0, positions along axis 1.
    segs = [
        ((10, 0), (50, 30)),  # growth: x 10 -> 50 over t 0..30
        ((50, 30), (10, 70)),  # shrink: x 50 -> 10 over t 30..70
        ((10, 70), (50, 110)),
    ]  # growth: x 10 -> 50 over t 70..110
    img = _render_kymograph_with_segments(segs)
    pts = extract_kymograph_points(img)
    assert pts.shape[0] > 20, f"too few points extracted: {pts.shape}"

    rng = np.random.default_rng(0)
    ransac = Ransac(
        data_points=pts.tolist(),
        model_class=LinearFunction,
        degree=2,  # LinearFunction(...,degree) is unused but kept for signature
        min_samples=10,
        max_trials=120,
        iterations=6,
        residual_threshold=2.0,
        timeindex=0,
        random_state=int(rng.integers(1 << 31)),
    )
    estimators, inliers = ransac.extract_multiple_lines()
    # Allow a little over-segmentation (RANSAC can split a single segment
    # into two if residuals fluctuate), so we expect 3 to 5 segments.
    assert (
        3 <= len(estimators) <= 5
    ), f"expected 3-5 segments, got {len(estimators)}"

    segments = classify_segments(
        estimators, inliers, timeindex=0, slope_threshold=0.4
    )
    # We should see at least one growth and one shrinkage segment.
    kinds = {s.kind for s in segments}
    assert "growth" in kinds, f"no growth segment in {kinds}"
    assert "shrinkage" in kinds, f"no shrinkage segment in {kinds}"


def test_dynamic_instability_counts_catastrophe_and_rescue():
    # Hand-craft three segments in chronological order: growth, shrink, growth.
    from kapoorlabs_mtrack.ransac import Segment

    segments = [
        Segment(
            slope=+1.0,
            intercept=10.0,
            t_start=0.0,
            t_end=30.0,
            kind="growth",
            n_inliers=15,
        ),
        Segment(
            slope=-1.0,
            intercept=70.0,
            t_start=30.0,
            t_end=70.0,
            kind="shrinkage",
            n_inliers=20,
        ),
        Segment(
            slope=+1.0,
            intercept=-60.0,
            t_start=70.0,
            t_end=110.0,
            kind="growth",
            n_inliers=20,
        ),
    ]
    di = dynamic_instability(segments)
    assert di.n_catastrophes == 1
    assert di.n_rescues == 1
    assert di.time_in_growth == 70.0  # 30 + 40
    assert di.time_in_shrinkage == 40.0
    # Frequencies: 1 catastrophe per 70 units of growth time.
    assert abs(di.catastrophe_frequency - 1.0 / 70.0) < 1e-9
    assert abs(di.rescue_frequency - 1.0 / 40.0) < 1e-9
    # Mean rates: growth +1.0, shrink -1.0.
    assert abs(di.mean_growth_rate - 1.0) < 1e-9
    assert abs(di.mean_shrinkage_rate - (-1.0)) < 1e-9


def test_pause_between_growth_and_shrink_still_counts_as_catastrophe():
    from kapoorlabs_mtrack.ransac import Segment

    segments = [
        Segment(
            slope=+1.0,
            intercept=10.0,
            t_start=0.0,
            t_end=30.0,
            kind="growth",
            n_inliers=15,
        ),
        Segment(
            slope=+0.0,
            intercept=40.0,
            t_start=30.0,
            t_end=40.0,
            kind="pause",
            n_inliers=8,
        ),
        Segment(
            slope=-1.0,
            intercept=80.0,
            t_start=40.0,
            t_end=70.0,
            kind="shrinkage",
            n_inliers=15,
        ),
    ]
    di = dynamic_instability(segments)
    assert di.n_catastrophes == 1
    assert di.n_rescues == 0
    assert di.time_in_pause == 10.0


if __name__ == "__main__":
    test_linear_function_distance_matches_geometric_formula()
    test_quadratic_function_distance_finite_on_offset_point()
    test_ransac_recovers_three_growth_shrink_growth_segments()
    test_dynamic_instability_counts_catastrophe_and_rescue()
    test_pause_between_growth_and_shrink_still_counts_as_catastrophe()
    print("\nOK")
