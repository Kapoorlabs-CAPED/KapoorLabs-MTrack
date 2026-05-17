"""RANSAC drivers: single-model + two-pass combo.

Port of ``caped_ai_mtrack.Fits.{ransac,comboransac}`` with the disk-
writing side effects removed (the original called ``plot_ransac_gt``
inside ``extract_multiple_lines`` which wrote PNGs to a hard-coded
path -- we just return the estimators and let the caller plot).
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np

from .models import GeneralizedFunction


def _check_consistent_length(*arrays) -> None:
    lengths = []
    for a in arrays:
        if a is None:
            continue
        try:
            lengths.append(int(a.shape[0]) if hasattr(a, "shape") else len(a))
        except TypeError:
            continue
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: {lengths}"
        )


def _dedup_estimators_by_envelope(estimators, estimator_inliers, timeindex):
    """Drop estimators whose inliers are strictly contained in another's.

    Mirrors ``caped_ai_mtrack.Fits.utils.clean_estimators`` step 1.
    A second pass (matching slope + endpoints) is left out -- it
    pruned too aggressively on synthetic data.
    """
    kept = []
    kept_inliers = []
    for i, (est, inl) in enumerate(zip(estimators, estimator_inliers)):
        if est is None or inl is None or len(inl) == 0:
            continue
        sorted_i = sorted(inl.tolist(), key=lambda p: p[timeindex])
        t_lo, t_hi = sorted_i[0][timeindex], sorted_i[-1][timeindex]
        contained_in_other = False
        for j, other_inl in enumerate(estimator_inliers):
            if j == i or other_inl is None or len(other_inl) == 0:
                continue
            sorted_j = sorted(other_inl.tolist(), key=lambda p: p[timeindex])
            t_lo2, t_hi2 = sorted_j[0][timeindex], sorted_j[-1][timeindex]
            if t_lo2 <= t_lo and t_hi2 >= t_hi and i > j:
                contained_in_other = True
                break
        if not contained_in_other:
            kept.append(est)
            kept_inliers.append(inl)
    return kept, kept_inliers


class Ransac:
    """Single-model RANSAC with sequential multi-segment extraction.

    Args:
        data_points: list of ``(y, x)`` tuples. (Kymograph convention:
            ``y`` is time, ``x`` is position along the line.)
        model_class: a subclass of :class:`models.GeneralizedFunction`.
        degree: degree of the model (passed to ``model_class``).
        min_samples: minimum number of points for one ``fit()``.
        max_trials: maximum random-sample iterations per segment.
        iterations: maximum number of segments to extract sequentially.
        residual_threshold: inlier distance threshold (in y units).
        timeindex: which coordinate of a point is "time" -- used by
            ``clean_estimators`` for envelope comparisons.
        stop_probability: dynamic-trial cap formula's confidence (1 →
            no early stopping by probability).
        random_state: seed for reproducibility.
    """

    def __init__(
        self,
        data_points,
        model_class: type[GeneralizedFunction],
        degree: int,
        min_samples: int,
        max_trials: int,
        iterations: int,
        residual_threshold: float,
        timeindex: int = 0,
        stop_probability: float = 1.0,
        stop_sample_num: float = np.inf,
        stop_residuals_sum: float = 0.0,
        random_state: Optional[int] = None,
    ):
        self.data_points = data_points
        self.model_class = model_class
        self.degree = degree
        self.max_trials = max_trials
        self.iterations = iterations
        self.residual_threshold = residual_threshold
        self.timeindex = timeindex
        self.stop_probability = stop_probability
        self.stop_sample_num = stop_sample_num
        self.stop_residuals_sum = stop_residuals_sum
        self.random_state = random_state

        y, X = zip(*self.data_points)
        self.y = np.asarray(y, dtype=float)
        self.X = np.asarray(X, dtype=float)
        _check_consistent_length(self.y, self.X)

        if min_samples is None:
            self.min_samples = self.X.shape[0] + 1
        elif 0 < min_samples < 1:
            self.min_samples = int(np.ceil(min_samples * self.X.shape[0]))
        else:
            self.min_samples = int(min_samples)

    @staticmethod
    def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
        if n_inliers == 0 or probability >= 1.0:
            return np.inf
        if n_inliers == n_samples:
            return 1
        nom = math.log(1.0 - probability)
        denom = math.log(1.0 - (n_inliers / n_samples) ** min_samples)
        return int(np.ceil(nom / denom))

    def _ransac_one(self, starting_points):
        """One RANSAC pass -- yields (best_estimator, best_inlier_mask)."""
        if isinstance(starting_points, np.ndarray):
            starting_points = starting_points.tolist()
        y, X = zip(*starting_points)
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        num_samples = len(starting_points)

        if not (0 < self.min_samples < num_samples):
            raise ValueError(f"min_samples must be in (0, {num_samples})")
        if self.residual_threshold < 0:
            raise ValueError("residual_threshold must be >= 0")
        if self.max_trials < 0:
            raise ValueError("max_trials must be >= 0")
        if not (0 <= self.stop_probability <= 1):
            raise ValueError("stop_probability must be in [0, 1]")

        rng = np.random.default_rng(self.random_state)
        best_inlier_num = 0
        best_residuals_sum = np.inf
        best_inliers = np.zeros(num_samples, dtype=bool)

        spl_idxs = rng.choice(num_samples, self.min_samples, replace=False)
        for trial in range(self.max_trials):
            X_sub, y_sub = X[spl_idxs], y[spl_idxs]
            samples = [(y_sub[i], X_sub[i]) for i in range(y_sub.shape[0])]
            spl_idxs = rng.choice(num_samples, self.min_samples, replace=False)

            estimator = self.model_class(samples, self.degree)
            success = estimator.fit()
            if success is False:
                continue

            residuals = np.abs(estimator.residuals(starting_points))
            inliers = residuals < self.residual_threshold
            residuals_sum = float(residuals @ residuals)
            inliers_count = int(np.count_nonzero(inliers))

            if inliers_count > best_inlier_num or (
                inliers_count == best_inlier_num
                and residuals_sum < best_residuals_sum
            ):
                best_inlier_num = inliers_count
                best_residuals_sum = residuals_sum
                best_inliers = inliers
                cap = self._dynamic_max_trials(
                    best_inlier_num,
                    num_samples,
                    self.min_samples,
                    self.stop_probability,
                )
                if (
                    best_inlier_num >= self.stop_sample_num
                    or best_residuals_sum <= self.stop_residuals_sum
                    or trial >= cap
                ):
                    break

        if best_inlier_num == 0:
            warnings.warn("RANSAC found no inliers; returning no estimator")
            return None, None

        in_pts_y = y[best_inliers]
        in_pts_x = X[best_inliers]
        in_samples = [
            (in_pts_y[i], in_pts_x[i]) for i in range(in_pts_y.shape[0])
        ]
        final = self.model_class(in_samples, self.degree)
        final.fit()
        return final, best_inliers

    def extract_first_ransac_line(self, starting_points):
        """One RANSAC + split inliers vs outliers."""
        est, inliers = self._ransac_one(starting_points)
        if est is None:
            return None
        starting_points = np.asarray(starting_points)
        keep = starting_points[inliers]
        drop = starting_points[~inliers]
        return keep, drop, est

    def extract_multiple_lines(self):
        """Sequentially peel off segments. Returns (estimators, inlier-lists)."""
        starting_points = np.asarray(self.data_points)
        estimators = []
        estimator_inliers = []
        for _ in range(self.iterations):
            if len(starting_points) <= self.min_samples:
                break
            res = self.extract_first_ransac_line(starting_points)
            if res is None:
                break
            inlier_points, outlier_points, est = res
            estimators.append(est)
            estimator_inliers.append(inlier_points)
            if len(outlier_points) < self.min_samples:
                break
            starting_points = outlier_points
        return _dedup_estimators_by_envelope(
            estimators, estimator_inliers, self.timeindex
        )


class ComboRansac:
    """Two-pass RANSAC: peel quadratic segments first, then linear.

    Useful for kymograph segments where the MT changes direction with a
    smoothly curved transition before settling into linear motion. The
    quadratic pass picks up the curved transition; the linear pass on
    the surviving inliers picks up the straight segments.
    """

    def __init__(
        self,
        data_points,
        model_linear: type[GeneralizedFunction],
        model_quadratic: type[GeneralizedFunction],
        min_samples: int,
        max_trials: int,
        iterations: int,
        residual_threshold: float,
        timeindex: int = 0,
        random_state: Optional[int] = None,
    ):
        self.data_points = data_points
        self.timeindex = timeindex
        self.min_samples = int(max(3, min_samples))
        self.ransac_quadratic = Ransac(
            data_points,
            model_quadratic,
            3,
            self.min_samples,
            max_trials,
            iterations,
            residual_threshold,
            timeindex,
            random_state=random_state,
        )
        self.ransac_linear = Ransac(
            data_points,
            model_linear,
            2,
            self.min_samples,
            max_trials,
            iterations,
            residual_threshold,
            timeindex,
            random_state=random_state,
        )

    def extract_multiple_lines(self):
        # Pass 1: quadratic peel.
        quad_est, quad_inliers = self.ransac_quadratic.extract_multiple_lines()
        # Flatten the inlier lists for the linear pass.
        if not quad_inliers:
            return self.ransac_linear.extract_multiple_lines()
        all_inliers = np.vstack(quad_inliers)
        self.ransac_linear.data_points = all_inliers.tolist()
        return self.ransac_linear.extract_multiple_lines()
