"""Closed-form 1-D models used as RANSAC inliers.

Each model takes a ``points`` list of ``(y, x)`` tuples and exposes:

- ``fit()``                -- least-squares fit; sets ``self.coeff``;
                              returns ``True`` on success.
- ``predict(x)``           -- model evaluation at scalar ``x``.
- ``distance(point)``      -- shortest 2-D distance from ``point`` to
                              the curve.
- ``residuals(samples)``   -- list of distances for a list of points.
- ``get_coefficients(j)``  -- ``j``-th coefficient (low → high order).

Direct re-port of ``caped_ai_mtrack.RansacModels`` with two changes:

1. The ``points`` constructor argument is converted to ``np.asarray``
   once so we don't repeatedly cast inside the hot fit loop.
2. The quadratic distance computation guards against ``_epsilon``
   divide-by-zero with the explicit constant from the original (kept
   for numerical bit-parity with the Java/imglib2 reference).
"""

from __future__ import annotations

import math

import numpy as np

_EPSILON = 1.0e-15


class GeneralizedFunction:
    """Shared base: stores points + degree, exposes ``get_num_points``."""

    def __init__(self, points, degree: int):
        self.points = np.asarray(points, dtype=float)
        self.degree = int(degree)

    def get_num_points(self) -> int:
        return int(self.points.shape[0])

    # Subclass contract.
    def fit(self) -> bool:
        raise NotImplementedError

    def predict(self, x: float) -> float:
        raise NotImplementedError

    def distance(self, point) -> float:
        raise NotImplementedError

    def residuals(self, samples) -> list[float]:
        return [self.distance(p) for p in samples]

    def get_coefficients(self, j: int) -> float:
        return float(self.coeff[j])


class LinearFunction(GeneralizedFunction):
    """``y = m x + b`` via 2x2 normal equations.

    Coefficient layout: ``coeff = [m, b]``.
    """

    def __init__(self, points, degree: int = 1):
        super().__init__(points, degree)
        self.num_points = self.get_num_points()
        self.min_num_points = 2
        self.coeff = np.zeros(self.min_num_points)

    def fit(self) -> bool:
        delta = np.zeros(4)
        theta = np.zeros(2)
        for i in range(self.num_points):
            y, x = self.points[i, 0], self.points[i, 1]
            delta[0] += x * x
            delta[1] += x
            delta[2] += x
            delta[3] += 1
            theta[0] += x * y
            theta[1] += y
        delta_inv = np.linalg.pinv(delta.reshape(2, 2))
        self.coeff[0] = delta_inv[0, 0] * theta[0] + delta_inv[0, 1] * theta[1]
        self.coeff[1] = delta_inv[1, 0] * theta[0] + delta_inv[1, 1] * theta[1]
        return True

    def predict(self, x: float) -> float:
        return float(self.coeff[0] * x + self.coeff[1])

    def distance(self, point) -> float:
        y1, x1 = float(point[0]), float(point[1])
        m, b = self.coeff[0], self.coeff[1]
        return abs(y1 - m * x1 - b) / math.sqrt(1.0 + m * m)


class QuadraticFunction(GeneralizedFunction):
    """``y = a x^2 + b x + c`` with exact shortest-distance via Cardano.

    Coefficient layout: ``coeff = [a, b, c]`` (highest order first).
    The exact distance from a point to a parabola requires the root
    of a cubic; we use the standard Cardano formula. This is what the
    original ``caped_ai_mtrack`` did, kept verbatim for bit-parity
    with the imglib2 reference -- the only edits are a guard against
    ``a == 0`` (degenerate to a line) and explicit ``float`` casts so
    NumPy doesn't return 0-d arrays.
    """

    def __init__(self, points, degree: int = 2):
        super().__init__(points, degree)
        self.num_points = self.get_num_points()
        self.min_num_points = 3
        self.coeff = np.zeros(3)

    def fit(self) -> bool:
        delta = np.zeros(9)
        theta = np.zeros(3)
        for i in range(self.num_points):
            y, x = self.points[i, 0], self.points[i, 1]
            xx = x * x
            xxx = xx * x
            delta[0] += xx * xx
            delta[1] += xxx
            delta[2] += xx
            delta[3] += xxx
            delta[4] += xx
            delta[5] += x
            delta[6] += xx
            delta[7] += x
            delta[8] += 1
            theta[0] += xx * y
            theta[1] += x * y
            theta[2] += y
        delta_inv = np.linalg.pinv(delta.reshape(3, 3))
        for k in range(3):
            self.coeff[k] = float(
                delta_inv[k, 0] * theta[0]
                + delta_inv[k, 1] * theta[1]
                + delta_inv[k, 2] * theta[2]
            )
        return True

    def predict(self, x: float) -> float:
        a, b, c = self.coeff
        return float(a * x * x + b * x + c)

    def distance(self, point) -> float:
        y1, x1 = float(point[0]), float(point[1])
        a, b, c = self.coeff[0], self.coeff[1], self.coeff[2]
        # Degenerate quadratic = line; fall back.
        if abs(a) < _EPSILON:
            return abs(y1 - b * x1 - c) / math.sqrt(1.0 + b * b)

        # Reduce ``d/dx |P - point|^2 = 0`` to a depressed cubic in xc.
        a3 = 2.0 * a * a
        a2 = (3.0 * b * a) / (a3 + _EPSILON)
        a1 = (2.0 * c * a - 2.0 * a * y1 + 1.0 + b * b) / (a3 + _EPSILON)
        a0 = (c * b - y1 * b - x1) / (a3 + _EPSILON)
        p = (3.0 * a1 - a2 * a2) / 3.0
        q = (-9.0 * a1 * a2 + 27.0 * a0 + 2.0 * a2 * a2 * a2) / 27.0
        tmp1 = math.sqrt(abs(-p) / 3.0)
        tmp2 = q * q / 4.0 + p * p * p / 27.0

        if tmp2 > 0:
            sq = math.sqrt(tmp2)
            aBar = np.cbrt(-q / 2.0 + sq)
            bBar = np.cbrt(-q / 2.0 - sq)
            xc1 = xc2 = xc3 = aBar + bBar - a2 / 3.0
        elif tmp2 == 0:
            if q > 0:
                xc1, xc2, xc3 = -2.0 * tmp1, tmp1, tmp1
            elif q < 0:
                xc1, xc2, xc3 = 2.0 * tmp1, -tmp1, -tmp1
            else:
                xc1 = xc2 = xc3 = 0.0
        else:
            arg = math.sqrt(q * q * 0.25 / (-p * p * p / 27.0))
            phi = math.acos(-arg) if q >= 0 else math.acos(arg)
            xc1 = 2.0 * tmp1 * math.cos(phi / 3.0) - a2 / 3.0
            xc2 = 2.0 * tmp1 * math.cos((phi + 2.0 * math.pi) / 3.0) - a2 / 3.0
            xc3 = 2.0 * tmp1 * math.cos((phi + 4.0 * math.pi) / 3.0) - a2 / 3.0

        def d(xc: float) -> float:
            yc = c + b * xc + a * xc * xc
            return math.hypot(xc - x1, yc - y1)

        return min(d(xc1), d(xc2), d(xc3))


class PolynomialFunction(GeneralizedFunction):
    """``y = sum_k c_k x^k`` for arbitrary ``degree``.

    Uses ``numpy.polyfit`` for least squares (orders of magnitude
    faster than the hand-rolled normal-equation accumulators for
    ``degree > 2``) and ``numpy.polyval`` for prediction. The shortest
    perpendicular distance from a point to a general polynomial has
    no closed form; we use the *vertical* distance instead, which is
    a reasonable approximation when the polynomial is monotonic in
    ``x`` -- the dominant case for kymograph segments.
    """

    def __init__(self, points, degree: int = 3):
        super().__init__(points, degree)
        self.num_points = self.get_num_points()
        self.min_num_points = degree + 1
        self.coeff = np.zeros(degree + 1)

    def fit(self) -> bool:
        ys = self.points[:, 0]
        xs = self.points[:, 1]
        # numpy.polyfit returns coefficients in DESCENDING order
        # (highest degree first), which matches our get_coefficients()
        # convention if we store as-is.
        self.coeff = np.polyfit(xs, ys, self.degree)
        return True

    def predict(self, x: float) -> float:
        return float(np.polyval(self.coeff, x))

    def distance(self, point) -> float:
        y1, x1 = float(point[0]), float(point[1])
        return abs(y1 - self.predict(x1))
