"""Dynamic-instability analysis: catastrophe / rescue frequencies.

Once RANSAC has decomposed a kymograph into a list of approximately
linear segments, this module turns those segments into the biology
report the user actually wants:

- **growth segments**: slope > +``slope_threshold`` (length increases
  with time -- microtubule polymerises)
- **shrinkage segments**: slope < −``slope_threshold`` (depolymerises)
- **pause segments**: |slope| ≤ ``slope_threshold`` (length stable)

A **catastrophe** is the transition from growth to shrinkage; a
**rescue** is the transition from shrinkage to growth. Pauses sit
between either side and are not counted as transitions.

Frequencies are reported as ``events / time_in_relevant_state`` so
"catastrophe frequency" has units of 1/time and is comparable across
movies of different durations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .models import LinearFunction

SegmentKind = Literal["growth", "shrinkage", "pause"]


@dataclass
class Segment:
    """One fitted linear segment of a kymograph track."""

    slope: float  # length / time -- positive = growth
    intercept: float
    t_start: float
    t_end: float
    kind: SegmentKind  # "growth" | "shrinkage" | "pause"
    n_inliers: int = 0

    @property
    def duration(self) -> float:
        return float(self.t_end - self.t_start)

    def length_change(self) -> float:
        """Net length change across the segment (in y-units)."""
        return float(self.slope * self.duration)


@dataclass
class DynamicInstability:
    """Summary statistics for one kymograph."""

    segments: list[Segment]
    n_catastrophes: int
    n_rescues: int
    time_in_growth: float
    time_in_shrinkage: float
    time_in_pause: float
    catastrophe_frequency: float  # events / time_in_growth
    rescue_frequency: float  # events / time_in_shrinkage
    mean_growth_rate: float  # length / time (positive)
    mean_shrinkage_rate: float  # length / time (negative)
    transitions: list[tuple[int, int, str]] = field(default_factory=list)
    # ^ list of (segment_idx_from, segment_idx_to, "catastrophe" | "rescue")


def _segments_from_ransac(
    estimators,
    estimator_inliers,
    timeindex: int = 0,
    slope_threshold: float = 0.5,
) -> list[Segment]:
    """Turn (estimators, inliers) from a RANSAC run into Segment dataclasses.

    Only linear estimators are kept -- quadratic / polynomial inliers
    can be re-fit linearly first via ``classify_segments`` if you ran
    ComboRansac. Here we assume the estimators are already linear.
    """
    segs: list[Segment] = []
    for est, inl in zip(estimators, estimator_inliers):
        if est is None or inl is None or len(inl) < 2:
            continue
        if not isinstance(est, LinearFunction):
            # Re-fit the inliers with a linear model so slope is well-defined.
            est = LinearFunction(np.asarray(inl))
            est.fit()
        ys = np.asarray(inl)[:, 0]
        xs = np.asarray(inl)[:, 1]
        # In our convention y = time, x = length-along-axis. The fit
        # gives x as a function of y (time), so slope = dx/dy = length
        # per unit time -- the physical growth rate.
        # The original RANSAC fits y as a function of x; we flip below.
        # For LinearFunction: y_pred = m*x + b, so we have time = m*length + b.
        # Rate (length / time) = 1 / m  if m != 0.
        m = est.get_coefficients(0)
        b = est.get_coefficients(1)
        # Convert "time = m*length + b" to "length = (time-b)/m".
        rate = 1.0 / m if abs(m) > 1e-12 else 0.0
        # Per-segment time bounds (from inlier y values, which are the
        # time coordinates of the input points).
        if timeindex == 0:
            t_lo, t_hi = float(ys.min()), float(ys.max())
        else:
            t_lo, t_hi = float(xs.min()), float(xs.max())
        if abs(rate) > slope_threshold:
            kind: SegmentKind = "growth" if rate > 0 else "shrinkage"
        else:
            kind = "pause"
        segs.append(
            Segment(
                slope=rate,
                intercept=float(b),
                t_start=t_lo,
                t_end=t_hi,
                kind=kind,
                n_inliers=int(len(inl)),
            )
        )
    # Sort chronologically so transitions are detectable.
    segs.sort(key=lambda s: s.t_start)
    return segs


def classify_segments(
    estimators,
    estimator_inliers,
    timeindex: int = 0,
    slope_threshold: float = 0.5,
) -> list[Segment]:
    """Build :class:`Segment` list from RANSAC output.

    ``slope_threshold`` is the absolute growth rate (length per unit
    time) below which a segment is called a *pause* rather than growth
    or shrinkage. Default 0.5 px/frame -- tune to your data.
    """
    return _segments_from_ransac(
        estimators, estimator_inliers, timeindex, slope_threshold
    )


def dynamic_instability(segments: list[Segment]) -> DynamicInstability:
    """Compute the dynamic-instability summary from classified segments."""
    n_cat = 0
    n_res = 0
    transitions: list[tuple[int, int, str]] = []

    # Time accumulators.
    t_growth = sum(s.duration for s in segments if s.kind == "growth")
    t_shrink = sum(s.duration for s in segments if s.kind == "shrinkage")
    t_pause = sum(s.duration for s in segments if s.kind == "pause")

    # Rate accumulators (weighted by duration).
    growth_rates = [s.slope for s in segments if s.kind == "growth"]
    shrink_rates = [s.slope for s in segments if s.kind == "shrinkage"]
    mean_growth = float(np.mean(growth_rates)) if growth_rates else 0.0
    mean_shrink = float(np.mean(shrink_rates)) if shrink_rates else 0.0

    # Walk the chronological list, counting growth→shrink and shrink→growth
    # transitions. Pauses in between don't break the chain -- a growth
    # followed by pause followed by shrink still counts as one catastrophe.
    last_directional: tuple[int, SegmentKind] | None = None
    for idx, s in enumerate(segments):
        if s.kind == "pause":
            continue
        if last_directional is not None:
            prev_idx, prev_kind = last_directional
            if prev_kind == "growth" and s.kind == "shrinkage":
                n_cat += 1
                transitions.append((prev_idx, idx, "catastrophe"))
            elif prev_kind == "shrinkage" and s.kind == "growth":
                n_res += 1
                transitions.append((prev_idx, idx, "rescue"))
        last_directional = (idx, s.kind)

    cat_freq = float(n_cat / t_growth) if t_growth > 0 else 0.0
    res_freq = float(n_res / t_shrink) if t_shrink > 0 else 0.0
    return DynamicInstability(
        segments=segments,
        n_catastrophes=n_cat,
        n_rescues=n_res,
        time_in_growth=float(t_growth),
        time_in_shrinkage=float(t_shrink),
        time_in_pause=float(t_pause),
        catastrophe_frequency=cat_freq,
        rescue_frequency=res_freq,
        mean_growth_rate=mean_growth,
        mean_shrinkage_rate=mean_shrink,
        transitions=transitions,
    )
