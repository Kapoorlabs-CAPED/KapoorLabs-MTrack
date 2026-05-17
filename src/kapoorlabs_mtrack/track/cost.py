"""Cost matrix building blocks for linking microtubules across frames.

Each link cost between an MT observed at frame ``t-1`` and a candidate
MT at frame ``t`` is a weighted sum of:

- **distance** (always on): squared distance between the predicted
  position and the observed position, summed over the two tips. Each
  tip's position is predicted by adding a velocity vector derived from
  the previous one or two timepoints of that MT's track.
- **intensity** (default on): squared difference of the fitted
  amplitudes, scaled.
- **curvature** (default on): squared difference of the fitted
  curvature parameters (a whole-MT property, same value for both tips
  of one MT).
- **ds** (default off): squared difference of the curve step lengths.

Because we don't know which tip of MT_t corresponds to which tip of
MT_{t-1}, the cost is computed for **both possible tip permutations**
(start↔start/end↔end, or start↔end/end↔start) and the lower of the
two is returned, along with the permutation that achieved it. The
permutation feeds back into the track so the same physical tip keeps
its identity over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrackingCost:
    """Toggleable cost-function configuration.

    Set each ``enable_*`` flag to ``False`` to drop that term from the
    weighted sum. Weights are only consulted for enabled terms.
    """

    enable_intensity: bool = True
    enable_curvature: bool = True
    enable_ds: bool = False
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "distance": 1.0,
            "intensity": 0.3,
            "curvature": 0.5,
            "ds": 0.2,
        }
    )
    amp_scale: float = 100.0  # divisor for intensity term, photon units
    gate: float = 50.0**2  # max linkable cost (squared px units)
    velocity_lookback: int = 2  # frames of history used for velocity


@dataclass
class TipObservation:
    """One MT observation packed for the tracker.

    Mirrors the fields of ``pipeline.fit_stack.MTSnapshot`` that the
    cost function actually needs.
    """

    frame: int
    label: int
    mt_in_label: int
    start: np.ndarray  # (2,) (x, y)
    end: np.ndarray  # (2,) (x, y)
    amplitude: float
    curvature: float
    ds: float


def _predict_velocity(history: list[np.ndarray], lookback: int) -> np.ndarray:
    """Linear velocity from the last ``lookback`` positions in a history."""
    if len(history) < 2:
        return np.zeros(2)
    recent = history[-lookback:]
    if len(recent) < 2:
        return np.zeros(2)
    return recent[-1] - recent[-2]


def _tip_pair_cost(
    prev_start: np.ndarray,
    prev_end: np.ndarray,
    pred_start: np.ndarray,
    pred_end: np.ndarray,
    curr_start: np.ndarray,
    curr_end: np.ndarray,
) -> tuple[float, bool]:
    """Squared distance summed over both tips, minimised over the two
    possible tip permutations. Returns ``(cost, swapped)`` where
    ``swapped=True`` means ``curr.start`` matches ``prev.end`` (i.e.
    the model's start/end labels flip across this link).
    """
    # Permutation 1: prev.start -> curr.start, prev.end -> curr.end
    p1 = np.sum((pred_start - curr_start) ** 2) + np.sum(
        (pred_end - curr_end) ** 2
    )
    # Permutation 2: prev.start -> curr.end, prev.end -> curr.start
    pred_start_swap = pred_end
    pred_end_swap = pred_start
    p2 = np.sum((pred_start_swap - curr_start) ** 2) + np.sum(
        (pred_end_swap - curr_end) ** 2
    )
    if p2 < p1:
        return float(p2), True
    return float(p1), False


def mt_to_mt_cost(
    prev: TipObservation,
    curr: TipObservation,
    cfg: TrackingCost,
    start_history: Optional[list[np.ndarray]] = None,
    end_history: Optional[list[np.ndarray]] = None,
) -> tuple[float, bool]:
    """Cost of linking ``prev`` (at frame t-1) to ``curr`` (at frame t).

    ``start_history`` / ``end_history`` are the recent tip positions
    of the previous track, used to estimate per-tip velocity for
    motion prediction. Returns ``(cost, swapped)`` with the same
    semantics as :func:`_tip_pair_cost` for the position component;
    the scalar shape terms (intensity, curvature, ds) don't depend on
    the permutation.
    """
    if start_history is None:
        start_history = [prev.start]
    if end_history is None:
        end_history = [prev.end]

    vel_start = _predict_velocity(start_history, cfg.velocity_lookback)
    vel_end = _predict_velocity(end_history, cfg.velocity_lookback)
    pred_start = prev.start + vel_start
    pred_end = prev.end + vel_end

    dist_cost, swapped = _tip_pair_cost(
        prev.start, prev.end, pred_start, pred_end, curr.start, curr.end
    )
    total = cfg.weights.get("distance", 1.0) * dist_cost

    if cfg.enable_intensity:
        d = (prev.amplitude - curr.amplitude) / max(cfg.amp_scale, 1e-6)
        total += cfg.weights.get("intensity", 0.0) * float(d * d)

    if cfg.enable_curvature:
        d = prev.curvature - curr.curvature
        total += cfg.weights.get("curvature", 0.0) * float(d * d)

    if cfg.enable_ds:
        d = prev.ds - curr.ds
        total += cfg.weights.get("ds", 0.0) * float(d * d)

    return total, swapped
