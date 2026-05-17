"""Post-tracking: plus/minus labelling and length-profile generation.

Once the Hungarian tracker has produced per-MT trajectories (with
stable tip_A / tip_B identity), we still don't know which physical
tip is the plus end (dynamic) and which is the minus end (anchored).

We use the empirical rule: **the more dynamic tip is plus**. For each
track, sum the per-frame displacement of tip_A and of tip_B; whichever
has larger total path length gets labelled ``plus``.

Length profiles per MT:

- ``frame``                -- timepoint
- ``mt_id``                -- track identifier
- ``plus_x, plus_y``       -- plus-tip position
- ``minus_x, minus_y``     -- minus-tip position
- ``tip_distance``         -- straight-line ‖plus − minus‖
- ``arc_length``           -- length of the swept curve at this frame
                              (sum of step distances along the walked
                              curve). Falls back to ``tip_distance``
                              when the curve walker yields no
                              intermediate points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..models.spline_third_order import walk_curve

TipLabel = Literal["plus", "minus"]


@dataclass
class LengthProfile:
    """One MT track's plus/minus length profile across time."""

    mt_id: int
    frames: np.ndarray  # (T,)
    plus_xy: np.ndarray  # (T, 2)
    minus_xy: np.ndarray  # (T, 2)
    tip_distance: np.ndarray  # (T,)
    arc_length: np.ndarray  # (T,)
    plus_was_tip: TipLabel  # "A" or "B" -- which raw tip became plus


def _path_length(history: list[np.ndarray]) -> float:
    if len(history) < 2:
        return 0.0
    pts = np.asarray(history)
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def label_plus_minus(track) -> str:
    """Return ``"A"`` if tip_A is plus (more dynamic), else ``"B"``.

    Falls back to ``"A"`` on a perfect tie.
    """
    la = _path_length(track.tip_a_history)
    lb = _path_length(track.tip_b_history)
    return "B" if lb > la else "A"


def _arc_length_from_frame(frame) -> float:
    """Sum step distances along the swept curve described by ``frame``.

    Frame's stored ds + tips are sufficient -- we synthesise a 9-vector
    that ``walk_curve`` can consume. Curve direction must be
    start.x < end.x (model contract); if the stored tips violate this
    (happens when the track's plus was tip_A and tip_A < tip_B in y but
    > in x) we fall back to the straight tip-tip distance.
    """
    start = frame.tip_a
    end = frame.tip_b
    if end[0] - start[0] <= 1e-6:
        return float(np.linalg.norm(end - start))
    a9 = np.array(
        [
            start[0],
            start[1],
            end[0],
            end[1],
            frame.ds,
            frame.curvature,
            0.0,
            1.0,
            0.0,
        ]
    )
    pts = walk_curve(a9)
    if pts.shape[0] == 0:
        return float(np.linalg.norm(end - start))
    # Prepend the start, append the end so the arc length covers the
    # full curve from one tip to the other.
    full = np.vstack([start[None, :], pts, end[None, :]])
    diffs = np.diff(full, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def build_length_profiles(tracks) -> list[LengthProfile]:
    """One :class:`LengthProfile` per :class:`MTTrack`."""
    profiles: list[LengthProfile] = []
    for tr in tracks:
        if not tr.frames:
            continue
        which = label_plus_minus(tr)
        frames = np.array([f.frame for f in tr.frames])
        T = len(tr.frames)
        plus = np.empty((T, 2))
        minus = np.empty((T, 2))
        tip_d = np.empty(T)
        arc = np.empty(T)
        for i, f in enumerate(tr.frames):
            if which == "A":
                p, m = f.tip_a, f.tip_b
            else:
                p, m = f.tip_b, f.tip_a
            plus[i] = p
            minus[i] = m
            tip_d[i] = float(np.linalg.norm(p - m))
            arc[i] = _arc_length_from_frame(f)
        profiles.append(
            LengthProfile(
                mt_id=tr.mt_id,
                frames=frames,
                plus_xy=plus,
                minus_xy=minus,
                tip_distance=tip_d,
                arc_length=arc,
                plus_was_tip=which,
            )
        )
    return profiles
