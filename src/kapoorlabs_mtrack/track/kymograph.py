"""Per-microtubule kymograph builder.

A kymograph is a 2-D image of an MT's extent over time:

- **y-axis**: time (frames)
- **x-axis**: signed position along the MT's reference axis, in pixels.
  Zero is the minus tip at frame 0; positive values move toward the
  plus tip's initial direction.
- **value**: by default, the local fitted intensity (``amplitude``)
  where the MT exists at that timepoint, and zero elsewhere. Pass
  ``mode="binary"`` for a 0/1 occupancy image.

The reference axis is the unit vector from the **minus tip at frame 0**
to the **plus tip at frame 0**. The MT's plus and minus tips at every
later frame get projected onto this axis. The resulting kymograph
shows the classic "growing line" pattern -- the minus end stays near
zero, the plus end's column position grows / shrinks over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class Kymograph:
    """Per-MT kymograph image plus the coordinate metadata to plot it."""

    mt_id: int
    image: np.ndarray  # shape (T, n_pos_bins)
    pos_axis: np.ndarray  # shape (n_pos_bins,), signed pixel along ref axis
    frames: np.ndarray  # shape (T,)
    plus_along_axis: np.ndarray  # shape (T,), plus tip's projection per frame
    minus_along_axis: (
        np.ndarray
    )  # shape (T,), minus tip's projection per frame


def _project_onto_axis(
    point: np.ndarray, origin: np.ndarray, unit_axis: np.ndarray
) -> float:
    """Signed projection of ``point`` onto the line through ``origin``."""
    return float(np.dot(point - origin, unit_axis))


def build_kymograph(
    profile,
    n_pos_bins: int = 200,
    pad_fraction: float = 0.2,
    mode: Literal["intensity", "binary"] = "intensity",
    amplitudes: np.ndarray | None = None,
) -> Kymograph:
    """Build one MT's kymograph from a :class:`track.profile.LengthProfile`.

    Args:
        profile: ``LengthProfile`` from :func:`track.build_length_profiles`.
        n_pos_bins: horizontal resolution of the kymograph (pixels).
        pad_fraction: extra extent on each side of the observed
            position range, as a fraction of the total range. Gives a
            margin so the plus tip doesn't crawl off the edge.
        mode: ``"intensity"`` -> fill the MT's extent with the
            amplitude value at that frame; ``"binary"`` -> 0/1
            occupancy.
        amplitudes: optional per-frame amplitude vector. If ``None``
            and ``mode == "intensity"``, the kymograph uses 1.0 (i.e.
            falls back to binary). Pass the ``amplitude`` column from
            the tracks CSV (or a smoothed version) for a nicer-looking
            kymograph.
    """
    minus0 = profile.minus_xy[0]
    plus0 = profile.plus_xy[0]
    axis_vec = plus0 - minus0
    axis_len = float(np.linalg.norm(axis_vec))
    if axis_len < 1e-6:
        # Degenerate: MT collapsed to a point at t=0. Fall back to +x.
        unit = np.array([1.0, 0.0])
    else:
        unit = axis_vec / axis_len

    T = len(profile.frames)
    plus_proj = np.empty(T)
    minus_proj = np.empty(T)
    for i in range(T):
        plus_proj[i] = _project_onto_axis(profile.plus_xy[i], minus0, unit)
        minus_proj[i] = _project_onto_axis(profile.minus_xy[i], minus0, unit)

    lo_obs = min(float(plus_proj.min()), float(minus_proj.min()))
    hi_obs = max(float(plus_proj.max()), float(minus_proj.max()))
    pad = max(1.0, (hi_obs - lo_obs) * pad_fraction)
    lo = lo_obs - pad
    hi = hi_obs + pad
    pos_axis = np.linspace(lo, hi, n_pos_bins)

    image = np.zeros((T, n_pos_bins), dtype=float)
    for i in range(T):
        a, b = sorted((minus_proj[i], plus_proj[i]))
        mask = (pos_axis >= a) & (pos_axis <= b)
        if mode == "binary":
            value = 1.0
        else:
            value = (
                float(amplitudes[i])
                if (amplitudes is not None and i < len(amplitudes))
                else 1.0
            )
        image[i, mask] = value

    return Kymograph(
        mt_id=int(profile.mt_id),
        image=image,
        pos_axis=pos_axis,
        frames=profile.frames,
        plus_along_axis=plus_proj,
        minus_along_axis=minus_proj,
    )
