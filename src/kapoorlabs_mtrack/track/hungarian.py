"""Frame-by-frame Hungarian tracker that links MT observations into tracks.

Algorithm:

1. Start with the first frame: every observation becomes a new track
   with a fresh ``mt_id``.
2. For each subsequent frame:
   a. Build the cost matrix ``C[i, j]`` = cost of linking active track
      ``i`` to observation ``j`` via :func:`cost.mt_to_mt_cost`.
   b. Pad with virtual rows / columns at cost ``cfg.gate`` so any link
      with cost above the gate is preferred-rejected -- the tracker
      will birth a new track or terminate the unmatched one instead.
   c. Run :func:`scipy.optimize.linear_sum_assignment` for the optimal
      assignment.
   d. Append matched observations to their track, with the chosen
      tip-permutation flag so start/end identity stays consistent.
   e. Unmatched observations become new tracks; unmatched tracks
      either go inactive immediately or after ``max_gap`` frames if
      gap-closing is enabled (default: no gap, tracks die immediately).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cost import TipObservation, TrackingCost, mt_to_mt_cost


@dataclass
class _TrackFrame:
    """One observation appended to an MTTrack, with provenance for plus/minus."""

    frame: int
    label: int
    mt_in_label: int
    # tip_a / tip_b are the two TIP POSITIONS aligned to the track's
    # stable tip-A/tip-B identity (NOT the observation's raw start/end,
    # which may have been swapped at this link).
    tip_a: np.ndarray
    tip_b: np.ndarray
    amplitude: float
    curvature: float
    ds: float


@dataclass
class MTTrack:
    """One microtubule's trajectory across multiple frames."""

    mt_id: int
    frames: list[_TrackFrame] = field(default_factory=list)
    last_obs_frame: int = -1
    # Whether the *current* observation's start corresponds to track tip_A
    # (False) or tip_B (True). Updated each time we append.
    _current_swap: bool = False

    @property
    def tip_a_history(self) -> list[np.ndarray]:
        return [f.tip_a for f in self.frames]

    @property
    def tip_b_history(self) -> list[np.ndarray]:
        return [f.tip_b for f in self.frames]

    def latest_observation(self) -> Optional[TipObservation]:
        """Reconstruct a :class:`TipObservation` representing the last frame.

        The reconstructed observation's ``start`` / ``end`` are aligned
        to the track's tip_A / tip_B (no swap), so subsequent
        cost-matrix calls see consistent tip identity.
        """
        if not self.frames:
            return None
        f = self.frames[-1]
        return TipObservation(
            frame=f.frame,
            label=f.label,
            mt_in_label=f.mt_in_label,
            start=f.tip_a.copy(),
            end=f.tip_b.copy(),
            amplitude=f.amplitude,
            curvature=f.curvature,
            ds=f.ds,
        )


def _make_track_frame_from_observation(
    obs: TipObservation, swapped: bool
) -> _TrackFrame:
    """Map raw start/end to stable tip_a/tip_b using the link's swap flag."""
    if swapped:
        tip_a, tip_b = obs.end, obs.start
    else:
        tip_a, tip_b = obs.start, obs.end
    return _TrackFrame(
        frame=obs.frame,
        label=obs.label,
        mt_in_label=obs.mt_in_label,
        tip_a=tip_a.copy(),
        tip_b=tip_b.copy(),
        amplitude=obs.amplitude,
        curvature=obs.curvature,
        ds=obs.ds,
    )


def _observation_from_snapshot(snap) -> TipObservation:
    """Adapt a :class:`pipeline.fit_stack.MTSnapshot` to :class:`TipObservation`."""
    return TipObservation(
        frame=snap.frame,
        label=snap.label,
        mt_in_label=snap.mt_in_label,
        start=np.asarray(snap.start, dtype=float),
        end=np.asarray(snap.end, dtype=float),
        amplitude=float(snap.amplitude),
        curvature=float(snap.curvature),
        ds=float(snap.ds),
    )


def track_snapshots(
    frame_snapshots,
    cfg: Optional[TrackingCost] = None,
    max_gap: int = 0,
) -> list[MTTrack]:
    """Link per-frame MT snapshots into across-frame tracks.

    Args:
        frame_snapshots: list of ``pipeline.fit_stack.FrameSnapshot``
            (or any iterable yielding objects with a ``.mts`` list of
            objects exposing the ``MTSnapshot`` fields).
        cfg: tracking-cost configuration. ``None`` → default
            (distance + intensity + curvature, all on).
        max_gap: how many empty frames a track may go unmatched before
            it can no longer claim a new observation. ``0`` (default)
            means a missed match terminates the track immediately.

    Returns: list of :class:`MTTrack`, one per microtubule trajectory.
    """
    if cfg is None:
        cfg = TrackingCost()

    tracks: list[MTTrack] = []
    next_mt_id = 0

    for fs in frame_snapshots:
        observations = [_observation_from_snapshot(m) for m in fs.mts]
        t = fs.frame

        # Active tracks: those whose last observation is no older than max_gap.
        active: list[MTTrack] = [
            tr
            for tr in tracks
            if (t - tr.last_obs_frame) <= max(1, max_gap + 1)
        ]

        if not active:
            # First frame, or every prior track has died: start fresh.
            for obs in observations:
                tr = MTTrack(mt_id=next_mt_id)
                next_mt_id += 1
                tr.frames.append(
                    _make_track_frame_from_observation(obs, swapped=False)
                )
                tr.last_obs_frame = t
                tracks.append(tr)
            continue

        if not observations:
            continue

        # Build cost & swap matrices.
        n_tracks, n_obs = len(active), len(observations)
        cost_mat = np.full((n_tracks, n_obs), cfg.gate, dtype=float)
        swap_mat = np.zeros((n_tracks, n_obs), dtype=bool)
        for i, tr in enumerate(active):
            prev_obs = tr.latest_observation()
            for j, obs in enumerate(observations):
                cost, swapped = mt_to_mt_cost(
                    prev_obs,
                    obs,
                    cfg,
                    start_history=tr.tip_a_history,
                    end_history=tr.tip_b_history,
                )
                if cost < cfg.gate:
                    cost_mat[i, j] = cost
                    swap_mat[i, j] = swapped

        # Augmented matrix for LAP-style assignment with birth/death
        # (Jaqaman 2008 formulation):
        #
        #   [  C   |  D  ]    real costs (n_tracks x n_obs)  | death diag
        #   [---+----    ]
        #   [  B   |  0  ]    birth diag (n_obs x n_tracks)  | filler 0
        #
        # Death / birth diagonals carry cost ``cfg.gate``. Real links
        # win when C[i,j] < 2 * gate (since deciding to break a link
        # costs death + birth = 2 * gate). Off-diagonal cells in the D
        # / B blocks get +inf so virtual obs / tracks pair only with
        # their own real partner.
        size = n_tracks + n_obs
        aug = np.zeros((size, size), dtype=float)
        aug[:n_tracks, :n_obs] = cost_mat
        # Death block: diagonal at gate, off-diagonal at +inf.
        death = np.full((n_tracks, n_tracks), np.inf)
        np.fill_diagonal(death, cfg.gate)
        aug[:n_tracks, n_obs:] = death
        # Birth block: diagonal at gate, off-diagonal at +inf.
        birth = np.full((n_obs, n_obs), np.inf)
        np.fill_diagonal(birth, cfg.gate)
        aug[n_tracks:, :n_obs] = birth
        # Bottom-right virtual-to-virtual filler stays 0.

        rows, cols = linear_sum_assignment(aug)
        matched_obs: set[int] = set()
        for i, j in zip(rows, cols):
            if i < n_tracks and j < n_obs and cost_mat[i, j] < cfg.gate:
                tr = active[i]
                obs = observations[j]
                tr.frames.append(
                    _make_track_frame_from_observation(
                        obs, swapped=bool(swap_mat[i, j])
                    )
                )
                tr.last_obs_frame = t
                matched_obs.add(j)
            # Unmatched track-rows = death (just don't update); virtual
            # rows have no real effect.

        # Unmatched observations → new tracks.
        for j, obs in enumerate(observations):
            if j in matched_obs:
                continue
            tr = MTTrack(mt_id=next_mt_id)
            next_mt_id += 1
            tr.frames.append(
                _make_track_frame_from_observation(obs, swapped=False)
            )
            tr.last_obs_frame = t
            tracks.append(tr)

    return tracks
