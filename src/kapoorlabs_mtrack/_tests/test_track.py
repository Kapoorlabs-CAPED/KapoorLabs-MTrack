"""Tracker tests: identity persistence + plus/minus labelling.

The tracker doesn't need a real image. We hand-build per-frame
snapshots that emulate the output of ``pipeline.fit_stack``:

- Two microtubules in every frame. One has its plus tip (defined here
  to be ``end``) growing rightward over time; the other has its plus
  tip (also ``end``) growing rightward but at a different position.
- Frame to frame, raw start/end labels may flip (the model has no
  preferred direction) -- the tracker should recover the same physical
  tip identity regardless.

Acceptance criteria:

1. ``track_snapshots`` produces exactly two tracks, each spanning all
   frames.
2. For each track, ``tip_a_history`` is a smooth trajectory (no jumps
   from the swap-handling logic).
3. ``label_plus_minus`` assigns the growing tip to ``plus`` for both
   tracks (it accumulates the most path length).
4. Length profiles' ``arc_length`` grows monotonically over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kapoorlabs_mtrack.track import (
    TrackingCost,
    build_length_profiles,
    label_plus_minus,
    track_snapshots,
)


@dataclass
class _Snap:
    """Stand-in for pipeline.fit_stack.MTSnapshot."""

    frame: int
    label: int
    mt_in_label: int
    start: np.ndarray
    end: np.ndarray
    amplitude: float
    curvature: float
    ds: float


@dataclass
class _FrameSnap:
    frame: int
    mts: list


def _two_mts_growing(swap_pattern: list[bool]) -> list[_FrameSnap]:
    """Build T frames of 2 MTs with the plus tip growing rightward.

    ``swap_pattern[t]`` toggles whether MT_b's raw start/end are
    flipped at frame t. Used to verify the tracker's
    permutation-handling.
    """
    frames: list[_FrameSnap] = []
    T = len(swap_pattern)
    for t in range(T):
        # MT a: minus tip stays at (10, 20), plus tip grows from x=30 -> 30+1.5*t
        a_minus = np.array([10.0, 20.0])
        a_plus = np.array([30.0 + 1.5 * t, 22.0])
        # MT b: minus tip stays at (60, 60), plus tip grows from x=40 -> 40-1.0*t
        b_minus = np.array([60.0, 60.0])
        b_plus = np.array([40.0 - 1.0 * t, 55.0])
        # Always store start with smaller x first (model contract),
        # then optionally swap for MT b at frames where swap_pattern[t].
        if swap_pattern[t]:
            mt_a = _Snap(
                t, 1, 0, a_plus.copy(), a_minus.copy(), 100.0, 0.0, 0.7
            )
            mt_b = _Snap(
                t, 2, 0, b_minus.copy(), b_plus.copy(), 95.0, 0.0, 0.7
            )
        else:
            mt_a = _Snap(
                t, 1, 0, a_minus.copy(), a_plus.copy(), 100.0, 0.0, 0.7
            )
            mt_b = _Snap(
                t, 2, 0, b_plus.copy(), b_minus.copy(), 95.0, 0.0, 0.7
            )
        frames.append(_FrameSnap(frame=t, mts=[mt_a, mt_b]))
    return frames


def test_two_mts_persist_identity_across_frames():
    swap = [False, True, False, True, False]  # alternate raw flips
    frames = _two_mts_growing(swap)
    cfg = TrackingCost(gate=20.0**2)  # generous gate for clear test
    tracks = track_snapshots(frames, cfg=cfg)

    assert len(tracks) == 2, f"expected 2 tracks, got {len(tracks)}"
    for tr in tracks:
        assert len(tr.frames) == len(
            frames
        ), f"track {tr.mt_id} only has {len(tr.frames)}/{len(frames)} frames"

    # Each track's tip_a should be a smooth trajectory: max single-frame
    # jump bounded by typical motion (here ~3 px including any noise).
    for tr in tracks:
        for i in range(1, len(tr.frames)):
            jump_a = np.linalg.norm(
                tr.frames[i].tip_a - tr.frames[i - 1].tip_a
            )
            jump_b = np.linalg.norm(
                tr.frames[i].tip_b - tr.frames[i - 1].tip_b
            )
            assert (
                jump_a < 3.0
            ), f"track {tr.mt_id} tip_a jumped {jump_a:.2f}px at f{i}"
            assert (
                jump_b < 3.0
            ), f"track {tr.mt_id} tip_b jumped {jump_b:.2f}px at f{i}"


def test_plus_minus_labelled_by_dynamics():
    swap = [False, False, False, False, False]
    frames = _two_mts_growing(swap)
    tracks = track_snapshots(frames)

    for tr in tracks:
        which_plus = label_plus_minus(tr)
        plus_history = (
            tr.tip_a_history if which_plus == "A" else tr.tip_b_history
        )
        minus_history = (
            tr.tip_b_history if which_plus == "A" else tr.tip_a_history
        )
        from kapoorlabs_mtrack.track.profile import _path_length

        plus_path = _path_length(plus_history)
        minus_path = _path_length(minus_history)
        assert plus_path > minus_path, (
            f"track {tr.mt_id}: plus path {plus_path:.2f} should exceed "
            f"minus {minus_path:.2f}"
        )


def test_length_profiles_grow_monotonically():
    swap = [False, False, False, False, False]
    frames = _two_mts_growing(swap)
    tracks = track_snapshots(frames)
    profiles = build_length_profiles(tracks)

    assert len(profiles) == 2
    for p in profiles:
        # tip_distance grows over time because plus moves and minus is fixed.
        assert np.all(
            np.diff(p.tip_distance) > 0
        ), f"mt {p.mt_id} tip_distance not monotonic: {p.tip_distance}"


def test_intensity_cost_can_discriminate_when_positions_collide():
    """Two MTs in nearly the same place but very different amplitudes.

    Distance-only would assign arbitrarily; intensity term should
    pick the right partners.
    """
    f0 = _FrameSnap(
        0,
        mts=[
            _Snap(
                0,
                1,
                0,
                np.array([10.0, 10.0]),
                np.array([20.0, 10.0]),
                amplitude=200.0,
                curvature=0.0,
                ds=0.7,
            ),
            _Snap(
                0,
                2,
                0,
                np.array([10.5, 10.5]),
                np.array([20.5, 10.5]),
                amplitude=50.0,
                curvature=0.0,
                ds=0.7,
            ),
        ],
    )
    f1 = _FrameSnap(
        1,
        mts=[
            # Permuted in input order -- amplitudes must match across frames.
            _Snap(
                1,
                3,
                0,
                np.array([10.7, 10.7]),
                np.array([20.7, 10.7]),
                amplitude=51.0,
                curvature=0.0,
                ds=0.7,
            ),
            _Snap(
                1,
                4,
                0,
                np.array([10.2, 10.2]),
                np.array([20.2, 10.2]),
                amplitude=199.0,
                curvature=0.0,
                ds=0.7,
            ),
        ],
    )

    cfg = TrackingCost(
        enable_intensity=True,
        weights={
            "distance": 1.0,
            "intensity": 5.0,
            "curvature": 0.5,
            "ds": 0.2,
        },
        amp_scale=20.0,
        gate=20.0**2,
    )
    tracks = track_snapshots([f0, f1], cfg=cfg)
    assert len(tracks) == 2, f"expected 2 tracks, got {len(tracks)}"

    # The track that started with amplitude 200 should still have
    # amplitude near 199 at frame 1, not near 51.
    by_initial_amp = sorted(tracks, key=lambda t: t.frames[0].amplitude)
    low_amp_track, high_amp_track = by_initial_amp
    assert abs(low_amp_track.frames[1].amplitude - 51.0) < 5.0
    assert abs(high_amp_track.frames[1].amplitude - 199.0) < 5.0


if __name__ == "__main__":
    test_two_mts_persist_identity_across_frames()
    test_plus_minus_labelled_by_dynamics()
    test_length_profiles_grow_monotonically()
    test_intensity_cost_can_discriminate_when_positions_collide()
    print("\nOK")
