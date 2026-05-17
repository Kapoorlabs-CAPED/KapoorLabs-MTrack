"""Headless tests for the plugin's worker layer.

We avoid importing the widget (which needs Qt) -- this test only
exercises ``_worker.fit_stack_stream`` and ``track_and_profile``,
which are pure-python wrappers over the core. A full
``pytest-qt``-driven widget test belongs in the plugin install matrix
and is left for the next push.
"""

from __future__ import annotations

import numpy as np
from kapoorlabs_mtrack_optimizer._worker import (
    fit_stack_stream,
    track_and_profile,
)

from kapoorlabs_mtrack.simulate import MTRecipe, generate_movie
from kapoorlabs_mtrack.track import TrackingCost


def test_fit_stream_yields_per_frame():
    rng = np.random.default_rng(2)
    recipes = [
        MTRecipe(
            start_xy=np.array([10.0, 14.0]),
            end_xy0=np.array([30.0, 22.0]),
            vel_xy=np.array([0.5, 0.0]),
            amplitude=100.0,
        ),
    ]
    raw, lab, _truth = generate_movie(
        recipes,
        shape=(50, 70),
        n_frames=3,
        sigma=(1.6, 1.6),
        background=6.0,
        read_noise_sigma=2.0,
        rng=rng,
    )

    seen = []
    for fs in fit_stack_stream(raw, lab, sigma=(1.6, 1.6), jac_mode="hybrid"):
        seen.append(fs.frame)
    assert seen == [0, 1, 2], f"expected frames 0,1,2; got {seen}"

    # Confirm at least one MT was fit per frame.
    snapshots = list(
        fit_stack_stream(raw, lab, sigma=(1.6, 1.6), jac_mode="hybrid")
    )
    assert all(len(fs.mts) >= 1 for fs in snapshots)


def test_track_and_profile_returns_compatible_objects():
    rng = np.random.default_rng(3)
    recipes = [
        MTRecipe(
            start_xy=np.array([8.0, 14.0]),
            end_xy0=np.array([28.0, 22.0]),
            vel_xy=np.array([1.0, 0.0]),
            amplitude=110.0,
        ),
    ]
    raw, lab, _t = generate_movie(
        recipes,
        shape=(50, 70),
        n_frames=4,
        sigma=(1.6, 1.6),
        rng=rng,
    )
    snapshots = list(
        fit_stack_stream(raw, lab, sigma=(1.6, 1.6), jac_mode="hybrid")
    )
    tracks, profiles = track_and_profile(
        snapshots, cfg=TrackingCost(gate=30.0**2)
    )
    assert tracks, "tracker should produce at least one track"
    assert len(profiles) == len(tracks)
    p = profiles[0]
    assert p.frames.size >= 2
    assert p.plus_xy.shape == (p.frames.size, 2)


if __name__ == "__main__":
    test_fit_stream_yields_per_frame()
    test_track_and_profile_returns_compatible_objects()
    print("\nOK")
