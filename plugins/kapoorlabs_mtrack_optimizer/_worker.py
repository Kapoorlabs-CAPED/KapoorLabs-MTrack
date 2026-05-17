"""Generator wrappers around the core pipeline for live progress in napari.

``fit_stack`` and ``track_snapshots`` in the core are batch APIs --
they only return after every frame is done. In a GUI we want to render
each frame's endpoints as soon as they're fitted, and update the tip
trajectories as the tracker links them. These generators do exactly
that: they replicate the core's logic but ``yield`` after each frame
so a ``napari.qt.threading.thread_worker`` can stream updates back to
the main thread.

Keeping the live-progress glue here, rather than inside ``pipeline``
and ``track``, lets the core stay batch-shaped and avoids a
dependency on Qt / napari from the optimizer.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
from skimage.measure import regionprops

from kapoorlabs_mtrack.fit import fit_endpoints, fit_endpoints_joint
from kapoorlabs_mtrack.models import multi
from kapoorlabs_mtrack.pipeline.fit_stack import (
    FrameSnapshot,
    MTSnapshot,
    _bbox_with_pad,
    _seed_from_region,
)
from kapoorlabs_mtrack.pipeline.skeleton import region_seeds_from_label
from kapoorlabs_mtrack.track import (
    TrackingCost,
    build_length_profiles,
    track_snapshots,
)


def fit_stack_stream(
    raw_stack: np.ndarray,
    label_stack: np.ndarray,
    sigma: tuple[float, float],
    pad: Optional[int] = None,
    ds_seed: float = 0.7,
    jac_mode: str = "hybrid",
) -> Iterator[FrameSnapshot]:
    """Same outputs as :func:`pipeline.fit_stack.fit_stack`, yielded per frame."""
    sigma = np.asarray(sigma, dtype=float)
    b = 1.0 / (sigma * sigma)
    if pad is None:
        pad = int(np.ceil(3.0 * float(sigma.max())))

    T = raw_stack.shape[0]
    for t in range(T):
        raw = raw_stack[t]
        labels = label_stack[t]
        mts: list[MTSnapshot] = []
        skipped: list[tuple[int, str]] = []
        for prop in regionprops(labels):
            label_id = int(prop.label)
            bbox = _bbox_with_pad(prop.bbox, pad, raw.shape)
            r0, c0, r1, c1 = bbox
            crop_raw = raw[r0:r1, c0:c1].astype(float)
            crop_mask = labels[r0:r1, c0:c1] == label_id
            seeds = region_seeds_from_label(crop_mask)
            if not seeds.status.startswith("ok"):
                skipped.append((label_id, seeds.status))
                continue

            seed_packed, bg_guess, amp_guess = _seed_from_region(
                crop_raw, seeds, ds_seed
            )
            weights = 1.0 / np.sqrt(np.clip(crop_raw, 1.0, None))

            try:
                if seeds.n_mt == 1:
                    a9 = np.concatenate(
                        [seed_packed[: multi.PER_MT], [bg_guess]]
                    )
                    fit = fit_endpoints(
                        crop_raw, a9, b, weights=weights, jac_mode=jac_mode
                    )
                    per_mt_blocks = [fit.a[: multi.PER_MT]]
                    bg_fit = float(fit.a[-1])
                    cost = fit.cost
                    success = fit.success
                else:
                    fit = fit_endpoints_joint(
                        crop_raw,
                        seed_packed,
                        b,
                        n_mt=seeds.n_mt,
                        weights=weights,
                        jac_mode=jac_mode,
                    )
                    per_mt_blocks = fit.per_mt
                    bg_fit = fit.background
                    cost = fit.cost
                    success = fit.success
            except Exception as exc:
                skipped.append(
                    (label_id, f"skip:fit-error({type(exc).__name__})")
                )
                continue

            if not success:
                skipped.append((label_id, "skip:fit-not-converged"))
                continue

            for k, a8 in enumerate(per_mt_blocks):
                start = np.array([a8[0] + c0, a8[1] + r0])
                end = np.array([a8[2] + c0, a8[3] + r0])
                mts.append(
                    MTSnapshot(
                        frame=t,
                        label=label_id,
                        mt_in_label=k,
                        n_mt_in_label=seeds.n_mt,
                        start=start,
                        end=end,
                        ds=float(a8[4]),
                        curvature=float(a8[5]),
                        inflection=float(a8[6]),
                        amplitude=float(a8[7]),
                        background=bg_fit,
                        fit_cost=cost,
                        status="ok",
                    )
                )
        yield FrameSnapshot(frame=t, mts=mts, skipped_labels=skipped)


def track_and_profile(snapshots, cfg: TrackingCost, max_gap: int = 0):
    """Batch track + length-profile from a complete list of FrameSnapshots."""
    tracks = track_snapshots(snapshots, cfg=cfg, max_gap=max_gap)
    profiles = build_length_profiles(tracks)
    return tracks, profiles
