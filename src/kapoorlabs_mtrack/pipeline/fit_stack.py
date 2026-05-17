"""Per-frame, per-label microtubule endpoint fitting orchestrator.

Takes a raw + label TIFF pair (already loaded as ``(T, H, W)`` arrays
by :mod:`io.tif`) and produces, for every frame and every label, a list
of fitted microtubule snapshots. The output is consumed by the tracker
(``track/``) and the CSV writer (``io.tif.save_endpoints_csv``).

Per label region:

- Skeletonise the binary mask; count endpoints to decide ``N`` (the
  number of microtubules sharing the region).
- Crop the raw image to the label's bounding box, padded by a few PSF
  widths so the Gaussian tails are inside the crop.
- Build seeds from the skeleton endpoints (in image coords) and a
  rough amplitude / background guess from the crop intensity stats.
- Single MT -> :func:`fit.fit_endpoints`; ``N >= 2`` -> joint fit via
  :func:`fit.fit_endpoints_joint`. Coordinates are translated back to
  the full-image frame before being recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from skimage.measure import regionprops

from ..fit import fit_endpoints, fit_endpoints_joint
from ..models import multi
from .skeleton import RegionSeeds, region_seeds_from_label


@dataclass
class MTSnapshot:
    """One fitted microtubule at one timepoint."""

    frame: int
    label: int
    mt_in_label: int
    n_mt_in_label: int
    start: np.ndarray  # (2,) in image (x, y)
    end: np.ndarray  # (2,) in image (x, y)
    ds: float
    curvature: float
    inflection: float
    amplitude: float
    background: float
    fit_cost: float
    status: str  # "ok" | "skip:<reason>"


@dataclass
class FrameSnapshot:
    """All microtubule snapshots for one frame."""

    frame: int
    mts: list[MTSnapshot]
    skipped_labels: list[tuple[int, str]]  # (label_id, reason)


def _bbox_with_pad(
    bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Inflate a (minr, minc, maxr, maxc) bbox by ``pad`` pixels, clipped to image."""
    h, w = shape
    minr, minc, maxr, maxc = bbox
    return (
        max(0, minr - pad),
        max(0, minc - pad),
        min(h, maxr + pad),
        min(w, maxc + pad),
    )


def _seed_from_region(
    crop_raw: np.ndarray, seeds: RegionSeeds, ds_seed: float
) -> tuple[np.ndarray, float, float]:
    """Build a packed parameter vector from skeleton seeds + crop stats."""
    bg = float(np.median(crop_raw))
    amp = max(
        1.0, (float(crop_raw.max()) - bg) / 5.0
    )  # see README amp seeding note
    per_mt = []
    for i in range(seeds.n_mt):
        a8 = np.array(
            [
                seeds.starts[i, 0],
                seeds.starts[i, 1],
                seeds.ends[i, 0],
                seeds.ends[i, 1],
                ds_seed,
                0.0,  # curvature
                0.0,  # inflection
                amp,
            ]
        )
        per_mt.append(a8)
    packed = multi.pack(per_mt, bg=bg)
    return packed, bg, amp


def fit_stack(
    raw_stack: np.ndarray,
    label_stack: np.ndarray,
    sigma: tuple[float, float] | np.ndarray,
    pad: Optional[int] = None,
    ds_seed: float = 0.7,
    jac_mode: str = "hybrid",
) -> list[FrameSnapshot]:
    """Fit microtubules across a ``(T, H, W)`` stack.

    Args:
        raw_stack: ``(T, H, W)`` raw image.
        label_stack: ``(T, H, W)`` integer label image, one label per
            microtubule (or per crossing cluster of microtubules).
        sigma: PSF widths ``(sigma_x, sigma_y)`` in pixels. Fixed
            microscope property (see ``models/__init__.py``).
        pad: padding (px) around each label bbox before fitting. If
            ``None``, uses ``ceil(3 * max(sigma))`` so the Gaussian
            tails are inside the crop.
        ds_seed: initial value for the curve step length parameter.
        jac_mode: ``"analytic"`` / ``"hybrid"`` / ``"numeric"`` -- see
            :func:`fit.fit_endpoints` for the speed / accuracy tradeoff.
    """
    if raw_stack.shape != label_stack.shape:
        raise ValueError(
            f"raw and label shapes differ: {raw_stack.shape} vs {label_stack.shape}"
        )
    sigma = np.asarray(sigma, dtype=float)
    b = 1.0 / (sigma * sigma)
    if pad is None:
        pad = int(np.ceil(3.0 * float(sigma.max())))

    T = raw_stack.shape[0]
    snapshots: list[FrameSnapshot] = []
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

            # Translate skeleton seeds (crop-local) into crop coords.
            # region_seeds_from_label already returns (x, y) in the
            # crop's own frame, so we just record offsets to map back.
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
            except Exception as exc:  # one bad region shouldn't kill the run
                skipped.append(
                    (label_id, f"skip:fit-error({type(exc).__name__})")
                )
                continue

            if not success:
                skipped.append((label_id, "skip:fit-not-converged"))
                continue

            for k, a8 in enumerate(per_mt_blocks):
                # Translate crop-local (x, y) back to full-image (x, y).
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
        snapshots.append(
            FrameSnapshot(frame=t, mts=mts, skipped_labels=skipped)
        )
    return snapshots


def snapshots_to_csv_rows(snapshots: list[FrameSnapshot]) -> list[dict]:
    """Flatten :func:`fit_stack` output into rows for ``save_endpoints_csv``."""
    rows: list[dict] = []
    for fs in snapshots:
        for m in fs.mts:
            rows.append(
                dict(
                    frame=m.frame,
                    label=m.label,
                    mt_in_label=m.mt_in_label,
                    n_mt_in_label=m.n_mt_in_label,
                    start_x=m.start[0],
                    start_y=m.start[1],
                    end_x=m.end[0],
                    end_y=m.end[1],
                    ds=m.ds,
                    curvature=m.curvature,
                    inflection=m.inflection,
                    amplitude=m.amplitude,
                    background=m.background,
                    fit_cost=m.fit_cost,
                    status=m.status,
                )
            )
        for label_id, reason in fs.skipped_labels:
            rows.append(
                dict(
                    frame=fs.frame,
                    label=label_id,
                    mt_in_label=0,
                    n_mt_in_label=0,
                    start_x="",
                    start_y="",
                    end_x="",
                    end_y="",
                    ds="",
                    curvature="",
                    inflection="",
                    amplitude="",
                    background="",
                    fit_cost="",
                    status=reason,
                )
            )
    return rows
