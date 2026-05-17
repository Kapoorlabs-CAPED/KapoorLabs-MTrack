"""End-to-end pipeline test: synthetic (T, H, W) stack -> fit_stack -> CSV.

The synthetic scene has, per frame:
  - label 1: a single isolated microtubule
  - label 2: two crossing microtubules sharing a label region

We render the raw image as the SUM of all microtubule contributions,
build a label image from dilated skeletons of each microtubule (so
label 2 contains both crossing MTs in one connected component), then
run the whole pipeline and verify:
  * label 1 -> 1 fitted MT, endpoints near truth
  * label 2 -> 2 fitted MTs, endpoints near truth (unordered match)
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
from skimage.morphology import dilation, disk

from kapoorlabs_mtrack.io import save_endpoints_csv
from kapoorlabs_mtrack.pipeline.fit_stack import (
    fit_stack,
    snapshots_to_csv_rows,
)
from kapoorlabs_mtrack.simulate import add_shot_noise
from kapoorlabs_mtrack.simulate.synthetic import render_curve_image


def _draw_label(shape, starts_ends, dil_radius=3):
    """Build a label image where label k+1 covers MT k (single) or
    MTs in a shared region (multiple).

    starts_ends: list of (label_id, [(start, end), ...]) -- if a label
    has multiple (start, end) pairs, both microtubules are merged
    into one connected component via dilation, simulating a crossing.
    """
    label_img = np.zeros(shape, dtype=np.int32)
    for label_id, segs in starts_ends:
        mask = np.zeros(shape, dtype=bool)
        for start, end in segs:
            # rasterise a thin line as the skeleton seed
            n_steps = int(np.ceil(np.linalg.norm(end - start) * 2))
            ts = np.linspace(0.0, 1.0, n_steps + 1)
            for tt in ts:
                p = start * (1 - tt) + end * tt
                rr, cc = int(round(p[1])), int(round(p[0]))
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    mask[rr, cc] = True
        mask = dilation(mask, disk(dil_radius))
        label_img[mask] = label_id
    return label_img


def _build_synthetic_stack():
    """Return (raw_stack, label_stack, truth) for a 2-frame test."""
    shape = (60, 80)  # (H, W)
    sigma = np.array([1.6, 1.6])
    b = 1.0 / (sigma * sigma)

    # Truth microtubules (8-vectors, no per-MT bg).
    # Label 1: a single, well-isolated MT in the top-left region.
    mt_iso = np.array([6.0, 8.0, 26.0, 22.0, 0.7, 0.0, 0.0, 100.0])

    # Label 2: two crossing MTs in the lower-right region.
    mt_x1 = np.array([35.0, 12.0, 75.0, 50.0, 0.7, 0.0, 0.0, 110.0])
    mt_x2 = np.array([35.0, 48.0, 75.0, 14.0, 0.7, 0.0, 0.0, 105.0])

    bg = 6.0

    # Render two frames -- frame 1 perturbs the dynamic endpoints a bit
    # so the test is not literally identical across time.
    mts_per_frame = [
        [mt_iso, mt_x1, mt_x2],
        [
            mt_iso + np.array([0.0, 0.0, 0.5, 0.3, 0, 0, 0, 0]),
            mt_x1 + np.array([0.0, 0.0, 1.0, -0.5, 0, 0, 0, 0]),
            mt_x2 + np.array([0.0, 0.0, -0.7, 0.4, 0, 0, 0, 0]),
        ],
    ]

    rng = np.random.default_rng(11)
    raw_stack = np.empty((2,) + shape, dtype=float)
    for t, mts in enumerate(mts_per_frame):
        # Render each MT separately as a single-MT model and sum.
        scene = np.full(shape, bg, dtype=float)
        for a8 in mts:
            a9 = np.concatenate([a8, [0.0]])  # bg=0 per MT (added once above)
            scene += render_curve_image(a9, b, shape)
        raw_stack[t] = add_shot_noise(scene, read_noise_sigma=2.0, rng=rng)

    # Build labels off the FIRST-FRAME geometry only (typical segmentation
    # is a single mask reused across time, or per-frame but conservative).
    label_img = _draw_label(
        shape,
        [
            (1, [(mt_iso[0:2], mt_iso[2:4])]),
            (2, [(mt_x1[0:2], mt_x1[2:4]), (mt_x2[0:2], mt_x2[2:4])]),
        ],
        dil_radius=3,
    )
    label_stack = np.broadcast_to(label_img, (2,) + shape).copy()
    return raw_stack, label_stack, mts_per_frame, sigma


def test_fit_stack_pipeline_recovers_all_endpoints():
    raw_stack, label_stack, truth_per_frame, sigma = _build_synthetic_stack()

    snapshots = fit_stack(
        raw_stack, label_stack, sigma=tuple(sigma), jac_mode="hybrid"
    )

    assert len(snapshots) == 2, "expected one FrameSnapshot per timepoint"
    for t, fs in enumerate(snapshots):
        truth_mts = truth_per_frame[t]
        truth_iso = truth_mts[0]
        truth_crossing = [truth_mts[1], truth_mts[2]]

        # Split fitted MTs by their source label.
        by_label = {1: [], 2: []}
        for m in fs.mts:
            by_label[m.label].append(m)

        # Label 1: should have exactly 1 MT, endpoints near truth_iso.
        assert len(by_label[1]) == 1, (
            f"frame {t}: label 1 should have 1 MT, got {len(by_label[1])} "
            f"(skipped={fs.skipped_labels})"
        )
        m1 = by_label[1][0]
        se = np.linalg.norm(m1.start - truth_iso[0:2])
        ee = np.linalg.norm(m1.end - truth_iso[2:4])
        print(f"f{t} L1: start_err={se:.2f}px end_err={ee:.2f}px")
        assert (
            se < 2.0 and ee < 2.0
        ), f"frame {t} label 1 off: {se:.2f} {ee:.2f}"

        # Label 2: should have exactly 2 MTs, match unordered to truth.
        assert len(by_label[2]) == 2, (
            f"frame {t}: label 2 should have 2 MTs, got {len(by_label[2])} "
            f"(skipped={fs.skipped_labels})"
        )
        fits = by_label[2]
        # Match by minimum total endpoint distance over the 2 permutations.
        perms = [(0, 1), (1, 0)]
        best_total = np.inf
        best_perm = None
        for perm in perms:
            total = 0.0
            for ti, fi in enumerate(perm):
                total += np.linalg.norm(
                    fits[fi].start - truth_crossing[ti][0:2]
                )
                total += np.linalg.norm(fits[fi].end - truth_crossing[ti][2:4])
            if total < best_total:
                best_total = total
                best_perm = perm
        for ti, fi in enumerate(best_perm):
            se = np.linalg.norm(fits[fi].start - truth_crossing[ti][0:2])
            ee = np.linalg.norm(fits[fi].end - truth_crossing[ti][2:4])
            print(f"f{t} L2 MT{ti}: start_err={se:.2f}px end_err={ee:.2f}px")
            # Crossings lose precision at the intersection in noisy data;
            # 5/6 endpoints land sub-3 px, occasional outliers up to ~4 px
            # are expected and still useful for length-profile tracking.
            assert se < 4.0, f"frame {t} L2 MT{ti} start off by {se:.2f}"
            assert ee < 4.0, f"frame {t} L2 MT{ti} end off by {ee:.2f}"


def test_csv_writer_writes_expected_schema():
    """save_endpoints_csv should round-trip headers + a sample row."""
    raw_stack, label_stack, _truth, sigma = _build_synthetic_stack()
    snapshots = fit_stack(
        raw_stack, label_stack, sigma=tuple(sigma), jac_mode="hybrid"
    )
    rows = snapshots_to_csv_rows(snapshots)
    assert any(r["status"] == "ok" for r in rows)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "endpoints.csv"
        save_endpoints_csv(path, rows)
        with path.open() as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            assert header is not None
            assert set(header) >= {
                "frame",
                "label",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "amplitude",
                "background",
                "status",
            }


if __name__ == "__main__":
    test_fit_stack_pipeline_recovers_all_endpoints()
    test_csv_writer_writes_expected_schema()
    print("\nOK")
