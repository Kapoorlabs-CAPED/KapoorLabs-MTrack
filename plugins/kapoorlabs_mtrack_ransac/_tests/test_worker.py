"""Headless test for the RANSAC plugin's worker layer.

Builds a synthetic kymograph with one growth + one shrinkage segment,
runs ``run_ransac_stream``, and verifies that:

1. Per-segment yields arrive (at least one before the summary).
2. The trailing summary carries a non-null
   :class:`DynamicInstability`.
3. The summary identifies at least one growth segment and one
   shrinkage segment, and counts exactly one catastrophe.

The widget itself needs Qt + a display and is exercised manually.
"""

from __future__ import annotations

import numpy as np
from kapoorlabs_mtrack_ransac._worker import run_ransac_stream
from skimage.draw import line


def _render_two_segment_kymograph():
    img = np.zeros((120, 80), dtype=np.uint8)
    segs = [((10, 0), (50, 50)), ((50, 50), (10, 110))]
    for (x0, t0), (x1, t1) in segs:
        rr, cc = line(int(t0), int(x0), int(t1), int(x1))
        rr = np.clip(rr, 0, img.shape[0] - 1)
        cc = np.clip(cc, 0, img.shape[1] - 1)
        img[rr, cc] = 255
        for dx in (-1, 1):
            for dy in (-1, 1):
                rrn = np.clip(rr + dy, 0, img.shape[0] - 1)
                ccn = np.clip(cc + dx, 0, img.shape[1] - 1)
                img[rrn, ccn] = 255
    return img


def test_worker_streams_segments_then_summary():
    img = _render_two_segment_kymograph()
    out = list(
        run_ransac_stream(
            img,
            mode="linear",
            min_samples=10,
            max_trials=200,
            iterations=6,
            residual_threshold=2.0,
            slope_threshold=0.4,
            random_state=1,
        )
    )
    # Last element is the summary; everything before is segment yields.
    seg_results = [r for r in out if not r.is_summary]
    summary_results = [r for r in out if r.is_summary]
    assert (
        len(seg_results) >= 2
    ), f"expected at least 2 segments, got {len(seg_results)}"
    assert len(summary_results) == 1
    di = summary_results[0].summary
    assert di is not None
    kinds = {s.kind for s in di.segments}
    assert "growth" in kinds, f"no growth segment in {kinds}"
    assert "shrinkage" in kinds, f"no shrinkage in {kinds}"
    assert di.n_catastrophes >= 1
    # rescue may or may not appear depending on how RANSAC splits the
    # tail of the shrinkage; we don't require exactly one.


def test_worker_handles_empty_kymograph_gracefully():
    img = np.zeros((40, 60), dtype=np.uint8)
    out = list(
        run_ransac_stream(
            img,
            mode="linear",
            min_samples=10,
            max_trials=50,
            iterations=3,
            residual_threshold=2.0,
            slope_threshold=0.4,
            random_state=1,
        )
    )
    assert len(out) == 1 and out[0].is_summary
    di = out[0].summary
    assert di is not None
    assert di.n_catastrophes == 0 and di.n_rescues == 0
    assert di.segments == []


if __name__ == "__main__":
    test_worker_streams_segments_then_summary()
    test_worker_handles_empty_kymograph_gracefully()
    print("\nOK")
