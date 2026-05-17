"""Thread-worker glue: extract -> RANSAC -> classify -> dynamic instability.

Wraps :mod:`kapoorlabs_mtrack.ransac` for the napari plugin so each
extracted segment can be streamed back to the GUI thread as soon as
it's available. The core stays batch-shaped (no Qt dep); this module
is the live-progress adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from kapoorlabs_mtrack.ransac import (
    ComboRansac,
    DynamicInstability,
    LinearFunction,
    QuadraticFunction,
    Ransac,
    classify_segments,
    dynamic_instability,
    extract_kymograph_points,
)


def segment_kymograph(
    kymograph: np.ndarray,
    source: str,
    pretrained_name: str | None = None,
    custom_path: str | None = None,
    user_mask: np.ndarray | None = None,
    n_tiles: tuple = (1, 1),
) -> np.ndarray | None:
    """Dispatch a segmentation source to a binary mask.

    Sources:

    - ``"otsu"`` -- return ``None``; caller falls through to
      ``extract_kymograph_points``'s built-in Otsu + skeletonise path.
    - ``"vollseg_pretrained"`` -- run the named vollseg UNET model.
      Requires ``pip install KapoorLabs-MTrack[vollseg]``.
    - ``"vollseg_custom"`` -- load a vollseg model from a directory.
    - ``"user_mask"`` -- accept ``user_mask`` as the segmentation,
      after casting to ``bool`` + shape-checking.
    """
    if source == "otsu":
        return None
    if source == "user_mask":
        if user_mask is None:
            raise ValueError(
                "source='user_mask' but no user_mask was provided"
            )
        m = np.asarray(user_mask).astype(bool)
        if m.shape != kymograph.shape:
            raise ValueError(
                f"user_mask shape {m.shape} != kymograph {kymograph.shape}"
            )
        return m
    if source == "vollseg_pretrained":
        from kapoorlabs_mtrack.ransac.vollseg_segment import (
            segment_kymograph_pretrained,
        )

        return segment_kymograph_pretrained(
            kymograph, model_name=pretrained_name, n_tiles=n_tiles
        )
    if source == "vollseg_custom":
        from kapoorlabs_mtrack.ransac.vollseg_segment import (
            segment_kymograph_custom,
        )

        return segment_kymograph_custom(
            kymograph, model_dir=custom_path, n_tiles=n_tiles
        )
    raise ValueError(f"unknown segmentation source: {source!r}")


@dataclass
class RansacRunResult:
    """Per-segment intermediate result yielded by :func:`run_ransac_stream`."""

    segment_index: int
    inliers: np.ndarray  # (N, 2) in (time, position)
    slope_y_over_x: float  # raw RANSAC slope in y/x units
    intercept: float
    t_start: float
    t_end: float
    # When the run finishes, the worker yields a final FINAL_RESULT
    # sentinel with these summary fields set.
    summary: Optional[DynamicInstability] = None
    is_summary: bool = False


def run_ransac_stream(
    kymograph: np.ndarray,
    *,
    mode: str = "linear",  # "linear" or "combo"
    min_samples: int = 10,
    max_trials: int = 200,
    iterations: int = 8,
    residual_threshold: float = 2.0,
    slope_threshold: float = 0.4,
    mask: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> Iterator[RansacRunResult]:
    """Extract points, run RANSAC, yield segments one at a time, then summary.

    For ``mode="combo"`` the two passes are completed before any
    segments are yielded (the quadratic peel must finish first); for
    ``mode="linear"`` segments are yielded as they come out of the
    sequential extractor.
    """
    pts = extract_kymograph_points(kymograph, mask=mask)
    if pts.shape[0] < min_samples + 1:
        # Nothing to fit; yield an empty summary.
        di = dynamic_instability([])
        yield RansacRunResult(
            segment_index=-1,
            inliers=pts,
            slope_y_over_x=0.0,
            intercept=0.0,
            t_start=0.0,
            t_end=0.0,
            summary=di,
            is_summary=True,
        )
        return

    if mode == "combo":
        cr = ComboRansac(
            data_points=pts.tolist(),
            model_linear=LinearFunction,
            model_quadratic=QuadraticFunction,
            min_samples=min_samples,
            max_trials=max_trials,
            iterations=iterations,
            residual_threshold=residual_threshold,
            timeindex=0,
            random_state=random_state,
        )
        estimators, inliers = cr.extract_multiple_lines()
    else:
        rs = Ransac(
            data_points=pts.tolist(),
            model_class=LinearFunction,
            degree=2,
            min_samples=min_samples,
            max_trials=max_trials,
            iterations=iterations,
            residual_threshold=residual_threshold,
            timeindex=0,
            random_state=random_state,
        )
        estimators, inliers = rs.extract_multiple_lines()

    # Stream the per-segment results.
    for i, (est, inl) in enumerate(zip(estimators, inliers)):
        if est is None or inl is None or len(inl) == 0:
            continue
        m = est.get_coefficients(0) if hasattr(est, "coeff") else 0.0
        b = est.get_coefficients(1) if hasattr(est, "coeff") else 0.0
        arr = np.asarray(inl)
        ts = arr[:, 0]
        yield RansacRunResult(
            segment_index=i,
            inliers=arr,
            slope_y_over_x=float(m),
            intercept=float(b),
            t_start=float(ts.min()),
            t_end=float(ts.max()),
            is_summary=False,
        )

    # Final summary.
    segments = classify_segments(
        estimators, inliers, timeindex=0, slope_threshold=slope_threshold
    )
    di = dynamic_instability(segments)
    yield RansacRunResult(
        segment_index=-1,
        inliers=np.empty((0, 2)),
        slope_y_over_x=0.0,
        intercept=0.0,
        t_start=0.0,
        t_end=0.0,
        summary=di,
        is_summary=True,
    )
