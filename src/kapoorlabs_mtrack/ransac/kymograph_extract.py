"""Extract a ``(time, position)`` point cloud from a kymograph image.

Threshold (Otsu by default) and skeletonise the kymograph, then
return the on-pixel coordinates as ``(y=time, x=position)`` tuples
ready for the RANSAC drivers. No ``vollseg`` / ``stardist`` dependency
-- pure scikit-image. Pass a precomputed binary mask via ``mask`` if
you have a better segmentation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize


def extract_kymograph_points(
    kymograph: np.ndarray,
    mask: Optional[np.ndarray] = None,
    do_skeletonize: bool = True,
    min_intensity_fraction: float = 0.05,
) -> np.ndarray:
    """Return ``(N, 2)`` array of ``(time_row, position_col)`` points.

    Args:
        kymograph: 2-D image, axis 0 = time, axis 1 = position.
        mask: optional precomputed binary mask. If ``None``, an Otsu
            threshold is applied to ``kymograph``.
        do_skeletonize: if True (default), skeletonise the mask so
            the RANSAC fits the trajectory's midline rather than the
            blob interior.
        min_intensity_fraction: rows with no skeleton pixels above
            this fraction of the global max are ignored (kills hot
            outliers near the image border).
    """
    img = np.asarray(kymograph, dtype=float)
    if mask is None:
        thr = threshold_otsu(img)
        bin_ = img > thr
    else:
        bin_ = np.asarray(mask, dtype=bool)

    if do_skeletonize:
        bin_ = skeletonize(bin_)

    coords = np.argwhere(bin_)  # (N, 2) in (row, col) = (time, position)
    if coords.size == 0:
        return coords

    if min_intensity_fraction > 0:
        vals = img[coords[:, 0], coords[:, 1]]
        thr = min_intensity_fraction * img.max()
        coords = coords[vals >= thr]
    return coords.astype(float)
