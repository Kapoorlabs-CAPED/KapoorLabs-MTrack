"""Multi-microtubule movie generator for tracker / kymograph demos.

Each microtubule is described by:

- ``start_xy`` (fixed minus-tip position; doesn't move over time)
- ``end_xy0`` (plus-tip position at frame 0)
- ``vel_xy`` (per-frame plus-tip velocity; ``(0, 0)`` for a static MT)
- ``ds`` (curve step length; passed through to the model)
- ``curvature`` (3rd-order spline coefficient)
- ``amplitude`` (peak photon coefficient)

At each timepoint we render every MT with the single-MT
spline-Gaussian model and sum the contributions, add a shared
background, then apply Poisson + read noise. The synthetic label
image is built from dilated rasterisations of each MT's swept curve
-- so MTs whose paths cross at any frame share a single label, which
is exactly what triggers the joint-fit path in
:func:`pipeline.fit_stack.fit_stack`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from skimage.morphology import dilation, disk

from .synthetic import add_shot_noise, render_curve_image


@dataclass
class MTRecipe:
    """Truth recipe for a single microtubule across a movie."""

    start_xy: np.ndarray  # (2,) fixed minus-tip
    end_xy0: np.ndarray  # (2,) plus-tip at frame 0
    vel_xy: np.ndarray  # (2,) plus-tip velocity per frame
    ds: float = 0.7
    curvature: float = 0.0
    amplitude: float = 100.0
    label_id: int | None = None  # if None, assigned during generation


def _rasterise_segment(
    mask: np.ndarray, p0: np.ndarray, p1: np.ndarray
) -> None:
    """Plot a thin line p0 -> p1 onto ``mask`` (in place)."""
    h, w = mask.shape
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) * 2)))
    for t in np.linspace(0.0, 1.0, n + 1):
        p = p0 * (1 - t) + p1 * t
        rr, cc = int(round(p[1])), int(round(p[0]))
        if 0 <= rr < h and 0 <= cc < w:
            mask[rr, cc] = True


def generate_movie(
    recipes: Sequence[MTRecipe],
    shape: tuple[int, int],
    n_frames: int,
    sigma: tuple[float, float] = (1.6, 1.6),
    background: float = 6.0,
    read_noise_sigma: float = 2.0,
    label_dilation: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, list[list[MTRecipe]]]:
    """Render a 2-D + time microtubule movie.

    Returns:
        raw_stack: ``(T, H, W)`` noisy raw image stack (float).
        label_stack: ``(T, H, W)`` integer label image. Labels are
            assigned per-frame so a crossing pair occupies a single
            label when their rasterisations touch after dilation.
        truth_per_frame: per-frame list of :class:`MTRecipe` instances
            with their plus tip at the time-evolved position. Use to
            check fit / tracker accuracy.
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma_arr = np.asarray(sigma, dtype=float)
    b = 1.0 / (sigma_arr * sigma_arr)

    T = n_frames
    raw_stack = np.empty((T,) + shape, dtype=float)
    label_stack = np.zeros((T,) + shape, dtype=np.int32)
    truth_per_frame: list[list[MTRecipe]] = []

    for t in range(T):
        scene = np.full(shape, background, dtype=float)
        # Per-MT masks; merge into label_stack so touching MTs share an id.
        binary_masks = []
        recipes_t: list[MTRecipe] = []
        for r in recipes:
            end_xy = r.end_xy0 + r.vel_xy * t
            recipes_t.append(
                MTRecipe(
                    start_xy=r.start_xy.copy(),
                    end_xy0=end_xy.copy(),
                    vel_xy=r.vel_xy.copy(),
                    ds=r.ds,
                    curvature=r.curvature,
                    amplitude=r.amplitude,
                    label_id=r.label_id,
                )
            )
            a9 = np.array(
                [
                    r.start_xy[0],
                    r.start_xy[1],
                    end_xy[0],
                    end_xy[1],
                    r.ds,
                    r.curvature,
                    0.0,
                    r.amplitude,
                    0.0,
                ]
            )
            # Enforce model contract: start.x < end.x for the renderer.
            if a9[2] - a9[0] <= 1e-6:
                a9[[0, 2]] = a9[[2, 0]]
                a9[[1, 3]] = a9[[3, 1]]
            scene += render_curve_image(a9, b, shape)

            mask = np.zeros(shape, dtype=bool)
            _rasterise_segment(mask, r.start_xy, end_xy)
            mask = dilation(mask, disk(label_dilation))
            binary_masks.append(mask)

        # Build label image: connected components of the union of all
        # masks. Touching masks merge into one label, which is how the
        # downstream pipeline detects crossings.
        union = np.zeros(shape, dtype=bool)
        for m in binary_masks:
            union |= m
        from skimage.measure import label as cc_label

        label_stack[t] = cc_label(union, connectivity=2).astype(np.int32)
        raw_stack[t] = add_shot_noise(
            scene, read_noise_sigma=read_noise_sigma, rng=rng
        )
        truth_per_frame.append(recipes_t)

    return raw_stack, label_stack, truth_per_frame
