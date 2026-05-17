"""Skeleton-based endpoint extraction and pairing.

Given a binary mask of one segmentation label, we want:

1. How many microtubules are inside it (``N``).
2. A pair of seed endpoint coordinates for each microtubule.

The recipe:

- Skeletonise the mask.
- Find skeleton endpoints (pixels with exactly one skeleton neighbour
  in 8-connectivity). A clean single MT has 2 endpoints; ``N`` crossing
  MTs typically expose ``2N`` skeleton endpoints around the crop
  perimeter.
- Pair endpoints into ``(start, end)`` tuples by **entry-tangent
  alignment**: for each endpoint we walk ``K`` skeleton pixels along
  its arm (away from the endpoint) and record the unit tangent that
  enters the skeleton interior. Two endpoints belong to the same
  microtubule when their tangents are roughly **antiparallel** -- the
  arm of MT_i leaves endpoint A pointing inward, continues through
  any junctions, and emerges at endpoint B pointing inward in the
  opposite global direction. The pair cost is ``1 + dot(t_A, t_B)``
  (zero for perfect continuation, two for same-direction tangents),
  and we pick the matching with the smallest total cost. This is
  robust to the multi-pixel junction zones that ``skeletonize``
  produces on dilated masks, which broke a path-integrated angle
  cost.

Coordinates returned in **image XY convention**: ``(x, y) = (column,
row)`` so they slot directly into the model's ``a[0..1]`` /
``a[2..3]`` slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from skimage.morphology import skeletonize

# 8-connectivity neighbour offsets.
_NEI8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class RegionSeeds:
    """Output of :func:`region_seeds_from_label` for one label region."""

    n_mt: int
    starts: np.ndarray  # (N, 2) in (x, y) order
    ends: np.ndarray  # (N, 2) in (x, y) order
    status: str  # "ok" or "skip:<reason>"


def _skeleton_endpoints_and_junctions(
    skel: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return lists of (row, col) for endpoints (degree 1) and junctions (degree >= 3)."""
    h, w = skel.shape
    endpoints: list[tuple[int, int]] = []
    junctions: list[tuple[int, int]] = []
    coords = np.argwhere(skel)
    for r, c in coords:
        deg = 0
        for dr, dc in _NEI8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and skel[rr, cc]:
                deg += 1
        if deg == 1:
            endpoints.append((int(r), int(c)))
        elif deg >= 3:
            junctions.append((int(r), int(c)))
    return endpoints, junctions


def _entry_tangent(
    skel: np.ndarray,
    endpoint: tuple[int, int],
    junctions: set[tuple[int, int]],
    max_steps: int = 5,
) -> np.ndarray:
    """Walk inward from ``endpoint`` along the skeleton arm and return
    the unit vector ``(end_of_walk - endpoint)`` in (row, col) order.

    The walk stops at a junction (skeleton pixel with degree >= 3) or
    after ``max_steps`` steps, whichever comes first. The returned
    tangent therefore captures the local arm direction, ignoring the
    multi-pixel ambiguity of the junction zone itself.
    """
    h, w = skel.shape
    prev = None
    cur = endpoint
    end_of_walk = endpoint
    for _ in range(max_steps):
        # Step to the unique skeleton neighbour that isn't ``prev``.
        next_pix = None
        for dr, dc in _NEI8:
            rr, cc = cur[0] + dr, cur[1] + dc
            if 0 <= rr < h and 0 <= cc < w and skel[rr, cc]:
                if (rr, cc) != prev:
                    next_pix = (rr, cc)
                    break
        if next_pix is None or next_pix in junctions:
            end_of_walk = next_pix if next_pix is not None else cur
            break
        prev = cur
        cur = next_pix
        end_of_walk = cur
    v = np.array(
        [
            float(end_of_walk[0] - endpoint[0]),
            float(end_of_walk[1] - endpoint[1]),
        ]
    )
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.array([0.0, 0.0])
    return v / norm


def _best_pairing(
    endpoints: list[tuple[int, int]],
    skel: np.ndarray,
    junctions: list[tuple[int, int]],
    max_steps: int = 5,
) -> list[tuple[int, int]]:
    """Pair endpoints by tangent alignment.

    Cost of pairing endpoints ``i`` and ``j`` is
    ``1 + dot(t_i, t_j)`` where ``t_i`` is the inward unit tangent at
    endpoint ``i``. Antiparallel tangents (collinear arms through a
    junction = same MT) give cost 0; parallel tangents (both arms
    pointing the same way = different MTs) give cost 2.
    """
    n_eps = len(endpoints)
    assert n_eps >= 2 and n_eps % 2 == 0

    junc_set = set(junctions)
    tangents = [
        _entry_tangent(skel, ep, junc_set, max_steps) for ep in endpoints
    ]

    pair_cost: dict[tuple[int, int], float] = {}
    for ii in range(n_eps):
        for jj in range(ii + 1, n_eps):
            pair_cost[(ii, jj)] = 1.0 + float(
                np.dot(tangents[ii], tangents[jj])
            )

    def _enumerate(remaining: list[int]) -> Iterable[list[tuple[int, int]]]:
        if not remaining:
            yield []
            return
        first = remaining[0]
        for partner_pos in range(1, len(remaining)):
            partner = remaining[partner_pos]
            rest = remaining[1:partner_pos] + remaining[partner_pos + 1 :]
            for tail in _enumerate(rest):
                yield [(first, partner)] + tail

    best_cost = np.inf
    best_match: list[tuple[int, int]] = []
    for matching in _enumerate(list(range(n_eps))):
        total = 0.0
        for a, c in matching:
            i, j = (a, c) if a < c else (c, a)
            total += pair_cost[(i, j)]
        if total < best_cost:
            best_cost = total
            best_match = matching
    return best_match


def _rc_to_xy(rc: tuple[int, int]) -> np.ndarray:
    """(row, col) skeleton-pixel → (x, y) image coord for the model."""
    r, c = rc
    return np.array([float(c), float(r)])


def region_seeds_from_label(
    mask: np.ndarray, tangent_steps: int = 5
) -> RegionSeeds:
    """Extract endpoint seeds from a binary mask of one label region.

    Args:
        mask: 2-D bool / 0-1 array isolating a single segmentation label.
        tangent_steps: how many skeleton pixels to walk inward from each
            endpoint when computing the entry tangent used for pairing.
            Larger values smooth out local skeleton wiggles but risk
            running into / past the junction zone.

    Returns: :class:`RegionSeeds`. If the region's skeleton yields an
    odd number of endpoints, or fewer than 2 (a blob with no clear
    line), the result has ``status`` starting with ``"skip:"`` and the
    caller should not fit it.
    """
    binary = mask.astype(bool)
    skel = skeletonize(binary)
    endpoints, junctions = _skeleton_endpoints_and_junctions(skel)
    n_eps = len(endpoints)

    if n_eps < 2:
        return RegionSeeds(
            0,
            np.empty((0, 2)),
            np.empty((0, 2)),
            status=f"skip:not-a-line(eps={n_eps})",
        )
    if n_eps % 2 != 0:
        return RegionSeeds(
            0,
            np.empty((0, 2)),
            np.empty((0, 2)),
            status=f"skip:odd-endpoints(eps={n_eps})",
        )

    n_mt = n_eps // 2
    if n_mt == 1:
        a, b = endpoints
        # Ensure start.x <= end.x for the model's left-to-right walk.
        p1 = _rc_to_xy(a)
        p2 = _rc_to_xy(b)
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        return RegionSeeds(1, p1[None, :], p2[None, :], status="ok")

    pairs = _best_pairing(endpoints, skel, junctions, max_steps=tangent_steps)
    starts, ends = [], []
    for i, j in pairs:
        p1 = _rc_to_xy(endpoints[i])
        p2 = _rc_to_xy(endpoints[j])
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        starts.append(p1)
        ends.append(p2)
    return RegionSeeds(
        n_mt=n_mt,
        starts=np.array(starts),
        ends=np.array(ends),
        status="ok",
    )
