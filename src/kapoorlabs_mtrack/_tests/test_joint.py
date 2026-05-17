"""Two-microtubule joint fit: simulate a crossing, fit jointly, recover both.

The synthetic scene contains two microtubules whose curves intersect
inside the crop -- exactly the case where a single segmentation label
would contain both. We render the sum of their forward models, add
shot + read noise, then run the joint LM fitter with seeds perturbed
by ~1 px and verify each pair of endpoints is recovered.
"""

from __future__ import annotations

import numpy as np

from kapoorlabs_mtrack.fit import fit_endpoints_joint
from kapoorlabs_mtrack.models import multi, val
from kapoorlabs_mtrack.simulate import add_shot_noise


def _render_joint(
    a_concat: np.ndarray, b: np.ndarray, n_mt: int, shape: tuple[int, int]
) -> np.ndarray:
    """Render the joint-N-MT model on a (H, W) grid."""
    h, w = shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([jj, ii], axis=-1).astype(float)  # (H, W, 2)
    return multi.val(coords, a_concat, b, n_mt)


def _match_unordered(fit_starts, fit_ends, truth_starts, truth_ends):
    """Match fitted MTs to truth MTs by minimum endpoint distance.

    Because the joint model is symmetric under permutation of the MT
    blocks, we have to figure out which fit MT corresponds to which
    truth MT before computing per-MT errors. Returns (perm, errors)
    where ``perm[i]`` is the fit-index assigned to truth-index ``i``
    and ``errors`` is a list of (start_err, end_err) per truth MT.
    """
    n = len(truth_starts)
    from itertools import permutations

    best_perm = None
    best_total = np.inf
    for perm in permutations(range(n)):
        total = 0.0
        for i, j in enumerate(perm):
            total += np.linalg.norm(fit_starts[j] - truth_starts[i])
            total += np.linalg.norm(fit_ends[j] - truth_ends[i])
        if total < best_total:
            best_total = total
            best_perm = perm
    errors = []
    for i, j in enumerate(best_perm):
        errors.append(
            (
                float(np.linalg.norm(fit_starts[j] - truth_starts[i])),
                float(np.linalg.norm(fit_ends[j] - truth_ends[i])),
            )
        )
    return best_perm, errors


def test_joint_fit_recovers_two_crossing_mts():
    rng = np.random.default_rng(7)
    sigma = np.array([1.6, 1.6])
    b = 1.0 / (sigma * sigma)

    # Two MTs crossing inside a 50x60 crop.
    mt1 = np.array([5.0, 8.0, 45.0, 32.0, 0.7, 0.0, 0.0, 100.0])
    mt2 = np.array([5.0, 30.0, 45.0, 10.0, 0.7, 0.0, 0.0, 110.0])
    truth = multi.pack([mt1, mt2], bg=8.0)
    n_mt = 2
    shape = (40, 50)

    clean = _render_joint(truth, b, n_mt, shape)
    noisy = add_shot_noise(clean, read_noise_sigma=2.0, rng=rng)

    # Seed: perturb each endpoint by < 1 px, zero curvature/inflection,
    # rough amp / bg.
    seed_mt1 = mt1.copy()
    seed_mt1[0:2] += np.array([0.6, -0.4])
    seed_mt1[2:4] += np.array([-0.5, 0.7])
    seed_mt2 = mt2.copy()
    seed_mt2[0:2] += np.array([-0.4, 0.5])
    seed_mt2[2:4] += np.array([0.6, -0.6])
    bg_guess = float(np.median(noisy))
    seed_mt1[7] = max(1.0, (noisy.max() - bg_guess) / 5.0)
    seed_mt2[7] = max(1.0, (noisy.max() - bg_guess) / 5.0)
    seed = multi.pack([seed_mt1, seed_mt2], bg=bg_guess)

    weights = 1.0 / np.sqrt(np.clip(noisy, 1.0, None))
    result = fit_endpoints_joint(
        noisy, seed, b, n_mt=n_mt, weights=weights, jac_mode="hybrid"
    )

    truth_starts = [mt1[0:2], mt2[0:2]]
    truth_ends = [mt1[2:4], mt2[2:4]]
    perm, errors = _match_unordered(
        result.starts, result.ends, truth_starts, truth_ends
    )

    print()
    print(
        f"joint converged: {result.success}  nfev={result.nfev}  cost={result.cost:.2f}"
    )
    print(f"bg fit={result.background:.2f}  truth=8.0")
    print(f"perm (truth->fit) = {perm}")
    for i, (se, ee) in enumerate(errors):
        print(f"  MT{i}: start_err={se:.3f} px  end_err={ee:.3f} px")

    assert result.success
    # Crossings inherently lose ~1 px precision at the intersection where
    # the two MTs' Gaussians overlap.
    for i, (se, ee) in enumerate(errors):
        assert se < 1.5, f"MT{i} start off by {se:.3f} px"
        assert ee < 1.5, f"MT{i} end off by {ee:.3f} px"


def test_joint_fit_single_mt_matches_single_fitter():
    """N=1 joint fit must reduce to the single-MT model (sanity check)."""
    sigma = np.array([1.5, 1.5])
    b = 1.0 / (sigma * sigma)
    a8 = np.array([6.0, 9.0, 30.0, 22.0, 0.7, 0.005, 0.0, 100.0])
    bg = 6.0
    a_joint = multi.pack([a8], bg=bg)

    # Compare model val at a probe pixel: joint(N=1) == single
    x = np.array([[12.0, 14.0], [20.0, 17.0]])
    a_single = np.concatenate([a8, [bg]])
    v_single = val(x, a_single, b)
    v_joint = multi.val(x, a_joint, b, n_mt=1)
    assert np.allclose(v_single, v_joint, atol=1e-10)


if __name__ == "__main__":
    test_joint_fit_single_mt_matches_single_fitter()
    test_joint_fit_recovers_two_crossing_mts()
    print("\nOK")
