"""End-to-end smoke test: simulate -> fit -> assert endpoints recovered.

This is the golden-path test for the math core. A synthetic microtubule
image is rendered from known truth parameters, Poisson + read noise is
added, the fit is seeded with perturbed endpoints, and we check that
LM recovers the original start/end within sub-pixel tolerance.
"""

from __future__ import annotations

import numpy as np

from kapoorlabs_mtrack.fit import fit_endpoints
from kapoorlabs_mtrack.simulate import add_shot_noise, render_curve_image


def test_simulate_and_fit_recovers_endpoints():
    rng = np.random.default_rng(42)

    sigma = np.array([1.6, 1.6])
    b = 1.0 / (sigma * sigma)
    truth = np.array(
        [
            8.0,  # start x
            12.0,  # start y
            42.0,  # end   x
            28.0,  # end   y
            0.7,  # ds
            0.005,  # curvature
            0.0,  # inflection
            120.0,  # amplitude (peak photons)
            8.0,  # background
        ]
    )

    image_shape = (40, 50)  # (H, W)
    clean = render_curve_image(truth, b, image_shape)
    noisy = add_shot_noise(clean, read_noise_sigma=2.0, rng=rng)

    # Seed the way the real pipeline does: a Hough / RANSAC stage gives
    # endpoints already within ~1 px of truth. Curvature / inflection /
    # amplitude / background are rough guesses. The amplitude is the
    # coefficient on the *sum* of Gaussians (peak pixel ~ amp * 2*sigma/ds
    # for overlapping steps), so seed with peak / 5 not raw peak.
    seed = truth.copy()
    seed[0:2] += np.array([0.8, -0.6])  # perturb start ~1 px
    seed[2:4] += np.array([-0.7, 0.9])  # perturb end ~1 px
    seed[5] = 0.0  # curvature unknown
    seed[6] = 0.0  # inflection unknown
    bg_guess = float(np.median(noisy))
    seed[8] = bg_guess
    seed[7] = max(1.0, (noisy.max() - bg_guess) / 5.0)

    # Poisson-style weighting: 1 / sqrt(I+1) makes the loss approximately
    # log-likelihood for photon-count data and is markedly more robust to
    # an imperfect amplitude / background seed than unit weights.
    weights = 1.0 / np.sqrt(np.clip(noisy, 1.0, None))
    result = fit_endpoints(noisy, seed, b, weights=weights)

    print()
    print(
        f"converged: {result.success}  nfev={result.nfev}  cost={result.cost:.2f}"
    )
    print(f"truth  start = {truth[0:2]}   end = {truth[2:4]}")
    print(f"seed   start = {seed[0:2]}   end = {seed[2:4]}")
    print(f"fit    start = {result.start}   end = {result.end}")
    start_err = np.linalg.norm(result.start - truth[0:2])
    end_err = np.linalg.norm(result.end - truth[2:4])
    print(f"|start_err|={start_err:.3f} px   |end_err|={end_err:.3f} px")

    assert result.success
    # Analytic-mode endpoint bias is ~1-2 px (Java-faithful approximation
    # ignores the swept-curve coupling to the endpoints).
    assert start_err < 2.0, f"start endpoint off by {start_err:.3f} px"
    assert end_err < 2.0, f"end endpoint off by {end_err:.3f} px"


def test_simulate_and_fit_recovers_endpoints_hybrid():
    """Same simulation, hybrid jacobian -> sub-pixel endpoint recovery."""
    rng = np.random.default_rng(42)
    sigma = np.array([1.6, 1.6])
    b = 1.0 / (sigma * sigma)
    truth = np.array([8.0, 12.0, 42.0, 28.0, 0.7, 0.005, 0.0, 120.0, 8.0])
    image_shape = (40, 50)
    clean = render_curve_image(truth, b, image_shape)
    noisy = add_shot_noise(clean, read_noise_sigma=2.0, rng=rng)

    seed = truth.copy()
    seed[0:2] += np.array([0.8, -0.6])
    seed[2:4] += np.array([-0.7, 0.9])
    seed[5] = 0.0
    seed[6] = 0.0
    bg_guess = float(np.median(noisy))
    seed[8] = bg_guess
    seed[7] = max(1.0, (noisy.max() - bg_guess) / 5.0)
    weights = 1.0 / np.sqrt(np.clip(noisy, 1.0, None))

    result = fit_endpoints(noisy, seed, b, weights=weights, jac_mode="hybrid")

    start_err = np.linalg.norm(result.start - truth[0:2])
    end_err = np.linalg.norm(result.end - truth[2:4])
    print()
    print(
        f"[hybrid] converged: {result.success}  nfev={result.nfev}  cost={result.cost:.2f}"
    )
    print(f"[hybrid] fit  start={result.start}  end={result.end}")
    print(
        f"[hybrid] |start_err|={start_err:.3f} px  |end_err|={end_err:.3f} px"
    )

    assert result.success
    assert start_err < 0.5, f"hybrid start off by {start_err:.3f} px"
    assert end_err < 0.5, f"hybrid end off by {end_err:.3f} px"


if __name__ == "__main__":
    test_simulate_and_fit_recovers_endpoints()
    test_simulate_and_fit_recovers_endpoints_hybrid()
    print("\nOK")
