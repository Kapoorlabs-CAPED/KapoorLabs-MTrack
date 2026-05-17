# KapoorLabs-MTrack

[![License BSD-3](https://img.shields.io/pypi/l/KapoorLabs-MTrack.svg?color=green)](https://github.com/Kapoorlabs-CAPED/KapoorLabs-MTrack/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/KapoorLabs-MTrack.svg?color=green)](https://pypi.org/project/KapoorLabs-MTrack)
[![Python Version](https://img.shields.io/pypi/pyversions/KapoorLabs-MTrack.svg?color=green)](https://python.org)
[![tests](https://github.com/Kapoorlabs-CAPED/KapoorLabs-MTrack/workflows/tests/badge.svg)](https://github.com/Kapoorlabs-CAPED/KapoorLabs-MTrack/actions)
[![codecov](https://codecov.io/gh/Kapoorlabs-CAPED/KapoorLabs-MTrack/branch/main/graph/badge.svg)](https://codecov.io/gh/Kapoorlabs-CAPED/KapoorLabs-MTrack)

Python port of MTrack — model-based sub-pixel localisation of microtubule
endpoints. Models the microtubule as a 3rd-order polynomial curve, swept
by a sum of anisotropic Gaussians matching the imaging PSF, fitted
against the observed image with Levenberg–Marquardt. Deprecates the
original Fiji / imglib2 plugin.

---

## The model

A microtubule between fixed start point $(x_0, y_0)$ and end point
$(x_1, y_1)$ is represented as a cubic curve

$$y(x) = y_0 + m\,x + C\,x^2 + I\,x^3$$

where the slope $m$ is chosen so the polynomial passes through both
endpoints given $C$ and $I$. The curve is sampled at step length $ds$
along its tangent; at each sample the imaging PSF is an anisotropic 2-D
Gaussian:

$$g(\mathbf{x}; \boldsymbol{\gamma}) = \exp\!\Big(-\sum_d b_d\,(x_d - \gamma_d)^2\Big)$$

with $b_d = 1/\sigma_d^2$ where $\sigma_d$ is the microscope PSF along
axis $d$. The observed intensity at pixel $\mathbf{x}$ is

$$I(\mathbf{x}) = A\,\Big[g(\mathbf{x};\,\text{start}) + \sum_{t}\,g(\mathbf{x};\,\boldsymbol{\gamma}(t)) + g(\mathbf{x};\,\text{end})\Big] + B$$

i.e. amplitude $A$ times the sum of: a Gaussian at the start, one
Gaussian per walked-curve sample $\boldsymbol{\gamma}(t)$, and one at
the end; plus a constant background $B$.

### Parameter layout

The Levenberg–Marquardt solver refines a 9-vector `a`:

| Index | Symbol | Meaning | Status |
|-------|--------|---------|--------|
| `a[0..1]` | $(x_0, y_0)$ | start point | **free** |
| `a[2..3]` | $(x_1, y_1)$ | end point | **free** |
| `a[4]` | $ds$ | curve step length (|ds| used) | **free** |
| `a[5]` | $C$ | curvature (2nd-order coefficient) | **free** |
| `a[6]` | $I$ | inflection (3rd-order coefficient) | **free** |
| `a[7]` | $A$ | amplitude | **free** |
| `a[8]` | $B$ | background | **free** |

The PSF widths live in the fixed 2-vector `b = [1/σx², 1/σy²]`. **`b`
is a property of the microscope, not an optimisation target** — pass it
in once and the solver never touches it. The step length `ds` is fitted
because it controls how densely the curve is sampled and trades against
amplitude (a tighter sampling with smaller amp can model the same image
as a coarser one with larger amp).

This layout is identical to `GaussianSplinethirdorder.java` in the
original imglib2 implementation, so the hand-derived analytic gradients
port one-for-one.

---

## Quickstart

### Simulate a synthetic microtubule and fit it

```python
import numpy as np
from kapoorlabs_mtrack.simulate import render_curve_image, add_shot_noise
from kapoorlabs_mtrack.fit import fit_endpoints

# Microscope PSF — fixed, NOT fitted.
sigma = np.array([1.6, 1.6])
b = 1.0 / (sigma * sigma)

# Truth parameters of the microtubule we are about to simulate.
truth = np.array([
    8.0,   # start_x
    12.0,  # start_y
    42.0,  # end_x
    28.0,  # end_y
    0.7,   # ds (curve step length)
    0.005, # curvature
    0.0,   # inflection
    120.0, # amplitude
    8.0,   # background
])

# Render clean image + add Poisson + read noise.
clean = render_curve_image(truth, b, shape=(40, 50))
noisy = add_shot_noise(clean, read_noise_sigma=2.0,
                       rng=np.random.default_rng(42))

# Seed: in production this comes from a Hough / RANSAC line-finder.
# Endpoint seeds need to be within ~1 px of truth for the analytic LM.
seed = truth.copy()
seed[0:2] += [0.8, -0.6]
seed[2:4] += [-0.7, 0.9]
seed[5] = seed[6] = 0.0                      # shape unknowns
bg = float(np.median(noisy))
seed[8] = bg
seed[7] = max(1.0, (noisy.max() - bg) / 5.0)  # rough amp guess

# Poisson-style weighting for photon-count data.
weights = 1.0 / np.sqrt(np.clip(noisy, 1.0, None))

result = fit_endpoints(noisy, seed, b, weights=weights)
print("start:", result.start)   # ~ [8, 12]
print("end:",   result.end)     # ~ [42, 28]
print("cost:",  result.cost)
```

### Seeding the amplitude correctly

The model amplitude is the coefficient on the **sum** of Gaussians, not
the peak-pixel intensity. With step $ds$ and PSF $\sigma$, roughly
$2\sigma/ds$ overlapping Gaussians contribute to each on-curve pixel,
so peak ≈ $A \cdot 2\sigma/ds$. Seed `a[7]` with `(peak − bg) / (2σ/ds)`
rather than the raw peak — otherwise LM lands in a wider-shallower
local minimum.

---

## Package layout

```
src/kapoorlabs_mtrack/
├── models/                       # forward model + analytic Jacobian
│   ├── __init__.py
│   └── spline_third_order.py     # val, jac, walk_curve
├── verify/                       # analytic-vs-numeric gradient checks
│   ├── __init__.py
│   └── gradient_check.py         # numeric_jacobian, check_jacobian
├── simulate/                     # synthetic image generation
│   ├── __init__.py
│   └── synthetic.py              # render_curve_image, add_shot_noise
├── fit/                          # scipy LM wrapper
│   ├── __init__.py
│   └── lm.py                     # fit_endpoints, FitResult
└── _tests/
    ├── test_gradient.py          # analytic vs numeric per-parameter table
    └── test_pipeline.py          # simulate → fit → assert endpoints
```

---

## Gradients: analytic, hybrid, numeric

`fit_endpoints` takes a `jac_mode` argument with three settings:

| `jac_mode` | What it does | Speed | Endpoint bias |
|------------|--------------|-------|----------------|
| `"analytic"` (default) | Java-faithful: exact analytic columns for ds, curvature, inflection, amp, bg; **approximate** columns for the four endpoint coordinates (uses only the endpoint-Gaussian gradient, ignores how moving an endpoint shifts the swept curve). | fastest | ~1–2 px |
| `"hybrid"` | Analytic for shape parameters; central-difference (4 extra `val` calls per LM iteration) for the four endpoint coordinates. | ~1.3× analytic | sub-pixel |
| `"numeric"` | Central differences for every column. | slowest | sub-pixel |

Pick `analytic` when you are fitting thousands of microtubules and have
good seeds. Pick `hybrid` when sub-pixel endpoint localisation matters
more than a small speed hit. `numeric` exists as a reference for
verification.

---

## Verifying the analytic gradients

Whenever you touch the model, re-run the gradient check. It prints a
per-parameter analytic-vs-numeric report:

```python
from kapoorlabs_mtrack.models import val, jac
from kapoorlabs_mtrack.verify import check_jacobian

reports = check_jacobian(val, jac, x, a, b, eps=1e-5)
for r in reports:
    print(r)
```

Expected behaviour:

- **Shape parameters** (`ds`, `curvature`, `inflection`, `amplitude`,
  `background`) — analytic and numeric agree to round-off (per-column
  abs error / column norm $\lesssim 10^{-4}$).
- **Endpoint parameters** (`start_x/y`, `end_x/y`) — disagree by ~90%
  relative error. **This is expected** — the analytic columns drop the
  swept-curve coupling on purpose, exactly as the Java original does.
  If you see a regression here it means the *Java approximation* is
  broken, not the math; use `verify` to localise which column changed.

The shipped `_tests/test_gradient.py` codifies these expectations as
assertions.

---

## Levenberg–Marquardt details

The fitter wraps `scipy.optimize.least_squares`. Default settings:

- `method="trf"` (trust-region reflective, supports box bounds) when
  `bounds="auto"` (the default).
- `method="lm"` (plain LM, unconstrained) when `bounds=None` — closest
  to the original Java solver's behaviour.
- Auto-bounds keep `ds`, amplitude, and inflection in sane ranges and
  the endpoints inside the image; pass an explicit `(lo, hi)` tuple to
  override.
- `xtol = ftol = 1e-8`, `max_nfev = 500`.

The Jacobian is provided analytically (see `jac_mode` above) — scipy
will not recompute it numerically unless you select `jac_mode="numeric"`.

---

## Installation

```bash
pip install KapoorLabs-MTrack
```

Or for the development version:

```bash
pip install git+https://github.com/Kapoorlabs-CAPED/KapoorLabs-MTrack.git
```

## Contributing

Contributions are very welcome. Tests can be run with [tox]; please
ensure coverage at least stays the same before submitting a PR. If you
modify `models/spline_third_order.py`, the `test_gradient.py` check is
mandatory — that file is the contract that the analytic port still
matches the Java math.

## License

Distributed under the terms of the [BSD-3] license, KapoorLabs-MTrack is
free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a
detailed description.

[pip]: https://pypi.org/project/pip/
[caped]: https://github.com/Kapoorlabs-CAPED
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@caped]: https://github.com/Kapoorlabs-CAPED
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-template]: https://github.com/Kapoorlabs-CAPED/cookiecutter-template
[file an issue]: https://github.com/Kapoorlabs-CAPED/KapoorLabs-MTrack/issues
[tox]: https://tox.readthedocs.io/en/latest/
[PyPI]: https://pypi.org/
