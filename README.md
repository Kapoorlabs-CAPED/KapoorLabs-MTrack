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

**End goal**: per-microtubule length profiles over time, separated into
plus-end and minus-end trajectories. The pipeline here delivers stage 1
(per-frame, per-MT endpoints from a TIFF stack with segmentation
labels); the tracker that links those endpoints into trajectories with
a Hungarian + two-frame velocity cost is the next push.

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

## End-to-end on a TIFF stack

For real data the pipeline takes a raw TIFF and a label TIFF (one
integer per microtubule, shared across timepoints if you have a single
segmentation), and produces a per-microtubule, per-frame CSV of fitted
endpoints. Both inputs may be 2-D `(H, W)` or 2-D + time `(T, H, W)`.

### As a script

```bash
python scripts/fit_endpoints.py \
    --raw     movie_raw.tif \
    --labels  movie_labels.tif \
    --sigma   1.6 1.6 \
    --out     endpoints.csv \
    --jac-mode hybrid
```

### As a library call

```python
from kapoorlabs_mtrack.io import load_pair, save_endpoints_csv
from kapoorlabs_mtrack.pipeline import fit_stack
from kapoorlabs_mtrack.pipeline.fit_stack import snapshots_to_csv_rows

raw, labels = load_pair("movie_raw.tif", "movie_labels.tif")
snapshots = fit_stack(raw.array, labels.array, sigma=(1.6, 1.6),
                      jac_mode="hybrid")

# snapshots is list[FrameSnapshot] -- one per timepoint, each with
# a list of fitted microtubules (.mts) and any skipped labels.
for fs in snapshots:
    for m in fs.mts:
        print(fs.frame, m.label, m.mt_in_label, m.start, m.end, m.fit_cost)

save_endpoints_csv("endpoints.csv", snapshots_to_csv_rows(snapshots))
```

### Per-label workflow

For each `(frame, label)`:

1. The label's bounding box is cropped with `pad = ceil(3 * max(σ))` so
   the Gaussian tails are inside the crop.
2. The cropped binary mask is skeletonised (`scikit-image`'s
   `skeletonize`). The number of skeleton endpoints determines how many
   microtubules share this region: 2 → single, 4 → two crossing,
   6 → three, etc. **Odd counts skip the region** with status
   `skip:odd-endpoints(eps=N)`.
3. Endpoints are paired into `(start, end)` tuples by **entry-tangent
   alignment** (see "Endpoint pairing" below).
4. A seed `a` is built: start/end from the skeleton, `ds = 0.7`,
   curvature/inflection = 0, amplitude from `(peak − bg) / 5`, bg from
   the crop median.
5. Single MT → `fit.fit_endpoints`; two-or-more → `fit.fit_endpoints_joint`
   with the per-MT 8-vectors packed via `models.multi.pack`.
6. Output coordinates are translated from crop-local back to full-image
   `(x, y)`.

### CSV schema (`endpoints.csv`)

| Column | Meaning |
|---|---|
| `frame` | timepoint index (0 for a 2-D input) |
| `label` | source segmentation label id |
| `mt_in_label` | 0..N-1 index within a joint-fit region (0 for single MTs) |
| `n_mt_in_label` | N microtubules detected in this label |
| `start_x`, `start_y` | fitted start coordinate in full-image pixels |
| `end_x`, `end_y` | fitted end coordinate in full-image pixels |
| `ds`, `curvature`, `inflection`, `amplitude` | refined model parameters |
| `background` | shared background for the region (same value across N MTs of one joint fit) |
| `fit_cost` | final `0.5 * Σ residuals²` from scipy |
| `status` | `ok` for fitted rows, `skip:<reason>` for regions that were not fitted |

The exact column list is in
`kapoorlabs_mtrack.io.tif.ENDPOINT_CSV_COLUMNS`. The tracker (next
push) consumes this schema directly.

---

## Joint fits for crossings / overlapping microtubules

When a single label region contains multiple microtubules (X- or
T-crossing, or partial overlap), we fit them jointly. The joint model
(`models.multi`) packs `N` microtubules into one parameter vector and
shares **one background scalar** across the crop:

```
a_concat[0..7]     MT 1: start_x, start_y, end_x, end_y, ds, curv, infl, amp
a_concat[8..15]    MT 2: same fields
...
a_concat[-1]       shared background
```

Length = `8N + 1`. Use `models.multi.pack(per_mt_8vecs, bg)` to build
this vector, and `fit.fit_endpoints_joint(crop, seed, b, n_mt=N)` to
fit it. The analytic Jacobian is block-diagonal in the per-MT columns
plus a column of `1`s for the background, so the speed vs accuracy
tradeoff (`jac_mode`) carries over identically.

**Accuracy at crossings.** Endpoints inherently lose ~1–2 px precision
at the pixel where two MTs' Gaussian envelopes overlap most strongly.
The shipped `test_joint.py` confirms 3 of 4 endpoints sub-pixel and 1
endpoint at ~1 px on a clean two-MT crossing; on noisy crop boundaries
this can grow to ~3–4 px. The downstream tracker is designed to
absorb this — the length profile is robust even with a few-pixel
endpoint jitter.

### Endpoint pairing at junctions

With `N ≥ 2` microtubules in one label, the skeleton has `2N`
endpoints around the region perimeter — but no labels telling us which
endpoint belongs to which MT. We pair them by **entry-tangent
alignment**:

1. From each endpoint, walk `K=5` skeleton pixels inward along the arm.
   Stop early if a junction (degree ≥ 3) is hit.
2. The unit vector from `endpoint` → `end_of_walk` is the entry
   tangent, capturing the local arm direction *before* it gets lost in
   the multi-pixel junction zone.
3. Pair cost: `1 + dot(t_i, t_j)`. Antiparallel tangents (the same MT
   continues from `i` through the junction to `j`) → cost 0. Parallel
   tangents (different MTs pointing the same way) → cost 2.
4. Enumerate all perfect matchings of endpoints into pairs and pick the
   one with minimum total cost. For `N=2` there are 3 matchings; for
   `N=3`, 15. Fine for hand-segmented data.

We tried a path-integrated angle-change cost first; it failed because
`skeletonize` of a dilated mask produces a thick, multi-pixel junction
zone whose accumulated wiggle saturates the angle integral. The
entry-tangent approach sidesteps the junction entirely. See
`pipeline/skeleton.py` for the rationale.

---

## Package layout

```
KapoorLabs-MTrack/
├── src/kapoorlabs_mtrack/
│   ├── models/                       # forward model + analytic Jacobian
│   │   ├── __init__.py
│   │   ├── spline_third_order.py     # single-MT: val, jac, walk_curve
│   │   └── multi.py                  # joint N-MT model (shared bg)
│   ├── verify/                       # analytic-vs-numeric gradient checks
│   │   ├── __init__.py
│   │   └── gradient_check.py         # numeric_jacobian, check_jacobian
│   ├── simulate/                     # synthetic image generation
│   │   ├── __init__.py
│   │   └── synthetic.py              # render_curve_image, add_shot_noise
│   ├── fit/                          # scipy LM wrappers
│   │   ├── __init__.py
│   │   ├── lm.py                     # fit_endpoints (single MT)
│   │   └── joint.py                  # fit_endpoints_joint (N MTs in crop)
│   ├── io/                           # TIFF I/O for raw + label pairs
│   │   ├── __init__.py
│   │   └── tif.py                    # load_pair, save_endpoints_csv
│   ├── pipeline/                     # orchestration: stack → snapshots
│   │   ├── __init__.py
│   │   ├── skeleton.py               # region_seeds_from_label
│   │   └── fit_stack.py              # fit_stack, snapshots_to_csv_rows
│   └── _tests/
│       ├── test_gradient.py          # analytic vs numeric per-parameter table
│       ├── test_pipeline.py          # simulate → fit → assert endpoints
│       ├── test_joint.py             # two crossing MTs jointly fit
│       └── test_fit_stack.py         # full (T,H,W) pipeline end-to-end
├── scripts/                          # CLI tools that import the package
│   └── fit_endpoints.py              # raw + label TIFF → endpoints.csv
└── notebooks/                        # interactive walk-throughs
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

## Roadmap

| Stage | Status | Module |
|---|---|---|
| 1. Single-MT forward model + analytic Jacobian | ✅ shipped | `models.spline_third_order` |
| 2. Gradient verifier (analytic vs numeric) | ✅ shipped | `verify.gradient_check` |
| 3. Synthetic image generator | ✅ shipped | `simulate.synthetic` |
| 4. Single-MT LM fitter (scipy) | ✅ shipped | `fit.lm` |
| 5. Joint N-MT model + fitter (crossings / overlaps) | ✅ shipped | `models.multi`, `fit.joint` |
| 6. TIFF I/O for raw + label pairs | ✅ shipped | `io.tif` |
| 7. Skeleton-based per-region seeding + endpoint pairing | ✅ shipped | `pipeline.skeleton` |
| 8. Stack orchestrator (per-frame, per-label fits → CSV) | ✅ shipped | `pipeline.fit_stack`, `scripts/fit_endpoints.py` |
| 9. Hungarian tracker (cost: distance + intensity + curvature, 2-frame velocity prediction, gating, plus/minus labelling) | ⏳ next | `track/` |
| 10. Length-profile output (per-MT plus-end and minus-end length-vs-time) | ⏳ after tracker | |
| 11. Multi-frame synthetic movie generator (for tracker tests) | ⏳ alongside tracker | `scripts/simulate_movie.py` |
| 12. Interactive notebooks driving the full chain | ⏳ alongside tracker | `notebooks/` |

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
