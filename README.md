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
plus-end and minus-end trajectories, with per-MT kymographs. The
pipeline runs in three stages:

```
TIFF stack + label TIFF
        │
        │  scripts/fit_endpoints.py     (stage 8)
        ▼
endpoints.csv                           per-frame, per-MT model fits
        │
        │  scripts/track_endpoints.py   (stages 9-10)
        ▼
tracks.csv + length_profiles.csv        plus/minus tip trajectories
        │
        │  notebooks/02_*.ipynb         (visualisation)
        ▼
length plots + kymographs               per microtubule
```

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
│   │   ├── synthetic.py              # render_curve_image, add_shot_noise
│   │   └── movie.py                  # multi-MT (T,H,W) movies + labels
│   ├── fit/                          # scipy LM wrappers
│   │   ├── __init__.py
│   │   ├── lm.py                     # fit_endpoints (single MT)
│   │   └── joint.py                  # fit_endpoints_joint (N MTs in crop)
│   ├── io/                           # TIFF I/O + CSV writers
│   │   ├── __init__.py
│   │   └── tif.py                    # load_pair, save_endpoints_csv,
│   │                                 # save_length_profiles_csv
│   ├── pipeline/                     # orchestration: stack → snapshots
│   │   ├── __init__.py
│   │   ├── skeleton.py               # region_seeds_from_label
│   │   └── fit_stack.py              # fit_stack, snapshots_to_csv_rows
│   ├── track/                        # link snapshots into per-MT tracks
│   │   ├── __init__.py
│   │   ├── cost.py                   # TrackingCost, mt_to_mt_cost
│   │   ├── hungarian.py              # track_snapshots, MTTrack
│   │   ├── profile.py                # build_length_profiles, label_plus_minus
│   │   └── kymograph.py              # build_kymograph
│   ├── ransac/                       # kymograph-level RANSAC + DI analysis
│   │   ├── __init__.py
│   │   ├── models.py                 # LinearFunction, QuadraticFunction, ...
│   │   ├── fits.py                   # Ransac, ComboRansac
│   │   ├── dynamics.py               # Segment, classify_segments,
│   │   │                             # dynamic_instability
│   │   └── kymograph_extract.py      # extract_kymograph_points
│   └── _tests/
│       ├── test_gradient.py          # analytic vs numeric per-parameter table
│       ├── test_pipeline.py          # simulate → fit → assert endpoints
│       ├── test_joint.py             # two crossing MTs jointly fit
│       ├── test_fit_stack.py         # full (T,H,W) pipeline end-to-end
│       ├── test_track.py             # tracker identity + plus/minus + intensity
│       └── test_ransac.py            # RANSAC math + dynamic-instability counts
├── scripts/                          # CLI tools that import the package
│   ├── fit_endpoints.py              # raw + label TIFF → endpoints.csv
│   └── track_endpoints.py            # endpoints.csv → tracks + length profiles
├── notebooks/                        # interactive walk-throughs
│   ├── 01_simulate_and_fit.ipynb     # one MT: simulate, fit, gradient check
│   └── 02_real_data_pipeline.ipynb   # multi-MT movie → fit → track → kymograph
└── plugins/                          # optional install via pip ...[napari]
    ├── kapoorlabs_mtrack_optimizer/  # napari widget (Input | Fit | Track | Results)
    │   ├── napari.yaml               # plugin manifest
    │   ├── _widget.py                # tabbed magicgui UI
    │   ├── _worker.py                # streaming fit_stack for live overlays
    │   ├── _reader.py                # TIFF reader for raw + label pairs
    │   └── _tests/test_worker.py     # headless smoke test
    └── kapoorlabs_mtrack_ransac/     # kymograph RANSAC + dynamic instability
        ├── napari.yaml               # plugin manifest
        ├── _widget.py                # tabbed UI (Input | RANSAC | Results)
        ├── _worker.py                # run_ransac_stream (segment-by-segment)
        ├── _reader.py                # kymograph TIFF reader
        └── _tests/test_worker.py     # headless smoke test
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

## Tracking microtubules across frames

`track.track_snapshots` links per-frame `MTSnapshot`s into per-MT
trajectories with the Jaqaman-style LAP formulation (Hungarian over an
augmented matrix with explicit birth and death). For each candidate
link `prev → curr` the cost is a **weighted sum of toggleable terms**:

| Term | Default | Formula |
|---|---|---|
| `distance` | always on | $\|\text{predicted}_\text{tip} - \text{observed}_\text{tip}\|^2$, summed over both tips, **minimised over the two tip permutations** |
| `intensity` | on | $w_\text{int}\cdot(\Delta A / s)^2$ where $s$ = `amp_scale` |
| `curvature` | on | $w_\text{cur}\cdot(\Delta C)^2$ |
| `ds` | off | $w_\text{ds}\cdot(\Delta\,ds)^2$ |

Tip positions are predicted by adding a 2-frame velocity to the
previous tip position (controlled by `velocity_lookback`). Links with
total cost above `gate` are forbidden, forcing the LAP to choose
birth + death instead.

### Toggling cost components

```python
from kapoorlabs_mtrack.track import TrackingCost, track_snapshots

cfg = TrackingCost(
    enable_intensity=True,
    enable_curvature=True,
    enable_ds=False,
    weights={"distance": 1.0, "intensity": 0.3, "curvature": 0.5, "ds": 0.2},
    amp_scale=100.0,
    gate=50.0**2,
    velocity_lookback=2,
)
tracks = track_snapshots(frame_snapshots, cfg=cfg)
```

### Tip identity across frames

The model's `start` / `end` labels are arbitrary — the same physical
tip can be `start` in one frame and `end` in the next. The cost
function evaluates **both possible tip permutations** for every
candidate link, and the winning permutation is fed back into the track
so each track exposes a stable `tip_a` / `tip_b` identity. Post-hoc,
`track.label_plus_minus` labels the more dynamic tip (longer total
path length) as **plus**.

### Length profiles

`track.build_length_profiles(tracks)` returns one `LengthProfile` per
track, containing per-frame:

- `plus_xy`, `minus_xy` — tip positions after plus/minus assignment
- `tip_distance` — straight-line distance between tips
- `arc_length` — length of the swept curve at this frame, computed by
  re-walking the spline with the fitted `ds` / `curvature`

These ship to disk via `io.save_length_profiles_csv` for downstream
plotting.

### Kymographs

`track.build_kymograph(profile)` returns a `Kymograph` with a 2-D
image where time runs down the y-axis and the **signed position along
the MT's initial axis** runs along the x-axis. The axis is defined as
the unit vector from the minus tip at frame 0 to the plus tip at
frame 0; both tips' positions at every later frame are projected onto
this axis to populate the image. Pass `mode="intensity"` with an
`amplitudes` array to weight the kymograph by fitted intensity per
frame, or `mode="binary"` for an occupancy heatmap.

The notebook `notebooks/02_real_data_pipeline.ipynb` plots these for
every fitted MT — the classic "growing line" shape for a polymerising
MT, "shrinking line" for a depolymerising one, and any combination
thereof.

### CLI

```bash
python scripts/track_endpoints.py \
    --endpoints endpoints.csv \
    --out-tracks tracks.csv \
    --out-profiles length_profiles.csv \
    --gate 50 --velocity-lookback 2
```

Toggle terms from the CLI with `--no-intensity`, `--no-curvature`,
`--enable-ds`. `--max-gap N` lets a track survive N empty frames
before it is terminated (default 0).

### Length-profile CSV schema (`length_profiles.csv`)

| Column | Meaning |
|---|---|
| `mt_id` | track identifier (assigned by the tracker) |
| `frame` | timepoint |
| `plus_x`, `plus_y` | plus-tip position in full-image pixels |
| `minus_x`, `minus_y` | minus-tip position |
| `tip_distance` | straight-line ‖plus − minus‖ |
| `arc_length` | length of the fitted spline curve between the tips |
| `plus_was_tip` | `A` or `B` — which raw tip-track was labelled plus |

---

## Kymograph RANSAC and dynamic-instability analysis

The `kapoorlabs_mtrack.ransac` subpackage is a **clean port** of the
RANSAC fits from `caped-ai-mtrack` plus a fresh dynamic-instability
analyzer on top. It operates on **kymograph images** (one MT per
image, x = position along the MT, y = time, intensity = fluorescence)
and produces the canonical biology report: growth/shrinkage segments,
catastrophe / rescue counts, and frequencies.

### Pipeline

```
kymograph TIFF
      │
      │  ransac.extract_kymograph_points()           (threshold + skeletonise)
      ▼
(t, x) point cloud
      │
      │  ransac.Ransac(...).extract_multiple_lines() (sequential RANSAC peel)
      ▼
list of (estimator, inlier_points) segments
      │
      │  ransac.classify_segments()                  (label growth/shrink/pause)
      │  ransac.dynamic_instability()
      ▼
DynamicInstability summary
   • n_catastrophes,  n_rescues
   • catastrophe_frequency, rescue_frequency
   • mean_growth_rate, mean_shrinkage_rate
   • time_in_growth, time_in_shrinkage, time_in_pause
```

### Quickstart

```python
import numpy as np
from kapoorlabs_mtrack.ransac import (
    LinearFunction, Ransac,
    classify_segments, dynamic_instability,
    extract_kymograph_points,
)

img = np.load("my_kymograph.npy")           # 2-D: (time, position)
points = extract_kymograph_points(img)       # (N, 2) of (t, x)

rs = Ransac(
    data_points=points.tolist(),
    model_class=LinearFunction, degree=2,
    min_samples=10, max_trials=200, iterations=8,
    residual_threshold=2.0, timeindex=0, random_state=0,
)
estimators, inliers = rs.extract_multiple_lines()

segments = classify_segments(estimators, inliers,
                              slope_threshold=0.4)
di = dynamic_instability(segments)

print(f"catastrophes: {di.n_catastrophes}, rescues: {di.n_rescues}")
print(f"f_cat = {di.catastrophe_frequency:.4f}/frame, "
      f"f_res = {di.rescue_frequency:.4f}/frame")
print(f"growth rate {di.mean_growth_rate:+.2f}, "
      f"shrink rate {di.mean_shrinkage_rate:+.2f}")
```

### Two-pass ComboRansac for curved + linear segments

When the MT changes direction with a curved transition before settling
into linear motion (transit between depolymerising and polymerising
phases), use `ComboRansac` -- it peels quadratic segments first then
re-fits linear segments over those inliers:

```python
from kapoorlabs_mtrack.ransac import (
    ComboRansac, LinearFunction, QuadraticFunction,
)

cr = ComboRansac(
    data_points=points.tolist(),
    model_linear=LinearFunction,
    model_quadratic=QuadraticFunction,
    min_samples=10, max_trials=200, iterations=8,
    residual_threshold=2.0,
)
estimators, inliers = cr.extract_multiple_lines()
```

### What was ported, and what got dropped

Ported (from `caped-ai-mtrack`):

- `RansacModels/*` → `ransac/models.py` -- `LinearFunction`,
  `QuadraticFunction` (with the exact cubic-root distance via Cardano),
  `PolynomialFunction` (now uses `numpy.polyfit` for speed at higher
  degrees), `GeneralizedFunction` base.
- `Fits/ransac.py` + `Fits/comboransac.py` → `ransac/fits.py` --
  `Ransac` and `ComboRansac` with the **PNG-writing side effects
  removed**: the original called `plot_ransac_gt(...)` to a hard-coded
  path inside the extraction loop; we just return estimators and let
  the caller plot.
- `Fits/utils.py:clean_estimators` → kept as private
  `_dedup_estimators_by_envelope`; the second-pass slope/endpoint
  prune was dropped (it over-segmented on synthetic data).

Dropped:

- `vollseg`/`stardist` segmentation dependency -- replaced with
  `skimage.filters.threshold_otsu` + `skimage.morphology.skeletonize`
  in `extract_kymograph_points`.
- `Fits/regression.py`, `Solvers/newton_raphson.py` -- unused by the
  public RANSAC API.
- PNG writing in `clean_ransac`/`plot_ransac_gt` -- replaced with
  pure data-returning functions; callers do their own plotting.
- The hand-rolled normal-equation accumulator for higher-degree
  polynomials -- `PolynomialFunction` now defers to `numpy.polyfit`.

Added (not in original):

- `ransac/dynamics.py` -- `Segment` dataclass + `classify_segments`
  + `dynamic_instability` to turn RANSAC output into the
  catastrophe-frequency / rescue-frequency / mean-rate report the
  biology actually needs.
- Pause-aware transition counting -- a pause segment between growth
  and shrinkage still counts as one catastrophe (the directional
  change is what matters).

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
| 9. Hungarian tracker (cost: distance + intensity + curvature, 2-frame velocity prediction, gating, plus/minus labelling) | ✅ shipped | `track.hungarian`, `track.cost` |
| 10. Length-profile output (per-MT plus-end and minus-end length-vs-time) | ✅ shipped | `track.profile`, `scripts/track_endpoints.py` |
| 11. Multi-frame synthetic movie generator | ✅ shipped | `simulate.movie` |
| 12. Per-MT kymograph builder | ✅ shipped | `track.kymograph` |
| 13. Interactive notebooks driving the full chain | ✅ shipped | `notebooks/` |
| 14. Napari plugin (live fit + track + kymograph; optional install) | ✅ shipped | `plugins/kapoorlabs_mtrack_optimizer/` |
| 15. RANSAC kymograph segmentation + dynamic-instability analysis (ported & cleaned from `caped-ai-mtrack`) | ✅ shipped | `kapoorlabs_mtrack.ransac` |
| 16. Napari plugin `kapoorlabs-mtrack-ransac` (RANSAC fits + catastrophe/rescue UI) | ✅ shipped | `plugins/kapoorlabs_mtrack_ransac/` |

---

## Installation

The package ships in two layers — pick the install that matches your
use case:

| Goal | Install command | What you get |
|---|---|---|
| Use the optimizer / pipeline from scripts and notebooks | `pip install KapoorLabs-MTrack` | core: models, fit, simulate, pipeline, track, io |
| Add the **napari plugin** for interactive fitting + tracking | `pip install KapoorLabs-MTrack[napari]` | core + napari + magicgui + Qt |
| Everything, including dev / test deps | `pip install KapoorLabs-MTrack[all]` | core + napari + pytest + pytest-qt |

Or for the development version:

```bash
pip install -e .            # core only
pip install -e .[napari]    # core + plugin
```

### Launching the napari plugin

```bash
napari -w kapoorlabs-mtrack-optimizer "MTrack Optimizer"
```

(or open napari and pick **Plugins → MTrack Optimizer**). The widget
has four tabs — see "Napari plugin (kapoorlabs-mtrack-optimizer)"
below.

---

## Napari plugin (`kapoorlabs-mtrack-optimizer`)

The plugin lives at `plugins/kapoorlabs_mtrack_optimizer/` and is
exposed as a separate distribution entry-point so users can opt in via
`pip install KapoorLabs-MTrack[napari]`. It wraps every step of the
pipeline behind a tabbed magicgui UI:

| Tab | What it does |
|---|---|
| **Input** | File pickers for the raw + label TIFFs (2-D or 2-D+T), microscope PSF (`sigma_x`, `sigma_y`), LM Jacobian mode. The **Load into napari** button drops both stacks into the viewer as `image` and `labels` layers and resets any prior fit overlays. |
| **Fit** | `ds` seed, optional bbox padding, run / stop buttons. Pressing **Run fit (live)** spawns a `napari.qt.threading.thread_worker` driven by `kapoorlabs_mtrack_optimizer._worker.fit_stack_stream`, which yields one `FrameSnapshot` per frame. Each yield appends to two overlay layers — **MT tips** (`Points`) and **MT curves** (`Shapes`, with the walked-curve polyline) — both indexed on the time axis so the napari time slider scrubs them. A progress bar tracks the per-frame count. |
| **Track** | Toggles for `intensity` / `curvature` / `ds` cost terms, weight spin boxes, gate distance, velocity lookback, max gap. **Run tracking** consumes the per-frame snapshots, runs `track_snapshots`, and adds a **MT tip tracks** layer (a napari `Tracks` layer with 2 rows per MT per frame — one per physical tip). |
| **Results** | Per-MT summary (arc length t=0 → end, plus-was-tip), a microtubule picker, **Show kymograph for selected MT** (adds the kymograph as an inferno-colormapped image layer), and **Export CSVs** which writes `endpoints.csv` + `length_profiles.csv` via the existing `io` writers. |

### Live overlays — what's actually rendered

- **MT tips** layer: one cyan point per fitted tip per frame, sized
  6 px, white-edged for visibility on bright frames.
- **MT curves** layer: yellow polylines per fitted curve, walked at
  the fitted `ds` so curvature/inflection are visible.
- **MT tip tracks** layer (after tracking): napari's built-in Tracks
  layer so you get connected colored trails per tip-track.
- **kymograph mt N** layer (on demand): the inferno-colormapped
  kymograph image generated by `build_kymograph(profile)`.

### Architecture: why a generator and not a callback

The core `pipeline.fit_stack.fit_stack` is intentionally a **batch**
function — it returns the full snapshot list, with no GUI dependency.
The plugin's `_worker.fit_stack_stream` replicates the same per-frame
loop but **yields** after each frame, which `thread_worker(connect={
"yielded": ...})` then dispatches to the main thread for layer
updates. This keeps the core importable in headless environments
(servers, CI, plain Python scripts) while letting the plugin scrub
frames live.

---

## Napari plugin (`kapoorlabs-mtrack-ransac`)

Companion plugin to the optimizer — operates on **kymograph images**
(one MT per image, x = position along the MT, y = time) and reports
the canonical dynamic-instability statistics. Same `pip install
KapoorLabs-MTrack[napari]` brings both plugins in.

Launch:

```bash
napari -w kapoorlabs-mtrack-ransac "MTrack RANSAC"
```

### Tabs

| Tab | Contents |
|---|---|
| **Input** | `FileEdit` for the kymograph TIFF, optional `FileEdit` for a precomputed mask. **Load** drops the kymograph into napari as an `inferno`-colormapped image layer (and the mask as `labels` if provided). |
| **RANSAC** | `mode` (linear vs combo), `min_samples`, `max_trials`, `iterations`, `residual_threshold` (px), `slope_threshold` (px/frame; below this a segment is called a *pause*), `random_state`. **Run RANSAC (live)** spawns a `thread_worker` driven by `kapoorlabs_mtrack_ransac._worker.run_ransac_stream`. Each extracted segment is yielded immediately and appended as a yellow line to a **RANSAC segments** Shapes overlay on the kymograph, so the user watches the segments accumulate. The trailing summary yield carries the full `DynamicInstability` report. |
| **Results** | Text panel with catastrophe / rescue counts, frequencies (`events / time_in_state`), mean growth and shrinkage rates, time-in-state totals; per-segment listing (kind, slope, time bounds, inlier count); **Export CSV** writes `segments.csv` + `summary.csv`. |

### Reusing the math without napari

Every Results-tab number is computed by the headless core; nothing in
the plotting layer is original to the plugin. To do the same analysis
from a script:

```python
from kapoorlabs_mtrack.ransac import (
    Ransac, LinearFunction,
    classify_segments, dynamic_instability,
    extract_kymograph_points,
)

img = tifffile.imread("kymograph.tif")
pts = extract_kymograph_points(img)
estimators, inliers = Ransac(
    pts.tolist(), LinearFunction, degree=2,
    min_samples=10, max_trials=200, iterations=8,
    residual_threshold=2.0, timeindex=0,
).extract_multiple_lines()
di = dynamic_instability(
    classify_segments(estimators, inliers, slope_threshold=0.4)
)
```

### What the rewrite changes vs `vollseg-napari-mtrack`

This plugin replaces the original `vollseg-napari-mtrack` and removes
three weight bearing dependencies the original carried:

1. **No `vollseg` / `stardist` dependency.** The original called
   `VollSeg` to segment kymographs before RANSAC; this plugin uses
   `skimage.filters.threshold_otsu` + `skeletonize` (already required
   for the rest of the pipeline) and lets the user pass a precomputed
   mask if they have a better segmentation.
2. **No `caped_ai_tabulour` custom Qt table widget.** Replaced with a
   plain `magicgui.Label` block plus CSV export — far smaller dep
   surface, fewer style issues, copy-pastable into any other GUI.
3. **No `seaborn` dependency for the plots.** The new widget displays
   results inline as text + a Shapes overlay; users who want fancier
   plots can use the exported `segments.csv` + `summary.csv` in any
   notebook (matplotlib is already a core dep).

The 1600-line monolithic `_widget.py` of the original is now split
into `_widget.py` (UI), `_worker.py` (streaming RANSAC runner), and
`_reader.py` (TIFF loader), totalling ~350 lines.

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
