"""KapoorLabs MTrack RANSAC napari widget.

Tabbed UI matching the optimizer plugin's style:

- **Input**  : load a kymograph TIFF (single file or directory),
  optionally a mask. The kymograph appears as a napari image layer
  so the user can scrub colormap / contrast before fitting.
- **RANSAC** : pick the model (linear vs quadratic+linear combo),
  set ``min_samples``, ``max_trials``, ``iterations``,
  ``residual_threshold``, and the dynamic-instability
  ``slope_threshold``. **Run** spawns a ``thread_worker`` driven by
  :func:`_worker.run_ransac_stream`; each yielded segment becomes a
  yellow line in a Shapes overlay so the user sees fits accumulate.
- **Results**: dynamic-instability summary text (catastrophe /
  rescue counts, frequencies, mean growth + shrinkage rates), an
  inline matplotlib plot of the kymograph with all segments
  superimposed and transitions marked, and CSV export.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tifffile
from magicgui import widgets as mw
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from kapoorlabs_mtrack.ransac import (
    calibrate,
    dynamic_instability,
    segments_from_polylines,
)

from ._worker import RansacRunResult, run_ransac_stream, segment_kymograph

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _load_kymograph(path: Path) -> np.ndarray:
    """Load a kymograph TIFF as a 2-D array."""
    arr = tifffile.imread(str(path))
    if arr.ndim != 2:
        raise ValueError(
            f"{path}: kymograph must be 2-D (time, position), got {arr.shape}"
        )
    return arr


def _segment_polyline(res: RansacRunResult) -> np.ndarray:
    """Build a 2-point polyline ``(y, x)`` spanning the segment's time range."""
    # res.slope_y_over_x is dy/dx (time per position). To draw a line on
    # the kymograph we need two (t, x) endpoints. Solve x = (t - b)/m.
    m = res.slope_y_over_x
    b = res.intercept
    t0, t1 = res.t_start, res.t_end
    if abs(m) < 1e-9:
        # Degenerate; use the inliers' x range.
        x0 = float(res.inliers[:, 1].min())
        x1 = float(res.inliers[:, 1].max())
    else:
        x0 = (t0 - b) / m
        x1 = (t1 - b) / m
    return np.array(
        [[t0, x0], [t1, x1]]
    )  # napari shapes expect (row, col) = (t, x)


# ---------------------------------------------------------------------------
# Widget factory.
# ---------------------------------------------------------------------------


def plugin_wrapper_ransac():
    """Return the QWidget napari embeds in the right dock."""
    import napari

    viewer = napari.current_viewer()

    state = {
        "kymograph": None,  # 2-D array
        "mask": None,  # optional binary mask
        "kymo_layer": None,
        "segments_layer": None,
        "results": [],  # list[RansacRunResult] (segments)
        "summary": None,  # DynamicInstability
    }

    # ---------- Input tab -----------------------------------------------

    kymo_path = mw.FileEdit(
        label="Kymograph TIFF", mode="r", filter="*.tif *.tiff"
    )
    mask_path = mw.FileEdit(
        label="Mask TIFF (optional, pre-computed)",
        mode="r",
        filter="*.tif *.tiff",
    )
    # Segmentation source for the RANSAC run. "otsu" is the old default
    # and needs no extra deps; "vollseg_pretrained" / "vollseg_custom"
    # require the [vollseg] extra; "user_mask" lets the user paint or
    # edit a labels layer in napari and feed that as the segmentation.
    seg_source = mw.ComboBox(
        label="Segmentation source",
        choices=[
            ("Otsu + skeletonize (no model)", "otsu"),
            ("vollseg pretrained model", "vollseg_pretrained"),
            ("vollseg custom model path", "vollseg_custom"),
            ("user mask layer in napari", "user_mask"),
        ],
        value="otsu",
    )
    vollseg_model_name = mw.LineEdit(
        label="vollseg pretrained name",
        value="",
    )
    vollseg_custom_dir = mw.FileEdit(
        label="vollseg custom model dir",
        mode="d",
    )
    user_mask_layer_name = mw.LineEdit(
        label="user mask layer name",
        value="mask",
    )
    load_button = mw.PushButton(text="Load into napari")
    input_status = mw.Label(value="(no kymograph)")
    input_container = mw.Container(
        widgets=[
            kymo_path,
            mask_path,
            seg_source,
            vollseg_model_name,
            vollseg_custom_dir,
            user_mask_layer_name,
            load_button,
            input_status,
        ]
    )

    @load_button.changed.connect
    def _on_load():
        try:
            kymo = _load_kymograph(Path(str(kymo_path.value)))
        except Exception as exc:
            input_status.value = f"ERROR: {exc}"
            return
        mask = None
        mp = str(mask_path.value).strip()
        if mp and mp != ".":
            try:
                m = tifffile.imread(mp)
                mask = m.astype(bool)
                if mask.shape != kymo.shape:
                    input_status.value = (
                        f"ERROR: mask shape {mask.shape} != kymo {kymo.shape}"
                    )
                    return
            except Exception as exc:
                input_status.value = f"ERROR loading mask: {exc}"
                return

        state["kymograph"] = kymo
        state["mask"] = mask
        for k in ("kymo_layer", "segments_layer"):
            layer = state.get(k)
            if layer is not None and layer in viewer.layers:
                viewer.layers.remove(layer)
            state[k] = None
        state["results"] = []
        state["summary"] = None
        state["kymo_layer"] = viewer.add_image(
            kymo,
            name=Path(str(kymo_path.value)).stem,
            colormap="inferno",
        )
        if mask is not None:
            viewer.add_labels(mask.astype(np.int32), name="mask")
        input_status.value = f"loaded kymograph {kymo.shape}"

    # ---------- RANSAC tab ----------------------------------------------

    mode = mw.ComboBox(
        label="Mode", choices=["linear", "combo"], value="linear"
    )
    min_samples = mw.SpinBox(label="min_samples", value=10, min=3, step=1)
    max_trials = mw.SpinBox(label="max_trials", value=200, min=10, step=10)
    iterations = mw.SpinBox(
        label="iterations (max segments)", value=8, min=1, step=1
    )
    residual_threshold = mw.FloatSpinBox(
        label="residual threshold (px)", value=2.0, min=0.1, step=0.1
    )
    slope_threshold = mw.FloatSpinBox(
        label="slope threshold (px/frame, for pause vs growth)",
        value=0.4,
        min=0.0,
        step=0.05,
    )
    random_state = mw.SpinBox(
        label="random seed (0 = nondeterministic)", value=0, min=0, step=1
    )
    run_button = mw.PushButton(text="Run RANSAC (live)")
    ransac_status = mw.Label(value="(no run)")
    ransac_progress = mw.ProgressBar(value=0, max=1)
    ransac_container = mw.Container(
        widgets=[
            mode,
            min_samples,
            max_trials,
            iterations,
            residual_threshold,
            slope_threshold,
            random_state,
            run_button,
            ransac_status,
            ransac_progress,
        ]
    )

    def _ensure_segments_layer():
        if state["segments_layer"] is None:
            state["segments_layer"] = viewer.add_shapes(
                [],
                shape_type="line",
                ndim=2,
                name="RANSAC segments",
                edge_color="yellow",
                edge_width=1.5,
            )

    def _on_yield(res: RansacRunResult):
        if res.is_summary:
            state["summary"] = res.summary
            results_refresh()
            return
        state["results"].append(res)
        _ensure_segments_layer()
        poly = _segment_polyline(res)
        # napari Shapes line: pass list of (N, 2) arrays via add_lines.
        state["segments_layer"].add_lines([poly])
        ransac_status.value = (
            f"segment {len(state['results'])}: "
            f"{res.t_start:.0f}..{res.t_end:.0f} t  "
            f"({res.inliers.shape[0]} inliers)"
        )
        ransac_progress.value = min(
            int(iterations.value), len(state["results"])
        )

    def _on_done(_result):
        run_button.enabled = True
        n = len(state["results"])
        ransac_status.value = f"done. {n} segments extracted; see Results tab."

    def _resolve_user_mask():
        """Resolve the napari layer named in ``user_mask_layer_name`` to a 2-D bool array."""
        name = str(user_mask_layer_name.value).strip()
        if not name:
            return None
        for layer in viewer.layers:
            if layer.name == name:
                data = np.asarray(layer.data)
                if data.ndim != 2:
                    raise ValueError(
                        f"layer {name!r} is {data.ndim}-D, need 2-D mask"
                    )
                return data.astype(bool)
        raise ValueError(f"no napari layer named {name!r}")

    @run_button.changed.connect
    def _on_run():
        if state["kymograph"] is None:
            ransac_status.value = "load a kymograph first"
            return

        # Resolve the chosen segmentation source into a mask (or None
        # for the Otsu fall-through inside extract_kymograph_points).
        source = str(seg_source.value)
        try:
            if source == "otsu":
                mask = state["mask"]  # may be None or a TIFF-supplied one
            elif source == "user_mask":
                mask = _resolve_user_mask()
            else:
                mask = segment_kymograph(
                    state["kymograph"],
                    source,
                    pretrained_name=str(vollseg_model_name.value).strip()
                    or None,
                    custom_path=str(vollseg_custom_dir.value).strip() or None,
                )
        except Exception as exc:
            ransac_status.value = f"segmentation error: {exc}"
            return

        # If we produced a mask, show it as a labels layer.
        if mask is not None and source != "otsu":
            viewer.add_labels(mask.astype(np.int32), name=f"seg:{source}")

        # Clear prior segments layer.
        if (
            state["segments_layer"] is not None
            and state["segments_layer"] in viewer.layers
        ):
            viewer.layers.remove(state["segments_layer"])
        state["segments_layer"] = None
        state["results"] = []
        state["summary"] = None
        ransac_progress.max = max(1, int(iterations.value))
        ransac_progress.value = 0
        run_button.enabled = False

        seed = int(random_state.value)
        seed_arg = seed if seed > 0 else None

        @thread_worker(connect={"yielded": _on_yield, "returned": _on_done})
        def _run():
            yield from run_ransac_stream(
                state["kymograph"],
                mode=str(mode.value),
                min_samples=int(min_samples.value),
                max_trials=int(max_trials.value),
                iterations=int(iterations.value),
                residual_threshold=float(residual_threshold.value),
                slope_threshold=float(slope_threshold.value),
                mask=mask,
                random_state=seed_arg,
            )
            return state["results"]

        _run()

    # ---------- Results tab ---------------------------------------------

    # Microscope calibration -- physical-unit conversion for the DI report.
    pixel_size = mw.FloatSpinBox(
        label="pixel size",
        value=1.0,
        min=1e-9,
        step=0.01,
    )
    space_unit = mw.LineEdit(label="space unit", value="µm")
    time_per_frame = mw.FloatSpinBox(
        label="seconds per frame",
        value=1.0,
        min=1e-9,
        step=0.01,
    )
    time_unit = mw.LineEdit(label="time unit", value="s")

    # User-line override: pick a napari Shapes layer of polylines drawn
    # on the kymograph, replace the auto-detected segments with those.
    user_lines_layer_name = mw.LineEdit(
        label="user lines layer name",
        value="user lines",
        tooltip=(
            "Name of a napari Shapes layer holding line/path shapes "
            "on the kymograph. Each shape becomes one segment."
        ),
    )
    recompute_button = mw.PushButton(text="Recompute DI from user lines")
    apply_calibration_button = mw.PushButton(
        text="Refresh report with calibration"
    )

    summary_label = mw.Label(value="(no results)")
    segments_label = mw.Label(value="")
    export_dir = mw.FileEdit(label="Export to dir", mode="d")
    export_button = mw.PushButton(text="Export CSV (segments + summary)")
    export_status = mw.Label(value="")
    results_container = mw.Container(
        widgets=[
            pixel_size,
            space_unit,
            time_per_frame,
            time_unit,
            user_lines_layer_name,
            recompute_button,
            apply_calibration_button,
            summary_label,
            segments_label,
            export_dir,
            export_button,
            export_status,
        ]
    )

    def results_refresh():
        di = state.get("summary")
        if di is None:
            summary_label.value = "(no results)"
            segments_label.value = ""
            return
        # Apply microscope calibration on top of the raw px / frame
        # numbers, then render both for clarity.
        cal = calibrate(
            di,
            pixel_size=float(pixel_size.value),
            time_per_frame=float(time_per_frame.value),
            space_unit=str(space_unit.value),
            time_unit=str(time_unit.value),
        )
        su, tu = cal.space_unit, cal.time_unit
        summary_label.value = (
            f"catastrophes: {di.n_catastrophes}   rescues: {di.n_rescues}\n"
            f"f_cat = {di.catastrophe_frequency:.4f}/frame  "
            f"= {cal.catastrophe_frequency:.4f}/{tu}\n"
            f"f_res = {di.rescue_frequency:.4f}/frame  "
            f"= {cal.rescue_frequency:.4f}/{tu}\n"
            f"mean growth rate:    {di.mean_growth_rate:+.3f} px/frame  "
            f"= {cal.mean_growth_rate:+.3f} {su}/{tu}\n"
            f"mean shrinkage rate: {di.mean_shrinkage_rate:+.3f} px/frame  "
            f"= {cal.mean_shrinkage_rate:+.3f} {su}/{tu}\n"
            f"time in growth:    {di.time_in_growth:.1f} frames  "
            f"= {cal.time_in_growth:.1f} {tu}\n"
            f"time in shrinkage: {di.time_in_shrinkage:.1f} frames  "
            f"= {cal.time_in_shrinkage:.1f} {tu}\n"
            f"time in pause:     {di.time_in_pause:.1f} frames  "
            f"= {cal.time_in_pause:.1f} {tu}"
        )
        rows = [
            f"  [{i}] {s.kind:9s}  slope={s.slope:+.3f}  "
            f"t={s.t_start:.0f}..{s.t_end:.0f}  ({s.n_inliers} pts)"
            for i, s in enumerate(di.segments)
        ]
        segments_label.value = "Segments:\n" + "\n".join(rows) if rows else ""
        state["calibrated"] = cal

    @apply_calibration_button.changed.connect
    def _on_apply_calibration():
        results_refresh()

    @recompute_button.changed.connect
    def _on_recompute_from_user_lines():
        """Replace auto-detected segments with the user's drawn polylines."""
        name = str(user_lines_layer_name.value).strip()
        layer = next((L for L in viewer.layers if L.name == name), None)
        if layer is None:
            summary_label.value = f"no napari layer named {name!r}"
            return
        # napari Shapes.data is a list of (N, 2) arrays. We accept
        # any shape type whose vertex array is (N, 2) with N >= 2.
        polylines = []
        for shape in getattr(layer, "data", []):
            arr = np.asarray(shape)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 2:
                polylines.append(arr[:, :2])  # (t, x) per napari convention
        if not polylines:
            summary_label.value = f"layer {name!r} has no usable polylines"
            return
        segments = segments_from_polylines(
            polylines,
            slope_threshold=float(slope_threshold.value),
        )
        di = dynamic_instability(segments)
        state["summary"] = di
        state["results"] = []  # auto-detected results are replaced
        results_refresh()

    @export_button.changed.connect
    def _on_export():
        out_dir = Path(str(export_dir.value))
        if not out_dir.is_dir():
            export_status.value = "select a valid directory"
            return
        di = state.get("summary")
        if di is None:
            export_status.value = "run RANSAC first"
            return
        # segments.csv
        with (out_dir / "segments.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "index",
                    "kind",
                    "slope",
                    "intercept",
                    "t_start",
                    "t_end",
                    "n_inliers",
                ]
            )
            for i, s in enumerate(di.segments):
                w.writerow(
                    [
                        i,
                        s.kind,
                        s.slope,
                        s.intercept,
                        s.t_start,
                        s.t_end,
                        s.n_inliers,
                    ]
                )
        # summary.csv -- raw px / frame numbers in column 'value', then
        # calibrated values in their physical units in column 'value_calibrated'.
        cal = calibrate(
            di,
            pixel_size=float(pixel_size.value),
            time_per_frame=float(time_per_frame.value),
            space_unit=str(space_unit.value),
            time_unit=str(time_unit.value),
        )
        su, tu = cal.space_unit, cal.time_unit
        with (out_dir / "summary.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["metric", "value", "value_calibrated", "unit_raw", "unit_cal"]
            )
            w.writerow(
                [
                    "n_catastrophes",
                    di.n_catastrophes,
                    cal.n_catastrophes,
                    "count",
                    "count",
                ]
            )
            w.writerow(
                ["n_rescues", di.n_rescues, cal.n_rescues, "count", "count"]
            )
            w.writerow(
                [
                    "catastrophe_frequency",
                    di.catastrophe_frequency,
                    cal.catastrophe_frequency,
                    "1/frame",
                    f"1/{tu}",
                ]
            )
            w.writerow(
                [
                    "rescue_frequency",
                    di.rescue_frequency,
                    cal.rescue_frequency,
                    "1/frame",
                    f"1/{tu}",
                ]
            )
            w.writerow(
                [
                    "mean_growth_rate",
                    di.mean_growth_rate,
                    cal.mean_growth_rate,
                    "px/frame",
                    f"{su}/{tu}",
                ]
            )
            w.writerow(
                [
                    "mean_shrinkage_rate",
                    di.mean_shrinkage_rate,
                    cal.mean_shrinkage_rate,
                    "px/frame",
                    f"{su}/{tu}",
                ]
            )
            w.writerow(
                [
                    "time_in_growth",
                    di.time_in_growth,
                    cal.time_in_growth,
                    "frames",
                    tu,
                ]
            )
            w.writerow(
                [
                    "time_in_shrinkage",
                    di.time_in_shrinkage,
                    cal.time_in_shrinkage,
                    "frames",
                    tu,
                ]
            )
            w.writerow(
                [
                    "time_in_pause",
                    di.time_in_pause,
                    cal.time_in_pause,
                    "frames",
                    tu,
                ]
            )
            w.writerow(["pixel_size", cal.pixel_size, "", f"{su}/px", ""])
            w.writerow(
                ["time_per_frame", cal.time_per_frame, "", f"{tu}/frame", ""]
            )
        export_status.value = f"wrote segments.csv + summary.csv to {out_dir}"

    # ---------- Assemble -------------------------------------------------

    tabs = QTabWidget()
    tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    tabs.addTab(input_container.native, "Input")
    tabs.addTab(ransac_container.native, "RANSAC")
    tabs.addTab(results_container.native, "Results")

    outer = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(tabs)
    outer.setLayout(layout)
    return outer
