"""KapoorLabs MTrack napari widget.

Tabbed UI matching the style of ``vollseg-napari-mtrack``:

- **Input** : load raw + label TIFFs (single file or directory) and
  set the microscope PSF + Jacobian mode.
- **Fit**   : per-frame fitting parameters. Run starts a
  ``thread_worker`` that streams per-frame ``FrameSnapshot``s back
  to the GUI thread, updating two napari layers as it goes -- a
  Points layer for tips and a Shapes layer for the fitted curve
  polylines, both indexed by frame so the time slider scrubs them.
- **Track**: linking cost configuration (toggles + weights + gate)
  and a "Run tracking" button. Builds tracks + length profiles
  once all frames have been fitted.
- **Results**: a per-MT summary table + length-vs-time plot +
  per-MT kymograph picker (matplotlib FigureCanvas inside Qt).
"""

from __future__ import annotations

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

from kapoorlabs_mtrack.models import walk_curve
from kapoorlabs_mtrack.track import (
    TrackingCost,
    build_kymograph,
)

from ._worker import fit_stack_stream, track_and_profile

# ---------------------------------------------------------------------------
# Helpers (kept module-level so they're easy to unit-test).
# ---------------------------------------------------------------------------


def _curve_polyline(snap) -> np.ndarray:
    """Walked-curve polyline (Nx2, rows = (y, x)) for a fitted MT snapshot."""
    a9 = np.array(
        [
            snap.start[0],
            snap.start[1],
            snap.end[0],
            snap.end[1],
            snap.ds,
            snap.curvature,
            snap.inflection,
            1.0,
            0.0,
        ]
    )
    if a9[2] - a9[0] <= 1e-6:
        pts_xy = np.vstack([snap.start, snap.end])
    else:
        mid = walk_curve(a9)
        pts_xy = np.vstack([snap.start[None, :], mid, snap.end[None, :]])
    # napari shapes expect (y, x) ordering.
    return np.column_stack([pts_xy[:, 1], pts_xy[:, 0]])


def _load_stack(path: Path) -> np.ndarray:
    """Load a TIFF as a ``(T, H, W)`` stack (broadcasts 2-D inputs to T=1)."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"{path}: expected 2-D or 2-D+T TIFF, got {arr.shape}")


def _resolve_pair(
    raw_path: str, label_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load and shape-check a raw + label TIFF pair."""
    raw = _load_stack(Path(raw_path))
    lab = _load_stack(Path(label_path))
    if raw.shape != lab.shape:
        raise ValueError(
            f"raw shape {raw.shape} != label shape {lab.shape} "
            "(both must be 2-D or 2-D+T with identical (T,H,W))"
        )
    return raw, lab


# ---------------------------------------------------------------------------
# Widget factory.
# ---------------------------------------------------------------------------


def plugin_wrapper_mtrack():
    """Build and return the QWidget napari embeds in its right dock."""
    import napari

    viewer = napari.current_viewer()

    # Mutable state shared between tabs.
    state = {
        "raw": None,  # (T, H, W)
        "labels": None,  # (T, H, W)
        "snapshots": [],  # list[FrameSnapshot]
        "tracks": [],
        "profiles": [],
        "tips_layer": None,
        "curves_layer": None,
        "tracks_layer": None,
    }

    # ---------- Input tab ------------------------------------------------

    raw_path = mw.FileEdit(label="Raw TIFF", mode="r", filter="*.tif *.tiff")
    label_path = mw.FileEdit(
        label="Label TIFF", mode="r", filter="*.tif *.tiff"
    )
    sigma_x = mw.FloatSpinBox(
        label="PSF sigma_x (px)", value=1.6, min=0.1, step=0.1
    )
    sigma_y = mw.FloatSpinBox(
        label="PSF sigma_y (px)", value=1.6, min=0.1, step=0.1
    )
    jac_mode = mw.ComboBox(
        label="LM Jacobian mode",
        choices=["analytic", "hybrid", "numeric"],
        value="hybrid",
    )
    load_button = mw.PushButton(text="Load into napari")
    input_status = mw.Label(value="(no data loaded)")
    input_container = mw.Container(
        widgets=[
            raw_path,
            label_path,
            sigma_x,
            sigma_y,
            jac_mode,
            load_button,
            input_status,
        ]
    )

    @load_button.changed.connect
    def _on_load():
        try:
            raw, lab = _resolve_pair(
                str(raw_path.value), str(label_path.value)
            )
        except Exception as exc:
            input_status.value = f"ERROR: {exc}"
            return
        state["raw"] = raw
        state["labels"] = lab
        # Reset any prior fit overlays.
        for k in ("tips_layer", "curves_layer", "tracks_layer"):
            layer = state.get(k)
            if layer is not None and layer in viewer.layers:
                viewer.layers.remove(layer)
            state[k] = None
        state["snapshots"] = []
        state["tracks"] = []
        state["profiles"] = []

        viewer.add_image(raw, name="raw")
        viewer.add_labels(lab, name="labels")
        input_status.value = f"loaded raw {raw.shape}, labels {lab.shape}"

    # ---------- Fit tab --------------------------------------------------

    ds_seed = mw.FloatSpinBox(label="ds seed", value=0.7, min=0.05, step=0.05)
    pad = mw.SpinBox(label="Bbox pad (px, 0 = auto)", value=0, min=0)
    run_fit_button = mw.PushButton(text="Run fit (live)")
    stop_fit_button = mw.PushButton(text="Stop")
    stop_fit_button.enabled = False
    fit_status = mw.Label(value="(no fit run)")
    fit_progress = mw.ProgressBar(value=0, max=1)
    fit_container = mw.Container(
        widgets=[
            ds_seed,
            pad,
            run_fit_button,
            stop_fit_button,
            fit_status,
            fit_progress,
        ]
    )

    def _ensure_layers():
        """Create the Points + Shapes overlays once we know the canvas shape."""
        if state["tips_layer"] is None:
            state["tips_layer"] = viewer.add_points(
                np.zeros((0, 3)),
                ndim=3,
                name="MT tips",
                size=6,
                face_color="cyan",
                edge_color="white",
            )
        if state["curves_layer"] is None:
            state["curves_layer"] = viewer.add_shapes(
                [],
                shape_type="path",
                ndim=3,
                name="MT curves",
                edge_color="yellow",
                edge_width=1,
            )

    def _push_frame(fs):
        """Append one FrameSnapshot's overlays into the napari layers."""
        if state["tips_layer"] is None:
            _ensure_layers()
        t = fs.frame
        tip_pts = []
        curve_paths = []
        for m in fs.mts:
            tip_pts.append([t, float(m.start[1]), float(m.start[0])])
            tip_pts.append([t, float(m.end[1]), float(m.end[0])])
            poly = _curve_polyline(m)  # (N, 2) in (y, x)
            t_col = np.full((poly.shape[0], 1), t)
            curve_paths.append(np.hstack([t_col, poly]))
        if tip_pts:
            existing = state["tips_layer"].data
            new = np.vstack([existing, np.asarray(tip_pts)])
            state["tips_layer"].data = new
        for path in curve_paths:
            state["curves_layer"].add_paths([path])
        fit_status.value = (
            f"frame {t}: {len(fs.mts)} fits, {len(fs.skipped_labels)} skipped"
        )

    def _fit_done():
        run_fit_button.enabled = True
        stop_fit_button.enabled = False
        n = sum(len(fs.mts) for fs in state["snapshots"])
        fit_status.value = (
            f"done. {len(state['snapshots'])} frames, {n} MT fits."
        )

    current_worker = {"w": None}

    @run_fit_button.changed.connect
    def _on_run_fit():
        if state["raw"] is None:
            fit_status.value = "load raw + labels first"
            return
        state["snapshots"] = []
        if (
            state["tips_layer"] is not None
            and state["tips_layer"] in viewer.layers
        ):
            viewer.layers.remove(state["tips_layer"])
            state["tips_layer"] = None
        if (
            state["curves_layer"] is not None
            and state["curves_layer"] in viewer.layers
        ):
            viewer.layers.remove(state["curves_layer"])
            state["curves_layer"] = None

        fit_progress.max = max(1, state["raw"].shape[0])
        fit_progress.value = 0
        run_fit_button.enabled = False
        stop_fit_button.enabled = True

        sigma_t = (float(sigma_x.value), float(sigma_y.value))
        pad_v = int(pad.value) if pad.value > 0 else None

        @thread_worker(
            connect={
                "yielded": _on_fit_yield,
                "returned": lambda _r: _fit_done(),
            }
        )
        def _run():
            for fs in fit_stack_stream(
                state["raw"],
                state["labels"],
                sigma=sigma_t,
                pad=pad_v,
                ds_seed=float(ds_seed.value),
                jac_mode=str(jac_mode.value),
            ):
                state["snapshots"].append(fs)
                yield fs
            return state["snapshots"]

        current_worker["w"] = _run()

    def _on_fit_yield(fs):
        _push_frame(fs)
        fit_progress.value = fs.frame + 1

    @stop_fit_button.changed.connect
    def _on_stop_fit():
        w = current_worker.get("w")
        if w is not None:
            try:
                w.quit()
            except Exception:
                pass
            stop_fit_button.enabled = False

    # ---------- Track tab ------------------------------------------------

    enable_intensity = mw.CheckBox(text="Use intensity term", value=True)
    enable_curvature = mw.CheckBox(text="Use curvature term", value=True)
    enable_ds = mw.CheckBox(text="Use ds term", value=False)
    gate_px = mw.FloatSpinBox(label="Gate (px)", value=50.0, min=1.0, step=1.0)
    velocity_lookback = mw.SpinBox(
        label="Velocity lookback (frames)", value=2, min=1
    )
    max_gap = mw.SpinBox(label="Max gap (frames)", value=0, min=0)
    w_distance = mw.FloatSpinBox(
        label="weight distance", value=1.0, min=0.0, step=0.1
    )
    w_intensity = mw.FloatSpinBox(
        label="weight intensity", value=0.3, min=0.0, step=0.1
    )
    w_curvature = mw.FloatSpinBox(
        label="weight curvature", value=0.5, min=0.0, step=0.1
    )
    w_ds = mw.FloatSpinBox(label="weight ds", value=0.2, min=0.0, step=0.1)
    run_track_button = mw.PushButton(text="Run tracking")
    track_status = mw.Label(value="(no tracks)")
    track_container = mw.Container(
        widgets=[
            enable_intensity,
            enable_curvature,
            enable_ds,
            gate_px,
            velocity_lookback,
            max_gap,
            w_distance,
            w_intensity,
            w_curvature,
            w_ds,
            run_track_button,
            track_status,
        ]
    )

    @run_track_button.changed.connect
    def _on_run_track():
        if not state["snapshots"]:
            track_status.value = "run fit first"
            return
        cfg = TrackingCost(
            enable_intensity=bool(enable_intensity.value),
            enable_curvature=bool(enable_curvature.value),
            enable_ds=bool(enable_ds.value),
            weights={
                "distance": float(w_distance.value),
                "intensity": float(w_intensity.value),
                "curvature": float(w_curvature.value),
                "ds": float(w_ds.value),
            },
            gate=float(gate_px.value) ** 2,
            velocity_lookback=int(velocity_lookback.value),
        )
        tracks, profiles = track_and_profile(
            state["snapshots"], cfg=cfg, max_gap=int(max_gap.value)
        )
        state["tracks"] = tracks
        state["profiles"] = profiles
        track_status.value = (
            f"{len(tracks)} tracks, {len(profiles)} length profiles"
        )

        # Show tip trajectories as a Tracks layer (3-D: t, y, x).
        if (
            state["tracks_layer"] is not None
            and state["tracks_layer"] in viewer.layers
        ):
            viewer.layers.remove(state["tracks_layer"])
        track_rows = []
        for tr in tracks:
            for f in tr.frames:
                # Two rows per frame -- one per physical tip.
                track_rows.append(
                    [2 * tr.mt_id, f.frame, f.tip_a[1], f.tip_a[0]]
                )
                track_rows.append(
                    [2 * tr.mt_id + 1, f.frame, f.tip_b[1], f.tip_b[0]]
                )
        if track_rows:
            state["tracks_layer"] = viewer.add_tracks(
                np.asarray(track_rows),
                name="MT tip tracks",
            )
        results_refresh()

    # ---------- Results tab ----------------------------------------------

    results_summary = mw.Label(value="(no results)")
    mt_picker = mw.ComboBox(label="Microtubule", choices=[])
    show_kymograph_button = mw.PushButton(
        text="Show kymograph for selected MT"
    )
    export_dir = mw.FileEdit(label="Export to dir", mode="d")
    export_button = mw.PushButton(text="Export CSVs")
    export_status = mw.Label(value="")
    results_container = mw.Container(
        widgets=[
            results_summary,
            mt_picker,
            show_kymograph_button,
            export_dir,
            export_button,
            export_status,
        ]
    )

    def results_refresh():
        profiles = state.get("profiles", [])
        if not profiles:
            results_summary.value = "(no results)"
            mt_picker.choices = []
            return
        rows = []
        for p in profiles:
            rows.append(
                f"mt {p.mt_id}: {len(p.frames)} frames, "
                f"arc {p.arc_length[0]:.1f}→{p.arc_length[-1]:.1f} px, "
                f"plus={p.plus_was_tip}"
            )
        results_summary.value = "\n".join(rows)
        mt_picker.choices = [(f"mt {p.mt_id}", p.mt_id) for p in profiles]

    @show_kymograph_button.changed.connect
    def _on_show_kymograph():
        profiles = state.get("profiles", [])
        if not profiles:
            return
        sel_id = mt_picker.value
        if sel_id is None:
            return
        prof = next((p for p in profiles if p.mt_id == sel_id), None)
        if prof is None:
            return
        tr = next((t for t in state["tracks"] if t.mt_id == sel_id), None)
        amps = (
            np.array([f.amplitude for f in tr.frames])
            if tr is not None
            else None
        )
        kymo = build_kymograph(prof, n_pos_bins=240, amplitudes=amps)
        # Add as an image layer so napari's own canvas renders it; the user
        # can then take a screenshot or do further analysis.
        viewer.add_image(
            kymo.image,
            name=f"kymograph mt {sel_id}",
            colormap="inferno",
        )

    @export_button.changed.connect
    def _on_export():
        from kapoorlabs_mtrack.io import (
            save_endpoints_csv,
            save_length_profiles_csv,
        )
        from kapoorlabs_mtrack.pipeline.fit_stack import snapshots_to_csv_rows

        out_dir = Path(str(export_dir.value))
        if not out_dir.is_dir():
            export_status.value = "select a valid directory"
            return
        if state["snapshots"]:
            save_endpoints_csv(
                out_dir / "endpoints.csv",
                snapshots_to_csv_rows(state["snapshots"]),
            )
        if state["profiles"]:
            save_length_profiles_csv(
                out_dir / "length_profiles.csv", state["profiles"]
            )
        export_status.value = f"wrote CSVs to {out_dir}"

    # ---------- Assemble tabs -------------------------------------------

    tabs = QTabWidget()
    tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    tabs.addTab(input_container.native, "Input")
    tabs.addTab(fit_container.native, "Fit")
    tabs.addTab(track_container.native, "Track")
    tabs.addTab(results_container.native, "Results")

    outer = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(tabs)
    outer.setLayout(layout)
    return outer
