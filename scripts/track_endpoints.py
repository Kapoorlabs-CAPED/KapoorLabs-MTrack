"""Link per-frame endpoint snapshots into tracks and emit length profiles.

Usage:
    python track_endpoints.py \\
        --endpoints endpoints.csv \\
        --out-tracks tracks.csv \\
        --out-profiles length_profiles.csv

Input is the CSV produced by ``fit_endpoints.py`` (schema in
``kapoorlabs_mtrack.io.tif.ENDPOINT_CSV_COLUMNS``). Output is two CSVs:

- ``tracks.csv``           -- per-frame, per-MT track membership
- ``length_profiles.csv``  -- per-frame plus/minus positions + arc length
                              for each MT (the deliverable of stage 10).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from kapoorlabs_mtrack.io import save_length_profiles_csv
from kapoorlabs_mtrack.track import (
    TrackingCost,
    build_length_profiles,
    track_snapshots,
)


@dataclass
class _MinimalSnapshot:
    """Subset of ``MTSnapshot`` fields needed by the tracker."""

    frame: int
    label: int
    mt_in_label: int
    start: np.ndarray
    end: np.ndarray
    amplitude: float
    curvature: float
    ds: float


@dataclass
class _MinimalFrameSnapshot:
    frame: int
    mts: list


def _load_snapshots_from_csv(path: str) -> list[_MinimalFrameSnapshot]:
    """Read ``endpoints.csv`` back into the shape ``track_snapshots`` wants."""
    by_frame: dict[int, list[_MinimalSnapshot]] = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("status", "").startswith("ok"):
                continue  # skip rows that were skipped at fit time
            t = int(row["frame"])
            snap = _MinimalSnapshot(
                frame=t,
                label=int(row["label"]),
                mt_in_label=int(row["mt_in_label"]),
                start=np.array([float(row["start_x"]), float(row["start_y"])]),
                end=np.array([float(row["end_x"]), float(row["end_y"])]),
                amplitude=float(row["amplitude"]),
                curvature=float(row["curvature"]),
                ds=float(row["ds"]),
            )
            by_frame[t].append(snap)
    frames = sorted(by_frame)
    return [_MinimalFrameSnapshot(frame=t, mts=by_frame[t]) for t in frames]


def _save_tracks_csv(path: str, tracks) -> None:
    cols = (
        "mt_id",
        "frame",
        "label_source",
        "mt_in_label",
        "tip_a_x",
        "tip_a_y",
        "tip_b_x",
        "tip_b_y",
        "amplitude",
        "curvature",
        "ds",
    )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for tr in tracks:
            for fr in tr.frames:
                w.writerow(
                    {
                        "mt_id": tr.mt_id,
                        "frame": fr.frame,
                        "label_source": fr.label,
                        "mt_in_label": fr.mt_in_label,
                        "tip_a_x": float(fr.tip_a[0]),
                        "tip_a_y": float(fr.tip_a[1]),
                        "tip_b_x": float(fr.tip_b[0]),
                        "tip_b_y": float(fr.tip_b[1]),
                        "amplitude": fr.amplitude,
                        "curvature": fr.curvature,
                        "ds": fr.ds,
                    }
                )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--endpoints",
        required=True,
        help="endpoints.csv from fit_endpoints.py",
    )
    p.add_argument("--out-tracks", required=True, help="output tracks CSV")
    p.add_argument(
        "--out-profiles", required=True, help="output length-profile CSV"
    )
    p.add_argument(
        "--gate",
        type=float,
        default=50.0,
        help="max linking distance (px). Defaults to 50.",
    )
    p.add_argument(
        "--no-intensity",
        action="store_true",
        help="drop the amplitude-difference term from the cost.",
    )
    p.add_argument(
        "--no-curvature",
        action="store_true",
        help="drop the curvature-difference term from the cost.",
    )
    p.add_argument(
        "--enable-ds",
        action="store_true",
        help="add the |ds_prev - ds_curr| term to the cost.",
    )
    p.add_argument(
        "--velocity-lookback",
        type=int,
        default=2,
        help="frames of history for tip velocity prediction (default 2).",
    )
    p.add_argument(
        "--max-gap",
        type=int,
        default=0,
        help="frames a track can survive without an observation.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrackingCost(
        enable_intensity=not args.no_intensity,
        enable_curvature=not args.no_curvature,
        enable_ds=args.enable_ds,
        gate=float(args.gate) ** 2,
        velocity_lookback=int(args.velocity_lookback),
    )

    frame_snapshots = _load_snapshots_from_csv(args.endpoints)
    n_obs = sum(len(fs.mts) for fs in frame_snapshots)
    print(f"loaded {len(frame_snapshots)} frames, {n_obs} 'ok' observations")

    tracks = track_snapshots(frame_snapshots, cfg=cfg, max_gap=args.max_gap)
    _save_tracks_csv(args.out_tracks, tracks)
    print(f"wrote {len(tracks)} tracks -> {args.out_tracks}")

    profiles = build_length_profiles(tracks)
    save_length_profiles_csv(args.out_profiles, profiles)
    n_rows = sum(len(p.frames) for p in profiles)
    print(f"wrote {n_rows} length-profile rows -> {args.out_profiles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
