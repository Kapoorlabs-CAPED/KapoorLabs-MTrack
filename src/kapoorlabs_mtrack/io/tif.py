"""Read raw + label TIFF pairs (2-D or 2-D + time) and write endpoint CSVs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


@dataclass
class StackInfo:
    """What we needed to know about a TIFF after loading it."""

    array: np.ndarray  # always (T, H, W) -- T=1 for a 2-D input
    is_2d_only: bool  # True if the on-disk file was a single 2-D frame


def load_tif_as_stack(path: str | Path) -> StackInfo:
    """Load a TIFF as a ``(T, H, W)`` stack.

    Accepts a 2-D ``(H, W)`` image (returned as ``T=1``) or a 2-D + time
    ``(T, H, W)`` stack. Other shapes raise -- multi-channel and 3-D
    volumes are intentionally out of scope for the v1 microtubule
    pipeline.
    """
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        return StackInfo(array=arr[None, ...], is_2d_only=True)
    if arr.ndim == 3:
        return StackInfo(array=arr, is_2d_only=False)
    raise ValueError(
        f"{path}: expected 2-D (H,W) or 2-D+T (T,H,W) TIFF, got shape {arr.shape}"
    )


def load_pair(
    raw_path: str | Path, label_path: str | Path
) -> tuple[StackInfo, StackInfo]:
    """Load a raw / label TIFF pair, checking shape compatibility."""
    raw = load_tif_as_stack(raw_path)
    lab = load_tif_as_stack(label_path)
    if raw.array.shape != lab.array.shape:
        raise ValueError(
            f"raw and label shapes differ: {raw.array.shape} vs {lab.array.shape}"
        )
    return raw, lab


# Columns of the per-frame, per-MT endpoint snapshot CSV. This is the
# schema the tracker will consume.
ENDPOINT_CSV_COLUMNS = (
    "frame",
    "label",  # source segmentation label id
    "mt_in_label",  # 0..N-1 within that label (1 for single-MT, 2+ for joint fits)
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "ds",
    "curvature",
    "inflection",
    "amplitude",
    "background",
    "fit_cost",
    "n_mt_in_label",
    "status",
)


def save_endpoints_csv(path: str | Path, rows: Iterable[dict]) -> None:
    """Write endpoint snapshots to CSV using :data:`ENDPOINT_CSV_COLUMNS`."""
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ENDPOINT_CSV_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ENDPOINT_CSV_COLUMNS})


# Length-profile CSV: one row per (mt_id, frame), wide format so a
# downstream notebook can pivot straight into a per-MT line plot.
LENGTH_PROFILE_COLUMNS = (
    "mt_id",
    "frame",
    "plus_x",
    "plus_y",
    "minus_x",
    "minus_y",
    "tip_distance",
    "arc_length",
    "plus_was_tip",  # "A" or "B" -- which raw tip was labelled plus
)


def save_length_profiles_csv(path: str | Path, profiles) -> None:
    """Flatten a list of :class:`track.profile.LengthProfile` into CSV rows."""
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LENGTH_PROFILE_COLUMNS))
        w.writeheader()
        for p in profiles:
            for i, fr in enumerate(p.frames):
                w.writerow(
                    {
                        "mt_id": int(p.mt_id),
                        "frame": int(fr),
                        "plus_x": float(p.plus_xy[i, 0]),
                        "plus_y": float(p.plus_xy[i, 1]),
                        "minus_x": float(p.minus_xy[i, 0]),
                        "minus_y": float(p.minus_xy[i, 1]),
                        "tip_distance": float(p.tip_distance[i]),
                        "arc_length": float(p.arc_length[i]),
                        "plus_was_tip": p.plus_was_tip,
                    }
                )
