"""Trivial kymograph reader -- thin wrapper around ``tifffile``."""

from __future__ import annotations

from pathlib import Path

import tifffile


def napari_get_reader(path):
    p = Path(path) if isinstance(path, (str, Path)) else None
    if p is None:
        return None
    if p.is_dir():
        return _read_dir
    if p.suffix.lower() in {".tif", ".tiff"}:
        return _read_single
    return None


def _read_single(path):
    arr = tifffile.imread(str(path))
    name = Path(path).stem
    is_mask = any(t in name.lower() for t in ("mask", "label", "seg"))
    return [(arr, {"name": name}, "labels" if is_mask else "image")]


def _read_dir(path):
    p = Path(path)
    tifs = sorted(
        [f for f in p.iterdir() if f.suffix.lower() in {".tif", ".tiff"}]
    )
    out = []
    for f in tifs:
        arr = tifffile.imread(str(f))
        name = f.stem
        is_mask = any(t in name.lower() for t in ("mask", "label", "seg"))
        out.append((arr, {"name": name}, "labels" if is_mask else "image"))
    return out or None
