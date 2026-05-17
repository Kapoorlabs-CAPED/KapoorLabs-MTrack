"""napari reader for raw + label TIFF pairs.

If the user opens a single TIFF, we load it as one image layer. If
they open a directory, we look for any ``*labels*.tif`` / ``*mask*.tif``
naming convention and pair each raw frame TIFF with its label sibling.
Falling back to "best-effort, load anything that's a TIFF" so dragging
a folder onto napari at least gets the user some data.
"""

from __future__ import annotations

from pathlib import Path

import tifffile


def napari_get_reader(path):
    p = Path(path) if isinstance(path, (str, Path)) else None
    if p is None:
        return None
    if p.is_dir():
        return read_directory
    if p.suffix.lower() in {".tif", ".tiff"}:
        return read_single_tif
    return None


def read_single_tif(path):
    arr = tifffile.imread(str(path))
    name = Path(path).stem
    is_label = any(t in name.lower() for t in ("label", "mask", "seg"))
    layer_type = "labels" if is_label else "image"
    return [(arr, {"name": name}, layer_type)]


def read_directory(path):
    p = Path(path)
    tifs = sorted(
        [f for f in p.iterdir() if f.suffix.lower() in {".tif", ".tiff"}]
    )
    layers = []
    for f in tifs:
        arr = tifffile.imread(str(f))
        name = f.stem
        is_label = any(t in name.lower() for t in ("label", "mask", "seg"))
        layers.append((arr, {"name": name}, "labels" if is_label else "image"))
    return layers or None
