"""Vollseg-based kymograph segmentation (optional dependency).

The original ``vollseg-napari-mtrack`` plugin used a UNET model to
segment microtubule traces on a kymograph. We keep that capability
behind a **lazy import** -- ``vollseg`` is *not* a core dependency of
``kapoorlabs_mtrack``. Install it via the ``[vollseg]`` extra
(``pip install KapoorLabs-MTrack[vollseg]``) to enable this path; the
default ``ransac.extract_kymograph_points`` still uses
Otsu + skeletonize and has no extra deps.

Two entry points:

- :func:`segment_kymograph_pretrained` -- pick a name from
  ``vollseg.pretrained.get_registered_models(UNET)``.
- :func:`segment_kymograph_custom`     -- point at a local model
  directory ``<basedir>/<name>/``.

Both return a 2-D ``bool`` mask the same shape as the input
kymograph, ready to feed to ``extract_kymograph_points(..., mask=...)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VollsegNotInstalledError(ImportError):
    """Raised when a vollseg-backed function is called without vollseg."""


def _require_vollseg():
    try:
        from vollseg import UNET, VollSeg
    except ImportError as exc:
        raise VollsegNotInstalledError(
            "vollseg is not installed. `pip install KapoorLabs-MTrack[vollseg]` "
            "to enable UNET-based kymograph segmentation."
        ) from exc
    return UNET, VollSeg


def list_pretrained_models() -> list[tuple[str, str]]:
    """Return ``[(alias_or_name, key), ...]`` for every registered UNET model."""
    UNET, _ = _require_vollseg()
    from vollseg.pretrained import get_registered_models

    models, aliases = get_registered_models(UNET)
    return [((aliases[m][0] if len(aliases[m]) > 0 else m), m) for m in models]


def _run_vollseg(
    model, kymograph: np.ndarray, n_tiles: tuple = (1, 1)
) -> np.ndarray:
    UNET, VollSeg = _require_vollseg()
    res = VollSeg(kymograph, unet_model=model, n_tiles=n_tiles, axes="YX")
    # Original returns (unet_mask, skeleton); we want the mask.
    if isinstance(res, tuple):
        unet_mask = res[0]
    else:
        unet_mask = res
    return np.asarray(unet_mask) > 0


def segment_kymograph_pretrained(
    kymograph: np.ndarray,
    model_name: str,
    n_tiles: tuple = (1, 1),
) -> np.ndarray:
    """Segment with a pretrained vollseg UNET model.

    Args:
        kymograph: 2-D image (time × position).
        model_name: a key from :func:`list_pretrained_models`.
        n_tiles: tiling for large images (passed through to ``VollSeg``).

    Returns: 2-D ``bool`` mask of the same shape.
    """
    UNET, _ = _require_vollseg()
    model = UNET.local_from_pretrained(model_name)
    return _run_vollseg(model, kymograph, n_tiles=n_tiles)


def segment_kymograph_custom(
    kymograph: np.ndarray,
    model_dir: str | Path,
    n_tiles: tuple = (1, 1),
) -> np.ndarray:
    """Segment with a locally-trained vollseg UNET model.

    ``model_dir`` is the directory that contains ``config.json`` /
    ``weights_best.h5`` -- the same layout vollseg's training scripts
    produce.
    """
    UNET, _ = _require_vollseg()
    p = Path(model_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"{p} is not a directory")
    model = UNET(None, name=p.name, basedir=str(p.parent))
    return _run_vollseg(model, kymograph, n_tiles=n_tiles)
