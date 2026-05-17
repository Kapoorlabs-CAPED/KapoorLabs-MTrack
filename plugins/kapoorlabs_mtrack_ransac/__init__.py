"""KapoorLabs MTrack RANSAC napari plugin.

Companion plugin to ``kapoorlabs-mtrack-optimizer``. Where the
optimizer fits per-frame spline-Gaussian endpoints on 2-D + time
microscopy data, this plugin operates on **kymograph images** -- one
microtubule per image, with the time axis along rows -- and produces
dynamic-instability statistics (catastrophe / rescue frequencies,
growth + shrinkage rates) via the ``kapoorlabs_mtrack.ransac`` core.

Install with ``pip install KapoorLabs-MTrack[napari]`` and launch via
``napari -w kapoorlabs-mtrack-ransac "MTrack RANSAC"``.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["__version__"]
