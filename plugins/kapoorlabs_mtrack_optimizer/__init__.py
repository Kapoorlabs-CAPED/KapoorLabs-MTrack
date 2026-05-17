"""KapoorLabs MTrack napari plugin.

Optional companion to the core ``kapoorlabs_mtrack`` package. Install
via ``pip install kapoorlabs-mtrack[napari]`` to pull in napari +
magicgui + Qt deps. The plugin itself ships with the core package's
source tree under ``plugins/`` so users with the core install can
opt-in without a separate distribution.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["__version__"]
