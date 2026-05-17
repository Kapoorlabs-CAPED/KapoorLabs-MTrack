"""TIFF I/O for raw + label image pairs.

Supports 2-D single frames and 2-D + time stacks. The pipeline treats
any input as a stack: a 2-D image becomes a length-1 stack so the
per-frame loop in :mod:`pipeline.fit_stack` doesn't need a special
case.
"""

from .tif import (
    StackInfo,
    load_pair,
    load_tif_as_stack,
    save_endpoints_csv,
    save_length_profiles_csv,
)

__all__ = [
    "load_pair",
    "load_tif_as_stack",
    "save_endpoints_csv",
    "save_length_profiles_csv",
    "StackInfo",
]
