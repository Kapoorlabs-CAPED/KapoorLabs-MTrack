"""Tip tracking across frames + length-profile output.

Stage 9 of the pipeline (see README roadmap). Consumes the per-frame
endpoint snapshots produced by ``pipeline.fit_stack`` and links
microtubules across timepoints with a Hungarian assignment whose cost
function combines distance (mandatory, with 2-frame velocity
prediction), intensity, and curvature -- all toggleable.

Post-tracking, each microtubule track gets its two tips labelled
``plus`` vs ``minus`` by total path length (the more dynamic tip is
plus), and per-frame length profiles are emitted.
"""

from .cost import TrackingCost, mt_to_mt_cost
from .hungarian import MTTrack, TipObservation, track_snapshots
from .kymograph import Kymograph, build_kymograph
from .profile import LengthProfile, build_length_profiles, label_plus_minus

__all__ = [
    "TrackingCost",
    "mt_to_mt_cost",
    "MTTrack",
    "TipObservation",
    "track_snapshots",
    "LengthProfile",
    "build_length_profiles",
    "label_plus_minus",
    "Kymograph",
    "build_kymograph",
]
