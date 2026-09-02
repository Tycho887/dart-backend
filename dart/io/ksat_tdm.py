"""Canonical public facade for the supported KSAT TDM products.

The legacy :mod:`dart.tdm.legacy` module remains a solver diagnostic format.  This
module deliberately exposes only the reviewed KSAT TRACK mode-4 and ANGLE AZEL
writers used by the service and command line.
"""

from dart.tdm.angle import AngleColumns, AngleRequest, AngleResult, write_angle_tdm
from dart.tdm.ranging import (
    FrequencySource,
    TrackColumns,
    TrackRequest,
    TrackResult,
    write_track_tdm,
)

__all__ = [
    "AngleColumns",
    "AngleRequest",
    "AngleResult",
    "FrequencySource",
    "TrackColumns",
    "TrackRequest",
    "TrackResult",
    "write_angle_tdm",
    "write_track_tdm",
]
